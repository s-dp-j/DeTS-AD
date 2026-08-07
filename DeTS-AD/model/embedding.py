import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from model.Attn_module import (PositionalEmbedding,
                              AttentionLayer,
                              complex_dropout,
                              complex_operator)
from model.ConvModule import Initi_Block
from model.MA_module import Initial_Att
from model.deTS import DeTSABlock

from model.ModiFy import ModiFy

class EncoderLayer(nn.Module):
    def __init__(self, attn, d_model, d_ff=None, dropout=0.1, activation='relu'):
        super(EncoderLayer, self).__init__()

        self.Attn_module = attn
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.activation = F.relu if activation == 'relu' else F.gelu

    def forward(self, x):
        """
        x : N x L x C(=d_model)
        """

        out, attn = self.Attn_module(x)
        y = complex_dropout(self.dropout, out)

        return y


class Encoder(nn.Module):
    def __init__(self, Attn_modules, norm_layer=None):
        super(Encoder, self).__init__()
        self.Attn_modules = nn.ModuleList(Attn_modules)
        self.norm = norm_layer

    def forward(self, x):
        """
        x : N x L x C(=d_model)
        """

        for Attn_module in self.Attn_modules:
            x, _ = Attn_module(x)

        if self.norm is not None:
            x = self.norm(x)

        return x


class TokenEmbedding(nn.Module):
    def __init__(self, in_dim, d_model, n_window=100, n_layers=1, branch_layers=['st_linear', 'inner_tf'],
                 group_embedding='False', match_dimension='first', kernel_size=[5], multiscale_patch_size=[10, 20],
                 init_type='normal', gain=0.02, dropout=0.1):
        super(TokenEmbedding, self).__init__()

        self.window_size = n_window
        self.d_model = d_model
        self.n_layers = n_layers
        self.branch_layers = branch_layers
        self.group_embedding = group_embedding
        self.match_dimension = match_dimension
        self.kernel_size = kernel_size
        self.multiscale_patch_size = multiscale_patch_size

        component_network = ['real_part', 'imaginary_part']
        num_in_fc_networks = len(component_network)

        self.encoder_layers = nn.ModuleList([])
        self.norm_layers = nn.ModuleList([])


        self.semantic_embeddings = nn.ParameterList([])
        self.use_dual_block = False

        for i, e_layer in enumerate(branch_layers):
            if self.match_dimension == 'none':
                updated_in_dim = in_dim
                extended_dim = in_dim
            elif (i == 0 and self.match_dimension == 'first') or (len(branch_layers) < 2):
                updated_in_dim = in_dim
                extended_dim = d_model
            elif (i == 0) and (not self.match_dimension == 'first'):
                updated_in_dim = in_dim
                extended_dim = in_dim
            elif (i + 1 < len(branch_layers)) and (self.match_dimension == 'middle'):
                updated_in_dim = extended_dim
                extended_dim = d_model
            elif i + 1 == len(branch_layers):
                updated_in_dim = extended_dim
                extended_dim = d_model
            else:
                updated_in_dim = extended_dim
                extended_dim = extended_dim

            if 'conv1d' in e_layer or 'deconv1d' in e_layer:
                if self.group_embedding == 'False':
                    groups = 1
                else:
                    if extended_dim >= updated_in_dim and extended_dim % updated_in_dim == 0:
                        groups = updated_in_dim
                    elif extended_dim < updated_in_dim and updated_in_dim % extended_dim == 0:
                        groups = extended_dim
                    else:
                        print(f"The conv1d/deconv1d layer {i} of encoder is non-grouped convolution!")
                        groups = 1

            if e_layer == 'dropout':
                self.encoder_layers.append(nn.Dropout(p=dropout))
                self.norm_layers.append(nn.Identity())
            elif e_layer == 'st_linear':
                self.encoder_layers.append(nn.ModuleList([nn.Linear(updated_in_dim, extended_dim, bias=False)
                                                         for _ in range(num_in_fc_networks)])
                                           )
                self.norm_layers.append(nn.ModuleList([nn.LayerNorm(extended_dim) for _ in range(num_in_fc_networks)]))
            elif e_layer == 'linear':
                self.encoder_layers.append(nn.ModuleList([nn.Linear(updated_in_dim, extended_dim, bias=False)
                                                          for _ in range(num_in_fc_networks)]))
                self.norm_layers.append(nn.ModuleList([nn.LayerNorm(extended_dim) for _ in range(num_in_fc_networks)]))
            elif e_layer == 'multiatt_conv':
                for _ in range(n_layers):
                    self.encoder_layers.append(Initi_Block(in_channels=updated_in_dim,
                                                               out_channels=extended_dim,
                                                               kernel_list=kernel_size,
                                                               groups=groups
                                                               )
                                                   )
                    self.norm_layers.append(nn.ModuleList([nn.LayerNorm(self.window_size)
                                                           for _ in range(num_in_fc_networks)]))
            elif e_layer == 'in_tf':
                w_model = self.window_size // 2 + 1
        
                self.use_dual_block = True
                
         
                num_heads = 4  
                while extended_dim % num_heads != 0 and num_heads > 0:
                    num_heads -= 1
                
          
                if num_heads == 0:
                    num_heads = 1
                
                self.encoder_layers.append(DeTSABlock(
                    dim=extended_dim,
                    num_heads=num_heads,
                    mlp_ratio=4,
                    drop_path=dropout,
                    norm_layer=nn.LayerNorm
                ))
            
                self.semantic_embeddings.append(nn.Parameter(
                    torch.zeros(1, self.window_size // 10, extended_dim),
                    requires_grad=True
                ))
            
                self.norm_layers.append(nn.LayerNorm(extended_dim))

            elif e_layer == 'inner_tf':
                w_model = self.window_size // 2 + 1
          
                self.use_dual_block = True
                
            
                num_heads = 4  
                while extended_dim % num_heads != 0 and num_heads > 0:
                    num_heads -= 1
                
             
                if num_heads == 0:
                    num_heads = 1
                
                self.encoder_layers.append(DeTSABlock(
                    dim=extended_dim,
                    num_heads=num_heads,
                    mlp_ratio=4,
                    drop_path=dropout,
                    norm_layer=nn.LayerNorm
                ))
       
                self.semantic_embeddings.append(nn.Parameter(
                    torch.zeros(1, self.window_size // 10, extended_dim),
                    requires_grad=True
                ))
           
                self.norm_layers.append(nn.LayerNorm(extended_dim))

            elif e_layer == 'multiscale_att':
                self.encoder_layers.append(Initial_Att(w_size=self.window_size,
                                                                     in_dim=extended_dim,
                                                                     d_model=extended_dim,
                                                                     patch_list=multiscale_patch_size,
                                                                     use_semantics=True))  
       
                self.norm_layers.append(nn.Identity())
            else:
                raise ValueError(f'The specified model {e_layer} is not supported!')

        self.dropout = nn.Dropout(p=dropout)
        self.criterion = nn.MSELoss(reduction='none')
        self.activation = nn.GELU()
  

        for m in self.modules():
            if isinstance(m, nn.Linear):
                if init_type == 'normal':
                    torch.nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    torch.nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    torch.nn.init.orthogonal_(m.weight.data, gain=gain)
                else:
                    torch.nn.init.uniform_(m.weight.data, a=-0.5, b=0.5)

                if hasattr(m, 'bias') and m.bias is not None:
                    torch.nn.init.constant_(m.bias.data, 0.0)

            elif isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):

                if init_type == 'normal':
                    torch.nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    torch.nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    torch.nn.init.orthogonal_(m.weight.data, gain=gain)
                else:
                    torch.nn.init.uniform_(m.weight.data, a=-0.5, b=0.5)
                if hasattr(m, 'bias') and m.bias is not None:
                    torch.nn.init.constant_(m.bias.data, 0.0)

    def forward(self, x):

        B, L, C = x.size()

        latent_list = []
        semantics_list = []  
        attention_maps = {}  

        residual = None

        amplitudeModiFy = ModiFy(int(L//2 + 1))


        dual_block_idx = 0
        

        last_semantics = None


        n_layers = min(len(self.encoder_layers), len(self.norm_layers), len(self.branch_layers))
        
        for i in range(n_layers):
            embedding_layer = self.encoder_layers[i]
            norm_layer = self.norm_layers[i]
            branch_layer = self.branch_layers[i]
            
            if branch_layer not in ['linear', 'st_linear', 'multiscale_att']:
                x = x.permute(0, 2, 1)

            if branch_layer == 'multiatt_conv':
                x = complex_operator(embedding_layer, x)
            elif branch_layer == 'multiscale_att' and last_semantics is not None:
           
                x, attn_maps = embedding_layer(x, semantics=last_semantics)
                attention_maps = attn_maps 
            elif branch_layer == 'multiscale_att':
   
                x, attn_maps = embedding_layer(x)
                attention_maps = attn_maps 
            elif branch_layer in ['st_linear']:
                x = torch.fft.rfft(x, dim=-2)
                x = complex_operator(embedding_layer, x)
                x = torch.fft.irfft(x, dim=-2)
            elif branch_layer in ['in_tf', 'inner_tf']:
            
                x = torch.fft.rfft(x, dim=-1)
                if branch_layer == 'inner_tf':
                    x = x.permute(0, 2, 1)
                
                if not torch.is_complex(x):
               
                    semantics = self.semantic_embeddings[dual_block_idx].expand(B, -1, -1)
                    
                   
                    if hasattr(embedding_layer, 'norm1'):
                        dual_block_dim = embedding_layer.norm1.normalized_shape[0]
                    else:
                    
                        if isinstance(embedding_layer, Initial_Att):
                            dual_block_dim = embedding_layer.d_model
                        else:
                            dual_block_dim = x.shape[2]
                        
                    if x.shape[2] != dual_block_dim:
                    
                        proj = nn.Linear(x.shape[2], dual_block_dim, device=x.device)
                        x = proj(x)
             
                    x, semantics_output = embedding_layer(x, semantics)
                
                    if semantics_output is not None:
                        last_semantics = semantics_output
                    else:
                        last_semantics = semantics
                    semantics_list.append(last_semantics)
                    dual_block_idx += 1
                else:
                    semantics = self.semantic_embeddings[dual_block_idx].expand(B, -1, -1)
                    
             
                    if hasattr(embedding_layer, 'norm1'):
                        dual_block_dim = embedding_layer.norm1.normalized_shape[0]
                    else:
                  
                        if isinstance(embedding_layer, Initial_Att):
                            dual_block_dim = embedding_layer.d_model
                        else:
                            dual_block_dim = x.real.shape[2]
                            
                    if x.real.shape[2] != dual_block_dim:
              
                        proj = nn.Linear(x.real.shape[2], dual_block_dim, device=x.device)
                        x_real_proj = proj(x.real)
                        x_imag_proj = proj(x.imag)
                    else:
                        x_real_proj = x.real
                        x_imag_proj = x.imag
                    
                
                    real_part, real_semantics = embedding_layer(x_real_proj, semantics)
                    imag_part, imag_semantics = embedding_layer(x_imag_proj, semantics)
                    x = torch.complex(real_part, imag_part)
               
                    if real_semantics is not None and imag_semantics is not None:
                        last_semantics = (real_semantics + imag_semantics) / 2
                    else:
                      
                        last_semantics = semantics
                    semantics_list.append(last_semantics)
                    dual_block_idx += 1
                    
                if branch_layer == 'inner_tf':
                    x = x.permute(0, 2, 1)
                x = torch.fft.irfft(x, dim=-1)
                
          
                latent_list.append(x)
          
                if residual is not None:
                    if x.shape == residual.shape and 'transformer' in branch_layer:
                        x += residual
                residual = x
              
                continue
            else:
                x = complex_operator(embedding_layer, x)

            x = complex_operator(norm_layer, x)

            if branch_layer not in ['linear', 'st_linear', 'multiscale_att']:
                x = x.permute(0, 2, 1)

            latent_list.append(x)

            if residual is not None:
                if x.shape == residual.shape and 'transformer' in branch_layer:
                    x += residual

                residual = x

        return x, latent_list, semantics_list, attention_maps 

class InputEmbedding(nn.Module):
    def __init__(self, in_dim, d_model, n_window, device, dropout=0.1, n_layers=1, use_pos_embedding='False',
                 group_embedding='False', kernel_size=5, init_type='kaiming', match_dimension='first',  branch_layers=['linear']):
        super(InputEmbedding, self).__init__()
        self.device = device
        self.token_embedding = TokenEmbedding(in_dim=in_dim, d_model=d_model, n_window=n_window,
                                              n_layers=n_layers, branch_layers=branch_layers,
                                              group_embedding=group_embedding, match_dimension=match_dimension,
                                              init_type=init_type, kernel_size=kernel_size,
                                              dropout=0.1)
        self.pos_embedding = PositionalEmbedding(d_model=d_model)
        self.use_pos_embedding = use_pos_embedding
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = x.to(self.device)

        x, latent_list, semantics_list, attention_maps = self.token_embedding(x)

        if self.use_pos_embedding == 'True':
            x = x + self.pos_embedding(x).to(self.device)

        return self.dropout(x), latent_list, semantics_list, attention_maps


