import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange
from timm.models.layers import DropPath, trunc_normal_
from model.embedding import InputEmbedding
from model.detTS import DeTSABlock, MergeBlock, DualAttention, Attention, PrototypeMatch
from model.Transformer import Decoder



class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv1d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H):
        B, N, C = x.shape
        x = x.transpose(1, 2).contiguous()  
        x = self.dwconv(x)
        x = x.transpose(1, 2).contiguous()  
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()

        original_num_heads = num_heads
        while dim % num_heads != 0 and num_heads > 0:
            num_heads -= 1

        if num_heads == 0:
            num_heads = 1
            
        if num_heads != original_num_heads:
            print(f"Warning: Adjusted num_heads from {original_num_heads} to {num_heads} to ensure dim {dim} is divisible")

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            fan_out = m.kernel_size[0] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class DualAttention(nn.Module):
    def __init__(self, dim, num_heads, drop_path=0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.q_proxy = nn.Linear(dim, dim)
        self.kv_proxy = nn.Linear(dim, dim * 2)
        self.qkv_proxy = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        

        self.proxy_ln = nn.Sequential(nn.LayerNorm(dim))
        self.p_ln = nn.Sequential(nn.LayerNorm(dim))
        self.q_proxy_ln = nn.Sequential(nn.LayerNorm(dim))
        
        self.mlp_proxy = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
        )
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        layer_scale_init_value = 1e-6
        self.gamma1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
            requires_grad=True) if layer_scale_init_value > 0 else None
        self.gamma2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
            requires_grad=True) if layer_scale_init_value > 0 else None
        self.gamma3 = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
            requires_grad=True) if layer_scale_init_value > 0 else None
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            fan_out = m.kernel_size[0] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def selfatt(self, semantics):
        B, N, C = semantics.shape
        
     
        if C != self.dim:
          
            projection = nn.Linear(C, self.dim, device=semantics.device)
            semantics = projection(semantics)
            C = self.dim
   
        qkv = self.qkv_proxy(semantics).reshape(B, -1, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        semantics = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return semantics

    def forward(self, x, semantics):
  
        B_s, N_s, C_s = semantics.shape
        if C_s != self.dim:
     
            projection = nn.Linear(C_s, self.dim, device=semantics.device)
            semantics = projection(semantics)
            C_s = self.dim
            
 
        semantics = semantics + self.drop_path(self.gamma1 * self.selfatt(semantics))


        B, N, C = x.shape
        

        if C != self.dim:

            projection = nn.Linear(C, self.dim, device=x.device)
            x = projection(x)
            C = self.dim
            

        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        

        norm_semantics = semantics
        if C_s != self.dim:
            norm_semantics = nn.LayerNorm(C_s)(semantics)
        else:
            for layer in self.q_proxy_ln:
                norm_semantics = layer(norm_semantics)
                
 
        q_semantics = self.q_proxy(norm_semantics).reshape(B_s, N_s, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)


        kv_semantics = self.kv_proxy(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        kp, vp = kv_semantics[0], kv_semantics[1]
        attn = (q_semantics @ kp.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        _semantics = (attn @ vp).transpose(1, 2).reshape(B, N_s, C) * self.gamma2
        semantics = semantics + self.drop_path(_semantics)
        

        norm_semantics = semantics
        if C_s != self.dim:
            norm_semantics = nn.LayerNorm(C_s)(semantics)
        else:
            for layer in self.p_ln:
                norm_semantics = layer(norm_semantics)
                
    
        semantics = semantics + self.drop_path(self.gamma3 * self.mlp_proxy(norm_semantics))

 
        proxy_semantics = semantics
        if C_s != self.dim:
            proxy_semantics = nn.LayerNorm(C_s)(semantics)
        else:
            for layer in self.proxy_ln:
                proxy_semantics = layer(proxy_semantics)
  
        kv = self.kv(proxy_semantics).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x, semantics


class PVT2FFN(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            fan_out = m.kernel_size[0] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H):
        x = self.fc1(x)
        x = self.dwconv(x, H)
        x = self.act(x)
        x = self.fc2(x)
        return x


class MergeFFN(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)

        self.fc_proxy = nn.Sequential(
            nn.Linear(in_features, 2*in_features),
            nn.GELU(),
            nn.Linear(2*in_features, in_features),
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            fan_out = m.kernel_size[0] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, semantic_len=None):

        if semantic_len is None:
            semantic_len = x.shape[1] - H

        
        sequence_feat = x[:, :H, :]  
        semantics = x[:, H:, :]

      
        semantics = self.fc_proxy(semantics)
        
       
        sequence_feat = self.fc1(sequence_feat)
        sequence_feat = self.dwconv(sequence_feat, H)
        sequence_feat = self.act(sequence_feat)
        sequence_feat = self.fc2(sequence_feat)
        
  
        x = torch.cat([sequence_feat, semantics], dim=1)
        return x


class DeTSAD(nn.Module):
    def __init__(self, win_size, enc_in, dec_in, c_out, d_model=512,
                 dropout=0.0, kernel_size=1, d_ff=512, memory_guided='sinusoid',
                 temperature=0.1, branch1_networks=[], branch2_networks=[],
                 branch1_match_dimension='first', branch2_match_dimension='first',
                 decoder_networks=['linear'], decoder_layers=1, embedding_init='normal',
                 decoder_group_embedding='False', branches_group_embedding='False_False',
                 multiscale_kernel_size=[5], multiscale_patch_size=[10, 20], **kwargs):
  
        super(DeTSAD, self).__init__()
        

        self.base_dim = d_model  
        self.feature_dim = enc_in  
        self.transformer_dim = self.base_dim  
        self.prototype_dim = self.base_dim  
        self.branch1_networks = branch1_networks
        self.branch2_networks = branch2_networks
        self.branch1_match_dimension = branch1_match_dimension
        self.branch2_match_dimension = branch2_match_dimension
        self.win_size = win_size
        self.enc_in = enc_in
        self.d_model = d_model
        self.score_alpha = nn.Parameter(torch.zeros(1))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        

        self.embedding = InputEmbedding(
            in_dim=enc_in, 
            d_model=d_model, 
            n_window=win_size,
            device=self.device,
            dropout=dropout,
            n_layers=1,
            use_pos_embedding='False',
            group_embedding=branches_group_embedding.split('_')[0],
            kernel_size=multiscale_kernel_size,
            init_type=embedding_init,
            match_dimension=branch1_match_dimension,
            branch_layers=branch1_networks
        )
        

        self.feature_prj = nn.Linear(self.transformer_dim, self.feature_dim)

        self.weak_decoder = Decoder(win_size, self.transformer_dim, c_out, 
                                   networks=decoder_networks, n_layers=decoder_layers,
                                   group_embedding=decoder_group_embedding)
        

        self.prototype_matcher = PrototypeMatch(self.prototype_dim, temperature)
        

        self.mem = nn.Parameter(torch.randn(self.prototype_dim, 10))  
        
     
        self.merge_block = MergeBlock(in_features=self.base_dim, hidden_features=self.base_dim*4)
        
       
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            fan_out = m.kernel_size[0] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
   
        B, L, C = x.shape
        input_data = x  
        

        t_query, latent_list, semantics_list, attention_maps = self.embedding(x)
        i_query = t_query  
        semantics = None if not semantics_list else semantics_list[-1]  
        

        if t_query.shape[-1] != self.feature_prj.in_features:
   
            temp_proj = nn.Linear(t_query.shape[-1], self.feature_prj.in_features, device=t_query.device)
            t_query = temp_proj(t_query)
        
    
        t_query_proj = self.feature_prj(t_query)
        out, _ = self.weak_decoder(t_query_proj)
        
   
        if out.shape[1] != input_data.shape[1]:
           
            out = F.interpolate(out.transpose(1, 2), size=input_data.shape[1], mode='linear').transpose(1, 2)
        
 
        td_scores = torch.sum((out - input_data) ** 2, dim=-1)  
        
      
        if semantics is not None:
         
            if i_query.shape[-1] != semantics.shape[-1]:
               
                if i_query.shape[-1] < semantics.shape[-1]:
                   
                    temp_proj = nn.Linear(i_query.shape[-1], semantics.shape[-1], device=i_query.device)
                    i_query = temp_proj(i_query)
                else:
                
                    temp_proj = nn.Linear(semantics.shape[-1], i_query.shape[-1], device=semantics.device)
                    semantics = temp_proj(semantics)
            
         
            merged_features = self.merge_block(i_query, H=i_query.shape[1], semantics=semantics)
            
       
            i_query_enhanced = merged_features[:, :i_query.shape[1], :]
            global_semantics = merged_features[:, i_query.shape[1]:, :]
        else:
        
            i_query_enhanced = i_query
            global_semantics = None
        
   
        mem = self.mem  
        rd_scores = self.prototype_matcher(i_query_enhanced, mem, global_semantics)
        

        if rd_scores.shape[1] != td_scores.shape[1]:
   
            rd_scores = F.interpolate(rd_scores.unsqueeze(1), size=td_scores.shape[1], 
                                     mode='linear', align_corners=False).squeeze(1)

        anomaly_score_method = getattr(self, 'anomaly_score_method', 'product')
        anomaly_score_alpha = getattr(self, 'anomaly_score_alpha', 0.5)
        
        if anomaly_score_method == 'maximum':
           
            anomaly_scores = torch.maximum(td_scores, rd_scores)
        elif anomaly_score_method == 'weighted_sum':
            
            anomaly_scores = anomaly_score_alpha * td_scores + (1 - anomaly_score_alpha) * rd_scores
        elif anomaly_score_method == 'norm_sum':
       
            td_norm = (td_scores - td_scores.mean(dim=1, keepdim=True)) / (td_scores.std(dim=1, keepdim=True) + 1e-10)
            rd_norm = (rd_scores - rd_scores.mean(dim=1, keepdim=True)) / (rd_scores.std(dim=1, keepdim=True) + 1e-10)
            anomaly_scores = td_norm + rd_norm
        elif anomaly_score_method == 'learnable_weight':
        
            alpha = torch.sigmoid(self.score_alpha)  
            anomaly_scores = alpha * td_scores + (1 - alpha) * rd_scores
        elif anomaly_score_method == 'learnable_norm_weight':
            
            alpha = torch.sigmoid(self.score_alpha)  
        
            td_norm = (td_scores - td_scores.mean(dim=1, keepdim=True)) / (td_scores.std(dim=1, keepdim=True) + 1e-10)
            rd_norm = (rd_scores - rd_scores.mean(dim=1, keepdim=True)) / (rd_scores.std(dim=1, keepdim=True) + 1e-10)
            anomaly_scores = alpha * td_norm + (1 - alpha) * rd_norm
        else:
           
            anomaly_scores = td_scores * rd_scores
        

        return {
            'reconstruction': out,
            't_query': t_query,
            'i_query': i_query_enhanced,
            'mem': mem,
            'td_scores': td_scores,
            'rd_scores': rd_scores,
            'anomaly_scores': anomaly_scores,
            'semantics': semantics
        } 
