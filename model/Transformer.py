
import torch.nn as nn
import torch.nn.functional as F
import types
from math import sqrt
from einops import rearrange


from model.embedding import InputEmbedding
from model.deTS import DeTSABlock, MergeBlock, PrototypeMatch


class Decoder(nn.Module):
    def __init__(self, w_size, d_model, c_out, networks=['linear'], n_layers=1,
                 group_embedding='False', kernel_size=[1], patch_size=-1, activation='gelu', dropout=0.0, device='cpu'):
        super().__init__()
        self.networks = networks
        self.device = device
        self.d_model = d_model
        self.w_size = w_size
        self.c_out = c_out

        self.projection = nn.Linear(d_model, c_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
  
        B, L, C = x.shape
        if C != self.projection.weight.shape[1]:
      
            temp_proj = nn.Linear(C, self.projection.weight.shape[1], device=x.device)
      
            with torch.no_grad():
                if C < self.projection.weight.shape[1]:
                  
                    temp_proj.weight.zero_()
                    for i in range(min(C, self.projection.weight.shape[1])):
                        temp_proj.weight[i, i] = 1.0
                else:
                    temp_proj.weight.normal_(0, 0.02)
                
            x = temp_proj(x)
            

        x = self.dropout(x)
        out = self.projection(x)
        return out, None


class TfVar(nn.Module):

    DEFAULTS = {}

    def __init__(self, config, n_heads=1, d_ff=128, dropout=0.3, activation='gelu', gain=0.02):
        super(TfVar, self).__init__()

        self.__dict__.update(TfVar.DEFAULTS, **config)

        
        branch1_group = self.branches_group_embedding.split('_')[0]
        branch2_group = self.branches_group_embedding.split('_')[1]

        branch1_dim = self.input_c if self.branch1_match_dimension == 'none' else self.d_model
        branch2_dim = self.input_c if self.branch2_match_dimension == 'none' else self.d_model


        self.score_alpha = nn.Parameter(torch.zeros(1))
        

        self.encoder_branch1 = InputEmbedding(in_dim=self.input_c, d_model=branch1_dim, n_window=self.win_size,
                                              dropout=dropout, n_layers=self.encoder_layers,
                                              branch_layers=self.branch1_networks,
                                              match_dimension=self.branch1_match_dimension,
                                              group_embedding=branch1_group,
                                              kernel_size=self.multiscale_kernel_size, init_type=self.embedding_init,
                                              device=self.device)

        self.encoder_branch2 = InputEmbedding(in_dim=self.input_c, d_model=branch2_dim, n_window=self.win_size,
                                              dropout=dropout, n_layers=self.encoder_layers,
                                              branch_layers=self.branch2_networks,
                                              match_dimension=self.branch2_match_dimension,
                                              group_embedding=branch2_group,
                                              kernel_size=self.multiscale_kernel_size,
                                              init_type=self.embedding_init, device=self.device)

        self.activate_func = nn.GELU()
        self.dropout = nn.AlphaDropout(p=dropout)
        self.loss_func = nn.MSELoss(reduction='none')


        memory_size = 10
        memory_dim = branch2_dim
        self.mem_R, self.mem_I = create_memory_matrix(N=memory_size,
                                                       L=memory_dim,
                                                      mem_type=self.memory_guided,
                                                       option='option1')

        branch1_out_dim = self.output_c if self.branch1_match_dimension == 'none' else self.d_model
        model_dim = branch1_out_dim


        num_heads = 4  
        while branch2_dim % num_heads != 0 and num_heads > 0:
            num_heads -= 1
        

        if num_heads == 0:
            num_heads = 1
            
        self.dual_block = DeTSABlock(
            dim=branch2_dim,
            num_heads=num_heads,
            mlp_ratio=4,
            drop_path=0.1,
            norm_layer=nn.LayerNorm
        )

        self.merge_block = MergeBlock(
            in_features=branch2_dim,
            hidden_features=branch2_dim * 4
        )


        self.prototype_matcher = PrototypeMatch(
            dim=branch2_dim,
            temperature=self.temperature
        )
        

        self.weak_decoder = Decoder(w_size=self.win_size,
                                    d_model=model_dim,
                                    c_out=self.output_c,
                                    networks=self.decoder_networks,
                                    n_layers=self.decoder_layers,
                                    group_embedding=self.decoder_group_embedding,
                                    kernel_size=self.multiscale_kernel_size,
                                    activation='gelu',
                                    dropout=0.0,   
                                    device=self.device)

        if self.branch1_match_dimension == 'none':
            self.feature_prj = lambda x: x
        else:
            self.feature_prj = nn.Linear(branch1_out_dim, self.output_c)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                if self.embedding_init == 'normal':
                    torch.nn.init.normal_(m.weight.data, 0.0, gain)
                elif self.embedding_init == 'xavier':
                    torch.nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif self.embedding_init == 'kaiming':
                    torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif self.embedding_init == 'orthogonal':
                    torch.nn.init.orthogonal_(m.weight.data, gain=gain)
                else:
                    torch.nn.init.uniform_(m.weight.data, a=-0.5, b=0.5)

                if hasattr(m, 'bias') and m.bias is not None:
                    torch.nn.init.constant_(m.bias.data, 0.0)

            elif isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                if self.embedding_init == 'normal':
                    torch.nn.init.normal_(m.weight.data, 0.0, gain)
                elif self.embedding_init == 'xavier':
                    torch.nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif self.embedding_init == 'kaiming':
                    torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif self.embedding_init == 'orthogonal':
                    torch.nn.init.orthogonal_(m.weight.data, gain=gain)
                else:
                    torch.nn.init.uniform_(m.weight.data, a=-0.5, b=0.5)

                if hasattr(m, 'bias') and m.bias is not None:
                    torch.nn.init.constant_(m.bias.data, 0.0)

    def forward(self, input_data, mode='train'):

        B, L, C = input_data.shape
        z1 = z2 = input_data

        t_query, t_latent_list, t_semantics_list, attention_maps = self.encoder_branch1(z1)


        i_query, _, i_semantics_list, _ = self.encoder_branch2(z2)

        semantic_size = max(1, L//10)
        semantics = torch.zeros((B, semantic_size, i_query.shape[2]), device=self.device)

        i_query, semantics = self.dual_block(i_query, semantics, L)
 
        merged_features = self.merge_block(i_query, L, semantics=semantics)

        i_query_enhanced = merged_features[:, :L, :]

        if semantics is not None and merged_features.shape[1] > L:
            global_semantics = merged_features[:, L:, :]
        else:

            global_semantics = semantics

        if t_semantics_list and len(t_semantics_list) > 0:
    
            t_semantics = t_semantics_list[-1]
            

            if t_semantics.shape[1] != i_query_enhanced.shape[1]:
                resized_t_semantics = torch.zeros((B, i_query_enhanced.shape[1], t_semantics.shape[2]), device=i_query_enhanced.device)
                for b in range(B):
                    for c in range(t_semantics.shape[2]):
                        resized_t_semantics[b, :, c] = F.interpolate(
                            t_semantics[b, :, c].unsqueeze(0).unsqueeze(0),
                            size=i_query_enhanced.shape[1],
                            mode='linear',
                            align_corners=False
                        ).squeeze()
                t_semantics = resized_t_semantics
            

            if t_semantics.shape[2] != i_query_enhanced.shape[2]:
                projection = nn.Linear(t_semantics.shape[2], i_query_enhanced.shape[2], device=i_query_enhanced.device)
                t_semantics = projection(t_semantics)
            

            beta = 0.3 
            i_query_enhanced = (1 - beta) * i_query_enhanced + beta * t_semantics

        mem = self.mem_R.T.to(self.device)
        rd_scores = self.prototype_matcher(i_query_enhanced, mem, global_semantics)


        B_t, L_t, C_t = t_query.shape
        

        if isinstance(self.feature_prj, types.LambdaType):
            combined_z = self.feature_prj(t_query)
        else:

            if hasattr(self.feature_prj, 'weight'):
                input_dim = self.feature_prj.weight.shape[1]
                if C_t != input_dim:
                    temp_proj = nn.Linear(C_t, input_dim, device=t_query.device)
           
                    with torch.no_grad():
                        if C_t < input_dim:
                        
                            temp_proj.weight.zero_()
                            for i in range(min(C_t, input_dim)):
                                temp_proj.weight[i, i] = 1.0
                        else:
                        
                            temp_proj.weight.zero_()
                            for i in range(input_dim):
                                temp_proj.weight[i, i] = 1.0
                    t_query = temp_proj(t_query)
            
            
            combined_z = self.feature_prj(t_query)

        out, _ = self.weak_decoder(combined_z)


        B_o, L_o, C_o = out.shape
        if L_o != L:
  
            out_resized = torch.zeros((B_o, L, C_o), device=out.device)
            for b in range(B_o):
                for c in range(C_o):
                    out_resized[b, :, c] = F.interpolate(
                        out[b, :, c].unsqueeze(0).unsqueeze(0),
                        size=L,
                        mode='linear',
                        align_corners=False
                    ).squeeze()
            out = out_resized
        

        td_scores = torch.sum((out - input_data) ** 2, dim=-1)  
        
 
        if rd_scores.shape[1] != td_scores.shape[1]:
  
            rd_scores_resized = torch.zeros((B, L), device=rd_scores.device)
            for b in range(B):
                rd_scores_resized[b] = F.interpolate(
                    rd_scores[b].unsqueeze(0).unsqueeze(0),
                    size=L,
                    mode='linear',
                    align_corners=False
                ).squeeze()
            rd_scores = rd_scores_resized
        
   
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
            "out": out,
            "queries": i_query_enhanced,
            "mem": mem,
            "td_scores": td_scores,
            "rd_scores": rd_scores,
            "anomaly_scores": anomaly_scores,
            "semantics": global_semantics,
            "attention_maps": attention_maps
        }

    def get_attn_score(self, query, key, scale=None):

        scale = 1. / sqrt(query.size(-1)) if scale is None else 1. / scale

        attn = torch.matmul(query, torch.t(key.to(self.device)))

        attn = attn * scale


        return attn

def generate_rolling_matrix(input_matrix):
    F, L = input_matrix.size()

    output_matrix = torch.empty(L, F, L)


    for step in range(L):

        rolled_matrix = input_matrix.roll(shifts=step, dims=1)

        output_matrix[step] = rolled_matrix

    return output_matrix

def create_memory_matrix(N, L, mem_type='sinusoid', option='option1'):

    with torch.no_grad():
        if mem_type  == 'sinusoid' or mem_type  == 'cosine_only':
            row_indices = torch.arange(N).reshape(-1, 1)
            col_indices = torch.arange(L)
            grid = row_indices * col_indices

            init_matrix_r = torch.cos((1 / L) * 2 * torch.tensor([torch.pi]) * grid)
            init_matrix_i = torch.sin((1 / L) * 2 * torch.tensor([torch.pi]) * grid)
        elif mem_type  == 'uniform' or mem_type  == 'uniform_only':
            init_matrix_r = torch.rand((N, L), dtype=torch.float)
            init_matrix_i = torch.rand((N, L), dtype=torch.float)
        elif mem_type  == 'orthogonal_uniform' or mem_type  == 'orthogonal_uniform_only':
            init_matrix_r = torch.nn.init.orthogonal_(torch.rand((N, L), dtype=torch.float))
            init_matrix_i = torch.nn.init.orthogonal_(torch.rand((N, L), dtype=torch.float))
        elif mem_type  == 'normal' or mem_type  == 'normal_only':
            init_matrix_r = torch.randn((N, L), dtype=torch.float)
            init_matrix_i = torch.randn((N, L), dtype=torch.float)
        elif mem_type  == 'orthogonal_normal' or mem_type  == 'orthogonal_normal_only':
            init_matrix_r = torch.nn.init.orthogonal_(torch.randn((N, L), dtype=torch.float))
            init_matrix_i = torch.nn.init.orthogonal_(torch.randn((N, L), dtype=torch.float))

        if option == 'option4':
            init_matrix_r = generate_rolling_matrix(init_matrix_r)
            init_matrix_i = generate_rolling_matrix(init_matrix_i)

        if 'only' not in mem_type:
            return init_matrix_r, init_matrix_i
        else:
            return init_matrix_r, torch.zeros_like(init_matrix_r)
