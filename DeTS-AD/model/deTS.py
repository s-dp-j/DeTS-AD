
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange
from timm.models.layers import DropPath, trunc_normal_

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


class DeTSABlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio, drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

        original_num_heads = num_heads
        while dim % num_heads != 0 and num_heads > 0:
            num_heads -= 1
        

        if num_heads == 0:
            num_heads = 1
            
        if num_heads != original_num_heads:
            print(f"Warning: Adjusted num_heads from {original_num_heads} to {num_heads} to ensure dim {dim} is divisible")

        self.attn = DualAttention(dim, num_heads, drop_path=drop_path)
        self.mlp = PVT2FFN(dim, int(dim * mlp_ratio))
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        layer_scale_init_value = 1e-6
        self.gamma1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
            requires_grad=True) if layer_scale_init_value > 0 else None
        self.gamma2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
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

    def forward(self, x, semantics, H=None):

        B, L, C = x.shape
        if C != self.norm1.normalized_shape[0]:
            if L == self.norm1.normalized_shape[0]:
                x = x.transpose(1, 2)
                transposed = True
            else:

                raise ValueError(f"Input shape {x.shape} and LayerNorm shape {(B, L, self.norm1.normalized_shape[0])} not match")
        else:
            transposed = False

        _x, semantics = self.attn(self.norm1(x), semantics)
        x = x + self.drop_path(self.gamma1 * _x)
        x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x), H if H is not None else x.shape[1]))

        if transposed:
            x = x.transpose(1, 2)
            
        return x, semantics


class MergeBlock(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()

        self.norm1 = nn.LayerNorm(in_features)
        self.norm2 = nn.LayerNorm(in_features)
        

        num_heads = 4 
        while in_features % num_heads != 0 and num_heads > 0:
            num_heads -= 1
        

        if num_heads == 0:
            num_heads = 1
            
        if num_heads != 4: 
            print(f"Warning: Adjusted num_heads from 4 to {num_heads} to ensure in_features {in_features} is divisible")
            
        self.attn = Attention(in_features, num_heads=num_heads)
        self.mlp = MergeFFN(in_features=in_features, hidden_features=hidden_features)
        self.drop_path = nn.Identity()
        
        layer_scale_init_value = 1e-6
        self.gamma1 = nn.Parameter(layer_scale_init_value * torch.ones((in_features)), 
            requires_grad=True) if layer_scale_init_value > 0 else None
        self.gamma2 = nn.Parameter(layer_scale_init_value * torch.ones((in_features)), 
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

    def forward(self, x, H=None, semantic_len=None, semantics=None):

        B, N, C = x.shape
        

        expected_dim = self.norm1.normalized_shape[0]
        if C != expected_dim:

            projection = nn.Linear(C, expected_dim, device=x.device)
            x = projection(x)
            C = expected_dim

        if semantics is not None:
           
            _, M, C_sem = semantics.shape
            
            if C_sem != expected_dim:
              
                projection = nn.Linear(C_sem, expected_dim, device=semantics.device)
                semantics = projection(semantics)
            
          
            x = torch.cat([x, semantics], dim=1)
            
   
            semantic_len = M
        

        x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x), H if H is not None else x.shape[1], semantic_len))
            
        return x


class PrototypeMatch(nn.Module):
    def __init__(self, dim, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        self.semantic_weight_proj = nn.Linear(dim, 1)

        self.proto_projection = None
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        
    def forward(self, queries, prototypes, global_token=None):

        B, L, C = queries.shape
        
 
        if len(prototypes.shape) == 2:
            if prototypes.shape[1] != C:

                if prototypes.shape[1] == L:
                    prototypes = prototypes.t()

                if prototypes.shape[1] != C:

                    if self.proto_projection is None or self.proto_projection.weight.shape[1] != prototypes.shape[1] or self.proto_projection.weight.shape[0] != C:
                        self.proto_projection = nn.Linear(prototypes.shape[1], C, device=queries.device)
                
                        with torch.no_grad():
                            if prototypes.shape[1] < C:
                          
                                nn.init.zeros_(self.proto_projection.weight)
                                for i in range(min(prototypes.shape[1], C)):
                                    self.proto_projection.weight[i, i] = 1.0
                            else:
                          
                                nn.init.xavier_uniform_(self.proto_projection.weight)
                    
                  
                    prototypes = self.proto_projection(prototypes)
        
   
        if global_token is not None:
       
            semantic_weights = torch.softmax(self.semantic_weight_proj(global_token), dim=1)  
            
            
            score = torch.einsum('blc,nc->bln', queries, prototypes)  
            score = torch.softmax(score / self.temperature, dim=-1)  
            
            
            global_semantic = global_token.mean(dim=1, keepdim=True)  
            proto_semantic_sim = torch.einsum('bkc,nc->bkn', global_semantic, prototypes)  
            proto_weights = torch.softmax(proto_semantic_sim / self.temperature, dim=-1)  
            
            
            weighted_score = score * proto_weights  
            
            
            _, indices = torch.topk(weighted_score, 1, dim=-1)  
            top_protos = torch.gather(prototypes.unsqueeze(0).repeat(B, 1, 1), 1, 
                                     indices.squeeze(-1).unsqueeze(-1).expand(-1, -1, C))  
        
            
            rd_score = torch.sum((queries - top_protos)**2, dim=-1)  
        else:
            
            score = torch.einsum('blc,nc->bln', queries, prototypes) 
            score = torch.softmax(score / self.temperature, dim=-1)  
            
            _, indices = torch.topk(score, 1, dim=-1)  
            top_protos = torch.gather(prototypes.unsqueeze(0).repeat(B, 1, 1), 1, 
                                     indices.squeeze(-1).unsqueeze(-1).expand(-1, -1, C))  
            
            
            rd_score = torch.sum((queries - top_protos)**2, dim=-1)  
            
        return rd_score 
