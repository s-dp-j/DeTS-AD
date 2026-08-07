import torch
import torch.nn.functional as F
from einops import rearrange
from model.Attn_module import AttentionLayer

class Initial_Att(nn.Module):
    def __init__(self, w_size, in_dim, d_model, patch_list=[10, 20], init_weight=True, use_semantics=False):
        super(Initial_Att, self).__init__()
        self.w_size = w_size
        self.in_dim = in_dim
        self.d_model = d_model
        self.patch_list = patch_list
        self.use_semantics = use_semantics
        
        self.padded_sizes = {}
        self.patch_numbers = {}
        
        patch_attention_layers = []
        linear_layers = []
        
        for patch_size in self.patch_list:
           
            if w_size % patch_size != 0:
                padded_w_size = w_size + (patch_size - (w_size % patch_size))
            else:
                padded_w_size = w_size
                
            patch_number = padded_w_size // patch_size
            
       
            self.padded_sizes[patch_size] = padded_w_size
            self.patch_numbers[patch_size] = patch_number
            
         
            patch_attention_layers.append(
                AttentionLayer(
                    w_size=patch_number,
                                          d_model=patch_size,
                                          n_heads=1
                                                         )
                                          )
            linear_layers.append(nn.Linear(patch_size, patch_number))
            
        self.patch_attention_layers = nn.ModuleList(patch_attention_layers)
        self.linear_layers = nn.ModuleList(linear_layers)
        
        
        if self.use_semantics:
        
            self.semantic_proj = nn.Linear(d_model, d_model)  
            self.feature_proj = nn.Linear(d_model, d_model)   
            self.gate_net = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
                nn.Sigmoid()
            )
            
        if init_weight:
            self._initialize_weights()
        self.attention_maps = {} 

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, semantics=None):
        B, L, C = x.size()
        res_list = []
        
       
        if semantics is not None and self.use_semantics:
       
            if semantics.shape[1] != L:
             
                resized_semantics = torch.zeros((B, L, semantics.shape[2]), device=x.device)
                for b in range(B):
                    for c in range(semantics.shape[2]):
                        resized_semantics[b, :, c] = torch.nn.functional.interpolate(
                            semantics[b, :, c].unsqueeze(0).unsqueeze(0),
                            size=L,
                            mode='linear',
                            align_corners=False
                        ).squeeze()
                semantics = resized_semantics
            
          
            if semantics.shape[2] != C:
          
                projection = nn.Linear(semantics.shape[2], C, device=x.device)
                semantics = projection(semantics)
            
   
            if x.shape[2] != self.d_model:
      
                tmp_feature_proj = nn.Linear(x.shape[2], self.d_model, device=x.device)
                x_proj = tmp_feature_proj(x)
            else:
                x_proj = self.feature_proj(x)
            
      
            if semantics.shape[2] != self.d_model:
            
                tmp_semantic_proj = nn.Linear(semantics.shape[2], self.d_model, device=x.device)
                semantics_proj = tmp_semantic_proj(semantics)
            else:
                semantics_proj = self.semantic_proj(semantics)
            
            gate = self.gate_net(x_proj + semantics_proj)
            

            if x.shape[2] != self.d_model:
                gate_proj = nn.Linear(self.d_model, x.shape[2], device=x.device)
                gate = gate_proj(gate)
            

            x = x * gate + semantics * (1 - gate)  
        
        attn_maps = {}
        for i, p_size in enumerate(self.patch_list):
            padded_size = self.padded_sizes[p_size]
            

            if padded_size > L:
                padding_size = padded_size - L
                padding = torch.zeros(B, padding_size, C, device=x.device)
                padded_x = torch.cat([x, padding], dim=1)
            else:
                padded_x = x
            

            transposed = False  
            try:
  
                if padded_x.shape[1] % p_size == 0:
   
                    z = rearrange(padded_x, 'b (w p) c -> (b c) w p', p=p_size)
                else:
       
                    padded_x_t = padded_x.transpose(1, 2)  
                    
 
                    if padded_x_t.shape[1] % p_size == 0:
              
                        z = rearrange(padded_x_t, 'b (w p) c -> (b c) w p', p=p_size)
                        transposed = True
                    else:
              
                        target_dim = (padded_x.shape[1] // p_size) * p_size
                        if target_dim == 0:
                            target_dim = p_size
                        
          
                        resized_x = torch.zeros((B, target_dim, C), device=padded_x.device)
                        for b in range(B):
                            for c in range(C):
                                resized_x[b, :, c] = F.interpolate(
                                    padded_x[b, :, c].unsqueeze(0).unsqueeze(0),
                                    size=target_dim,
                                    mode='linear',
                                    align_corners=False
                                ).squeeze()
                        
                        z = rearrange(resized_x, 'b (w p) c -> (b c) w p', p=p_size)
                        transposed = False
            except Exception as e:
                print(f"Error in rearrange: input shape={padded_x.shape}, p_size={p_size}")
                print(f"Exception details: {str(e)}")
                
      
                fallback_linear = nn.Linear(padded_x.shape[1], padded_x.shape[1], device=padded_x.device)
                z = fallback_linear(padded_x.transpose(1, 2)).transpose(1, 2)
                res_list.append(z)
                continue
                
            _, attn = self.patch_attention_layers[i](z)
            attn_maps[p_size] = attn.detach().cpu() 
            
   
            try:
                z = self.linear_layers[i](z)
            except Exception as e:
                print(f"Error in linear layer: z.shape={z.shape}, linear weight shape={self.linear_layers[i].weight.shape}")

                tmp_linear = nn.Linear(z.shape[-1], z.shape[-1], device=z.device)
                z = tmp_linear(z)
 
            if transposed:
             
                actual_b_c = z.shape[0]  
             
                possible_bs = [i for i in range(1, actual_b_c+1) if actual_b_c % i == 0]
   
                b_factor = min(possible_bs, key=lambda x: abs(x-B))
                c_factor = actual_b_c // b_factor
                
                z = rearrange(z, '(b c) w p -> b c (w p)', b=b_factor, c=c_factor)
  
                z = z.transpose(1, 2) 
            else:
           
                actual_b_c = z.shape[0] 
       
                possible_bs = [i for i in range(1, actual_b_c+1) if actual_b_c % i == 0]

                b_factor = min(possible_bs, key=lambda x: abs(x-B))
                c_factor = actual_b_c // b_factor
                
                z = rearrange(z, '(b c) w p -> b (w p) c', b=b_factor, c=c_factor)
            

            if z.shape[1] != L:

                resized_z = torch.zeros((B, L, C), device=z.device)
                for b in range(B):
                    for c in range(C):
                        resized_z[b, :, c] = F.interpolate(
                            z[b, :, c].unsqueeze(0).unsqueeze(0),
                            size=L,
                            mode='linear',
                            align_corners=False
                        ).squeeze()
                z = resized_z
                
            res_list.append(z)
            
  
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res, attn_maps



if __name__ == "__main__":
    kernel_sizes = [3, 60, 16]  
    model = Initial_Att(w_size=60, in_dim=16, d_model=32)

    input_tensor = torch.randn(3, 60, 32)  
    output, attn_maps = model(input_tensor)

