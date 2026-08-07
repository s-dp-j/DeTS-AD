
import torch.nn as nn


class Initi_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_list=[1, 3, 5], groups=1, init_weight=True):
        super(Initi_Block, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_list = kernel_list
        kernels = []
        for i in self.kernel_list:
            kernels.append(nn.Conv1d(in_channels,
                                     out_channels,
                                     kernel_size=i,
                                     padding='same',
                                     padding_mode='circular',
                                     bias=False,
                                     groups=groups))
        self.convs = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, *args, **kwargs):
   
        res_list = []
        for i in range(len(self.kernel_list)):
  
            if x.shape[1] != self.in_channels:
           
                temp_proj = nn.Conv1d(
                    in_channels=x.shape[1],
                    out_channels=self.in_channels,
                    kernel_size=1,
                    bias=False,
                    device=x.device
                )
         
                with torch.no_grad():
                    if x.shape[1] < self.in_channels:
            
                        temp_proj.weight.zero_()
                        for j in range(min(x.shape[1], self.in_channels)):
                            temp_proj.weight[j, j, 0] = 1.0
                    else:
                
                        temp_proj.weight.zero_()
                        for j in range(self.in_channels):
                            temp_proj.weight[j, j, 0] = 1.0
                            
                x_proj = temp_proj(x)
                res_list.append(self.convs[i](x_proj))
            else:
                res_list.append(self.convs[i](x))
                
        res = torch.stack(res_list, dim=-1).mean(-1)
        
    
        if 'semantics' in kwargs or len(args) > 0:
            return res, None
        return res



if __name__ == "__main__":
    kernel_sizes = [3, 5, 7] 
    model = Initi_Block(in_channels=3,
                            out_channels=6,
                            kernel_list=kernel_sizes,
                            groups=3
                            )
    print(model)


    input_tensor = torch.randn(5, 3, 32)  
    output = model(input_tensor)
    print(output.shape)  
