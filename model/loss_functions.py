from __future__ import absolute_import, print_function
import torch.nn as nn
from torch.nn import functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self, temp_param, eps=1e-12, reduce=True):
        super(ContrastiveLoss, self).__init__()
        self.temp_param = temp_param
        self.eps = eps
        self.reduce = reduce

    def get_score(self, query, key):
       
        qs = query.size()
        ks = key.size()

        score = torch.matmul(query, torch.t(key))   
        score = F.softmax(score, dim=1) 

        return score
    
    def forward(self, queries, items):

        batch_size = queries.size(0)
        d_model = queries.size(-1)

        loss = torch.nn.TripletMarginLoss(margin=1.0, reduce=self.reduce)

        queries = queries.contiguous().view(-1, d_model)   
        score = self.get_score(queries, items)    

        _, indices = torch.topk(score, 2, dim=1)


        pos = items[indices[:, 0]]  
        neg = items[indices[:, 1]]  
        anc = queries              

        spread_loss = loss(anc, pos, neg)

        if self.reduce:
            return spread_loss
        
        spread_loss = spread_loss.contiguous().view(batch_size, -1)       
        
        return spread_loss     

class GatheringLoss(nn.Module):
    def __init__(self, reduction='none', memto_framework=True):
        super(GatheringLoss, self).__init__()
        self.reduction = reduction
        self.memto_framework = memto_framework

    def get_score(self, query, key):

        score = torch.matmul(query, key.T) 
        score = F.softmax(score, dim=1)  
        return score
    
    def forward(self, queries, items):

        batch_size = queries.size(0)

        loss_mse = torch.nn.MSELoss(reduction=self.reduction)


        f = torch.fft.rfft(queries, dim=-2).permute(0, 2, 1)
        i_query_angle = torch.angle(f)
        unit_magnitude_queries = torch.fft.irfft(torch.exp(-1j * i_query_angle)).permute(0, 2, 1)

        if self.memto_framework:
            score = torch.einsum('bij,kj->bik', unit_magnitude_queries, items)

            _, indices = torch.topk(score, 1, dim=-1)
            step_basis = torch.gather(items.unsqueeze(0).repeat(batch_size, 1, 1), 1, indices.expand(-1, -1, items.size(-1)))
            gathering_loss = loss_mse(queries, step_basis)

        else:
            score = torch.einsum('bij,bkj->bik', unit_magnitude_queries, items)
 
            _, indices = torch.topk(score, 1, dim=-1)
            C = torch.gather(items, 1, indices.expand(-1, -1, items.size(-1)))
            gathering_loss = loss_mse(queries, C)

        if not self.reduction == 'none':
            return gathering_loss
        
        gathering_loss = torch.sum(gathering_loss, dim=-1)  
        gathering_loss = gathering_loss.contiguous().view(batch_size, -1)   

        return gathering_loss


class EntropyLoss(nn.Module):
    def __init__(self, eps=1e-12):
        super(EntropyLoss, self).__init__()
        self.eps = eps
    
    def forward(self, x):
     
        loss = -1 * x * torch.log(x + self.eps)
        loss = torch.sum(loss, dim=-1)
        loss = torch.mean(loss)
        return loss


class NearestSim(nn.Module):
    def __init__(self):
        super(NearestSim, self).__init__()
        
    def get_score(self, query, key):
      
        qs = query.size()
        ks = key.size()

        score = F.linear(query, key)   
        score = F.softmax(score, dim=1) 

        return score
    
    def forward(self, queries, items):

        batch_size = queries.size(0)
        d_model = queries.size(-1)

        queries = queries.contiguous().view(-1, d_model)    
        score = self.get_score(queries, items)     

       
        _, indices = torch.topk(score, 2, dim=1)

    
        pos = F.normalize(items[indices[:, 0]], p=2, dim=-1)  
        anc = F.normalize(queries, p=2, dim=-1)               

        similarity = -1 * torch.sum(pos * anc, dim=-1)        
        similarity = similarity.contiguous().view(batch_size, -1)  
    
        return similarity     

def sce_loss(x, y, alpha=3):
    x = F.normalize(x, p=1, dim=-1)
    y = F.normalize(y, p=1, dim=-1)

    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

    return loss


