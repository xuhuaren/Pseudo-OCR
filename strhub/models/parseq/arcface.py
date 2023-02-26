import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Parameter



# class ArcFace_bak(nn.Module):
#     def __init__(self, cin, cout, s=32, m=0.5):
#         super().__init__()
#         self.m = m
#         self.s = s
#         self.cout = cout
#         self.fc = nn.Linear(cin, cout, bias=False)                              

#     def forward(self, x, label=None):
#         if label is None:
#             w_L2 = torch_norm(self.fc.weight.detach(), dim=1, keepdim=True)
#             x_L2 = torch_norm(x, dim=1, keepdim=True)
#             logit = F.linear(x / x_L2, self.fc.weight / w_L2)
#         else:
#             one_hot = F.one_hot(label, num_classes=self.cout)
#             w_L2 = torch_norm(self.fc.weight.detach(), dim=1, keepdim=True)
#             x_L2 = torch_norm(x, dim=1, keepdim=True)
#             cos = F.linear(x / x_L2, self.fc.weight / w_L2)
#             theta_yi = torch.acos(cos * one_hot)
#             logit = torch.cos(theta_yi + self.m) * one_hot + cos * (1 - one_hot)
#             logit = logit * self.s

#         return logit
        
        
class Arcface(Module):
    def __init__(self, embedding_size=128, num_classes=10575, s=6., m=6):
        super().__init__()
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label=None):
        #print(input.shape)
        #input = input.flatten(end_dim=1)
        if label is None:
            return F.linear(F.normalize(input, dim = -1), F.normalize(self.weight)) 
        else:
            print(input.shape)
            print(self.weight.shape)
            print(label.shape)
            assert(0)
        
            cosine  = F.linear(F.normalize(input, dim = -1), F.normalize(self.weight))
            #print(cosine.shape)
            cosine = cosine.flatten(end_dim=1)
            label = label.flatten()
            sine    = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
            phi     = cosine * self.cos_m - sine * self.sin_m
            phi     = torch.where(cosine.float() > self.th, phi.float(), cosine.float() - self.mm)

    
            one_hot = torch.zeros((cosine.size()[0], cosine.size()[1] + 2)).type_as(phi).long()
 
            one_hot.scatter_(1, label.view(-1, 1).long(), 1)
            one_hot = one_hot[:, :-2]

            output  = (one_hot * phi) + ((1.0 - one_hot) * cosine) 
            #print(output.shape)
            output  *= self.s
            return output
