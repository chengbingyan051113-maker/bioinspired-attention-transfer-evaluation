"""Active-capacity controls. Compound engineering interventions, not physiological isolation."""
import torch
from torch import nn
from torch.nn import functional as F
from n3_attention import N3Attention
class GenericBudgetAttention(nn.Module):
 def __init__(self,channels=24):
  super().__init__()
  if channels!=24:raise ValueError('Require 24 stride-4 channels')
  self.channels=channels;self.channel=nn.Sequential(nn.Conv2d(channels,10,1),nn.ReLU(),nn.Conv2d(10,channels,1))
  nn.init.zeros_(self.channel[-1].weight);nn.init.zeros_(self.channel[-1].bias)
  self.kernel5=nn.Parameter(torch.zeros(1,1,5,5));self.kernel7=nn.Parameter(torch.zeros(1,1,7,7))
  with torch.no_grad():self.kernel5[0,0,2,2]=1;self.kernel7[0,0,3,3]=1
  self.spatial=nn.Conv2d(2,1,7,padding=3);nn.init.zeros_(self.spatial.weight);nn.init.zeros_(self.spatial.bias)
 def smooth(self,value,kernel):
  k=kernel.shape[-1]
  return F.conv2d(F.pad(value,(k//2,)*4,mode='replicate'),kernel.expand(self.channels,1,k,k).contiguous(),groups=self.channels)
 def forward(self,feature):
  logits=self.channel(F.adaptive_avg_pool2d(feature,1))+self.channel(F.adaptive_max_pool2d(feature,1));refined=feature*(2*torch.sigmoid(logits));spatial=refined
  for kernel,count in ((self.kernel5,2),(self.kernel7,13)):
   for _ in range(count):spatial=torch.tanh(self.smooth(spatial,kernel))
  stats=torch.cat((spatial.mean(1,keepdim=True),spatial.max(1,keepdim=True).values),dim=1)
  return refined*(2*torch.sigmoid(self.spatial(stats)))
class BudgetN3Attention(N3Attention):
 def __init__(self,channels=24,mode='late_pooled'):
  super().__init__(channels)
  if mode not in ('late_pooled','magnitude_basis'):raise ValueError('Unknown mode')
  self.mode=mode
 def responses(self,refined):
  on,off=super().responses(refined)
  if self.mode=='late_pooled':
   b,_,h,w=on.shape
   def pool(v):
    bank=v.reshape(b,self.channels,3,h,w)
    return bank.mean(1,keepdim=True).expand_as(bank).reshape_as(v)
   return pool(on),pool(off)
  magnitude=on+off;return magnitude,magnitude.square()
 def audit(self):
  a=super().audit();a.update(mode=self.mode,response_channels_before_filtering=self.channels,identical_unsigned_inputs=False,scope='compound budget control');return a
def make_budget_attention(variant):
 if variant=='M1_budget_s4':return GenericBudgetAttention()
 return BudgetN3Attention(mode={'N3_late_pooled_s4':'late_pooled','N3_magnitude_basis_s4':'magnitude_basis'}[variant])
