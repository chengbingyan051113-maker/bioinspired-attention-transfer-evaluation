"""N3: bounded adaptive contrast, channelwise sign-specific scale mixing.

This is an engineering hypothesis, not a physiological ON/OFF model.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

SCALES=((3,0.7,7,1.4),(5,1.0,9,2.0),(7,1.5,13,3.0))

class N3Attention(nn.Module):
    def __init__(self, channels=24):
        super().__init__()
        if channels != 24:
            raise ValueError('Only the prospectively specified stride-4 configuration is supported')
        self.channels=channels
        self.channel=nn.Sequential(nn.Conv2d(channels,10,1),nn.ReLU(),nn.Conv2d(10,channels,1))
        nn.init.zeros_(self.channel[-1].weight);nn.init.zeros_(self.channel[-1].bias)
        self.scale_raw=nn.Parameter(torch.zeros(3))
        self.rho_raw=nn.Parameter(torch.tensor(math.log(0.5),dtype=torch.float32))
        self.epsilon_raw=nn.Parameter(torch.tensor(0.0,dtype=torch.float32))
        self.on_mix=nn.Conv2d(3*channels,channels,1,groups=channels,bias=False)
        self.off_mix=nn.Conv2d(3*channels,channels,1,groups=channels,bias=False)
        self.gate_bias=nn.Parameter(torch.zeros(channels))
        nn.init.zeros_(self.on_mix.weight);nn.init.zeros_(self.off_mix.weight)
        for k in (3,5,7,9,13):
            axis=torch.arange(k,dtype=torch.float32)-(k-1)/2
            self.register_buffer('radius_'+str(k),axis[:,None].square()+axis[None,:].square())

    def coefficients(self):
        scale=0.75+0.50*torch.sigmoid(self.scale_raw)
        rho=0.05+0.15*torch.sigmoid(self.rho_raw)
        epsilon=torch.pow(self.epsilon_raw.new_tensor(10.0),-4+2*torch.sigmoid(self.epsilon_raw))
        return scale,rho,epsilon

    def kernel(self, width, sigma):
        radius=getattr(self,'radius_'+str(width))
        value=torch.exp(-radius/(2*sigma.square()))
        return (value/value.sum())[None,None]

    def smooth(self, feature, kernel):
        width=kernel.shape[-1]
        if feature.dtype != torch.float32:
            raise RuntimeError('N3 protocol requires FP32')
        return F.conv2d(F.pad(feature,(width//2,)*4,mode='replicate'),
                        kernel.expand(self.channels,1,width,width).contiguous(),groups=self.channels)

    def responses(self, refined):
        scale,rho,epsilon=self.coefficients()
        centered=refined-refined.mean(dim=(-2,-1),keepdim=True)
        global_var=centered.square().mean(dim=(-2,-1),keepdim=True)
        values=[]
        for i,(kc,sc,ks,ss) in enumerate(SCALES):
            center=self.smooth(centered,self.kernel(kc,scale[i]*sc))
            surround_kernel=self.kernel(ks,scale[i]*ss)
            local_mean=self.smooth(centered,surround_kernel)
            local_var=(self.smooth(centered.square(),surround_kernel)-local_mean.square()).clamp_min(0)
            values.append((center-local_mean)/torch.sqrt(local_var+rho.square()*global_var+epsilon.square()))
        # B,C,K,H,W: each grouped mixer receives the three scales of one channel.
        response=torch.stack(values,dim=2).flatten(1,2)
        return F.relu(response),F.relu(-response)

    def forward(self,feature):
        logits=self.channel(F.adaptive_avg_pool2d(feature,1))+self.channel(F.adaptive_max_pool2d(feature,1))
        refined=feature*(2*torch.sigmoid(logits))
        on,off=self.responses(refined)
        gate=self.on_mix(on)+self.off_mix(off)+self.gate_bias[None,:,None,None]
        return refined*(2*torch.sigmoid(gate))

    def audit(self):
        scale,rho,epsilon=self.coefficients()
        return {'scale_multipliers':scale.detach().cpu().tolist(),'rho':float(rho.detach().cpu()),
                'epsilon':float(epsilon.detach().cpu()),'attention_parameters':sum(p.numel() for p in self.parameters()),
                'physiologically_calibrated':False,'lesion_scale_calibrated':False}
