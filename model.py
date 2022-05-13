'''
Copyright (c) 2022 Zen Geng | All rights reserved | Contact gengzichenchin@gmail.com
Version: 1.1
model contain mainly 1 interface:
1. ContrastiveMAE
'''
from audioop import getsample
from sample import SampleGenerator
from libmain import *
from TSFMAE import *

class ContrastiveMAE(nn.Module):
    def __init__(self,
                 d_input = 2,
                 d_spatial = 4,
                 nhead = 2,
                 dim_feedforward = 512,
                 ped_size=81,
                 dropout = 0.2,
                 batch_first = True,
                 device = None) -> None:
        super(ContrastiveMAE, self).__init__()
        
        self.sample_generator = SampleGenerator()
        
        self.MAE = MaskedAutoEncoder(
            d_input=d_input,
            d_spatial=d_spatial,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            ped_size=ped_size,
            dropout=dropout,
            batch_first=batch_first,
            device=device
        )
        self.decoder_layer = TransformerDecoderLayer(
            d_model=d_spatial*ped_size,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            batch_first=batch_first
        )
        self.decoder = Decoder(
            d_input=2,
            d_model=d_spatial*ped_size,
            ped_size=ped_size,
            decoder_layer=self.decoder_layer
        )
    def forward(self, src:Tensor, tgt:Tensor, mask_prop:float):
        ''' Args:
            * src: graph like Tensor in shape [N,S,P+1,2]
            * tgt: Sequence Tensor in shape [N,S,2]
            * mask_prop: float for how much the src is masked.
        '''
        sample = self.get_sample(src)
        represent = self.MAE(src,mask_prop)
        represent_sample = self.MAE(sample,0)
        tgt_mask = get_attn_mask(tgt.size(1))
        recover = self.decoder.forward(tgt,represent,tgt_mask=tgt_mask)
        return represent, recover, represent_sample
    
    def get_sample(self, src:Tensor, neg_num=1):
        ''' Args:
            * src: Tensor in the form of [N,S,P+1,2]
            * neg_num: int, how many neg will be generated.
        '''
        sampleP = src
        sampleN = src.repeat(neg_num, 1, 1, 1)
        for i in range(src.shape[0]):
            neg_sample, pos_sample = self.sample_generator.generate(src[i,:,:,:])
            sampleP[i,:,0:1,:] = Tensor(pos_sample).reshape((src.shape[1], 1, -1))
            sampleN[neg_num*i:neg_num*i+neg_num, :, 0:,:] = neg_sample[:,0:neg_num,:].reshape((neg_num, src.shape[1], 1,src.shape[3]))            
        return torch.cat([sampleP, sampleN])
            
        
    
    def greedy_decode(self, src:Tensor, tgt_len:int):
        '''This function should be called after training'''
        memory = self.MAE(src, mask_prop=0)
        ys = src[:,-1:,0,:] # ys in math[N,S,2]
        for i in range(tgt_len):
            tgt_mask = get_attn_mask(ys.size(1))            
            out = self.decode(ys, memory)
            ys = torch.cat((ys, self.Generator(out)[:,-1:,:]), dim=1)
        return ys[:,1:,:]