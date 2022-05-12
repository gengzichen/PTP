'''
Copyright (c) 2022 Zen Geng | All rights reserved | Contact gengzichenchin@gmail.com
Version: 1.1
MAE contain mainly 1 interface:
1. MaskedAutoEncoder
2. SpatialEncoder
3. TemporalEncoder
4. Decoder
'''

from torch import DoubleTensor, batch_norm, float32
from libmain import *

class MaskedAutoEncoder(nn.Module):
    def __init__(self,
                 d_input = 2,
                 d_spatial = 16, 
                 nhead = 4,
                 dim_feedforward = 2048,
                 ped_size=81,
                 dropout = 0.2,
                 batch_first = True,
                 device = None
                 ) -> None:
        super(MaskedAutoEncoder, self).__init__()
        ''' Args:
            * d_input: input size of last dim, 2 by default.
            * d_spatial: embedding dim for spatial encoder.
            * nhead: head number of multihead attention in TransformerEncoder(decoder)
            * dim_feedforard: ffn layer dim in Transformer en(de)coder layer.
            * ped_size: max pedestrian number, 80+1 by default, eq [src.shape[2]]
            * dropout: dropout ratio in transformer en(de)coder layer.
            * batch_first: True for default, src in math:[N, L, P, E]
        '''
        self.pre_embedding = torch.nn.Linear(d_input, d_spatial)
        self.spatial_encoder_layer = TransformerEncoderLayer(
            d_model=d_spatial, dim_feedforward=dim_feedforward,
            nhead=nhead, batch_first=batch_first, 
            dropout=dropout, device=device,
            layer_norm_eps=0.000001
        )
        self.temporal_encoder_layer = TransformerEncoderLayer(
            d_model=d_spatial*ped_size, dim_feedforward=dim_feedforward,
            nhead=nhead, batch_first=batch_first,
            dropout=dropout, device=device,
            layer_norm_eps=0.000001
        )
        self.spatial_encoder = SpatialEncoder(encoder_layer=self.spatial_encoder_layer)
        self.temporal_encoder = TemporalEncoder(
            emb_size=ped_size * d_spatial,
            encoder_layer=self.temporal_encoder_layer)

    def _random_mask(self, observation, rseed=100, mask_prop=0.2):
        '''Input should in the form of [N, S, Ped+1,2]'''
        batch_size = observation.shape[0]
        seq_len = observation.shape[1]
        for i in range(batch_size):
            masked_index = torch.randperm(seq_len)[0:int(mask_prop*seq_len)]
            for j in masked_index:
                observation[i, j, 0,:] = torch.tensor([0.0,0.0]).to(float32)
        return observation

    def forward(self, sequence, masked_prop):
        ''' *sequence :math[N, S, Ped+1, 2], N for batch size, S for sequence length,
             Ped+1 is the Pedestrian number plus agent itself. 2 for [x,y]
            * mask_prop: mask proportion, if 0.0, no frame will be masked.
        '''

        masked_sequence = self._random_mask(sequence, mask_prop=masked_prop)

        spatial_padding_mask = get_padding_mask(
            sequence.reshape((sequence.shape[0]*sequence.shape[1],sequence.shape[2],-1)))
        
        pre_embedding = self.pre_embedding(masked_sequence)
        spatial_embedding = self.spatial_encoder.forward(
            src=pre_embedding,
            src_key_padding_mask=spatial_padding_mask)
        spatial_temporal_embedding = self.temporal_encoder(spatial_embedding)
        return spatial_temporal_embedding
        
    def extractEncoder(self):
        return self.unmaskedEncoder


class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        '''X (input embedding) in the form of [N,L,F]'''
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(0)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, input_embedding: Tensor):
        return self.dropout(input_embedding + self.pos_embedding[:, :input_embedding.size(1), :])


class SpatialEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers = 1, norm = None) -> None:
        super(SpatialEncoder, self).__init__()
        '''
        Spatial Encoder needs an external linear embedding layer ahead.
        It do not need positional encoding because spatial information is
        not in order. This encoder needs key padding mask to avoid extensive
        zeros padding. 
        '''
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.flattern = torch.nn.Flatten(start_dim=0, end_dim=1)
        
    def forward(self, src:torch.Tensor, src_key_padding_mask):
        batch_size = src.shape[0]
        seq_len = src.shape[1]
        output = self.flattern(src)
        for mod in self.layers:
            output = mod(output, src_mask=None,
                         src_key_padding_mask=src_key_padding_mask)
        if self.norm is not None:
            output = self.norm(output)
        return output.reshape((batch_size, seq_len,output.shape[-2],output.shape[-1]))
    
    
class TemporalEncoder(nn.Module):
    def __init__(self, emb_size, encoder_layer, num_layers = 1, norm = None) -> None:
        '''
        This encoder needs positional encoding and needs no padding mask
        emb_size should == flatten last dim
        '''
        super(TemporalEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.flatten = torch.nn.Flatten(start_dim=-2, end_dim=-1)
        self.emb_size = emb_size
        self.pos_encoding = PositionalEncoding(emb_size, 0.2, 5000)

    def forward(self, src:torch.Tensor):
        output = self.flatten(src)
        if output.shape[-1] != self.emb_size: print(output.shape,self.emb_size)
        assert output.shape[-1] == self.emb_size
        output = self.pos_encoding(output)
        
        for mod in self.layers:
            output = mod(output, src_mask=None, src_key_padding_mask=None)
        if self.norm is not None:
            output = self.norm(output)
        return output


class Decoder(nn.Module):
    def __init__(self, d_input, d_model, ped_size,
                 decoder_layer, num_layers = 6) -> None:
        super(Decoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.flatten = torch.nn.Flatten(start_dim=-2, end_dim=-1)
        self.embedding_layer = torch.nn.Linear(d_input*ped_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, 0.2, 5000)
        self.ffn = torch.nn.Linear(d_model,d_input)
        
    def forward(self, tgt:Tensor, memory:Tensor,
                tgt_mask:Tensor, tgt_key_padding_mask=None):
        output = self.embedding_layer(self.flatten(tgt))
        for mod in self.layers:
            output = mod(output, memory,tgt_mask=tgt_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask)
        return self.ffn(output)

    
def _get_clones(module, N):
    return torch.nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def get_padding_mask(key_seq):
    ''' If specified, a mask of shape :math:`(N, S)` indicating which elements within ``key``
        key_seq is math:[N,S,E], True for padding masked and False for unpadding unmasked.
        First element in each batch (:,0) needs no mask
    '''
    tail_mask = (torch.abs(key_seq[:,1:,:].sum(-1)) == 0.0)
    head_mask = (key_seq[:,0,:].sum(-1)==-1000).reshape((-1,1))
    return torch.cat([head_mask, tail_mask],dim=1)


def get_attn_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz)) == 1)).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask