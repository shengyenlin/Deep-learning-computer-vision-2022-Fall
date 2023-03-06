import copy
import math

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch import nn
import timm

from typing import Optional, List

from .utils import generate_square_subsequent_mask

# use pretrained encoder
class Encoder(nn.Module):
    def __init__(self, model_name):
        super(Encoder, self).__init__()
        self.model = timm.create_model(model_name, pretrained=True)
        self.remove_MLP()

    def remove_MLP(self):
        self.encoder_embed = self.model.patch_embed #use conv to turn images into image grids
        self.encoder_pos_drop = self.model.pos_drop
        self.encoder_norm_pre = self.model.norm_pre
        self.encoder_attn = self.model.blocks
        self.encoder_norm = self.model.norm
        self.encoder_fc_norm = self.model.fc_norm

    def forward(self, x):
        x = self.encoder_embed(x)
        x = self.encoder_pos_drop(x)
        x = self.encoder_norm_pre(x)
        x = self.encoder_attn(x)
        x = self.encoder_norm(x)
        x = self.encoder_fc_norm(x)
        # (nb, n_patches, dim) = (nb, seq, dim)
        return x 

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=True):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2[0])
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(
                query=self.with_pos_embed(tgt2, query_pos),
                key=self.with_pos_embed(memory, pos),
                value=memory, attn_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask
            )
        tgt = tgt + self.dropout2(tgt2[0])
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(
                tgt, memory, tgt_mask, memory_mask,
                tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos
                )
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)

class DecoderEmbeddings(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            # [PAD] = 0
            args.vocab_size, args.decoder_hid_dim, padding_idx=0)
        self.position_embeddings = nn.Embedding(
            args.max_len, args.decoder_hid_dim
        )

        self.LayerNorm = torch.nn.LayerNorm(
            args.decoder_hid_dim, eps=args.layer_norm_eps)
        self.dropout = nn.Dropout(args.decoder_dropout)

    def forward(self, x):
        # memory = (nb, seq_len, hid_dim)
        input_shape = x.size()
        seq_length = input_shape[1]
        device = x.device

        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(input_shape)

        input_embeds = self.word_embeddings(x)
        position_embeds = self.position_embeddings(position_ids)

        embeddings = input_embeds + position_embeds
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings

class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output

class DecoderCATR(nn.Module):
    def __init__(self, d_model=512, nhead=8,
                num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                activation="relu", normalize_before=True,
                return_intermediate_dec=False):
        super().__init__()
        decoder_layer = TransformerDecoderLayer(
            d_model, nhead, dim_feedforward,
            dropout, activation, normalize_before
            )
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                        return_intermediate=return_intermediate_dec)
        self.d_model = d_model
        self.nhead = nhead

    def forward(
        self, tgt, memory, memory_key_padding_mask, 
        tgt_key_padding_mask, pos, query_pos, tgt_mask
        ):
        # memory = (nb, seq_len, hid_dim)

        hs = self.decoder(
            tgt, memory, memory_key_padding_mask=memory_key_padding_mask, 
            tgt_key_padding_mask=tgt_key_padding_mask, pos=pos, 
            query_pos=query_pos, tgt_mask=tgt_mask
            )
        return hs

class Decoder(nn.Module):
    def __init__(self, d_model, n_head, num_layers, norm=None):
        '''
        d_model: the number of expected features in the input
        n_head: the number of heads in the multiheadattention models
        num_layers: the number of sub-decoder-layers in the decoder (required)
        norm: the layer normalization component (optional).
        '''
        super(Decoder, self).__init__()
        decoder_layer = nn.TransformerDecoderLayer(d_model, n_head, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)


    def forward(self, tgt, tgt_key_padding_mask, memory):
        seq_len = tgt.size(1)
        tgt_mask = generate_square_subsequent_mask(seq_len).to(tgt.device)
        #In Transformer class, non-zero values in mask are the values that will not be attended.
        out = self.decoder(
            tgt, memory, 
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
            )
        return out


class ImageCaptioningModel(nn.Module):
    def __init__(self, encoder, decoder, decoder_emb, generator):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.decoder_embed = decoder_emb
        self.generator = generator

    def forward(self, img, tgt, tgt_padding_mask):
        memory = self.encode(img)
        decoder_out = self.decode(tgt, tgt_padding_mask, memory)
        pred = self.generator(decoder_out)
        return pred

    def encode(self, img):
        memory = self.encoder(img)
        return memory

    def decode(self, tgt, tgt_padding_mask, memory):
        tgt_emb = self.decoder_embed(tgt) #(nb, seq_len, hid_dim)
        decoder_out = self.decoder(tgt_emb, tgt_padding_mask, memory) #(nb, seq_len, hid_dim)
        return decoder_out

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k)
                                    for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)

def make_model(args, len_tgt_vocab):
    c = copy.deepcopy
    position = PositionalEncoding(args.decoder_hid_dim, args.decoder_dropout)

    if args.use_catr:
        model = ImageCaptioningModelCATR(
            args,
            Encoder(args.encoder_model_name),
            DecoderCATR(args.decoder_hid_dim, args.decoder_n_head, args.decoder_num_layers, dropout=args.decoder_dropout)
        )
        
    else:
        model = ImageCaptioningModel(
            Encoder(args.encoder_model_name),
            Decoder(args.decoder_hid_dim, args.decoder_n_head, args.decoder_num_layers),
            nn.Sequential(Embeddings(args.decoder_hid_dim, len_tgt_vocab), c(position)),
            MLP(args.decoder_hid_dim, args.gen_hid_dim, len_tgt_vocab, args.gen_num_layers),
        )

    return model

class ImageCaptioningModelCATR(nn.Module):
    def __init__(self, args, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder_embedding = DecoderEmbeddings(args)
        self.decoder = decoder # DecoderCATR
        self.mlp = MLP(
                args.decoder_hid_dim, 512, args.vocab_size, 
                num_layers=3
            )

    def encode(self, img):
        bs = img.size(0)
        pos = self.encoder.model.pos_embed[:, 1: :].permute(1, 0, 2)
        pos = pos.repeat(1, bs, 1)
        memory = self.encoder(img).permute(1, 0, 2)
        return memory, pos     

    def decode(self, memory, pos, tgt, tgt_pad_mask):
        bs = tgt.size(0)
        tgt = self.decoder_embedding(tgt).permute(1, 0, 2)
        query_embed = self.decoder_embedding.position_embeddings.weight.unsqueeze(1)
        query_embed = query_embed.repeat(1, bs, 1)

        decode_out = self.decoder(
            tgt, memory, memory_key_padding_mask=None, 
            tgt_key_padding_mask=tgt_pad_mask,
            pos=pos, query_pos=query_embed,
            tgt_mask=generate_square_subsequent_mask(len(tgt)).to(tgt.device)
            )
        return decode_out

    def generator(self, decode_out):
        out = self.mlp(decode_out.permute(1, 0, 2))
        return out

    def forward(self, img, tgt, tgt_mask):
        # encoder
        memory, pos = self.encode(img)
        # decoder
        decode_out = self.decode(memory, pos, tgt, tgt_mask)
        # generator  
        out = self.generator(decode_out)
        return out

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")