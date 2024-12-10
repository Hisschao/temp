import sys
sys.path.append("../")
import os
import numpy as np
import torch
import time

# torch.set_default_tensor_type('torch.cuda.FloatTensor')

import torch
from torch import nn

import random

from einops import rearrange,repeat
import torch.nn.functional as F
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)



class SM_MAE(nn.Module): ## reconstruction without weather
    def __init__( self,*,image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.,decoder_dim, masking_ratio = 0.75, decoder_depth = 1, decoder_heads = 8, decoder_dim_head = 64, doy_embed_dim = 16, delt_embed_dim = 16, channels_w_in = 7,channels_w_out = 5, dim_w = 32):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer_enc = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.masking_ratio = masking_ratio
        self.to_latent = nn.Identity()
        num_patches, encoder_dim = self.pos_embedding.shape[-2:]

        self.to_patch = self.to_patch_embedding[0]
        self.patch_to_emb = nn.Sequential(*self.to_patch_embedding[1:])

        self.pixel_values_to_out = Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1 = patch_height, p2 = patch_width, h = image_height//patch_height, w = image_width // patch_width)

        pixel_values_per_patch = self.to_patch_embedding[2].weight.shape[-1]

        # decoder parameters
        self.decoder_dim = decoder_dim
        self.enc_to_dec = nn.Linear(encoder_dim, decoder_dim) if encoder_dim != decoder_dim else nn.Identity()
        self.mask_token = nn.Parameter(torch.randn(decoder_dim))
        self.decoder = Transformer(dim = decoder_dim, depth = decoder_depth, heads = decoder_heads, dim_head = decoder_dim_head, mlp_dim = mlp_dim)
        self.decoder_pos_emb = nn.Embedding(num_patches, decoder_dim)
        self.to_pixels = nn.Linear(decoder_dim, pixel_values_per_patch)

        self.create_embed_doy = torch.nn.Linear(1, dim)
        self.converterlin1 = torch.nn.Linear(dim, dim)
        self.converterlin2 = torch.nn.Linear(dim, dim)
        self.reprojecterlin1 = torch.nn.Linear(dim, dim)
        self.reprojecterlin2 = torch.nn.Linear(dim, dim)
        self.channels = channels

        self._mask = None

        self.temporal_encoder_layer = torch.nn.TransformerEncoderLayer(d_model=dim, nhead=8)
        self.temporal_transformer_encoder = torch.nn.TransformerEncoder(self.temporal_encoder_layer, num_layers=1)

        self.relu = torch.nn.ReLU()
        self.tanh = torch.nn.Tanh()

    def forward(self, img, timestamps):

        batch_size_s, seq_len_s, channels_s , input_patch_size, _ = img.shape #16,4,6,32,32

        ## create temporal positional embeddings
        timestamps = timestamps.unsqueeze(2)
        timestamps_embeds = self.create_embed_doy(timestamps)
        timestamps_embeds = self.tanh(timestamps_embeds)

        device = img.device

        ## send time to batch dim
        img = img.reshape(-1, img.shape[-3], img.shape[-2], img.shape[-1])

        # get patches
        patches = self.to_patch(img)
        batch, num_patches, *_ = patches.shape

        # patch to encoder tokens and add positions
        tokens = self.patch_to_emb(patches)
        if self.pool == "cls":
            tokens += self.pos_embedding[:, 1:(num_patches + 1)]
        elif self.pool == "mean":
            tokens += self.pos_embedding.to(device, dtype=tokens.dtype) 
    
        # calculate of patches needed to be masked, and get random indices, dividing it up for mask vs unmasked
        masking_seq_length = 0
        mask_binary = torch.zeros(batch_size_s, num_patches, seq_len_s).to(device)  
        patches_per_seq = int(num_patches*masking_seq_length/seq_len_s)
        
        mask_binary_reshaped = mask_binary.permute(0,2,1).reshape(-1,num_patches)

        unmasked_indices = torch.empty(batch_size_s, num_patches - patches_per_seq, seq_len_s).to(device) 
        masked_indices = torch.empty(batch_size_s, patches_per_seq, seq_len_s).to(device) 
        for b in range(batch_size_s):
            for t in range(seq_len_s):
                unmasked_indices[b,:,t] = torch.where(mask_binary[b,:,t] == 0)[0]
                masked_indices[b,:,t] = torch.where(mask_binary[b,:,t] == 1)[0]

        unmasked_indices_temporal = torch.empty(batch_size_s , num_patches , seq_len_s-masking_seq_length).to(device) 
        for b in range(batch_size_s):
            for p in range(num_patches):
                unmasked_indices_temporal[b,p,:] = torch.where(mask_binary[b,p,:] == 0)[0]

        unmasked_indices_permuted = unmasked_indices.permute(0,2,1)
        masked_indices_permuted = masked_indices.permute(0,2,1)
        unmasked_indices_reshaped = unmasked_indices_permuted.reshape(-1,unmasked_indices_permuted.shape[-1]).int()
        # masked_indices_reshaped = masked_indices_permuted.reshape(-1,masked_indices_permuted.shape[-1]).int()

        unmasked_indices_temporal_reshaped = unmasked_indices_temporal.reshape(-1,unmasked_indices_temporal.shape[-1]).int()

        # get the unmasked tokens to be encoded
        batch_range = torch.arange(batch, device = device)[:, None].to(device) 
        tokens = tokens[batch_range, unmasked_indices_reshaped]

        # attend with vision transformer
        encoded_tokens = self.transformer_enc(tokens)
        
        timestamps_in_embeds_repeat = timestamps_embeds[:,:seq_len_s,:].unsqueeze(2).repeat(1,1,encoded_tokens.shape[1],1)
        timestamps_in_embeds_repeat_reshaped = timestamps_in_embeds_repeat.reshape(-1,timestamps_in_embeds_repeat.shape[-2],timestamps_in_embeds_repeat.shape[-1])

        # weather_in_embeds_repeat = x_w_embed_subset[:,:seq_len_s,:].unsqueeze(2).repeat(1,1,encoded_tokens.shape[1],1)
        # weather_in_embeds_repeat_reshaped = weather_in_embeds_repeat.reshape(-1,weather_in_embeds_repeat.shape[-2],weather_in_embeds_repeat.shape[-1])
        
        encoded_tokens_sum = encoded_tokens + timestamps_in_embeds_repeat_reshaped

        encoded_tokens_sum = self.relu(self.converterlin2(self.relu(self.converterlin1(encoded_tokens_sum))))

        repop_encoder_tokens = torch.zeros(batch, num_patches, encoded_tokens_sum.shape[-1], device=device)
        repop_encoder_tokens[batch_range, unmasked_indices_reshaped] = encoded_tokens_sum


        repop_encoded_tokens_time_split = repop_encoder_tokens.view(-1,seq_len_s,repop_encoder_tokens.shape[-2],repop_encoder_tokens.shape[-1])
        repop_encoded_tokens_time_split = repop_encoded_tokens_time_split.permute(0,2,1,3)
        repop_encoded_tokens_time_split = repop_encoded_tokens_time_split.reshape(repop_encoded_tokens_time_split.shape[0]*repop_encoded_tokens_time_split.shape[1], seq_len_s, encoded_tokens.shape[-1])

        batch_range_temporal = torch.arange(repop_encoded_tokens_time_split.shape[0], device = device)[:, None]
        encoded_tokens_time_split_subset = repop_encoded_tokens_time_split[batch_range_temporal, unmasked_indices_temporal_reshaped]

        # mask out future values
        if self._mask is None or self._mask.size(0) != len(encoded_tokens_time_split_subset):
            self._mask = torch.triu(encoded_tokens_time_split_subset.new_full((len(encoded_tokens_time_split_subset), len(encoded_tokens_time_split_subset)), fill_value=float('-inf')), diagonal=1)

        encoded_tokens_temporal = self.temporal_transformer_encoder(encoded_tokens_time_split_subset,self._mask) 

        repop_encoder_tokens_temporal = torch.zeros(batch_size_s * num_patches, seq_len_s, encoded_tokens_sum.shape[-1], device=device)
        repop_encoder_tokens_temporal[batch_range_temporal, unmasked_indices_temporal_reshaped] = encoded_tokens_temporal

        encoded_tokens_temporal_spatial = repop_encoder_tokens_temporal.view(batch_size_s,-1,seq_len_s,encoded_tokens.shape[-1]).permute(0,2,1,3).reshape(-1, repop_encoder_tokens.shape[-2], repop_encoder_tokens.shape[-1])

        encoded_out_tokens_sum = encoded_tokens_temporal_spatial       
        encoded_out_tokens_sum_subset = encoded_out_tokens_sum[batch_range, unmasked_indices_reshaped]
        embeddings = encoded_out_tokens_sum_subset.view(-1,seq_len_s,encoded_out_tokens_sum_subset.shape[-2],encoded_out_tokens_sum_subset.shape[-1])
        encoded_out_tokens_sum_subset = embeddings.view(-1,embeddings.shape[-2],embeddings.shape[-1])
        
        ## decoder
        decoder_tokens = self.relu(self.reprojecterlin2(self.relu(self.reprojecterlin1(encoded_out_tokens_sum_subset))))
        
        # reapply decoder position embedding to unmasked tokens
        unmasked_decoder_tokens = decoder_tokens + self.decoder_pos_emb(unmasked_indices_reshaped)

        # concat the masked tokens to the decoder tokens and attend with decoder
        # mask_tokens = repeat(self.mask_token, 'd -> b n d', b = batch, n = masked_indices_reshaped.shape[-1])
        # mask_tokens = mask_tokens + self.decoder_pos_emb(masked_indices_reshaped)
        
        decoder_tokens = torch.zeros(batch, num_patches, self.decoder_dim, device=device)
        decoder_tokens[batch_range, unmasked_indices_reshaped] = unmasked_decoder_tokens
        # decoder_tokens[batch_range, masked_indices_reshaped] = mask_tokens

        decoded_tokens_out = self.decoder(decoder_tokens)
        # print('decoded_tokens_out',decoded_tokens_out.shape)

        pred_pixel_values = self.to_pixels(decoded_tokens_out)
        recon_output = self.pixel_values_to_out(pred_pixel_values)
        recon_output_reshape = recon_output.view(-1,seq_len_s,recon_output.shape[1],recon_output.shape[2],recon_output.shape[3])

        return recon_output_reshape, mask_binary, embeddings


class MM_MAE(nn.Module): ## reconstruction with weather
    def __init__( self,*,image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.,decoder_dim, masking_ratio = 0.75, decoder_depth = 1, decoder_heads = 8, decoder_dim_head = 64, doy_embed_dim = 16, delt_embed_dim = 16, channels_w_in = 7,channels_w_out = 5, dim_w = 32):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer_enc = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.masking_ratio = masking_ratio
        self.to_latent = nn.Identity()
        num_patches, encoder_dim = self.pos_embedding.shape[-2:]

        self.to_patch = self.to_patch_embedding[0]
        self.patch_to_emb = nn.Sequential(*self.to_patch_embedding[1:])

        self.pixel_values_to_out = Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1 = patch_height, p2 = patch_width, h = image_height//patch_height, w = image_width // patch_width)

        pixel_values_per_patch = self.to_patch_embedding[2].weight.shape[-1]

        # decoder parameters
        self.decoder_dim = decoder_dim
        self.enc_to_dec = nn.Linear(encoder_dim, decoder_dim) if encoder_dim != decoder_dim else nn.Identity()
        self.mask_token = nn.Parameter(torch.randn(decoder_dim))
        self.decoder = Transformer(dim = decoder_dim, depth = decoder_depth, heads = decoder_heads, dim_head = decoder_dim_head, mlp_dim = mlp_dim)
        self.decoder_pos_emb = nn.Embedding(num_patches, decoder_dim)
        self.to_pixels = nn.Linear(decoder_dim, pixel_values_per_patch)

        self.create_embed_doy = torch.nn.Linear(1, dim)
        self.converterlin1 = torch.nn.Linear(dim, dim)
        self.converterlin2 = torch.nn.Linear(dim, dim)
        self.reprojecterlin1 = torch.nn.Linear(dim, dim)
        self.reprojecterlin2 = torch.nn.Linear(dim, dim)
        self.channels = channels

        self._mask = None

        self.temporal_encoder_layer = torch.nn.TransformerEncoderLayer(d_model=dim, nhead=8)
        self.temporal_transformer_encoder = torch.nn.TransformerEncoder(self.temporal_encoder_layer, num_layers=1)

        self.relu = torch.nn.ReLU()
        self.tanh = torch.nn.Tanh()

        self.lstm_f_w = torch.nn.LSTM(channels_w_in, dim//2, batch_first=True, bidirectional=True)


    def forward(self, img, x_w, timestamps):

        batch_size_s, seq_len_s, channels_s , input_patch_size, _ = img.shape #16,4,6,32,32
        # batch_size_w, seq_len_w, channels_w , = x_w.shape #16,356,5

        ## create temporal positional embeddings
        timestamps = timestamps.unsqueeze(2)
        timestamps_embeds = self.create_embed_doy(timestamps)
        timestamps_embeds = self.tanh(timestamps_embeds)

        ## weather encoder
        x_w_embed,_ = self.lstm_f_w(x_w)
        x_w_embed = self.relu(x_w_embed)

        timestamps = timestamps.long()
        x_w_embed_subset = torch.gather(x_w_embed,1,timestamps.repeat(1,1,x_w_embed.shape[-1]))
        # print('x_w_embed_subset',x_w_embed_subset.shape)

        device = img.device

        ## send time to batch dim
        img = img.reshape(-1, img.shape[-3], img.shape[-2], img.shape[-1])

        # get patches
        patches = self.to_patch(img)
        batch, num_patches, *_ = patches.shape
        # print(patches.shape)

        # patch to encoder tokens and add positions
        tokens = self.patch_to_emb(patches)
        if self.pool == "cls":
            tokens += self.pos_embedding[:, 1:(num_patches + 1)]
        elif self.pool == "mean":
            tokens += self.pos_embedding.to(device, dtype=tokens.dtype) 
    
        # calculate of patches needed to be masked, and get random indices, dividing it up for mask vs unmasked
        masking_seq_length = 0
        mask_binary = torch.zeros(batch_size_s, num_patches, seq_len_s).to(device)  
        patches_per_seq = int(num_patches*masking_seq_length/seq_len_s)
        mask_binary_reshaped = mask_binary.permute(0,2,1).reshape(-1,num_patches)

        unmasked_indices = torch.empty(batch_size_s, num_patches - patches_per_seq, seq_len_s).to(device) 
        masked_indices = torch.empty(batch_size_s, patches_per_seq, seq_len_s).to(device) 
        for b in range(batch_size_s):
            for t in range(seq_len_s):
                unmasked_indices[b,:,t] = torch.where(mask_binary[b,:,t] == 0)[0]
                masked_indices[b,:,t] = torch.where(mask_binary[b,:,t] == 1)[0]

        unmasked_indices_temporal = torch.empty(batch_size_s , num_patches , seq_len_s-masking_seq_length).to(device) 
        for b in range(batch_size_s):
            for p in range(num_patches):
                unmasked_indices_temporal[b,p,:] = torch.where(mask_binary[b,p,:] == 0)[0]

        unmasked_indices_permuted = unmasked_indices.permute(0,2,1)
        masked_indices_permuted = masked_indices.permute(0,2,1)
        unmasked_indices_reshaped = unmasked_indices_permuted.reshape(-1,unmasked_indices_permuted.shape[-1]).int()
        # masked_indices_reshaped = masked_indices_permuted.reshape(-1,masked_indices_permuted.shape[-1]).int()

        unmasked_indices_temporal_reshaped = unmasked_indices_temporal.reshape(-1,unmasked_indices_temporal.shape[-1]).int()

        # get the unmasked tokens to be encoded
        batch_range = torch.arange(batch, device = device)[:, None].to(device) 
        tokens = tokens[batch_range, unmasked_indices_reshaped]

        # attend with vision transformer
        encoded_tokens = self.transformer_enc(tokens)
        
        timestamps_in_embeds_repeat = timestamps_embeds[:,:seq_len_s,:].unsqueeze(2).repeat(1,1,encoded_tokens.shape[1],1)
        timestamps_in_embeds_repeat_reshaped = timestamps_in_embeds_repeat.reshape(-1,timestamps_in_embeds_repeat.shape[-2],timestamps_in_embeds_repeat.shape[-1])

        weather_in_embeds_repeat = x_w_embed_subset[:,:seq_len_s,:].unsqueeze(2).repeat(1,1,encoded_tokens.shape[1],1)
        weather_in_embeds_repeat_reshaped = weather_in_embeds_repeat.reshape(-1,weather_in_embeds_repeat.shape[-2],weather_in_embeds_repeat.shape[-1])
        
        encoded_tokens_sum = encoded_tokens + timestamps_in_embeds_repeat_reshaped + weather_in_embeds_repeat_reshaped

        encoded_tokens_sum = self.relu(self.converterlin2(self.relu(self.converterlin1(encoded_tokens_sum))))

        repop_encoder_tokens = torch.zeros(batch, num_patches, encoded_tokens_sum.shape[-1], device=device)
        repop_encoder_tokens[batch_range, unmasked_indices_reshaped] = encoded_tokens_sum


        repop_encoded_tokens_time_split = repop_encoder_tokens.view(-1,seq_len_s,repop_encoder_tokens.shape[-2],repop_encoder_tokens.shape[-1])
        repop_encoded_tokens_time_split = repop_encoded_tokens_time_split.permute(0,2,1,3)
        repop_encoded_tokens_time_split = repop_encoded_tokens_time_split.reshape(repop_encoded_tokens_time_split.shape[0]*repop_encoded_tokens_time_split.shape[1], seq_len_s, encoded_tokens.shape[-1])

        batch_range_temporal = torch.arange(repop_encoded_tokens_time_split.shape[0], device = device)[:, None]
        encoded_tokens_time_split_subset = repop_encoded_tokens_time_split[batch_range_temporal, unmasked_indices_temporal_reshaped]

        # mask out future values
        if self._mask is None or self._mask.size(0) != len(encoded_tokens_time_split_subset):
            self._mask = torch.triu(encoded_tokens_time_split_subset.new_full((len(encoded_tokens_time_split_subset), len(encoded_tokens_time_split_subset)), fill_value=float('-inf')), diagonal=1)

        encoded_tokens_temporal = self.temporal_transformer_encoder(encoded_tokens_time_split_subset,self._mask) 

        repop_encoder_tokens_temporal = torch.zeros(batch_size_s * num_patches, seq_len_s, encoded_tokens_sum.shape[-1], device=device)
        repop_encoder_tokens_temporal[batch_range_temporal, unmasked_indices_temporal_reshaped] = encoded_tokens_temporal

        encoded_tokens_temporal_spatial = repop_encoder_tokens_temporal.view(batch_size_s,-1,seq_len_s,encoded_tokens.shape[-1]).permute(0,2,1,3).reshape(-1, repop_encoder_tokens.shape[-2], repop_encoder_tokens.shape[-1])

        encoded_out_tokens_sum = encoded_tokens_temporal_spatial       
        encoded_out_tokens_sum_subset = encoded_out_tokens_sum[batch_range, unmasked_indices_reshaped]
        embeddings = encoded_out_tokens_sum_subset.view(-1,seq_len_s,encoded_out_tokens_sum_subset.shape[-2],encoded_out_tokens_sum_subset.shape[-1])
        encoded_out_tokens_sum_subset = embeddings.view(-1,embeddings.shape[-2],embeddings.shape[-1])

        ## decoder
        decoder_tokens = self.relu(self.reprojecterlin2(self.relu(self.reprojecterlin1(encoded_out_tokens_sum_subset))))

        # reapply decoder position embedding to unmasked tokens
        unmasked_decoder_tokens = decoder_tokens + self.decoder_pos_emb(unmasked_indices_reshaped)

        # concat the masked tokens to the decoder tokens and attend with decoder
        # mask_tokens = repeat(self.mask_token, 'd -> b n d', b = batch, n = masked_indices_reshaped.shape[-1])
        # mask_tokens = mask_tokens + self.decoder_pos_emb(masked_indices_reshaped)
        
        decoder_tokens = torch.zeros(batch, num_patches, self.decoder_dim, device=device)
        decoder_tokens[batch_range, unmasked_indices_reshaped] = unmasked_decoder_tokens
        # decoder_tokens[batch_range, masked_indices_reshaped] = mask_tokens

        decoded_tokens_out = self.decoder(decoder_tokens)

        pred_pixel_values = self.to_pixels(decoded_tokens_out)
        recon_output = self.pixel_values_to_out(pred_pixel_values)
        recon_output_reshape = recon_output.view(-1,seq_len_s,recon_output.shape[1],recon_output.shape[2],recon_output.shape[3])

        return recon_output_reshape, mask_binary, embeddings


class SM_VSF(nn.Module): ## forecast without weather
    def __init__( self,*,image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.,decoder_dim, masking_ratio = 0.75, decoder_depth = 1, decoder_heads = 8, decoder_dim_head = 64, doy_embed_dim = 16, delt_embed_dim = 16, channels_w_in = 7,channels_w_out = 5, dim_w = 32):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer_enc = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.masking_ratio = masking_ratio
        self.to_latent = nn.Identity()
        num_patches, encoder_dim = self.pos_embedding.shape[-2:]

        self.to_patch = self.to_patch_embedding[0]
        self.patch_to_emb = nn.Sequential(*self.to_patch_embedding[1:])

        self.pixel_values_to_out = Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1 = patch_height, p2 = patch_width, h = image_height//patch_height, w = image_width // patch_width)

        pixel_values_per_patch = self.to_patch_embedding[2].weight.shape[-1]

        # decoder parameters
        self.decoder_dim = decoder_dim
        self.enc_to_dec = nn.Linear(encoder_dim, decoder_dim) if encoder_dim != decoder_dim else nn.Identity()
        self.mask_token = nn.Parameter(torch.randn(decoder_dim))
        self.decoder = Transformer(dim = decoder_dim, depth = decoder_depth, heads = decoder_heads, dim_head = decoder_dim_head, mlp_dim = mlp_dim)
        self.decoder_pos_emb = nn.Embedding(num_patches, decoder_dim)
        self.to_pixels = nn.Linear(decoder_dim, pixel_values_per_patch)

        self.create_embed_doy = torch.nn.Linear(1, dim)
        self.create_embed_delt = torch.nn.Linear(1, dim)
        self.converterlin1 = torch.nn.Linear(dim, dim)
        self.converterlin2 = torch.nn.Linear(dim, dim)
        self.forecasterlin1 = torch.nn.Linear(dim, dim)
        self.forecasterlin2 = torch.nn.Linear(dim, dim)
        self.channels = channels

        self._mask = None

        self.temporal_encoder_layer = torch.nn.TransformerEncoderLayer(d_model=dim, nhead=8)
        self.temporal_transformer_encoder = torch.nn.TransformerEncoder(self.temporal_encoder_layer, num_layers=1)

        self.relu = torch.nn.ReLU()
        self.tanh = torch.nn.Tanh()


    def forward(self, img, timestamps, delt_timestamps):

        batch_size_s, seq_len_s, channels_s , input_patch_size, _ = img.shape #16,4,6,32,32

        ## create temporal positional embeddings
        timestamps = timestamps.unsqueeze(2)
        timestamps_embeds = self.create_embed_doy(timestamps)
        timestamps_embeds = self.tanh(timestamps_embeds)

        ## create delta temporal positional embeddings
        delt_timestamps = delt_timestamps.unsqueeze(2)
        delt_timestamps_embeds = self.create_embed_delt(delt_timestamps)
        delt_timestamps_embeds = self.tanh(delt_timestamps_embeds)

        device = img.device

        ## send time to batch dim
        img = img.reshape(-1, img.shape[-3], img.shape[-2], img.shape[-1])

        # get patches
        patches = self.to_patch(img)
        batch, num_patches, *_ = patches.shape
        # print(patches.shape)

        # patch to encoder tokens and add positions
        tokens = self.patch_to_emb(patches)
        if self.pool == "cls":
            tokens += self.pos_embedding[:, 1:(num_patches + 1)]
        elif self.pool == "mean":
            tokens += self.pos_embedding.to(device, dtype=tokens.dtype) 
    
        # calculate of patches needed to be masked, and get random indices, dividing it up for mask vs unmasked
        masking_seq_length = 0
        mask_binary = torch.zeros(batch_size_s, num_patches, seq_len_s).to(device)  
        patches_per_seq = int(num_patches*masking_seq_length/seq_len_s)
       
        
        mask_binary_reshaped = mask_binary.permute(0,2,1).reshape(-1,num_patches)

        unmasked_indices = torch.empty(batch_size_s, num_patches - patches_per_seq, seq_len_s).to(device) 
        masked_indices = torch.empty(batch_size_s, patches_per_seq, seq_len_s).to(device) 
        for b in range(batch_size_s):
            for t in range(seq_len_s):
                unmasked_indices[b,:,t] = torch.where(mask_binary[b,:,t] == 0)[0]
                masked_indices[b,:,t] = torch.where(mask_binary[b,:,t] == 1)[0]

        unmasked_indices_temporal = torch.empty(batch_size_s , num_patches , seq_len_s-masking_seq_length).to(device) 
        for b in range(batch_size_s):
            for p in range(num_patches):
                unmasked_indices_temporal[b,p,:] = torch.where(mask_binary[b,p,:] == 0)[0]

        unmasked_indices_permuted = unmasked_indices.permute(0,2,1)
        masked_indices_permuted = masked_indices.permute(0,2,1)
        unmasked_indices_reshaped = unmasked_indices_permuted.reshape(-1,unmasked_indices_permuted.shape[-1]).int()
        # masked_indices_reshaped = masked_indices_permuted.reshape(-1,masked_indices_permuted.shape[-1]).int()

        unmasked_indices_temporal_reshaped = unmasked_indices_temporal.reshape(-1,unmasked_indices_temporal.shape[-1]).int()

        # get the unmasked tokens to be encoded
        batch_range = torch.arange(batch, device = device)[:, None].to(device) 
        # print(batch_range.shape,unmasked_indices_reshaped.shape)
        # tokens = tokens[batch_range, unmasked_indices]
        tokens = tokens[batch_range, unmasked_indices_reshaped]

        # attend with vision transformer
        encoded_tokens = self.transformer_enc(tokens)
        
        timestamps_in_embeds_repeat = timestamps_embeds[:,:seq_len_s,:].unsqueeze(2).repeat(1,1,encoded_tokens.shape[1],1)
        timestamps_in_embeds_repeat_reshaped = timestamps_in_embeds_repeat.reshape(-1,timestamps_in_embeds_repeat.shape[-2],timestamps_in_embeds_repeat.shape[-1])
        
        encoded_tokens_sum = encoded_tokens + timestamps_in_embeds_repeat_reshaped

        encoded_tokens_sum = self.relu(self.converterlin2(self.relu(self.converterlin1(encoded_tokens_sum))))

        repop_encoder_tokens = torch.zeros(batch, num_patches, encoded_tokens_sum.shape[-1], device=device)
        repop_encoder_tokens[batch_range, unmasked_indices_reshaped] = encoded_tokens_sum


        repop_encoded_tokens_time_split = repop_encoder_tokens.view(-1,seq_len_s,repop_encoder_tokens.shape[-2],repop_encoder_tokens.shape[-1])
        repop_encoded_tokens_time_split = repop_encoded_tokens_time_split.permute(0,2,1,3)
        repop_encoded_tokens_time_split = repop_encoded_tokens_time_split.reshape(repop_encoded_tokens_time_split.shape[0]*repop_encoded_tokens_time_split.shape[1], seq_len_s, encoded_tokens.shape[-1])

        batch_range_temporal = torch.arange(repop_encoded_tokens_time_split.shape[0], device = device)[:, None]
        encoded_tokens_time_split_subset = repop_encoded_tokens_time_split[batch_range_temporal, unmasked_indices_temporal_reshaped]

        # mask out future values
        if self._mask is None or self._mask.size(0) != len(encoded_tokens_time_split_subset):
            self._mask = torch.triu(encoded_tokens_time_split_subset.new_full((len(encoded_tokens_time_split_subset), len(encoded_tokens_time_split_subset)), fill_value=float('-inf')), diagonal=1)

        encoded_tokens_temporal = self.temporal_transformer_encoder(encoded_tokens_time_split_subset,self._mask) 

        repop_encoder_tokens_temporal = torch.zeros(batch_size_s * num_patches, seq_len_s, encoded_tokens_sum.shape[-1], device=device)
        repop_encoder_tokens_temporal[batch_range_temporal, unmasked_indices_temporal_reshaped] = encoded_tokens_temporal

        encoded_tokens_temporal_spatial = repop_encoder_tokens_temporal.view(batch_size_s,-1,seq_len_s,encoded_tokens.shape[-1]).permute(0,2,1,3).reshape(-1, repop_encoder_tokens.shape[-2], repop_encoder_tokens.shape[-1])

        timestamps_out_embeds_repeat = timestamps_embeds[:,-seq_len_s:,:].unsqueeze(2).repeat(1,1,repop_encoder_tokens.shape[1],1)
        timestamps_out_embeds_repeat_reshaped = timestamps_out_embeds_repeat.reshape(-1,timestamps_out_embeds_repeat.shape[-2],timestamps_out_embeds_repeat.shape[-1])

        delt_timestamps_out_embeds_repeat = delt_timestamps_embeds[:,-seq_len_s:,:].unsqueeze(2).repeat(1,1,repop_encoder_tokens.shape[1],1)
        delt_timestamps_out_embeds_repeat_reshaped = delt_timestamps_out_embeds_repeat.reshape(-1,delt_timestamps_out_embeds_repeat.shape[-2],delt_timestamps_out_embeds_repeat.shape[-1])

        encoded_out_tokens_sum = encoded_tokens_temporal_spatial + timestamps_out_embeds_repeat_reshaped + delt_timestamps_out_embeds_repeat_reshaped       

        encoded_out_tokens_sum_subset = encoded_out_tokens_sum[batch_range, unmasked_indices_reshaped]
        embeddings = encoded_out_tokens_sum_subset.view(-1,seq_len_s,encoded_out_tokens_sum_subset.shape[-2],encoded_out_tokens_sum_subset.shape[-1])
        encoded_out_tokens_sum_subset = embeddings.view(-1,embeddings.shape[-2],embeddings.shape[-1])

        decoder_tokens = self.relu(self.forecasterlin2(self.relu(self.forecasterlin1(encoded_out_tokens_sum_subset))))

        # reapply decoder position embedding to unmasked tokens
        unmasked_decoder_tokens = decoder_tokens + self.decoder_pos_emb(unmasked_indices_reshaped)

        # concat the masked tokens to the decoder tokens and attend with decoder
        # mask_tokens = repeat(self.mask_token, 'd -> b n d', b = batch, n = masked_indices_reshaped.shape[-1])
        # mask_tokens = mask_tokens + self.decoder_pos_emb(masked_indices_reshaped)
        
        decoder_tokens = torch.zeros(batch, num_patches, self.decoder_dim, device=device)
        decoder_tokens[batch_range, unmasked_indices_reshaped] = unmasked_decoder_tokens
        # decoder_tokens[batch_range, masked_indices_reshaped] = mask_tokens

        decoded_tokens_out = self.decoder(decoder_tokens)
        # print('decoded_tokens_out',decoded_tokens_out.shape)

        pred_pixel_values = self.to_pixels(decoded_tokens_out)
        recon_output = self.pixel_values_to_out(pred_pixel_values)
        recon_output_reshape = recon_output.view(-1,seq_len_s,recon_output.shape[1],recon_output.shape[2],recon_output.shape[3])

        return recon_output_reshape, mask_binary, embeddings

class MM_VSF(nn.Module): ## forecast with weather
    def __init__( self,*,image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.,decoder_dim, masking_ratio = 0.75, decoder_depth = 1, decoder_heads = 8, decoder_dim_head = 64, doy_embed_dim = 16, delt_embed_dim = 16, channels_w_in = 7,channels_w_out = 5, dim_w = 32):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer_enc = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.masking_ratio = masking_ratio
        self.to_latent = nn.Identity()
        num_patches, encoder_dim = self.pos_embedding.shape[-2:]

        self.to_patch = self.to_patch_embedding[0]
        self.patch_to_emb = nn.Sequential(*self.to_patch_embedding[1:])

        self.pixel_values_to_out = Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1 = patch_height, p2 = patch_width, h = image_height//patch_height, w = image_width // patch_width)

        pixel_values_per_patch = self.to_patch_embedding[2].weight.shape[-1]

        # decoder parameters
        self.decoder_dim = decoder_dim
        self.enc_to_dec = nn.Linear(encoder_dim, decoder_dim) if encoder_dim != decoder_dim else nn.Identity()
        self.mask_token = nn.Parameter(torch.randn(decoder_dim))
        self.decoder = Transformer(dim = decoder_dim, depth = decoder_depth, heads = decoder_heads, dim_head = decoder_dim_head, mlp_dim = mlp_dim)
        self.decoder_pos_emb = nn.Embedding(num_patches, decoder_dim)
        self.to_pixels = nn.Linear(decoder_dim, pixel_values_per_patch)

        self.create_embed_doy = torch.nn.Linear(1, dim)
        self.create_embed_delt = torch.nn.Linear(1, dim)
        self.converterlin1 = torch.nn.Linear(dim, dim)
        self.converterlin2 = torch.nn.Linear(dim, dim)
        self.forecasterlin1 = torch.nn.Linear(dim, dim)
        self.forecasterlin2 = torch.nn.Linear(dim, dim)
        self.channels = channels

        self._mask = None

        self.temporal_encoder_layer = torch.nn.TransformerEncoderLayer(d_model=dim, nhead=8)
        self.temporal_transformer_encoder = torch.nn.TransformerEncoder(self.temporal_encoder_layer, num_layers=1)
        self.relu = torch.nn.ReLU()
        self.tanh = torch.nn.Tanh()

        self.output_w_layer = nn.Linear(2*(dim//2), channels_w_out)

        self.lstm_f_w = torch.nn.LSTM(channels_w_in, dim//2, batch_first=True, bidirectional=True)


    def forward(self, img, x_w, timestamps, delt_timestamps):

        batch_size_s, seq_len_s, channels_s , input_patch_size, _ = img.shape #16,4,6,32,32

        ## create temporal positional embeddings
        timestamps = timestamps.unsqueeze(2)
        timestamps_embeds = self.create_embed_doy(timestamps)
        timestamps_embeds = self.tanh(timestamps_embeds)

        ## create delta temporal positional embeddings
        delt_timestamps = delt_timestamps.unsqueeze(2)
        delt_timestamps_embeds = self.create_embed_delt(delt_timestamps)
        delt_timestamps_embeds = self.tanh(delt_timestamps_embeds)

        ## weather encoder
        x_w_embed,_ = self.lstm_f_w(x_w)
        x_w_embed = self.relu(x_w_embed)

        timestamps = timestamps.long()
        x_w_embed_subset = torch.gather(x_w_embed,1,timestamps.repeat(1,1,x_w_embed.shape[-1]))

        device = img.device

        ## send time to batch dim
        img = img.reshape(-1, img.shape[-3], img.shape[-2], img.shape[-1])

        # get patches
        patches = self.to_patch(img)
        batch, num_patches, *_ = patches.shape

        # patch to encoder tokens and add positions
        tokens = self.patch_to_emb(patches)
        if self.pool == "cls":
            tokens += self.pos_embedding[:, 1:(num_patches + 1)]
        elif self.pool == "mean":
            tokens += self.pos_embedding.to(device, dtype=tokens.dtype) 
    
        # calculate of patches needed to be masked, and get random indices, dividing it up for mask vs unmasked
        masking_seq_length = 0
        mask_binary = torch.zeros(batch_size_s, num_patches, seq_len_s).to(device)  
        patches_per_seq = int(num_patches*masking_seq_length/seq_len_s)
        
        mask_binary_reshaped = mask_binary.permute(0,2,1).reshape(-1,num_patches)

        unmasked_indices = torch.empty(batch_size_s, num_patches - patches_per_seq, seq_len_s).to(device) 
        masked_indices = torch.empty(batch_size_s, patches_per_seq, seq_len_s).to(device) 
        for b in range(batch_size_s):
            for t in range(seq_len_s):
                unmasked_indices[b,:,t] = torch.where(mask_binary[b,:,t] == 0)[0]
                masked_indices[b,:,t] = torch.where(mask_binary[b,:,t] == 1)[0]

        unmasked_indices_temporal = torch.empty(batch_size_s , num_patches , seq_len_s-masking_seq_length).to(device) 
        for b in range(batch_size_s):
            for p in range(num_patches):
                unmasked_indices_temporal[b,p,:] = torch.where(mask_binary[b,p,:] == 0)[0]

        unmasked_indices_permuted = unmasked_indices.permute(0,2,1)
        masked_indices_permuted = masked_indices.permute(0,2,1)
        unmasked_indices_reshaped = unmasked_indices_permuted.reshape(-1,unmasked_indices_permuted.shape[-1]).int()
        # masked_indices_reshaped = masked_indices_permuted.reshape(-1,masked_indices_permuted.shape[-1]).int()

        unmasked_indices_temporal_reshaped = unmasked_indices_temporal.reshape(-1,unmasked_indices_temporal.shape[-1]).int()

        # get the unmasked tokens to be encoded
        batch_range = torch.arange(batch, device = device)[:, None].to(device) 
        # tokens = tokens[batch_range, unmasked_indices]
        tokens = tokens[batch_range, unmasked_indices_reshaped]

        # attend with vision transformer
        encoded_tokens = self.transformer_enc(tokens)
        
        timestamps_in_embeds_repeat = timestamps_embeds[:,:seq_len_s,:].unsqueeze(2).repeat(1,1,encoded_tokens.shape[1],1)
        timestamps_in_embeds_repeat_reshaped = timestamps_in_embeds_repeat.reshape(-1,timestamps_in_embeds_repeat.shape[-2],timestamps_in_embeds_repeat.shape[-1])

        weather_in_embeds_repeat = x_w_embed_subset[:,:seq_len_s,:].unsqueeze(2).repeat(1,1,encoded_tokens.shape[1],1)
        weather_in_embeds_repeat_reshaped = weather_in_embeds_repeat.reshape(-1,weather_in_embeds_repeat.shape[-2],weather_in_embeds_repeat.shape[-1])
        
        encoded_tokens_sum = encoded_tokens + timestamps_in_embeds_repeat_reshaped + weather_in_embeds_repeat_reshaped

        encoded_tokens_sum = self.relu(self.converterlin2(self.relu(self.converterlin1(encoded_tokens_sum))))

        repop_encoder_tokens = torch.zeros(batch, num_patches, encoded_tokens_sum.shape[-1], device=device)
        repop_encoder_tokens[batch_range, unmasked_indices_reshaped] = encoded_tokens_sum


        repop_encoded_tokens_time_split = repop_encoder_tokens.view(-1,seq_len_s,repop_encoder_tokens.shape[-2],repop_encoder_tokens.shape[-1])
        repop_encoded_tokens_time_split = repop_encoded_tokens_time_split.permute(0,2,1,3)
        repop_encoded_tokens_time_split = repop_encoded_tokens_time_split.reshape(repop_encoded_tokens_time_split.shape[0]*repop_encoded_tokens_time_split.shape[1], seq_len_s, encoded_tokens.shape[-1])

        batch_range_temporal = torch.arange(repop_encoded_tokens_time_split.shape[0], device = device)[:, None]
        encoded_tokens_time_split_subset = repop_encoded_tokens_time_split[batch_range_temporal, unmasked_indices_temporal_reshaped]

        # mask out future values
        if self._mask is None or self._mask.size(0) != len(encoded_tokens_time_split_subset):
            self._mask = torch.triu(encoded_tokens_time_split_subset.new_full((len(encoded_tokens_time_split_subset), len(encoded_tokens_time_split_subset)), fill_value=float('-inf')), diagonal=1)

        encoded_tokens_temporal = self.temporal_transformer_encoder(encoded_tokens_time_split_subset,self._mask) 

        repop_encoder_tokens_temporal = torch.zeros(batch_size_s * num_patches, seq_len_s, encoded_tokens_sum.shape[-1], device=device)
        repop_encoder_tokens_temporal[batch_range_temporal, unmasked_indices_temporal_reshaped] = encoded_tokens_temporal

        encoded_tokens_temporal_spatial = repop_encoder_tokens_temporal.view(batch_size_s,-1,seq_len_s,encoded_tokens.shape[-1]).permute(0,2,1,3).reshape(-1, repop_encoder_tokens.shape[-2], repop_encoder_tokens.shape[-1])

        timestamps_out_embeds_repeat = timestamps_embeds[:,-seq_len_s:,:].unsqueeze(2).repeat(1,1,repop_encoder_tokens.shape[1],1)
        timestamps_out_embeds_repeat_reshaped = timestamps_out_embeds_repeat.reshape(-1,timestamps_out_embeds_repeat.shape[-2],timestamps_out_embeds_repeat.shape[-1])

        weather_out_embeds_repeat = x_w_embed_subset[:,-seq_len_s:,:].unsqueeze(2).repeat(1,1,encoded_tokens_temporal_spatial.shape[1],1)
        weather_out_embeds_repeat_reshaped = weather_out_embeds_repeat.reshape(-1,weather_out_embeds_repeat.shape[-2],weather_out_embeds_repeat.shape[-1])

        delt_timestamps_out_embeds_repeat = delt_timestamps_embeds[:,-seq_len_s:,:].unsqueeze(2).repeat(1,1,repop_encoder_tokens.shape[1],1)
        delt_timestamps_out_embeds_repeat_reshaped = delt_timestamps_out_embeds_repeat.reshape(-1,delt_timestamps_out_embeds_repeat.shape[-2],delt_timestamps_out_embeds_repeat.shape[-1])

        encoded_out_tokens_sum = encoded_tokens_temporal_spatial + weather_out_embeds_repeat_reshaped + timestamps_out_embeds_repeat_reshaped + delt_timestamps_out_embeds_repeat_reshaped       

        encoded_out_tokens_sum_subset = encoded_out_tokens_sum[batch_range, unmasked_indices_reshaped]
        embeddings = encoded_out_tokens_sum_subset.view(-1,seq_len_s,encoded_out_tokens_sum_subset.shape[-2],encoded_out_tokens_sum_subset.shape[-1])
        encoded_out_tokens_sum_subset = embeddings.view(-1,embeddings.shape[-2],embeddings.shape[-1])

        decoder_tokens = self.relu(self.forecasterlin2(self.relu(self.forecasterlin1(encoded_out_tokens_sum_subset))))

        # reapply decoder position embedding to unmasked tokens
        unmasked_decoder_tokens = decoder_tokens + self.decoder_pos_emb(unmasked_indices_reshaped)

        # concat the masked tokens to the decoder tokens and attend with decoder
        # mask_tokens = repeat(self.mask_token, 'd -> b n d', b = batch, n = masked_indices_reshaped.shape[-1])
        # mask_tokens = mask_tokens + self.decoder_pos_emb(masked_indices_reshaped)
        
        # print('unmasked_decoder_tokens',unmasked_decoder_tokens[1,1,5:10],unmasked_decoder_tokens[2,1,5:10])
        decoder_tokens = torch.zeros(batch, num_patches, self.decoder_dim, device=device)
        decoder_tokens[batch_range, unmasked_indices_reshaped] = unmasked_decoder_tokens
        # decoder_tokens[batch_range, masked_indices_reshaped] = mask_tokens

        decoded_tokens_out = self.decoder(decoder_tokens)
        # print('decoded_tokens_out',decoded_tokens_out.shape)

        pred_pixel_values = self.to_pixels(decoded_tokens_out)
        recon_output = self.pixel_values_to_out(pred_pixel_values)
        recon_output_reshape = recon_output.view(-1,seq_len_s,recon_output.shape[1],recon_output.shape[2],recon_output.shape[3])

        return recon_output_reshape, mask_binary, embeddings

if __name__ == "__main__":
    device = 'cuda:0'
    bsize = 4
    t = 6
    data_sat = torch.randn(bsize, t, 6, 128, 128)
    data_weather = torch.randn(bsize,365,5)
    timestamps = torch.randint(3, 300, (bsize,t+1)).float()
    delt_timestamps = torch.randint(3, 20, (bsize,t)).float()

    model = MM_VSF(image_size = 128, patch_size = 8, num_classes = 1000, dim = 512, channels = 6, depth = 4, heads = 8, mlp_dim = 1024, pool = 'cls', decoder_dim = 512, masking_ratio = 0.75, decoder_depth = 1, decoder_heads = 8, decoder_dim_head = 64, doy_embed_dim = 16, delt_embed_dim = 16, channels_w_in = 5,channels_w_out = 5, dim_w = 32)
    model = model.to(device)
    out_f,mask,embs = model(data_sat.to(device),data_weather.to(device),timestamps.to(device),delt_timestamps.to(device))
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("MM_VSF")
    print("Total parameters:",pytorch_total_params)
    print("Input Data Shapes: ",data_sat.shape, data_weather.shape)
    print("Output shape: ",out_f.shape)
    print("Embeddings shape",embs.shape)


### code for spatiotemporal uniform masking
# indented code to be inserted in the appropiate place

# patches_per_seq = int(num_patches*masking_seq_length/seq_len_s)
        # for b in range(batch_size_s):
        #     for timestamp in range(seq_len_s):
        #         if(seq_len_s - timestamp - 1 < masking_seq_length):
        #             indices_to_fill = torch.where(torch.sum(mask_binary[b],dim = -1) == (timestamp - (seq_len_s -  masking_seq_length)))
        #             mask_binary[b,indices_to_fill[0],timestamp:] = 1
                    
        #         indices = torch.where(torch.sum(mask_binary[b],dim = -1) < masking_seq_length)
        #         indice = random.sample(range(len(indices[0])), patches_per_seq - int(torch.sum(mask_binary[b,:,timestamp])))
        #         indice = torch.tensor(indice).to(device) 
        #         if(patches_per_seq - int(torch.sum(mask_binary[b,:,timestamp])) > 0):
        #             sampled_values = indices[0][indice]
        #             mask_binary[b,sampled_values,timestamp] = 1
# mask_binary_reshaped = mask_binary.permute(0,2,1).reshape(-1,num_patches)



