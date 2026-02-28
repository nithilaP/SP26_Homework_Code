"""
Originally forked from Andrej Karpathy's minGPT,
Modified based on Stanford CS224N 2023-24: Homework 4

John Hewitt <johnhew@stanford.edu>
Ansh Khurana <anshk@stanford.edu>
Soumya Chatterjee <soumyac@stanford.edu>
"""

import math
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)


def precompute_rotary_emb(dim, max_positions):
    """
    RoPE uses the following sinusoidal functions to encode positions:

    cos(t theta_i) and sin(t theta_i)
        where t is the position and
              theta_i = 1/10000^(-2(i-1)/dim) for i in [1, dim/2]

    Since the maximum length of sequences is known, we can precompute
    these values to speed up training.

    Implement the precompute_rotary_emb function that returns a tensor of
    shape (max_positions, dim/2, 2) where the last dimension contains
    the cos and sin values for each position and each dimension of
    the embedding.
    """

    rope_cache = None
    # TODO: [part g]
    ### YOUR CODE HERE ###
    
    # HW CHECK IN (2 LINES) -> need to do arange for i-1 in theta calc
    # TODO: INSERT COMMENT
    positions = torch.arange(max_positions, device=torch.device("cpu"), dtype=torch.float32);
    i = torch.arange(1, dim//2 + 1, device=torch.device("cpu"), dtype=torch.float32); # from above: for i in [1, dim/2

    # dim 1 tensor : https://docs.pytorch.org/docs/main/generated/torch.unsqueeze.html
    positions = positions.unsqueeze(1); # [[a], [b], [c], ...]
    i = i.unsqueeze(0); # [[ a,  b,  c, ... ]]

    # FROM STAFF: 
    # cos(t theta_i) and sin(t theta_i)
    #   where t is the position and
    #         theta_i = 1/10000^(-2(i-1)/dim) for i in [1, dim/2] */
    theta_i = (1/(10000 ** ((2 * (i-1))/ dim))) # FIX: need decreasing freq -> ^ 2.. instead of -2
    angle_vals = positions * theta_i;

    # stack for apply rotary embedding
    rope_cache = torch.stack([torch.cos(angle_vals), torch.sin(angle_vals)], dim=-1);

    ### END YOUR CODE ###
    return rope_cache


def apply_rotary_emb(x, rope_cache):
    """Apply the RoPE to the input tensor x."""
    # TODO: [part g]
    # You might find the following functions useful to convert
    # between real and complex numbers:

    # torch.view_as_real - https://pytorch.org/docs/stable/generated/torch.view_as_real.html
    # torch.view_as_complex - https://pytorch.org/docs/stable/generated/torch.view_as_complex.html

    # Note that during inference, the length of the sequence might be different
    # from the length of the precomputed values. In this case, you should use
    # truncate the precomputed values to match the length of the sequence.

    rotated_x = None
    ### YOUR CODE HERE ###

    # ATTEMPT 1: Even, odd Split instead -> Gave me 10% accuracy!
    # rope_trimmed = rope_cache[:T] # size = [T, dimension pair = head_size / 2]
    # complex_rope = torch.view_as_complex(rope_trimmed.float())

    # even = x[:,:,:,::2]
    # odd = x[:,:,:,1::2]
    # x_out = torch.stack((even, odd), dim=4).contiguous()

    # complex_x = torch.view_as_complex(x_out) # FIX: neex x.float() here for view+as_complex
    # rotated_x = complex_rope * complex_x

    # rotated_x = torch.view_as_read(rotated_x)
    # rotated_x = rotated_x.reshape(B, n_head, T, head_size)

    # ATTEMPT 2: Use same process as eq from proof question 
    # https://aiexpjourney.substack.com/p/an-in-depth-exploration-of-rotary-position-embedding-rope-ac351a45c794
    # https://github.com/naver-ai/rope-vit/blob/main/deit/models_v2_rope.py

    # unpack x 
    B = x.shape[0]
    n_head = x.shape[1]
    T = x.shape[2]
    head_size = x.shape[3]

    # fit cache into length of seq T
    rope_trimmed = rope_cache[:T] # size = [T, dimension pair = head_size / 2]
    complex_rope = torch.view_as_complex(rope_trimmed.float()) # input cache as float
    # For comp for each batch & head -> when mult w complex atten tensor. -> https://docs.pytorch.org/docs/stable/generated/torch.Tensor.view.html
    complex_rope = complex_rope.view(1, 1, T, head_size // 2) # [1, T, dimension pair = head_size / 2]

    # torch.view_as_complex - https://pytorch.org/docs/stable/generated/torch.view_as_complex.html
    # when debugging, verified that last dim of x is pairs of real, img after reshaping
    complex_x = torch.view_as_complex(x.float().reshape(B, n_head, T, head_size // 2, 2)) # FIX: neex x.float() here -> last dim of x is pairs of real, img after reshaping, https://github.com/naver-ai/rope-vit/blob/main/deit/models_v2_rope.py

    # do elem wise rotation clac
    rotated_x = complex_x * complex_rope

    # convert back to real.
    rotated_x = torch.view_as_real(rotated_x)
    rotated_x = rotated_x.reshape(B, n_head, T, head_size) # sizeback from [...h/2, 2] -> h

    ### END YOUR CODE ###

    return rotated_x

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    I believe I could have just used torch.nn.MultiheadAttention but their documentation
    is all but absent and code ugly so I don't trust it, rolling my own here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd)

        self.rope = config.rope
        if self.rope:
            assert (config.n_embd % config.n_head) % 2 == 0

            # TODO: [part g] Precompute the cos and sin values for RoPE and
            # store them in rope_cache.
            # Hint: The maximum sequence length is given by config.block_size.
            rope_cache = None
            ### YOUR CODE HERE ###
            
            rope_cache = precompute_rotary_emb(dim=(config.n_embd // config.n_head), max_positions=config.block_size)

            ### END YOUR CODE ###

            self.register_buffer("rope_cache", rope_cache)

        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)

        # TODO: causal mask to ensure that attention is only applied to the left in the input sequence
        mask = None
        ### YOUR CODE HERE ###
        
        block_matrix = torch.ones(config.block_size, config.block_size)
        low_triangle_matrix = torch.tril(block_matrix) # makes sure that token n can only see token 0 - (n-1)
        mask = low_triangle_matrix.view(1, 1,config.block_size, config.block_size) # make it 4D now for mul for attn tens

        ### END YOUR CODE ###

        self.register_buffer("mask", mask)
        self.n_head = config.n_head

    def forward(self, x):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        qkv = self.qkv(x).view(B, T, self.n_head, 3 * (C // self.n_head)).transpose(1, 2) # (B, nh, T, 3*hs)
        q, k, v = qkv.chunk(3, dim=-1)  # each (B, nh, T, hs)

        if self.rope:
            # TODO: [part g] Apply RoPE to the query and key.
            ### YOUR CODE HERE ###
            
            # query 
            q = apply_rotary_emb(q, rope_cache=self.rope_cache)

            # key 
            k = apply_rotary_emb(k, rope_cache=self.rope_cache)

            ### END YOUR CODE ###

        # TODO: causal self-attention
        # 1. compute attention map (pre-softmax)
        # 2. apply attention mask to the attention map
        # 3. apply softmax to the attention map (hint: masked parts should not be included in the softmax)
        # 4. apply attention dropout to the attention map
        # 5. compute the output by applying the attention map to the value
        # 6. re-assemble all head outputs side by side
        y = None
        ### YOUR CODE HERE ###
        
        # 1. compute attention map (pre-softmax) -> (QK^T/sqrt(d_k))
        K_T = k.transpose(2, 3) # transpose to head_size, T
        d = C // self.n_head
        attention_map = (q @ K_T) / d**0.5

        # 2. apply attention mask to the attention map
        trim_mask = self.mask[:, :, :T, :T] # fit to current seq list
        # attention_map = attention_map + self.mask[:T, :T]
        attention_map = attention_map.masked_fill(trim_mask == 0, -1e9) # replace all 0 w -1e9 for softmax

        # 3. apply softmax to the attention map (hint: masked parts should not be included in the softmax)
        attention_map = F.softmax(attention_map, dim=3)

        # 4. apply attention dropout to the attention map
        attention_map = self.attn_drop(attention_map)

        # 5. compute the output by applying the attention map to the value
        y = attention_map @ v # current dim = B, n_head, T, head_size

        # 6. re-assemble all head outputs side by side -> set to (B, T, C)
        y = y.transpose(1, 2) # switch T & n_head
        y = y.reshape(B, T, C) # worked without .contiguous()

        ### END YOUR CODE ###

        # output projection
        y = self.resid_drop(self.proj(y))
        return y
