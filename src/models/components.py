from torch.utils.data import Dataset, DataLoader
import torch
from transformers import BertTokenizer
from random import randrange
import torch.nn.functional as F
from torch import nn, einsum
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.functional import to_map_style_dataset
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from torchviz import make_dot


import time
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

class SpatialGatingUnit(nn.Module):
    def __init__(self, d_ffn, 
                 seq_len, weight_value=0.05):
        super().__init__()
        self.norm = nn.LayerNorm(d_ffn//2)
        
        # Setup weight for the spatial projection
        self.weight = nn.Parameter(torch.zeros(seq_len,seq_len))
        nn.init.uniform_(self.weight, a=-weight_value, b=weight_value)
        
        # Setup bias for the spatial projection
        self.bias = nn.Parameter(torch.ones(seq_len))

    def forward(self, x):
        u, v = x.chunk(2, dim=-1)
        v = self.norm(v)
        
        weight, bias = self.weight, self.bias
        v = einsum('b n d, m n -> b m d', v, weight) + rearrange(bias, 'n -> () n ()')
        return u * v
    
class gMLPBlock(nn.Module):
    def __init__(self, d_model, d_ffn, seq_len):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.channel_proj_U = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.GELU()
        )
        self.sgu = SpatialGatingUnit(d_ffn, seq_len)
        self.channel_proj_V = nn.Sequential(
            nn.Linear(d_ffn//2, d_model),
            nn.GELU()
        )
        
    def forward(self, x):
        res = x
        x = self.norm(x)
        x = self.channel_proj_U(x)
        x = self.sgu(x)
        x = self.channel_proj_V(x)
        return x + res
    
    
class gMLP(nn.Module):
    def __init__(self, vocab_size, d_model, d_ffn, seq_len, num_layers, num_classes):
        super().__init__()
        self.Embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.layers.append(gMLPBlock(d_model, d_ffn, seq_len))
        self.classifier = nn.Linear(d_model, num_classes)
            
    def forward(self, x):
        x = self.Embedding(x)
        for layer in self.layers:
            x = layer(x)
        x = x.mean(dim=1)
        return self.classifier(x)