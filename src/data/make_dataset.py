import sys
import os
sys.path.append(r'C:\Users\84898\Desktop\project\Complete\Gated MLP')

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import torchtext 
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.functional import to_map_style_dataset
from torch.utils.data import DataLoader, Dataset

seq_length = 128
batch_size = 32
tokenizer = get_tokenizer("basic_english")

def load_data():
    train = pd.read_csv(r'C:\Users\84898\Desktop\project\Complete\Gated MLP\data\train.csv')
    test = pd.read_csv(r'C:\Users\84898\Desktop\project\Complete\Gated MLP\data\test.csv')
    return train, test

def yield_tokens(sentences, tokenizer):
    for sentence in sentences:
        yield tokenizer(sentence)
        

def prepare_dataset(df, vocabulary, tokenizer):
    for index, row in df.iterrows():
        sentence = row['text']
        encoded_sentence = vocabulary(tokenizer(sentence))
        label = torch.tensor(row['label'], dtype = torch.long)
        yield encoded_sentence, label
        
def collate_batch(batch):
    sentences, labels = list(zip(*batch))
    encoded_sentences = [
        sentence+([0]* (seq_length-len(sentence))) if len(sentence) < seq_length else sentence[:seq_length]
        for sentence in sentences
    ]

    encoded_sentences = torch.tensor(encoded_sentences, dtype=torch.int64)
    labels = torch.tensor(labels)

    return encoded_sentences, labels

def get_loaders(tokenizer, batch_size, vocab_size=1000):
    # Get data frame
    train_df, test_df = load_data()
    # Build vocabulary
    vocabulary = build_vocab_from_iterator(
        yield_tokens(train_df['text'], tokenizer),
        max_tokens=vocab_size,
        specials=["<unk>"]
    )
    vocabulary.set_default_index(vocabulary["<unk>"])
    # Build dataset
    train_dataset = prepare_dataset(train_df, vocabulary, tokenizer)
    train_dataset = to_map_style_dataset(train_dataset)
    test_dataset = prepare_dataset(test_df, vocabulary, tokenizer)
    test_dataset = to_map_style_dataset(test_dataset)
    # Get data loaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
    return train_dataloader, test_dataloader

