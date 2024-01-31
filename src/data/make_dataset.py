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
    train = pd.read_csv("../../data/train.csv")
    test = pd.read_csv("../../data/test.csv")
    return train, test

def yield_tokens(data_iter,tokenizer):
    for _, text in data_iter:
        yield tokenizer(text)
        
def build_vocab(data, tokenizer, vocab_size = 1000):

    vocab = build_vocab_from_iterator(yield_tokens(data['text']),
                                      tokenizer, 
                                      max_tokens=vocab_size,
                                      specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])
    return vocab

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

def get_loaders():
    # Get data frame
    train_df, test_df = load_data()
    # Build vocabulary
    vocab = build_vocab(train_df, tokenizer)
    # Build dataset
    train_dataset = prepare_dataset(train_df, vocab, tokenizer)
    train_dataset = to_map_style_dataset(train_dataset)
    test_dataset = prepare_dataset(test_df, vocab, tokenizer)
    test_dataset = to_map_style_dataset(test_dataset)
    # Get data loaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
    return train_dataloader, test_dataloader

