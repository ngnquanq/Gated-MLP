import sys
sys.path.append(r'C:\Users\84898\Desktop\project\Complete\Gated MLP')

import argparse
from src.data.make_dataset import *
from src.models.components import *
from src.models.predict_model import *
from src.models.train_model import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import logging
import os
import sys


import torch
import os

def main(args):
    model = None
    model_path = "model.pth"

    if args.create_model:
        print("Creating model...")
        model = gMLP(vocab_size=args.vocab_size, d_model=args.d_model, d_ffn=args.d_ffn, 
                     seq_len=args.seq_len, num_layers=args.num_layers, num_classes=args.num_classes)
        print(f"The model has {count_parameters(model):,} parameters")
        print("Completed creating model")
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    if args.train:
        if not os.path.isfile(model_path):
            print("No model has been created. Please create a model before training.")
        else:
            print("Loading model...")
            model = gMLP(vocab_size=args.vocab_size, d_model=args.d_model, d_ffn=args.d_ffn, 
                         seq_len=args.seq_len, num_layers=args.num_layers, num_classes=args.num_classes)
            model.load_state_dict(torch.load(model_path))
            print("Model loaded")
            print("Training...")
            print("The model Structure:")
            print(model)
            train_dataloader, val_dataloader = get_loaders(tokenizer=get_tokenizer("basic_english"),
                                                           vocab_size=args.vocab_size,
                                                           batch_size=32)
            model_name = "gMLP"
            save_model = os.path.join(os.getcwd(), r"src\features", model_name)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()
            num_epochs = 10
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            model, metrics = train(model, model_name, save_model, optimizer, criterion, 
                                   train_dataloader, val_dataloader, 
                                   num_epochs, device, 'gMLP_metrics.csv') 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLI for creating and training a model")
    parser.add_argument("-create_model", action="store_true", help="Create a new model")
    parser.add_argument("-train", action="store_true", help="Train the model")
    parser.add_argument("--vocab_size", type=int, default=1000, help="Vocabulary size")
    parser.add_argument("--d_model", type=int, default=256, help="Dimension of the model")
    parser.add_argument("--d_ffn", type=int, default=512, help="Dimension of the feedforward network")
    parser.add_argument("--seq_len", type=int, default=128, help="Sequence length")
    parser.add_argument("--num_layers", type=int, default=6, help="Number of layers")
    parser.add_argument("--num_classes", type=int, default=2, help="Number of classes")
    args = parser.parse_args()
    main(args)