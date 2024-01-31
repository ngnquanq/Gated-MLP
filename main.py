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

def main(args):
    if args.train:
        print("Training...")
        # Add your training code here

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLI for training and testing")
    parser.add_argument("-train", action="store_true", help="Train the model")
    args = parser.parse_args()
    main(args)