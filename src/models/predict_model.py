import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.data.make_dataset import *
from components import *

train_loader, test_loader = get_loaders()
print(train_loader.dataset[0])

gMLP = gMLP(vocab_size = 1000, d_model =  256, 
            d_ffn = 768, seq_len = 150, 
            num_layers = 128, num_classes = 2)

