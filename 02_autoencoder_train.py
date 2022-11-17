import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--title", type=str, default="autoencoder")
parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--root", type=str, default="data")
parser.add_argument("--num_workers", type=int, default=8)
parser.add_argument("--batch_size", type=int, default=100)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--lr", type=float, default=0.001)
args = parser.parse_args()


def main(args):
    # Fill this
    # 1. you should build data processing pipe line to train an autoencoder
    #    * transforms for data processing pipe line 
    #    - ToTensor (PIL image -> torch tensor)
    #    - Reshape (28 x 28 -> 784)
    #    - Normalize Tensors (mean=0.5, std=0.25)
    #    - Add Gaussian noise (mean=0.0, std=0.25)
    # 2. you should build the autoencoder in src.autoencoders.py
    # 3. you should build performenace metric (MSE), loss function (MSE), 
    #    optimizer (Adam), and learning rate scheduler (CosineSchedule)
    # 4. you should build training and evaluation loop
    # 5. you should train the autoencoder
    # 6. you should save the learning parameters of autoencoder 


if __name__=="__main__":
    main(args)