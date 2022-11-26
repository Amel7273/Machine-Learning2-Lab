import torch
import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T

from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
from torchvision.transforms.functional import normalize
from torchmetrics.functional.classification import accuracy

#from einops import rearrange

from src.autoencoders import Autoencoder
from src.engines import train
from src.utils import load_checkpoint, save_checkpoint



parser = argparse.ArgumentParser()
parser.add_argument("--title", type=str, default="autoencoder")
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--root", type=str, default="data")
parser.add_argument("--num_workers", type=int, default=2)
parser.add_argument("--batch_size", type=int, default=100)
parser.add_argument("--epochs", type=int, default=10) #원래는 100이지만 빠른 출력을 위해 10
parser.add_argument("--checkpoints", type=str, default='checkpoints')
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--resume", type=bool, default=False)

args = parser.parse_args()

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

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
    
    # Build dataset
    train_transform = T.Compose([
        T.ToTensor(),
        T.Resize((784)),
        T.Normalize((0.5), (0.25)),
        #AddGaussianNoise(0., 0.25),
    ])
    train_data = FashionMNIST(args.root, train=True, download=True, transform=train_transform)
    # print(f'type(train_data) : {type(train_data)}')
    # print(f'dir(train_data) : {dir(train_data)}')
    train_loader = DataLoader(train_data, args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    # print(f'type(train_loader) : {type(train_loader)}')
    # print(f'dir(train_loader) : {dir(train_loader)}')

    # img, label = train_data[0] 
    # plt.imshow(img.squeeze(), cmap="gray")
    # plt.show()

    # Build model
    model = Autoencoder()
    model = model.to(args.device)

     # Build optimizer 
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Build scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs * len(train_loader))

    # Build loss function
    loss_fn = nn.MSELoss()

    # Build metric function
    # metric_fn = accuracy

    # 원본 이미지를 시각화 하기 (첫번째 열)
    view_data = train_data.data[:5].view(-1, 28*28)
    # 복원이 어떻게 되는지 관찰하기 위해 5개의 이미지를 가져와 바로 넣어보겠습니다.
    view_data = view_data.type(torch.FloatTensor)/255.
    #픽셀의 색상값이 0~255이므로 모델이 인식하는 0부터 1사이의 값으로 만들기 위해 255로 나눠줍니다.

    # Load model
    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(args.checkpoints, args.title, model, optimizer)

    # Main loop
    for epoch in range(start_epoch, args.epochs):

        # train one epoch
        train_summary = train(train_loader, model, optimizer, scheduler, loss_fn, args.device) # train(train_loader, model, optimizer, scheduler, loss_fn, metric_fn, args.device)
        
        # 디코더에서 나온 이미지를 시각화 하기
        # 앞서 시각화를 위해 남겨둔 5개의 이미지를 한 에폭만큼 학습을 마친 모델에 넣어 복원이미지를 만듭니다.
        test_x = view_data.to(args.device)
        _, decoded_data = model(test_x)

        # 원본과 디코딩 결과 비교해보기
        f, a = plt.subplots(2, 5, figsize=(5, 2))
        print("[Epoch {}]".format(epoch))
        for i in range(5):
            img = np.reshape(view_data.data.numpy()[i],(28, 28)) #파이토치 텐서를 넘파이로 변환합니다.
            a[0][i].imshow(img, cmap='gray')
            a[0][i].set_xticks(()); a[0][i].set_yticks(())

        for i in range(5):
            img = np.reshape(decoded_data.to("cpu").data.numpy()[i], (28, 28)) 
            # CUDA를 사용하면 모델 출력값이 GPU에 남아있으므로 .to("cpu") 함수로 일반메모리로 가져와 numpy행렬로 변환합니다.
            # cpu를 사용할때에도 같은 코드를 사용해도 무방합니다.
            a[1][i].imshow(img, cmap='gray')
            a[1][i].set_xticks(()); a[1][i].set_yticks(())
        plt.show()

        # save model
        save_checkpoint(args.checkpoints, args.title, model, optimizer, epoch + 1)

        print('Loss', train_summary['loss'], epoch + 1)
        # print('Accuracy', train_summary['metric'], epoch + 1)

if __name__=="__main__":
    main(args)