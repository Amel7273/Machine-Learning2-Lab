import torch
import sys

from torchmetrics.aggregation import MeanMetric

def add_noise(x, mean = 0, std = 0.25):
    noise = torch.randn(x.size()) * std + mean
    # 무작위 작음은 torch.randn() 함수로 만들고 img.size()를 넣어 이미지와 같은 크기의 잡음을 만듭니다.
    # 잡음의 강도는 임의로 0.2로 정했습니다.
    noisy_x = x + noise
    return noisy_x

def train(loader, model, optimizer, scheduler, loss_fn, device): # train(loader, model, optimizer, scheduler, loss_fn, metric_fn, device):
    model.train()
    loss_mean = MeanMetric() # ??
    # metric_mean = MeanMetric() # ??
    cnt = 1
    for inputs, label in loader:
        print(f'cnt : {cnt}')
        cnt += 1
        targets = inputs
        inputs = add_noise(inputs)

        inputs = inputs.to(device)
        targets = targets.to(device)
        # label = label.to(device)

        encoded, decoded = model(inputs) # 별표 10개 ***********
        # print(f'type(targets) : {type(targets)}') # tensor
        # print(f'type(outputs) : {type(decoded)}') # tensor
        # print(f'len(targets) : {len(targets)}')
        # print(f'len(outputs) : {len(decoded)}')
        loss = loss_fn(decoded, targets)
        # metric = metric_fn(decoded, targets)

        optimizer.zero_grad() # 기울기에 대한 정보를 초기화합니다.
        loss.backward() # 기울기를 구합니다
        optimizer.step() # 최적화를 진행합니다
        
        loss_mean.update(loss.to('cpu'))
        # metric_mean.update(metric.to('cpu'))

        scheduler.step()

    summary = {'loss': loss_mean.compute()} # {'loss': loss_mean.compute(), 'metric': metric_mean.compute()}

    return summary