import argparse

from torchvision.datasets import FashionMNIST
from torchvision.transforms.functional import normalize
from src.kmeans import kmeans
from src.metrics import clustering_accuracy, compute_mean_and_std
from einops import rearrange

parser = argparse.ArgumentParser()
parser.add_argument("--root", type=str, default='data')
parser.add_argument("--num_clusters", type=int, default=10)
parser.add_argument("--num_iterations", type=int, default=20)
parser.add_argument("--num_trials", type=int, default=5)
args = parser.parse_args()

def load_fashionmnist(root):
    dataset = FashionMNIST(root=root, train=True, download=True)
    examples, targets = dataset.data, dataset.targets
    examples = examples.float()
    examples = rearrange(examples, 'n h w -> n 1 h w')
    examples = normalize(examples, mean=(0.5,), std=(0.25,))
    examples = rearrange(examples, 'n 1 h w -> n (1 h w)')
    return examples, targets

def main(args):
    examples, targets = load_fashionmnist(args.root)
    acc_list = []
    # print(f'type(examples) : {type(examples)}')
    # print(f'examples.shape : {examples.shape}') # torch.Size([60000, 784])
    # print(f'targets.shape : {targets.shape}') # torch.Size([60000])
    for _ in range(args.num_trials):
        _, predictions = kmeans(examples, args.num_clusters, args.num_iterations)
        accuracy = clustering_accuracy(predictions, targets, args.num_clusters)
        print(f'K: {args.num_clusters}, Acc.: {accuracy:.4f}')
        acc_list.append(accuracy)
    mean, std = compute_mean_and_std(acc_list)
    
    print(f'accuracy mean : {mean}')
    print(f'accuracy std : {std}')

if __name__=="__main__":
    main(args)