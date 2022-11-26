import torch
import random
import numpy as np
from einops import rearrange, reduce
        

def kmeans(examples, K, T):
    """ kmeans clustering
    Args:
        examples: (torch.Tensor) N x D tensor
        K: (int) The number of clusters
        T: (int) The number of iterations
    Returns:
        clusters: (torch.Tensor) K x D tensor
        predictions: (torch.Tensor) N tensor
    """
    clusters = init_clusters_advanced(examples, K)
    for _ in range(T):
        distances = compute_distance(examples, clusters)
        predictions = find_nearest_cluster(distances)
        clusters = update_clusters(examples, clusters, predictions, K)
    # print(f'clusters.shape : {clusters.shape}') # torch.Size([10, 784])
    # print(f'predictions.shape : {predictions.shape}') # torch.Size([60000])
    return clusters, predictions

def init_clusters(examples, K):
    clusters = torch.unbind(examples, dim=0) # 60000개의 튜플, 784차원의 텐서
    # print(f'type(clusters) - unbind : {type(clusters)}') # tuple
    # print(f'len(clusters) - unbind : {len(clusters)}') # 60000
    clusters = random.sample(clusters, k=K) # 60000개의 튜플 중 10개의 선택, 리스트 반환
    # print(f'type(clusters) - sample : {type(clusters)}') # list
    # print(f'len(clusters) - sample : {len(clusters)}') # 10
    clusters = torch.stack(clusters, dim=0) # 10개의 리스트를 stack
    # print(f'clusters.shape : {clusters.shape}') # torch.Size([10, 784])
    return clusters


def init_clusters_advanced(examples, K):
    """ Implement K-means ++ algorithm
    """
    print(f'examples.shape : {examples.shape}')
    clusters = torch.unbind(examples, dim=0) # (tuple) (60000,784)
    clusters = random.sample(clusters, k=1) # (list) (1, 784)
    clusters = torch.stack(clusters, dim=0) # (torch) (1, 784)

    for k in range(1,K):
        chosen_clusters = clusters # (torch) (1, 784)
        distances = compute_distance(examples, chosen_clusters) # (torch) (60000, 1) 
        nearest_distance, _ = torch.min(distances, dim=-1) # torch (60000)
        chosen_sample = random.choices(examples, weights = nearest_distance, k=1) # list (1, 784)
        chosen_sample = torch.stack(chosen_sample, dim=0) # tensor (1, 784)
        clusters = torch.cat([clusters,chosen_sample],dim=0)
    return clusters

def compute_distance(examples, clusters):
    # n : 샘플 개수, c : feature 수, k : cluster 개수
    examples = rearrange(examples, 'n c -> n 1 c')
    clusters = rearrange(clusters, 'k c -> 1 k c')
    distances = reduce((examples - clusters) ** 2, 'n k c -> n k', 'sum')
    return distances


def find_nearest_cluster(distances):
    cluster_ids = torch.argmin(distances, dim=-1)
    return cluster_ids


def update_clusters(examples, clusters, cluster_ids, K):
    for k in range(K):
        example_ids = torch.where(cluster_ids==k)[0]
        if len(example_ids) > 0:
            cluster_examples = examples[example_ids, ...]
            clusters[k] = reduce(cluster_examples, 'm c -> c', 'mean')
    return clusters


def compute_cost(distances):
    cost = reduce(distances, 'n m -> n', 'min')
    cost = reduce(cost ** 2, 'n -> 1', 'mean')
    return cost
