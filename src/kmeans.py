import torch
import random
import numpy as np
import sys
from einops import rearrange, reduce
        

def kmeans(examples, K, T):
    """ kmeans clustering
    Args:
        examples: (torch.Tensor) N x D tensor [60000, 784]
        K: (int) The number of clusters (10)
        T: (int) The number of iterations (20)
    Returns:
        clusters: (torch.Tensor) K x D tensor [10, 784]
        predictions: (torch.Tensor) N tensor [60000]
    """
    clusters = init_clusters(examples, K)
    for _ in range(T):
        distances = compute_distance(examples, clusters) # (tensor) [60000, 10]
        predictions = find_nearest_cluster(distances) # (tensor) [60000]
        clusters = update_clusters(examples, clusters, predictions, K) # (tensor) [10, 784]
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

    # examples : (tensor) [60000, 784], K : (int) 10
    clusters = torch.unbind(examples, dim=0) # (tuple) [60000,784]
    clusters = random.sample(clusters, k=1) # (list) [1, 784]
    clusters = torch.stack(clusters, dim=0) # (torch) [1, 784]

    for _ in range(1,K):
        chosen_clusters = clusters # (torch) [1, 784]
        distances = compute_distance(examples, chosen_clusters) # (torch) [60000, 1]
        nearest_distance = torch.min(distances, dim=-1)[0] # torch [60000]
        # d / sum(d)로 확률로 나타내는 대신, 각 d들을 가중치로 샘플링을 수행하는 random.choices 메서드를 사용했다.
        idx_list = range(examples.shape[0]) # (list) [60000]
        chosen_idx = random.choices(idx_list, weights = nearest_distance, k=1) # list [1, 784]
        clusters = torch.cat([clusters,examples[chosen_idx]],dim=0)
    return clusters

def compute_distance(examples, clusters):
    # exampels : (tensor) [60000, 784], clusters : (tensor) [10, 784]
    # n : 샘플 개수, c : feature 수, k : cluster 개수
    examples = rearrange(examples, 'n c -> n 1 c') # (tensor) [60000, 1, 784]
    clusters = rearrange(clusters, 'k c -> 1 k c') # (tensor) [1, 10, 784]
    distances = reduce((examples - clusters) ** 2, 'n k c -> n k', 'sum') # (tensor) [60000, 10]
    return distances


def find_nearest_cluster(distances):
    # distances : (tensor) [60000,10]
    cluster_ids = torch.argmin(distances, dim=-1) # (tensor) [60000]
    return cluster_ids


def update_clusters(examples, clusters, cluster_ids, K):
    # examples : (tensor) [60000, 784], clsuters : (tensor) [10, 784], clsuter_ids : (tensor) [60000], K : (int) 10
    for k in range(K):
        example_ids = torch.where(cluster_ids==k)[0]
        # print(f'exmple_ids.shape : {example_ids.shape}') # torch.Size([5037]) : k=0인 샘플의 개수
        # print(f'example_ids : {example_ids}') # tensor([   11,    23,    60,  ..., 59970, 59974, 59978]) : k=0인 샘플의 인덱스가 찍힌 텐서

        #continue
        if len(example_ids) > 0:
            cluster_examples = examples[example_ids, ...] 
            # print(f'cluster_examples.shape : {cluster_examples.shape}') # (tensor) [5037, 784]
            clusters[k] = reduce(cluster_examples, 'm c -> c', 'mean') # (tensor) [784]
    return clusters # (tensor) [10, 784]


def compute_cost(distances):
    cost = reduce(distances, 'n m -> n', 'min')
    cost = reduce(cost ** 2, 'n -> 1', 'mean')
    return cost
