import numpy as np
import sys

from scipy.optimize import linear_sum_assignment

def clustering_accuracy(predictions, targets, num_clusters=10):
    # predictions : (tensor) [60000], tragets : (tensor) [60000], num_clsuters : (int) 10
    # Torch tensor to Numpy array
    predictions = predictions.numpy()
    targets = targets.numpy()
    
    # Build graph
    matching_cost = np.zeros((num_clusters, num_clusters)) # (Ndarray) [10, 10]
    for i in range(num_clusters):
        for j in range(num_clusters):
            matching_cost[i][j] = -np.sum(np.logical_and(predictions == i, targets == j))
    
    # Bipartite graph matching (Hungarian algorithm)

    np.set_printoptions(precision=4, suppress=True)

    print("matching_cost : ")
    print(matching_cost)
    indices = linear_sum_assignment(matching_cost)
    print(f'type(indices) : {type(indices)}')
    print("indicies : ")
    print(indices)

    # Compute accuracy
    permuation = []
    for i in range(num_clusters):
        permuation.append(indices[1][i])
    
    pred_corresp = [permuation[int(p)] for p in predictions]
    print(f"pred_corresp : {len(pred_corresp)}")
    accuracy = np.sum(pred_corresp == targets) / float(len(targets))
    return accuracy
    
def compute_mean_and_std(accuracy):
    mean = np.mean(accuracy)
    std = np.std(accuracy)
    return mean, std