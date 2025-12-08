import numpy as np

def prior_adjustment(probabilities, train_counts):
    total_samples = sum(train_counts)
    p_train = np.array(train_counts) / total_samples
    p_balanced = np.ones_like(train_counts) / len(train_counts)
    adj_factors = p_balanced / p_train
    adjusted = probabilities * adj_factors
    adjusted /= adjusted.sum(axis=1, keepdims=True)
    return adjusted