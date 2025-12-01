import torch

def similarity_measure_loss(real_features, fake_features):
    real_mean = torch.mean(real_features, dim=0)
    fake_mean = torch.mean(fake_features, dim=0)    

    loss = torch.norm(real_mean - fake_mean, p=2)

    return loss