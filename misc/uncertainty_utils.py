import torch


def get_entropy(probs, dim=-1):
    # probs: arbitrary tensor, dimension dim contains the probabilities
    return -torch.sum(probs.log() * probs, dim=dim)


def calculate_uncertainties(candidate_probs, candidate_weights):
    # candidate_probs: (batch_size, vocab_size, num_candidates)
    # calculate average mean prediction
    probs = candidate_probs.mul(candidate_weights).div(candidate_weights.sum()).sum(-1)
    # total entropy 
    total = get_entropy(probs)    # size: (batch_size,)
    # aleatoric part
    aleatoric = get_entropy(candidate_probs, dim=1).mul(candidate_weights).div(candidate_weights.sum()).sum(-1)
    # epistemic part
    epistemic = total - aleatoric
    return aleatoric, epistemic
