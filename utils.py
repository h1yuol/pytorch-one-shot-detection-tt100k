import torch

def compute_mat_dist(a,b,squared=False):
    """
    Input 2 embedding matrices and output distance matrix
    """
    assert a.size(1) == b.size(1)
    dot_product = a.mm(b.t())
    a_square = torch.pow(a, 2).sum(dim=1, keepdim=True)
    b_square = torch.pow(b, 2).sum(dim=1, keepdim=True)
    dist = a_square - 2*dot_product + b_square.t()
    dist = dist.clamp(min=0)
    if not squared:
        epsilon = 1e-12
        mask = (dist.eq(0))
        dist += epsilon * mask.float()
        dist = dist.sqrt()
        dist *= (1-mask.float())
    return dist