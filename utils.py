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

def bb_intersection_over_union(boxA, boxB):
    """
    copied from https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
    """
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
 
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
 
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
 
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
 
    # return the intersection over union value
    return iou

def computeIOU_torch(boxA, boxB):
    """
    Input:
        - boxA: (m, 4) torch.Tensor
        - boxB: (n, 4) torch.Tensor
    Intermediate Variables:
        - xA, yA, xB, yB, interArea: (m, n)
        - boxAArea: (m)
        - boxBArea: (n)
    Output:
        - iou: (m, n)
    """
    xA = torch.max(boxA[:,0].unsqueeze(1), boxB[:,0].unsqueeze(1).t())
    yA = torch.max(boxA[:,1].unsqueeze(1), boxB[:,1].unsqueeze(1).t())
    xB = torch.min(boxA[:,2].unsqueeze(1), boxB[:,2].unsqueeze(1).t())
    yB = torch.min(boxA[:,3].unsqueeze(1), boxB[:,3].unsqueeze(1).t())

    interArea = (xB-xA+1).clamp(min=0) * (yB-yA+1).clamp(min=0)
    
    boxAArea = (boxA[:,2]-boxA[:,0]+1) * (boxA[:,3]-boxA[:,1]+1)
    boxBArea = (boxB[:,2]-boxB[:,0]+1) * (boxB[:,3]-boxB[:,1]+1)

    iou = interArea / (boxAArea.unsqueeze(1) + boxBArea.unsqueeze(1).t() - interArea).float()
    return iou

















