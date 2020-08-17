
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from .yi2018cvpr.model import Net as FeatureExtractor
from .yi2018cvpr.config import get_config, print_usage
from lib.optimal_transport import RegularisedTransport
from lib.nonlinear_weighted_blind_pnp import NonlinearWeightedBlindPnP

def pairwiseL2Dist(x1, x2):
    """ Computes the pairwise L2 distance between batches of feature vector sets

    res[..., i, j] = ||x1[..., i, :] - x2[..., j, :]||
    since 
    ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a^T*b

    Adapted to batch case from:
        jacobrgardner
        https://github.com/pytorch/pytorch/issues/15253#issuecomment-491467128
    """
    x1_norm2 = x1.pow(2).sum(dim=-1, keepdim=True)
    x2_norm2 = x2.pow(2).sum(dim=-1, keepdim=True)
    res = torch.baddbmm(
        x2_norm2.transpose(-2, -1),
        x1,
        x2.transpose(-2, -1),
        alpha=-2
    ).add_(x1_norm2).clamp_min_(1e-30).sqrt_()
    return res

def ransac_p3p(P, p2d, p3d, num_points_2d, num_points_3d):
    # 1. Choose top k correspondences:
    k = min(1000, round(1.5 * p2d.size(-2))) # Choose at most 1000 points
    _, P_topk_i = torch.topk(P.flatten(start_dim=-2), k=k, dim=-1, largest=True, sorted=True)
    p2d_indices = P_topk_i / P.size(-1) # bxk (integer division)
    p3d_indices = P_topk_i % P.size(-1) # bxk
    K = np.float32(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]))
    dist_coeff = np.float32(np.array([0.0, 0.0, 0.0, 0.0]))
    theta0 = P.new_zeros((P.size(0), 6))
    # 2. Loop over batch and run RANSAC:
    for i in range(P.size(0)):
        num_points_ransac = min(k, round(1.5 * num_points_2d[i].float().item()), round(1.5 * num_points_3d[i].float().item()))
        num_points_ransac = min(k, max(num_points_ransac, 10)) # At least 10 points
        p2d_np = p2d[i, p2d_indices[i, :num_points_ransac], :].cpu().numpy()
        p3d_np = p3d[i, p3d_indices[i, :num_points_ransac], :].cpu().numpy()
        retval, rvec, tvec, inliers = cv2.solvePnPRansac(
            p3d_np, p2d_np, K, dist_coeff,
            iterationsCount=1000,
            reprojectionError=0.01,
            flags=cv2.SOLVEPNP_P3P)
        # print(inliers.shape[0], '/',  num_points_2d[i].item())
        if rvec is not None and tvec is not None and retval:
            rvec = torch.as_tensor(rvec, dtype=P.dtype, device=P.device).squeeze(-1)
            tvec = torch.as_tensor(tvec, dtype=P.dtype, device=P.device).squeeze(-1)
            if torch.isfinite(rvec).all() and torch.isfinite(tvec).all():
                theta0[i, :3] = rvec
                theta0[i, 3:] = tvec
    return theta0

class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        iden = torch.eye(3, dtype=x.dtype, device=x.device).unsqueeze(0)
        x = x.view(-1, 3, 3) + iden
        return x

class DBPnP(nn.Module):
    def __init__(self, args):
        super(DBPnP, self).__init__()
        self.config_2d, _ = get_config()
        self.config_3d, _ = get_config()
        self.config_2d.in_channel = 2
        self.config_2d.gcn_in_channel = 2
        self.config_3d.in_channel = 3
        self.config_3d.gcn_in_channel = 3
        self.stn = STN3d()
        self.FeatureExtractor2d = FeatureExtractor(self.config_2d)
        self.FeatureExtractor3d = FeatureExtractor(self.config_3d)
        self.pairwiseL2Dist = pairwiseL2Dist
        self.sinkhorn_mu = 0.1
        self.sinkhorn_tolerance=1e-9
        self.sinkhorn = RegularisedTransport(self.sinkhorn_mu, self.sinkhorn_tolerance)
        self.ransac_p3p = ransac_p3p
        self.wbpnp = NonlinearWeightedBlindPnP()

    def forward(self, p2d, p3d, num_points_2d, num_points_3d, poseloss):
        f2d = p2d
        f3d = p3d
        # Transform f3d to canonical coordinate frame:
        trans = self.stn(f3d.transpose(-2, -1)) # bx3x3
        f3d = torch.bmm(f3d, trans) # bxnx3
        # Extract features:
        f2d = self.FeatureExtractor2d(f2d.transpose(-2,-1)).transpose(-2,-1) # b x m x 128
        f3d = self.FeatureExtractor3d(f3d.transpose(-2,-1)).transpose(-2,-1) # b x n x 128
        # L2 Normalise:
        f2d = torch.nn.functional.normalize(f2d, p=2, dim=-1)
        f3d = torch.nn.functional.normalize(f3d, p=2, dim=-1)
        # Compute pairwise L2 distance matrix:
        M = self.pairwiseL2Dist(f2d, f3d)

        # Sinkhorn:
        # Set replicated points to have a zero prior probability
        b, m, n = M.size()
        r = M.new_zeros((b, m)) # bxm
        c = M.new_zeros((b, n)) # bxn
        for i in range(b):
            r[i, :num_points_2d[i]] = 1.0 / num_points_2d[i].float()
            c[i, :num_points_3d[i]] = 1.0 / num_points_3d[i].float()
        P = self.sinkhorn(M, r, c)

        # Compute camera pose:
        theta = None
        theta0 = None
        if poseloss:
            # Skip entire batch if any point-set has < 4 points:
            if (num_points_2d < 4).any() or (num_points_3d < 4).any():
                theta0 = P.new_zeros((P.size(0), 6))
                theta = P.new_zeros((P.size(0), 6))
                return P, theta, theta0
            # RANSAC:
            theta0 = self.ransac_p3p(P, p2d, p3d, num_points_2d, num_points_3d)
            # Run Weighted BPnP Optimization:
            p2d_bearings = torch.nn.functional.pad(p2d, (0, 1), "constant", 1.0)
            p2d_bearings = torch.nn.functional.normalize(p2d_bearings, p=2, dim=-1)
            theta = self.wbpnp(P, p2d_bearings, p3d, theta0)

        return P, theta, theta0

