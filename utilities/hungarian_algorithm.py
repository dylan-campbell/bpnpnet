import torch
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

def hungarian(M):
    row_ind, col_ind = linear_sum_assignment(M)
    return row_ind, col_ind

def ransac_p3p(p2d, p3d):
    K = np.float32(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]))
    dist_coeff = np.float32(np.array([0.0, 0.0, 0.0, 0.0]))
    theta0 = np.zeros(6)
    retval, rvec, tvec, inliers = cv2.solvePnPRansac(
        p3d, p2d, K, dist_coeff,
        iterationsCount=1000,
        reprojectionError=0.01,
        flags=cv2.SOLVEPNP_P3P)
    # print(inliers.shape[0], '/',  num_points_2d[i].item())
    if rvec is not None and tvec is not None and retval:
        rvec = rvec.squeeze(-1)
        tvec = tvec.squeeze(-1)
        if np.isfinite(rvec).all() and np.isfinite(tvec).all():
            theta0[:3] = rvec
            theta0[3:] = tvec
    return theta0

def hungarian_ransac(M, p2d, p3d, num_points_2d, num_points_3d):
    b = M.size(0)
    theta0 = M.new_zeros((b, 6))
    P = torch.zeros_like(M) # bxmxn
    for i in range(b): # Loop over batch
        index_2d, index_3d = hungarian(M[i, :, :].cpu().numpy())
        # Form P:
        P[i, index_2d, index_3d] = 1.0 / float(index_2d.size) # P should sum to 1
        # Skip if any point-set has < 4 points:
        if (num_points_2d[i] < 4).any() or (num_points_3d[i] < 4).any():
            continue
        else:
            theta0_np = ransac_p3p(p2d[i, index_2d, :].cpu().numpy(), p3d[i, index_3d, :].cpu().numpy())
            theta0[i, :] = torch.as_tensor(theta0_np, dtype=theta0.dtype, device=theta0.device)
    return P, theta0
