import torch
import math
import utilities.geometry_utilities as geo

# Utility functions:

def correspondenceMatrices(R, t, p2d, p3d, threshold):
    # Form Boolean correspondence matrix given pose
    p2d_bearings = torch.nn.functional.pad(p2d, (0, 1), "constant", 1.0)
    p2d_bearings = torch.nn.functional.normalize(p2d_bearings, p=2, dim=-1)
    p3d_bearings = geo.transform_and_normalise_points(p3d, R, t)
    dot_product = torch.einsum('bmd,bnd->bmn', (p2d_bearings, p3d_bearings))
    return (dot_product >= math.cos(threshold)).float()

def correspondenceMatricesTheta(theta, p2d, p3d, threshold):
    R = geo.angle_axis_to_rotation_matrix(theta[..., :3])
    t = theta[..., 3:]
    return correspondenceMatrices(R, t, p2d, p3d, threshold)

# Error/success measures:

def correspondenceProbabilityDistances(P, C):
    """ Difference between the probability mass assigned to inlier and
        outlier correspondences
    """
    return ((1.0 - 2.0 * C) * P).sum(dim=(-2, -1))

def numInliers(R, t, p2d, p3d, threshold):
    return correspondenceMatrices(R, t, p2d, p3d, threshold).sum(dim=(-2, -1))

def numInliersTheta(theta, p2d, p3d, threshold):
    R = geo.angle_axis_to_rotation_matrix(theta[..., :3])
    t = theta[..., 3:]
    return numInliers(R, t, p2d, p3d, threshold)

def rotationErrors(R, R_gt, eps=1e-7):
    # Gradient is non-differentiable at acos(1) and acos(-1)
    max_dot_product = 1.0 - eps
    return (0.5 * ((R * R_gt).sum(dim=(-2, -1)) - 1.0)).clamp_(-max_dot_product, max_dot_product).acos()

def rotationErrorsTheta(theta, R_gt, eps=1e-7):
    R = geo.angle_axis_to_rotation_matrix(theta[..., :3])
    return rotationErrors(R, R_gt, eps)

def translationErrors(t, t_gt):
    return (t - t_gt).norm(dim=-1)

def translationErrorsTheta(theta, t_gt):
    t = theta[..., 3:]
    return translationErrors(t, t_gt)

def reprojectionErrors(R, t, p2d, p3d, P, eps=1e-7):
    """ Sum angular deviation between projected 3D points and 2D points
    Weighted by matrix P, which may be a correspondence probability matrix
    or a Boolean correspondence matrix C
    """
    max_dot_product = 1.0 - eps
    f = torch.nn.functional.pad(p2d, (0, 1), "constant", 1.0)
    f = torch.nn.functional.normalize(f, p=2, dim=-1)
    pt = geo.transform_and_normalise_points(p3d, R, t)
    dot_product = torch.einsum('bmd,bnd->bmn', (f, pt))
    angle = dot_product.clamp(-max_dot_product, max_dot_product).acos()
    P = P.div(P.sum(dim=(-2, -1), keepdim=True)) # Normalise P (sums to 1)
    return torch.einsum('bmn,bmn->b', (P, angle))

def reprojectionErrorsTheta(theta, p2d, p3d, P, eps=1e-7):
    R = geo.angle_axis_to_rotation_matrix(theta[..., :3])
    t = theta[..., 3:]
    return reprojectionErrors(R, t, p2d, p3d, P, eps)

def reconstructionErrors(R, t, R_gt, t_gt, p):
    """ Sum deviation between 3D points projected using ground-truth and estimated theta
    Dependent on the scale of the point-set
    """
    pt = geo.transform_and_normalise_points(p, R, t)
    pt_gt = geo.transform_and_normalise_points(p, R_gt, t_gt)
    return (pt - pt_gt).norm(dim=-1).mean(dim=-1) # Average over points

def angularReconstructionErrors(R, t, R_gt, t_gt, p):
    """ Sum angular deviation between 3D points projected using ground-truth and estimated theta
    Dependent on the scale of the point-set
    """
    pt = geo.transform_and_normalise_points(p, R, t)
    pt_gt = geo.transform_and_normalise_points(p, R_gt, t_gt)
    dot_product = torch.einsum('bnd,bnd->bn', (pt, pt_gt))
    return 1.0 - dot_product.mean(dim=-1) # Average over points

# Loss functions:

def rotationLoss(R, R_gt, max_rotation_angle=1.570796): # pi/2
    return rotationErrors(R, R_gt).clamp_max(max_rotation_angle).mean()

def weightedRotationLoss(R, R_gt, weights, max_rotation_angle=1.570796):
    return (weights * rotationErrors(R, R_gt).clamp_max(max_rotation_angle)).mean()

def translationLoss(t, t_gt, max_translation_error=1.0):
    return translationErrors(t, t_gt).clamp_max(max_translation_error).mean()

def weightedTranslationLoss(t, t_gt, weights, max_translation_error=1.0):
    return (weights * translationErrors(t, t_gt).clamp_max(max_translation_error)).mean()

def reprojectionLoss(R, t, p2d, p3d, P):
    return reprojectionErrors(R, t, p2d, p3d, P, eps=1e-7).mean() # Average over batch

def reconstructionLoss(R, t, R_gt, t_gt, p):
    return reconstructionErrors(R, t, R_gt, t_gt, p).mean() # Average over batch

def angularReconstructionLoss(R, t, R_gt, t_gt, p):
    return angularReconstructionErrors(R, t, R_gt, t_gt, p).mean() # Average over batch

def correspondenceLoss(P, C_gt):
    # Using precomputed C
    return correspondenceProbabilityDistances(P, C_gt).mean() # [-1, 1)

def correspondenceLossFromPose(P, R_gt, t_gt, p2d, p3d, threshold):
    # Computing C on the fly
    C = correspondenceMatrices(R_gt, t_gt, p2d, p3d, threshold)
    return correspondenceLoss(P, C)

def weightedCorrespondenceLoss(P, C_gt, weights):
    return (weights * correspondenceProbabilityDistances(P, C_gt)).mean() # [-1, 1)

class TotalLoss(torch.nn.Module):
    def __init__(self, gamma):
        super(TotalLoss, self).__init__()
        self.gamma = gamma
    def forward(self, theta, P, R_gt, t_gt, C_gt):
        if self.gamma > 0.0:
            R = geo.angle_axis_to_rotation_matrix(theta[..., :3])
            t = theta[..., 3:]
            losses = torch.cat((
                correspondenceLoss(P, C_gt).view(1),
                rotationLoss(R, R_gt).view(1), # [0, pi]
                translationLoss(t, t_gt).view(1), # [0, inf)
                ))
            loss = losses[0] + self.gamma * (losses[1] + losses[2])
        else:
            losses = correspondenceLoss(P, C_gt).view(1)
            loss = losses[0]
        return loss, losses

class WeightedTotalLoss(torch.nn.Module):
    def __init__(self, gamma):
        super(TotalLoss, self).__init__()
        self.gamma = gamma
    def forward(self, theta, P, R_gt, t_gt, C_gt, p2d, p3d, num_points_2d, num_points_3d):
        # Weight proportionally to the point-set size:
        b, m, n = P.size()
        weights = torch.min(num_points_2d, num_points_3d).float() / float(min(m, n)) # linear
        weights = (2.0 * weights).tanh() # nonlinear
        weights = weights.to(device=P.device)
        if self.gamma > 0.0:
            R = geo.angle_axis_to_rotation_matrix(theta[..., :3])
            t = theta[..., 3:]
            losses = torch.cat((
                weightedCorrespondenceLoss(P, C_gt, weights).view(1),
                weightedRotationLoss(R, R_gt, weights).view(1), # [0, pi]
                weightedTranslationLoss(t, t_gt, weights).view(1), # [0, inf)
                ))
            loss = losses[0] + self.gamma * (losses[1] + losses[2])
        else:
            losses = weightedCorrespondenceLoss(P, C_gt, weights).view(1)
            loss = losses[0]
        return loss, losses
