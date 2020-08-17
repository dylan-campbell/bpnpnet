#
# Nonlinear Weighted Blind PnP Layer
#
# theta^\star = argmin_theta f(W, p2d, p3d, theta)
#
# where f(W, p2d, p3d, theta) = \sum_{i=1}^m \sum_{j=1}^n w_ij (1 - p2d_i^T N(R(theta) (p3d_j - t(theta))))
# and N(p3d) = p3d / ||p3d||
#
# Dylan Campbell, Liu Liu, Stephen Gould, 2020,
# "Solving the Blind Perspective-n-Point Problem End-To-End With
# Robust Differentiable Geometric Optimization"
#
# Dylan Campbell <dylan.campbell@anu.edu.au>
#
# v0: 20190916
#

import torch
from torch.autograd import grad
import utilities.geometry_utilities as geo

def weightedAngularReprojectionError(W, p2d, p3d, theta):
    """ Weighted angular reprojection error

    f(W, p2d, p3d, theta) = sum_{i=1}^m sum_{j=1}^n w_ij (1 - p2d_i^T N(R(theta) p3d_j + t(theta)))
        N(p3d) = p3d / ||p3d||

    Arguments:
        W: (b, m*n) Torch tensor,
            batch of flattened weight matrices

        p2d: (b, m, 3) Torch tensor,
            batch of 3D bearing-vector-sets,
            assumption: unit norm

        p3d: (b, n, 3) Torch tensor,
            batch of 3D point-sets

        theta: (b, 6) Torch tensor,
            batch of transformation parameters
            assumptions:
                theta[:, 0:3]: angle-axis rotation vector
                theta[:, 3:6]: translation vector

    Return Values:
        error: (b, ) Torch tensor,
            sum cosine "distance"

    Complexity:
        O(bmn)
    """
    b = p2d.size(0)
    m = p2d.size(1)
    n = p3d.size(1)
    W = W.reshape(b, m, n)
    p3dt = geo.transform_and_normalise_points_by_theta(p3d, theta)
    error = torch.einsum('bmn,bmn->b', (W, 1.0 - torch.einsum('bmd,bnd->bmn', (p2d, p3dt))))
    return error

def robustWeightedAngularReprojectionError(W, p2d, p3d, theta, threshold=0.5):
    b = p2d.size(0)
    m = p2d.size(1)
    n = p3d.size(1)
    W = W.reshape(b, m, n)
    p3dt = geo.transform_and_normalise_points_by_theta(p3d, theta)
    error = 1.0 - torch.einsum('bmd,bnd->bmn', (p2d, p3dt)) # [0, 2]
    # Attenuate error to a maximum of threshold; acos(0.5) = 60deg
    # Approximately linear up to 0.5*threshold, then attenuates
    error = threshold * (error / threshold).tanh() # [0, threshold]
    error = torch.einsum('bmn,bmn->b', (W, error))
    # Alternative: Truncated L1
    # error = error.clamp_max(threshold)
    # error = torch.einsum('bmn,bmn->b', (W, error)) # [0, 2]
    return error

def Dy(f, x, y):
    """
    Dy(x) = -(D_YY^2 f(x, y))^-1 D_XY^2 f(x, y)
    Lemma 4.3 from
    Stephen Gould, Richard Hartley, and Dylan Campbell, 2019
    "Deep Declarative Networks: A New Hope", arXiv:1909.04866

    Arguments:
        f: (b, ) Torch tensor, with gradients
            batch of objective functions evaluated at (x, y)

        x: (b, n) Torch tensor, with gradients
            batch of input vectors

        y: (b, m) Torch tensor, with gradients
            batch of minima of f

    Return Values:
        Dy(x): (b, m, n) Torch tensor,
            batch of gradients of y with respect to x

    """
    with torch.enable_grad():
        grad_outputs = torch.ones_like(f)
        DYf = grad(f, y, grad_outputs=grad_outputs, create_graph=True)[0] # bxm
        DYYf = torch.empty_like(DYf.unsqueeze(-1).expand(-1, -1, y.size(-1))) # bxmxm
        DXYf = torch.empty_like(DYf.unsqueeze(-1).expand(-1, -1, x.size(-1))) # bxmxn
        grad_outputs = torch.ones_like(DYf[:, 0])
        for i in range(DYf.size(-1)): # [0,m)
            DYf_i = DYf[:, i] # b
            DYYf[:, i:(i+1), :] = grad(DYf_i, y, grad_outputs=grad_outputs, create_graph=True)[0].contiguous().unsqueeze(1) # bx1xm
            DXYf[:, i:(i+1), :] = grad(DYf_i, x, grad_outputs=grad_outputs, create_graph=True)[0].contiguous().unsqueeze(1) # bx1xn
    DYYf = DYYf.detach()
    DXYf = DXYf.detach()    
    DYYf = 0.5 * (DYYf + DYYf.transpose(1, 2)) # In case of floating point errors

    # Try a batchwise solve, otherwise revert to looping
    # Avoids cuda runtime error (9): invalid configuration argument
    try:
        U = torch.cholesky(DYYf, upper=True)
        Dy_at_x = torch.cholesky_solve(-1.0 * DXYf, U, upper=True) # bxmxn
    except:
        Dy_at_x = torch.empty_like(DXYf)
        for i in range(DYYf.size(0)): # For some reason, doing this in a loop doesn't crash
            try:
                U = torch.cholesky(DYYf[i, ...], upper=True)
                Dy_at_x[i, ...] = torch.cholesky_solve(-1.0 * DXYf[i, ...], U, upper=True)
            except:
                Dy_at_x[i, ...], _ = torch.solve(-1.0 * DXYf[i, ...], DYYf[i, ...])

    # Set NaNs to 0:
    if torch.isnan(Dy_at_x).any():
        Dy_at_x[torch.isnan(Dy_at_x)] = 0.0 # In-place
    # Clip gradient norms:
    max_norm = 100.0
    Dy_norm = Dy_at_x.norm(dim=-2, keepdim=True) # bxmxn
    if (Dy_norm > max_norm).any():
        clip_coef = (max_norm / (Dy_norm + 1e-6)).clamp_max_(1.0)
        Dy_at_x = clip_coef * Dy_at_x

    return Dy_at_x

class NonlinearWeightedBlindPnPFn(torch.autograd.Function):
    """
    A class to optimise the weighted angular reprojection error given
    a set of 3D point p3d, a set of bearing vectors p2d, a weight matrix W
    """
    @staticmethod
    def forward(ctx, W, p2d, p3d, theta0=None):
        """ Optimise the weighted angular reprojection error
        
        Arguments:
            W: (b, m, n) Torch tensor,
                batch of weight matrices,
                assumption: positive and sum to 1 per batch

            p2d: (b, m, 3) Torch tensor,
                batch of 3D bearing-vector-sets,
                assumption: unit norm

            p3d: (b, n, 3) Torch tensor,
                batch of 3D point-sets

            theta0: (b, 6) Torch tensor,
                batch of initial transformation parameters
                assumptions:
                    theta[:, 0:3]: angle-axis rotation vector
                    theta[:, 3:6]: translation vector

        Return Values:
            theta: (b, 6) Torch tensor,
                batch of optimal transformation parameters
        """
        W = W.detach()
        p2d = p2d.detach()
        p3d = p3d.detach()
        if theta0 is None:
            theta0 = W.new_zeros((W.size()[0], 6))
        theta0 = theta0.detach()
        W = W.flatten(start_dim=-2)

        # Use a variable maximum number of iterations (aim: ~1s per pair)
        max_num_points = max(p2d.size(-2), p3d.size(-2))
        max_iter = round(max(min(100, 75*pow(max_num_points / 1000.0, -1.5)), 1))

        with torch.enable_grad():
            theta = theta0.clone().requires_grad_(True) # bx6
            # Note: in batch-mode, stopping conditions are entangled across batches
            # It would be better to use group norms for the stopping conditions
            opt = torch.optim.LBFGS([theta],
                                    lr=1.0, # Default: 1
                                    max_iter=max_iter, # Default: 100
                                    max_eval=None,
                                    tolerance_grad=1e-40, # Default: 1e-05
                                    tolerance_change=1e-40, # Default: 1e-09
                                    history_size=100,
                                    line_search_fn="strong_wolfe",
                                    )
            def closure():
                if torch.is_grad_enabled():
                    opt.zero_grad()
                error = weightedAngularReprojectionError(W, p2d, p3d, theta).mean() # average over batch
                if error.requires_grad:
                    error.backward()
                return error
            opt.step(closure)
        theta = theta.detach()
        ctx.save_for_backward(W, p2d, p3d, theta)
        return theta.clone()

    @staticmethod
    def backward(ctx, grad_output):
        W, p2d, p3d, theta = ctx.saved_tensors
        b = p2d.size(0)
        m = p2d.size(1)
        n = p3d.size(1)
        grad_input = None
        if ctx.needs_input_grad[0]: # W only
            with torch.enable_grad():
                W = W.detach().requires_grad_()
                theta = theta.detach().requires_grad_()
                fn_at_theta = weightedAngularReprojectionError(W, p2d, p3d, theta) # b
            Dtheta = Dy(fn_at_theta, W, theta) # bx6xmn
            grad_input = torch.einsum("ab,abc->ac", (grad_output, Dtheta)) # bx6 * bx6xmn-> bxmn
            grad_input = grad_input.reshape(b, m, n)
        return grad_input, None, None, None

class NonlinearWeightedBlindPnP(torch.nn.Module):
    def __init__(self):
        super(NonlinearWeightedBlindPnP, self).__init__()
            
    def forward(self, W, p2d, p3d, theta0=None):
        return NonlinearWeightedBlindPnPFn.apply(W, p2d, p3d, theta0)
