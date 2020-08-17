#
# Entropy-regularised optimal transport layer
#
# y = argmin_u f(x, u; mu)
#     subject to h(u) = 0
#            and u_i >= 0
#
# where f(x, u) = \sum_{i=1}^n (x_i * u_i + mu * u_i * (log(u_i) - 1))
# and h(u) = Au - b
# for A = [
#          e_1 x n, ..., e_m x n,
#          I_n, ..., I_n
#         ]_{-1,.}
# where [.]_{-1,.} denotes removing the first row of the matrix
#
# Dylan Campbell, Liu Liu, Stephen Gould, 2020,
# "Solving the Blind Perspective-n-Point Problem End-To-End With
# Robust Differentiable Geometric Optimization"
#
# Dylan Campbell <dylan.campbell@anu.edu.au>
#
# v0: 20191018
#

import torch

class RegularisedTransportFn(torch.autograd.Function):
    """ Class for solving the entropy-regularised transport problem
    
    Finds the transport (or joint probability) matrix P that is the
    smallest Sinkhorn distance from cost/distance matrix D
    
    Using:
    [1] Sinkhorn distances: Lightspeed computation of optimal transport
        Marco Cuturi, 2013
        Advances in Neural Information Processing Systems
        https://papers.nips.cc/paper/4927-sinkhorn-distances-lightspeed-computation-of-optimal-transport.pdf
    """
    def objectiveFn(m, p, mu=0.1):
        """ Vectorised objective function

        Using:
            Equation (2) from [1]
        """
        logw = torch.where(p > 0.0, p.log(), torch.full_like(p, -1e9))
        return (p * m).sum(-1) + mu * (p * (logw - 1.0)).sum(-1)

    def sinkhorn(M, r=None, c=None, mu=0.1, tolerance=1e-9, iterations=None):
        """ Compute transport matrix P, given cost matrix M
        
        Using:
            Algorithm 1 from [1]
        """
        max_distance = 5.0
        K = (-M.clamp_max(max_distance) / mu).exp()
        if r is None:
            r = 1.0 / M.size()[-2]
            u = M.new_full(M.size()[:-1], r).unsqueeze(-1)
        else:
            r = r.unsqueeze(-1)
            u = r.clone()
        if c is None:
            c = 1.0 / M.size()[-1]
        else:
            c = c.unsqueeze(-1)
        if iterations is None:
            i = 0
            max_iterations = 100
            u_prev = torch.ones_like(u)
            while (u - u_prev).norm(dim=-1).max() > tolerance:
                if i > max_iterations:
                    break
                i += 1
                u_prev = u
                u = r / K.matmul(c / K.transpose(-2, -1).matmul(u))
        else:
            for i in range(iterations):
                u = r / K.matmul(c / K.transpose(-2, -1).matmul(u))
        v = c / K.transpose(-2, -1).matmul(u)
        P = (u * K) * v.transpose(-2, -1)
        return P

    def gradientFn(P, mu, v):
        """ Compute vector-Jacobian product DJ(M) = DJ(P) DP(M) [b x m*n]

        DP(M) = (H^-1 * A^T * (A * H^-1 * A^T)^-1 * A * H^-1 - H^-1) * B
        H = D_YY^2 f(x, y) = diag(mu / vec(P))
        B = D_XY^2 f(x, y) = I

        Using:
            Lemma 4.4 from
            Stephen Gould, Richard Hartley, and Dylan Campbell, 2019
            "Deep Declarative Networks: A New Hope", arXiv:1909.04866

        Arguments:
            P: (b, m, n) Torch tensor
                batch of transport matrices

            mu: float,
                regularisation factor

            v: (b, m*n) Torch tensor
                batch of gradients of J with respect to P

        Return Values:
            gradient: (b, m*n) Torch tensor,
                batch of gradients of J with respect to M

        """
        with torch.no_grad():
            b, m, n = P.size()
            B = P / mu
            hinv = B.flatten(start_dim=-2)
            d1inv = B.sum(-1)[:, 1:].reciprocal() # Remove first element
            d2 = B.sum(-2)
            B = B[:, 1:, :] # Remove top row
            S = -B.transpose(-2, -1).matmul(d1inv.unsqueeze(-1) * B)
            S[:, range(n), range(n)] += d2
            Su = torch.cholesky(S)
            Sinv = torch.zeros_like(S)
            for i in range (b):
                Sinv[i, ...] = torch.cholesky_inverse(Su[i, ...]) # Currently cannot handle batches
            R = -B.matmul(Sinv) * d1inv.unsqueeze(-1)
            Q = -R.matmul(B.transpose(-2, -1)  * d1inv.unsqueeze(-2))
            Q[:, range(m - 1), range(m - 1)] += d1inv
            # Build vector-Jacobian product from left to right:
            vHinv = v * hinv # bxmn * bxmn -> bxmn
            # Break vHinv into m blocks of n elements:
            u1 = vHinv.reshape((-1, m, n)).sum(-1)[:, 1:].unsqueeze(-2) # remove first element
            u2 = vHinv.reshape((-1, m, n)).sum(-2).unsqueeze(-2)
            u3 = u1.matmul(Q) + u2.matmul(R.transpose(-2, -1))
            u4 = u1.matmul(R) + u2.matmul(Sinv)
            u5 = u3.expand(-1, n, -1).transpose(-2, -1)+u4.expand(-1, m-1, -1)
            uHinv = torch.cat((u4, u5), dim=-2).flatten(start_dim=-2) * hinv
            gradient = uHinv - vHinv
        return gradient

    @staticmethod
    def forward(ctx, M, r=None, c=None, mu=0.1, tolerance=1e-9, iterations=None):
        """ Optimise the entropy-regularised Sinkhorn distance

        Solves:
            argmin_u   sum_{i=1}^n (x_i * u_i + mu * u_i * (log(u_i) - 1))
            subject to Au = 1, u_i >= 0 

        Using:
            Algorithm 1 from [1]
        
        Arguments:
            M: (b, m, n) Torch tensor,
                batch of cost matrices,
                assumption: non-negative

            mu: float,
                regularisation factor,
                assumption: positive,
                default: 0.1

            tolerance: float,
                stopping criteria for Sinkhorn algorithm,
                assumption: positive,
                default: 1e-9

            iterations: int,
                number of Sinkhorn iterations,
                assumption: positive,
                default: None

        Return Values:
            P: (b, m, n) Torch tensor,
                batch of transport (joint probability) matrices
        """
        M = M.detach()
        if r is not None:
            r = r.detach()
        if c is not None:
            c = c.detach()
        P = RegularisedTransportFn.sinkhorn(M, r, c, mu, tolerance, iterations)
        ctx.mu = mu
        ctx.save_for_backward(P, r, c)
        return P.clone()

    @staticmethod
    def backward(ctx, grad_output):
        P, r, c = ctx.saved_tensors
        mu = ctx.mu
        input_size = P.size()
        grad_input = None
        if ctx.needs_input_grad[0]:
            # Only compute gradient for non-zero rows and columns of P
            if r is None or c is None or ((r > 0.0).all() and (c > 0.0).all()):
                grad_output = grad_output.flatten(start_dim=-2) # bxmn
                grad_input = RegularisedTransportFn.gradientFn(P, mu, grad_output) # bxmn
                grad_input = grad_input.reshape(input_size) # bxmxn
            else:
                b, m, n = input_size
                r_num_nonzero = (r > 0).sum(dim=-1)
                c_num_nonzero = (c > 0).sum(dim=-1)
                grad_input = torch.empty_like(P)
                for i in range(b):
                    p = r_num_nonzero[i]
                    q = c_num_nonzero[i]
                    grad_output_i = grad_output[i:(i+1), :p, :q].flatten(start_dim=-2) # bxpq
                    grad_input_i = RegularisedTransportFn.gradientFn(P[i:(i+1), :p, :q], mu, grad_output_i)
                    grad_input_i = grad_input_i.reshape((1, p, q))
                    grad_input_i = torch.nn.functional.pad(grad_input_i, (0, n - q, 0, m - p), "constant", 0.0)
                    grad_input[i:(i+1), ...] = grad_input_i
        return grad_input, None, None, None, None, None

class RegularisedTransport(torch.nn.Module):
    def __init__(self, mu=0.1, tolerance=1e-9, iterations=None):
        super(RegularisedTransport, self).__init__()
        self.mu = mu
        self.tolerance = tolerance
        self.iterations = iterations
            
    def forward(self, M, r=None, c=None):
        return RegularisedTransportFn.apply(M, r, c, self.mu, self.tolerance, self.iterations)

