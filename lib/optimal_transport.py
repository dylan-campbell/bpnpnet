# Entropy-regularised optimal transport layer
#
# y = argmin_u f(x, u; lmbda)
#     subject to h(u) = 0
#            and u_i >= 0
#
# where f(x, u) = \sum_{i=1}^n (x_i * u_i + u_i * (log(u_i) - 1) / lmbda)
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
# v1: 20210325

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
    @staticmethod
    def objectiveFn(m, p, lmbda=10.0):
        """ Vectorised objective function

        Using:
            Equation (2) from [1]
        """
        return (p * m).sum(-1) + (p * (p.clamp_min(1e-36).log() - 1.0) / lmbda).sum(-1)

    @staticmethod
    def sinkhorn(M, r=None, c=None, lmbda=10.0, tolerance=1e-9, max_iterations=100, max_distance=5.0):
        """ Compute transport matrix P, given cost matrix M
        
        Using:
            Algorithm 1 from [1]
        """
        K = (-lmbda * M.clamp_max(max_distance)).exp()
        r = 1.0 / M.size(-2) if r is None else r.unsqueeze(-1) # Keep as scalar if uniform
        c = 1.0 / M.size(-1) if c is None else c.unsqueeze(-1) # Keep as scalar if uniform
        u = r.clone() if isinstance(r, torch.Tensor) else M.new_full(M.size()[:-1], r).unsqueeze(-1)
        u_prev = torch.ones_like(u)
        for i in range(max_iterations):
            if torch.all(torch.isclose(u, u_prev, atol=tolerance, rtol=0.0)):
                break
            u_prev = u
            u = r / K.matmul(c / K.transpose(-2, -1).matmul(u))
        v = c / K.transpose(-2, -1).matmul(u)
        P = (u * K) * v.transpose(-2, -1)
        return P

    @staticmethod
    def gradientFn(P, lmbda, v):
        """ Compute vector-Jacobian product DJ(M) = DJ(P) DP(M) [b x m*n]

        DP(M) = (H^-1 * A^T * (A * H^-1 * A^T)^-1 * A * H^-1 - H^-1) * B
        H = D_YY^2 f(x, y) = diag(1 / (lmbda * vec(P)))
        B = D_XY^2 f(x, y) = I

        Using:
            Lemma 4.4 from
            Stephen Gould, Richard Hartley, and Dylan Campbell, 2019
            "Deep Declarative Networks: A New Hope", arXiv:1909.04866

        Arguments:
            P: (b, m, n) Torch tensor
                batch of transport matrices

            lmbda: float,
                regularisation factor

            v: (b, m*n) Torch tensor
                batch of gradients of J with respect to P

        Return Values:
            gradient: (b, m*n) Torch tensor,
                batch of gradients of J with respect to M

        """
        with torch.no_grad():
            b, m, n = P.size()
            B = lmbda * P
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
    def forward(ctx, M, r=None, c=None, lmbda=10.0, tolerance=1e-9, max_iterations=100):
        """ Optimise the entropy-regularised Sinkhorn distance

        Solves:
            argmin_u   sum_{i=1}^n (x_i * u_i + u_i * (log(u_i) - 1) / lmbda)
            subject to Au = 1, u_i >= 0 

        Using:
            Algorithm 1 from [1]
        
        Arguments:
            M: (b, m, n) Torch tensor,
                batch of cost matrices,
                assumption: non-negative

            lmbda: float,
                regularisation factor,
                assumption: positive,
                default: 1.0

            tolerance: float,
                stopping criteria for Sinkhorn algorithm,
                assumption: positive,
                default: 1e-9

            max_iterations: int,
                max number of Sinkhorn iterations,
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
        P = RegularisedTransportFn.sinkhorn(M, r, c, lmbda, tolerance, max_iterations)
        ctx.lmbda = lmbda
        ctx.save_for_backward(P, r, c)
        return P.clone()

    @staticmethod
    def backward(ctx, grad_output):
        P, r, c = ctx.saved_tensors
        lmbda = ctx.lmbda
        input_size = P.size()
        grad_input = None
        if ctx.needs_input_grad[0]:
            # Only compute gradient for non-zero rows and columns of P
            if r is None or c is None or ((r > 0.0).all() and (c > 0.0).all()):
                grad_output = grad_output.flatten(start_dim=-2) # bxmn
                grad_input = RegularisedTransportFn.gradientFn(P, lmbda, grad_output) # bxmn
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
                    grad_input_i = RegularisedTransportFn.gradientFn(P[i:(i+1), :p, :q], lmbda, grad_output_i)
                    grad_input_i = grad_input_i.reshape((1, p, q))
                    grad_input_i = torch.nn.functional.pad(grad_input_i, (0, n - q, 0, m - p), "constant", 0.0)
                    grad_input[i:(i+1), ...] = grad_input_i
        return grad_input, None, None, None, None, None

class RegularisedTransport(torch.nn.Module):
    def __init__(self, lmbda=10.0, tolerance=1e-9, max_iterations=100):
        super(RegularisedTransport, self).__init__()
        self.lmbda = lmbda
        self.tolerance = tolerance
        self.max_iterations = max_iterations
            
    def forward(self, M, r=None, c=None):
        return RegularisedTransportFn.apply(M, r, c, self.lmbda, self.tolerance, self.max_iterations)

if __name__ == '__main__':
    # Test Sinkhorn
    torch.manual_seed(0)
    sinkhorn = RegularisedTransport(lmbda=10.0, tolerance=1e-9, max_iterations=100)
    b, m, n = 2, 5, 10
    M = torch.randn((b, m, n), dtype=torch.float).abs()
    P = sinkhorn(M)
    # r = M.new_ones((b, m)) / m # bxm
    # c = M.new_ones((b, n)) / n # bxn
    # P = sinkhorn(M, r, c)

    print(M)
    print(P)
    print(torch.sum(P, -1))
    print(torch.sum(P, -2))
    print(torch.einsum("bij,bij->b", M, P))
