import numpy as np
from math import sqrt
from scipy.signal import convolve2d
from scipy.interpolate import interp1d


class FastDeconvolution:
    def __init__(self, blurred, kernel, lambda_, alpha):
        if kernel.shape[0] % 2 != 1 or kernel.shape[1] % 2 != 1:
            raise ValueError("Error - blur kernel k must be odd-sized.")
        self.blurred = blurred
        self.deblurred = blurred.copy()
        self.kernel = kernel
        self.lambda_ = lambda_
        self.alpha = alpha
        self.imsize = blurred.shape
        self.lut = dict()

    def solve(
        self,
        beta_init=1,
        beta_rate=2 * sqrt(2),
        beta_max=256,
        iter_in=1,
        lut_v_range=np.arange(-10, 10, 1e-4),
        verbose=False,
    ):
        beta = beta_init
        # compute constant quantities
        nomin1, denom1, denom2 = self.compute_const_denominator()
        # compute v for equation 5
        gx, gy = gradient_x(self.deblurred)
        iter_out = 0
        while beta < beta_max:
            iter_out += 1
            gamma = beta / self.lambda_
            denom = denom1 + gamma * denom2
            for i in range(iter_in):
                # show cost
                if verbose:
                    print(f"Outer iteration {iter_out}, Inner iteration {i}, Cost {self.compute_cost(gx, gy)}\n")
                # w-subproblem
                wx = self.solve_w(gx, beta, lut_v_range)
                wy = self.solve_w(gy, beta, lut_v_range)
                # x-subproblem
                nomin2 = np.fft.fft2(gradient_wx(wx, wy))
                self.deblurred = np.real(np.fft.ifft2((nomin1 + gamma * nomin2) / denom))
                # update v for equation 5
                gx, gy = gradient_x(self.deblurred)
            beta *= beta_rate
        return self.deblurred

    def solve_w(self, v, beta, lut_v_range):
        if beta not in self.lut:
            print(f"Recomputing lookup table for new value of beta {beta:.3f}.")
            if abs(self.alpha - 1) < 1e-6:
                self.lut[beta] = compute_w1(lut_v_range, beta)
            elif abs(self.alpha - 2 / 3) < 1e-6:
                self.lut[beta] = compute_w23(lut_v_range, beta)
            elif abs(self.alpha - 1 / 2) < 1e-6:
                self.lut[beta] = compute_w12(lut_v_range, beta)
            else:
                self.lut[beta] = compute_w_newton(lut_v_range, beta)
        else:
            print(f"Reusing lookup table for beta {beta:.3f}.")
        interp = interp1d(lut_v_range, self.lut[beta], kind="linear", fill_value="extrapolate")
        return interp(v)

    def compute_const_denominator(self):
        """
        compute denominator and part of the numerator for equation 4
        """
        k_fft = psf2otf(self.kernel, self.imsize)
        kernel_gx = np.array([1, -1], dtype=np.float64).reshape(1, 2)
        kernel_gy = np.array([1, -1], dtype=np.float64).reshape(2, 1)
        nomin1 = np.conj(k_fft) * np.fft.fft2(self.blurred)
        denom1 = np.abs(k_fft) ** 2
        denom2 = np.abs(psf2otf(kernel_gx, self.imsize)) ** 2 + np.abs(psf2otf(kernel_gy, self.imsize)) ** 2
        return nomin1, denom1, denom2

    def compute_cost(self, gx, gy):
        """
        compute cost using equation 2
        """
        blurred_syn = convolve2d(self.deblurred, self.kernel, boundary="symm", mode="same")
        likelihood = np.sum(np.square(blurred_syn - self.blurred))
        return (self.lambda_ / 2) * likelihood + np.sum(np.abs(gx) ** self.alpha) + np.sum(np.abs(gy) ** self.alpha)


def psf2otf(psf, shape):
    return np.fft.fft2(np.fft.ifftshift(psf), shape)


def gradient_x(x):
    """
    compute (v1 = F1 * x) and (v2 = F2 * x) in equation 5
    """
    gx = np.zeros(x.shape, dtype=x.dtype)
    gy = np.zeros(x.shape, dtype=x.dtype)
    gx[:, :-1] = x[:, 1:] - x[:, :-1]
    gx[:, -1] = x[:, 0] - x[:, -1]
    gy[:-1, :] = x[1:, :] - x[:-1, :]
    gy[-1, :] = x[0, :] - x[-1, :]
    return gx, gy


def gradient_wx(wx, wy):
    """
    compute (F1^T * w1 + F2^T * w2) in equation 3
    """
    gx = np.zeros(wx.shape, dtype=wx.dtype)
    gy = np.zeros(wy.shape, dtype=wy.dtype)
    gx[:, 0] = wx[:, -1] - wx[:, 0]
    gx[:, 1:] = wx[:, :-1] - wx[:, 1:]
    gy[0, :] = wy[-1, :] - wy[0, :]
    gy[1:, :] = wy[:-1, :] - wy[1:, :]
    return gx + gy


def compute_w1(v, beta):
    pass


def compute_w23(v, beta):
    vsize = v.shape[0]
    v_complex = v.astype(np.complex128)
    epsilon = 1e-6

    m = np.full(vsize, 8 / (27 * beta**3), dtype=np.complex128)

    # precompute some terms
    v2 = v_complex * v_complex
    v3 = v2 * v_complex
    v4 = v3 * v_complex
    m2 = m * m
    m3 = m2 * m

    # t1 ~ t7
    t1 = -9 / 8 * v2
    t2 = v3 / 4
    t3 = -1 / 8 * (m * v2)
    t4 = -t3 / 2 + np.sqrt(-m3 / 27 + (m2 * v4) / 256)
    t5 = np.exp(np.log(t4) / 3)
    t6 = 2 * (-5 / 18 * t1 + t5 + (m / (3 * t5)))
    t7 = np.sqrt(t1 / 3 + t6)

    # compute 4 roots
    roots = np.zeros((4, vsize), dtype=np.complex128)
    term1 = 0.75 * v_complex
    term21 = np.sqrt(-(t1 + t6 + t2 / t7))
    term22 = np.sqrt(-(t1 + t6 - t2 / t7))
    roots[0, :] = term1 + 0.5 * (t7 + term21)
    roots[1, :] = term1 + 0.5 * (t7 - term21)
    roots[2, :] = term1 + 0.5 * (-t7 + term22)
    roots[3, :] = term1 + 0.5 * (-t7 - term22)

    # filter out roots
    vtile = np.tile(v, (4, 1))
    c1 = np.abs(np.imag(roots)) < epsilon
    c2 = np.real(roots) * np.sign(vtile) > np.abs(vtile) / 2
    c3 = np.real(roots) * np.sign(vtile) < np.abs(vtile)
    roots[~(c1 & c2 & c3)] = 0

    w = np.max(np.real(roots) * np.sign(vtile), axis=0) * np.sign(v)
    return w


def compute_w12(v, beta):
    pass


def compute_w_newton(v, beta):
    pass
