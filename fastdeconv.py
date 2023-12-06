import numpy as np
from math import sqrt
from numpy.fft import fft2, ifft2
from scipy.signal import convolve2d
from scipy.interpolate import interp1d


class FastDeconvolution:
    def __init__(self, blurred, kernel, lambda_, alpha, verbose=False):
        """
        Input Parameters:
        - blurred: observed blurry and noisy input grayscale image
        - kernel:  convolution kernel
        - lambda_: parameter that balances likelihood and prior term weighting
        - alpha: parameter between 0 and 2
        """
        assert blurred.ndim == 2, "Blurred image must be grayscale."
        assert kernel.shape[0] % 2 == kernel.shape[1] % 2 == 1, "Blur kernel k must be odd-sized."

        self.blurred = blurred
        self.deblurred = blurred.copy()
        self.kernel = kernel
        self.lambda_ = lambda_
        self.alpha = alpha
        self.verbose = verbose
        self.imsize = blurred.shape
        self.lut = dict()

    def solve(
        self,
        beta_init=1,
        beta_rate=2 * sqrt(2),
        beta_max=256,
        iter_in=1,
        lut_v_range=np.arange(-1, 1, 1e-4),
    ):
        beta = beta_init
        # compute constant quantities
        nomin1, denom1, denom2 = self.compute_const_denominator()
        # compute v for equation 5
        gx, gy = gradient_x(self.deblurred)
        iter_out = 0
        while beta < beta_max:
            iter_out += 1
            print(f"Outer iteration {iter_out}, Beta {beta:.2f}")
            gamma = beta / self.lambda_
            denom = denom1 + gamma * denom2
            for i in range(1, iter_in + 1):
                # show cost
                if self.verbose:
                    cost = self.compute_cost(gx, gy)
                    print(f"    Inner iteration {i}, Cost {cost:.4e}")
                # w-subproblem
                wx = self.solve_w(gx, beta, lut_v_range)
                wy = self.solve_w(gy, beta, lut_v_range)
                # x-subproblem
                nomin2 = fft2(gradient_wx(wx, wy))
                self.deblurred = np.real(ifft2((nomin1 + gamma * nomin2) / denom))
                # update v for equation 5
                gx, gy = gradient_x(self.deblurred)
            beta *= beta_rate
        return self.deblurred

    def solve_w(self, v, beta, lut_v_range):
        """
        solve w-subproblem
        """
        if beta not in self.lut:
            if abs(self.alpha - 1) < 1e-6:
                self.lut[beta] = compute_w1(lut_v_range, beta)
            elif abs(self.alpha - 2 / 3) < 1e-6:
                self.lut[beta] = compute_w23(lut_v_range, beta)
            elif abs(self.alpha - 1 / 2) < 1e-6:
                self.lut[beta] = compute_w12(lut_v_range, beta)
            else:
                self.lut[beta] = compute_w_newton(lut_v_range, beta, self.alpha)
        interp = interp1d(lut_v_range, self.lut[beta], kind="linear", fill_value="extrapolate")
        return interp(v)

    def compute_const_denominator(self):
        """
        compute denominator and part of the numerator for equation 4
        """
        k_fft = psf2otf(self.kernel, self.imsize)
        kernel_gx = np.array([1, -1], dtype=np.float64).reshape(1, 2)
        kernel_gy = np.array([1, -1], dtype=np.float64).reshape(2, 1)
        nomin1 = np.conj(k_fft) * fft2(self.blurred)
        denom1 = np.abs(k_fft) ** 2
        denom2 = np.abs(fft2(kernel_gx, self.imsize)) ** 2 + np.abs(fft2(kernel_gy, self.imsize)) ** 2
        return nomin1, denom1, denom2

    def compute_cost(self, gx, gy):
        """
        compute cost using equation 2
        """
        blurred_syn = convolve2d(self.deblurred, self.kernel, boundary="wrap", mode="same")
        likelihood = np.sum(np.square(blurred_syn - self.blurred))
        return (self.lambda_ / 2) * likelihood + np.sum(np.abs(gx) ** self.alpha) + np.sum(np.abs(gy) ** self.alpha)


def psf2otf(psf, shape):
    psf_pad = np.zeros(shape, dtype=np.float64)
    psf_pad[: psf.shape[0], : psf.shape[1]] = psf
    otf = np.roll(psf_pad, (-int(psf.shape[0] / 2), -int(psf.shape[1] / 2)), axis=(0, 1))
    return fft2(otf)


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
    """
    solve w-subproblem for alpha = 1
    """
    return np.maximum(np.abs(v) - 1 / beta, 0) * np.sign(v)


def compute_w23(v, beta):
    """
    solve w-subproblem for alpha = 2/3
    """
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
    """
    solve w-subproblem for alpha = 1/2
    """
    pass


def compute_w_newton(v, beta, alpha, iterations=4):
    """
    for a general alpha, use Newton-Raphson
    """
    w = v.copy()
    for _ in range(iterations):
        dw = alpha * np.sign(w) * np.abs(w) ** (alpha - 1) + beta * (w - v)
        ddw = alpha * (alpha - 1) * np.abs(w) ** (alpha - 2) + beta
        w -= dw / ddw

    w[np.isnan(w)] = 0

    filt = (abs(w) ** alpha + beta / 2 * (w - v) ** 2) >= (beta / 2 * v**2)
    w[filt] = 0
    return w
