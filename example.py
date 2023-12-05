import numpy as np
import cv2
import time
from scipy.signal import convolve2d

from fastdeconv import FastDeconvolution
from eval import ssim, nmse


def gaussian_kernel(size):
    kernel1d = cv2.getGaussianKernel(size, 0)
    kernel = kernel1d.dot(kernel1d.T)
    return kernel


def gaussian_noise(shape, amplitude=0.01, sigma=1):
    return np.random.normal(0, sigma, shape) * amplitude


def create_blur(image, kernel, noise):
    blurred = convolve2d(image, kernel, boundary="wrap", mode="same")
    return np.floor((blurred + noise) * 255.0) / 255.0


def resize(img, max_size):
    M = max(img.shape)
    ratio = float(max_size) / float(M)
    if M > max_size:
        img = cv2.resize(img, (0, 0), fx=ratio, fy=ratio)
    return img


if __name__ == "__main__":
    # grayscale image / 0 ~ 1
    original = cv2.imread("TRA.bmp", cv2.IMREAD_GRAYSCALE).astype(np.float64) / 255.0

    # create blurred image
    kernel = gaussian_kernel(11)
    noise = gaussian_noise(original.shape, amplitude=0.01, sigma=1)
    blurred = create_blur(original, kernel, noise)

    # deblur image
    fd = FastDeconvolution(blurred, kernel, 2000, 2 / 3)
    start = time.time()
    deblurred = fd.solve()
    end = time.time()
    print(f"Time taken for image of size {original.shape} is {end - start:.2f}s")

    # compute ssim and nmse
    nmse_blurred = nmse(blurred, original)
    nmse_deblurred = nmse(deblurred, original)
    ssim_blurred = ssim(blurred, original)
    ssim_deblurred = ssim(deblurred, original)

    print(f"NMSE blurred: {nmse_blurred:.4e}, SSIM blurred: {ssim_blurred:.4f}")
    print(f"NMSE deblurred: {nmse_deblurred:.4e}, SSIM deblurred: {ssim_deblurred:.4f}")

    cv2.imshow("original", original)
    cv2.imshow("blurred", blurred)
    cv2.imshow("deblurred", deblurred)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
