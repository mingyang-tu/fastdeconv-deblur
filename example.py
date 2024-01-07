import numpy as np
import cv2
import time
from scipy.signal import convolve2d

from fastdeconv import FastDeconvolution
from eval import ssim, psnr

np.random.seed(0)


def gaussian_kernel(size):
    kernel1d = cv2.getGaussianKernel(size, 0)
    kernel = kernel1d.dot(kernel1d.T)
    return kernel


def gaussian_noise(shape, amplitude=0.01, sigma=1.0):
    return np.random.normal(0, sigma, shape) * amplitude


def create_blur(image, kernel, noise):
    blurred = convolve2d(image, kernel, boundary="wrap", mode="same")
    return np.clip(np.floor((blurred + noise) * 255.0), 0, 255) / 255.0


if __name__ == "__main__":
    # grayscale image / 0 ~ 1
    original = cv2.imread("dsc_0085.jpg", cv2.IMREAD_GRAYSCALE).astype(np.float64) / 255.0
    original = cv2.resize(original, (0, 0), fx=0.25, fy=0.25)
    print(f"Image size: {original.shape}")

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

    # compute ssim and psnr
    psnr_blurred = psnr(blurred, original)
    psnr_deblurred = psnr(deblurred, original)
    ssim_blurred = ssim(blurred, original)
    ssim_deblurred = ssim(deblurred, original)

    print(f"PSNR blurred: {psnr_blurred:.4f}, SSIM blurred: {ssim_blurred:.4f}")
    print(f"PSNR deblurred: {psnr_deblurred:.4f}, SSIM deblurred: {ssim_deblurred:.4f}")

    cv2.imshow("original", original)
    cv2.imshow("blurred", blurred)
    cv2.imshow("deblurred", deblurred)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
