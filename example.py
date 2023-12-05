import numpy as np
import cv2
from scipy.signal import convolve2d

from fastdeconv import FastDeconvolution


def gaussian_kernel(size):
    kernel1d = cv2.getGaussianKernel(size, 0)
    kernel = kernel1d.dot(kernel1d.T)
    return kernel


def create_blur(image, kernel, amp=0.01, sigma=1):
    blurred = convolve2d(image, kernel, boundary="wrap", mode="same")
    return blurred + np.random.normal(0, sigma, image.shape) * amp


if __name__ == "__main__":
    original = cv2.imread("TRA.bmp", cv2.IMREAD_GRAYSCALE).astype(np.float64) / 255.0
    kernel = gaussian_kernel(11)

    blurred = create_blur(original, kernel)

    # deblur image
    fd = FastDeconvolution(blurred, kernel, 2000, 2 / 3, verbose=True)
    deblurred = fd.solve()

    cv2.imshow("original", original)
    cv2.imshow("blurred", blurred)
    cv2.imshow("deblurred", deblurred)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
