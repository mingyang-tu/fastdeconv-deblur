import numpy as np
import cv2
from scipy.signal import convolve2d

from fastdeconv import FastDeconvolution


if __name__ == "__main__":
    original = cv2.imread("TRA.bmp", cv2.IMREAD_GRAYSCALE)
    # gaussian blur kernel
    kernel1d = cv2.getGaussianKernel(11, 0)
    kernel = kernel1d.dot(kernel1d.T)
    blurred = convolve2d(original, kernel, boundary="symm", mode="same")

    # deblur image
    fd = FastDeconvolution(blurred, kernel, 2000, 2 / 3)
    deblurred = fd.solve(verbose=True)

    cv2.imshow("original", original)
    cv2.imshow("blurred", blurred / 255.)
    cv2.imshow("deblurred", deblurred / 255.)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
