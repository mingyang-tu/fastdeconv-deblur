import numpy as np


def ssim(img1, img2, c1=0.005, c2=0.005):
    """
    compute ssim between two images
    """
    assert len(img1.shape) == len(img2.shape)
    assert img1.shape[0] == img2.shape[0] and img1.shape[1] == img2.shape[1]

    mn = img1.shape[0] * img1.shape[1]

    mean1 = np.mean(img1)
    mean2 = np.mean(img2)

    img1_0mean = img1 - mean1
    img2_0mean = img2 - mean2

    var1 = np.sum(img1_0mean**2) / mn
    var2 = np.sum(img2_0mean**2) / mn
    covar12 = np.sum(img1_0mean * img2_0mean) / mn

    L = np.max(img1)
    nomin = (2 * mean1 * mean2 + (c1 * L) ** 2) * (2 * covar12 + (c2 * L) ** 2)
    denom = (mean1**2 + mean2**2 + (c1 * L) ** 2) * (var1 + var2 + (c2 * L) ** 2)
    return nomin / denom


def nmse(img, target):
    """
    compute nmse between two images
    """
    assert img.shape[0] == target.shape[0] and img.shape[1] == target.shape[1]

    return np.sum((img - target) ** 2) / np.sum(target**2)
