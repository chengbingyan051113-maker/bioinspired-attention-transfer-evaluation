"""Fixed candidate geometry; NumPy functions also support local mathematical checks."""
import numpy as np

SCALES = ((3, 0.7, 7, 1.4), (5, 1.0, 9, 2.0), (7, 1.5, 13, 3.0))


def gaussian_kernel(size, sigma):
    if size < 1 or size % 2 != 1 or sigma <= 0:
        raise ValueError('Gaussian kernel needs a positive odd size and positive sigma')
    coordinate = np.arange(size, dtype=np.float64) - size // 2
    y, x = np.meshgrid(coordinate, coordinate, indexing='ij')
    kernel = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    return kernel / kernel.sum()


def dog_kernel(center_size, center_sigma, surround_size, surround_sigma):
    center = gaussian_kernel(center_size, center_sigma)
    surround = gaussian_kernel(surround_size, surround_sigma)
    padded = np.zeros_like(surround)
    start = (surround_size - center_size) // 2
    padded[start:start + center_size, start:start + center_size] = center
    return padded - surround
