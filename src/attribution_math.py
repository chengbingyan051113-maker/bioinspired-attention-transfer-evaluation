"""Independent NumPy float64 oracle, not used in model training."""
import numpy as np

SCALES = ((3, .7, 7, 1.4), (5, 1., 9, 2.), (7, 1.5, 13, 3.))


def gaussian(width, sigma):
    axis = np.arange(width, dtype=np.float64) - (width - 1) / 2
    radius = axis[:, None] ** 2 + axis[None, :] ** 2
    value = np.exp(-radius / (2 * sigma ** 2))
    return value / value.sum()


def smooth(feature, kernel):
    radius = kernel.shape[0] // 2
    padded = np.pad(feature, ((0, 0), (0, 0), (radius, radius), (radius, radius)), mode='edge')
    height, width = feature.shape[-2:]
    value = np.zeros_like(feature, dtype=np.float64)
    for row in range(kernel.shape[0]):
        for col in range(kernel.shape[1]):
            value += kernel[row, col] * padded[:, :, row:row + height, col:col + width]
    return value


def responses(feature, mode, multipliers=(1., 1., 1.), rho=.1, epsilon=.001):
    feature = np.asarray(feature, dtype=np.float64)
    count = feature.shape[1]
    if mode == 'pooled':
        feature = np.concatenate((feature.mean(axis=1, keepdims=True),
                                  feature.max(axis=1, keepdims=True)), axis=1)
    centered = feature - feature.mean(axis=(-2, -1), keepdims=True)
    global_var = np.mean(centered ** 2, axis=(-2, -1), keepdims=True)
    maps = []
    for (kc, sc, ks, ss), factor in zip(SCALES, multipliers):
        center = smooth(centered, gaussian(kc, sc * factor))
        surround_kernel = gaussian(ks, ss * factor)
        local_mean = smooth(centered, surround_kernel)
        local_var = np.maximum(smooth(centered ** 2, surround_kernel) - local_mean ** 2, 0)
        maps.append((center - local_mean) / np.sqrt(local_var + rho ** 2 * global_var + epsilon ** 2))
    signed = np.stack(maps, axis=2)
    on, off = np.maximum(signed, 0), np.maximum(-signed, 0)
    if mode == 'pooled':
        on = np.broadcast_to(on.mean(axis=1, keepdims=True), (feature.shape[0], count, 3, *feature.shape[-2:]))
        off = np.broadcast_to(off.mean(axis=1, keepdims=True), (feature.shape[0], count, 3, *feature.shape[-2:]))
    if mode == 'unsigned':
        on = off = on + off
    shape = (feature.shape[0], 3 * count, *feature.shape[-2:])
    return on.reshape(shape), off.reshape(shape)
