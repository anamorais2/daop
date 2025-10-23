import numpy as np


def np_entropy(p):
    # flatten if not flat
    if len(p.shape) > 1:
        p = p.flatten()

    # avoid 0s
    p = p + 1e-10

    # normalize
    p = p / np.sum(p)

    # entropy
    return -np.sum(p * np.log2(p))


def np_neg_entropy(p):
    # flatten if not flat
    if len(p.shape) > 1:
        p = p.flatten()

    # avoid 0s
    p = p + 1e-10

    # normalize
    p = p / np.sum(p)

    # negative
    p = np.max(p) - p + np.min(p)

    # entropy
    return -np.sum(p * np.log2(p))



def np_centroid(hmap):
    y,x = np.indices(hmap.shape)
    total = np.sum(hmap)
    return np.sum(x*hmap)/total, np.sum(y*hmap)/total


def np_spatial_var(hmap):
    # normalize
    hmap = hmap / np.sum(hmap)

    x_cent, y_cent = np_centroid(hmap)

    y_ind, x_ind = np.indices(hmap.shape)

    total = np.sum(hmap)
    x_var = np.sum((x_ind - x_cent)**2 * hmap) / total
    y_var = np.sum((y_ind - y_cent)**2 * hmap) / total

    return x_var, y_var


def np_neg_spatial_var(hmap):
    # normalize
    hmap = hmap / np.sum(hmap)
    
    hmap = np.max(hmap) - hmap + np.min(hmap)

    x_cent, y_cent = np_centroid(hmap)

    y_ind, x_ind = np.indices(hmap.shape)

    total = np.sum(hmap)
    x_var = np.sum((x_ind - x_cent)**2 * hmap) / total
    y_var = np.sum((y_ind - y_cent)**2 * hmap) / total

    return x_var, y_var


def max_abs_val(a,b):
    if abs(a) > abs(b):
        return a
    else:
        return b
    

def min_abs_val(a,b):
    if abs(a) < abs(b):
        return a
    else:
        return b