from itertools import product
from numbers import Integral, Number, Real

import numpy as np
from numpy.lib.stride_tricks import as_strided

###############################################################################
# From an image to a set of small image patches

def _compute_n_patches(i_h, i_w, p_h, p_w, max_patches=None):
    """Compute the number of patches that will be extracted in an image.

    Read more in the :ref:`User Guide <image_feature_extraction>`.

    Parameters
    ----------
    i_h : int
        The image height
    i_w : int
        The image with
    p_h : int
        The height of a patch
    p_w : int
        The width of a patch
    max_patches : int or float, default=None
        The maximum number of patches to extract. If `max_patches` is a float
        between 0 and 1, it is taken to be a proportion of the total number
        of patches. If `max_patches` is None, all possible patches are extracted.
    """
    n_h = i_h - p_h + 1
    n_w = i_w - p_w + 1
    all_patches = n_h * n_w

    if max_patches:
        if isinstance(max_patches, (Integral)) and max_patches < all_patches:
            return max_patches
        elif isinstance(max_patches, (Integral)) and max_patches >= all_patches:
            return all_patches
        elif isinstance(max_patches, (Real)) and 0 < max_patches < 1:
            return int(max_patches * all_patches)
        else:
            raise ValueError("Invalid value for max_patches: %r" % max_patches)
    else:
        return all_patches


def _extract_patches(arr, patch_shape=8, extraction_step=1):
    """Extracts patches of any n-dimensional array in place using strides.

    Given an n-dimensional array it will return a 2n-dimensional array with
    the first n dimensions indexing patch position and the last n indexing
    the patch content. This operation is immediate (O(1)). A reshape
    performed on the first n dimensions will cause numpy to copy data, leading
    to a list of extracted patches.

    Read more in the :ref:`User Guide <image_feature_extraction>`.

    Parameters
    ----------
    arr : ndarray
        n-dimensional array of which patches are to be extracted

    patch_shape : int or tuple of length arr.ndim.default=8
        Indicates the shape of the patches to be extracted. If an
        integer is given, the shape will be a hypercube of
        sidelength given by its value.

    extraction_step : int or tuple of length arr.ndim, default=1
        Indicates step size at which extraction shall be performed.
        If integer is given, then the step is uniform in all dimensions.


    Returns
    -------
    patches : strided ndarray
        2n-dimensional array indexing patches on first n dimensions and
        containing patches on the last n dimensions. These dimensions
        are fake, but this way no data is copied. A simple reshape invokes
        a copying operation to obtain a list of patches:
        result.reshape([-1] + list(patch_shape))
    """

    arr_ndim = arr.ndim

    if isinstance(patch_shape, Number):
        patch_shape = tuple([patch_shape] * arr_ndim)
    if isinstance(extraction_step, Number):
        extraction_step = tuple([extraction_step] * arr_ndim)

    patch_strides = arr.strides

    slices = tuple(slice(None, None, st) for st in extraction_step)
    indexing_strides = arr[slices].strides

    patch_indices_shape = (
        (np.array(arr.shape) - np.array(patch_shape)) // np.array(extraction_step)
    ) + 1

    shape = tuple(list(patch_indices_shape) + list(patch_shape))
    strides = tuple(list(indexing_strides) + list(patch_strides))

    patches = as_strided(arr, shape=shape, strides=strides)
    return patches



def extract_patches_2d(image, patch_size, *, max_patches=None, random_state=None, extraction_step=1):
    """Reshape a 2D image into a collection of patches.

    The resulting patches are allocated in a dedicated array.

    Read more in the :ref:`User Guide <image_feature_extraction>`.

    Parameters
    ----------
    image : ndarray of shape (image_height, image_width) or \
        (image_height, image_width, n_channels)
        The original image data. For color images, the last dimension specifies
        the channel: a RGB image would have `n_channels=3`.

    patch_size : tuple of int (patch_height, patch_width)
        The dimensions of one patch.

    max_patches : int or float, default=None
        The maximum number of patches to extract. If `max_patches` is a float
        between 0 and 1, it is taken to be a proportion of the total number
        of patches. If `max_patches` is None it corresponds to the total number
        of patches that can be extracted.

    random_state : int, RandomState instance, default=None
        Determines the random number generator used for random sampling when
        `max_patches` is not None. Use an int to make the randomness
        deterministic.
        See :term:`Glossary <random_state>`.

    Returns
    -------
    patches : array of shape (n_patches, patch_height, patch_width) or \
        (n_patches, patch_height, patch_width, n_channels)
        The collection of patches extracted from the image, where `n_patches`
        is either `max_patches` or the total number of patches that can be
        extracted.

    """
    i_h, i_w = image.shape[:2]
    p_h, p_w = patch_size

    if p_h > i_h:
        raise ValueError(
            "Height of the patch should be less than the height of the image."
        )

    if p_w > i_w:
        raise ValueError(
            "Width of the patch should be less than the width of the image."
        )

   
    image = image.reshape((i_h, i_w, -1))
    n_colors = image.shape[-1]

    extracted_patches = _extract_patches(
        image, patch_shape=(p_h, p_w, n_colors), extraction_step=extraction_step
    )

    n_patches = _compute_n_patches(i_h, i_w, p_h, p_w, max_patches)
   
    patches = extracted_patches

    patches = patches.reshape(-1, p_h, p_w, n_colors)
    # remove the color dimension if useless
    if patches.shape[-1] == 1:
        return patches.reshape((n_patches, p_h, p_w))
    else:
        return patches


def reconstruct_from_patches_2d(patches, image_size):
    """Reconstruct the image from all of its patches.

    Patches are assumed to overlap and the image is constructed by filling in
    the patches from left to right, top to bottom, averaging the overlapping
    regions.

    Read more in the :ref:`User Guide <image_feature_extraction>`.

    Parameters
    ----------
    patches : ndarray of shape (n_patches, patch_height, patch_width) or \
        (n_patches, patch_height, patch_width, n_channels)
        The complete set of patches. If the patches contain colour information,
        channels are indexed along the last dimension: RGB patches would
        have `n_channels=3`.

    image_size : tuple of int (image_height, image_width) or \
        (image_height, image_width, n_channels)
        The size of the image that will be reconstructed.

    Returns
    -------
    image : ndarray of shape image_size
        The reconstructed image.
    """
    i_h, i_w = image_size[:2]
    p_h, p_w = patches.shape[1:3]
    img = np.zeros(image_size)
    # compute the dimensions of the patches array
    n_h = i_h - p_h + 1
    n_w = i_w - p_w + 1
    for p, (i, j) in zip(patches, product(range(n_h), range(n_w))):
        img[i : i + p_h, j : j + p_w] += p

    for i in range(i_h):
        for j in range(i_w):
            # divide by the amount of overlap
            # XXX: is this the most efficient way? memory-wise yes, cpu wise?
            img[i, j] /= float(min(i + 1, p_h, i_h - i) * min(j + 1, p_w, i_w - j))
    return img
