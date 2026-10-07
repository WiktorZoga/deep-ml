import numpy as np

def patch_embed_reconstruct(image: np.ndarray, patch_size: int):
    """
    Split image into patches, flatten to patch embedding matrix,
    then reconstruct the original image.

    Args:
        image: 2D numpy array of shape (H, W)
        patch_size: int, side length of each square patch

    Returns:
        Reconstructed image as a nested list of shape (H, W),
        or -1 if dimensions are invalid.
    """

    h, w = image.shape

    if h % patch_size != 0 or w % patch_size != 0:
        return -1

    h_patches = h // patch_size
    w_patches = w // patch_size

    patches = image.reshape(h_patches, patch_size, w_patches, patch_size).transpose(0, 2, 1, 3)
    flatten_paches = patches.reshape(-1, patch_size * patch_size)
    recon_patches = flatten_paches.reshape(h_patches, w_patches, patch_size, patch_size)
    recon_img = recon_patches.transpose(0, 2, 1, 3).reshape(h, w)

    return recon_img


