import numpy as np

def zero_pad_image(img, pad_width):
    """
    Add zero padding around a grayscale image.
    
    Args:
        img: 2D list or numpy array of pixel values
        pad_width: integer number of pixels to pad on each side
    
    Returns:
        Padded image as 2D list with integer values,
        or -1 if input is invalid
    """
    # Write your code here

    if pad_width < 0:
        return -1

    try:
        img = np.asarray(img)
    except ... :
        return -1

    if img.ndim != 2:
        return -1

    if pad_width == 0:
        return img

    h, w = img.shape

    pad = np.zeros(shape=(h, pad_width))

    img = np.concatenate([img, pad], axis=1)
    img = np.concatenate([pad, img], axis=1)

    h, w = img.shape

    pad = np.zeros(shape=(pad_width, w))

    img = np.concatenate([img, pad], axis=0)
    img = np.concatenate([pad, img], axis=0)

    return img
    
    