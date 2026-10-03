import numpy as np

def rgb_to_grayscale(image):
    """
    Convert an RGB image to grayscale using luminosity method.
    
    Args:
        image: RGB image as list or numpy array of shape (H, W, 3)
               with values in range [0, 255]
    
    Returns:
        Grayscale image as 2D list with integer values,
        or -1 if input is invalid
    """
    # Write your code here
    try:
        img = np.asarray(image, dtype=np.float32)
    except (ValueError, TypeError) as e:
        return -1

    if img.ndim != 3:
        return -1
    
    mini = np.min(img)
    maxi = np.max(img)

    if mini < 0 or maxi > 255:
        return -1

    weights = np.array([0.299, 0.587, 0.114]).reshape(3, 1)

    # img : (h, w, 3)
    # weights: (3,)

    h, w, _ = img.shape
    
    gray = np.matmul(img, weights).reshape(h, w)

    gray = np.round(gray).astype(np.uint8)

    return gray.tolist()
    