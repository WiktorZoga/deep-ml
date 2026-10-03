import numpy as np

def flip_image(image, direction):
    """
    Flip an image horizontally or vertically.
    
    Args:
        image: 2D or 3D list/array representing a grayscale or RGB image
        direction: string, either 'horizontal' or 'vertical'
    
    Returns:
        Flipped image as a nested list, or -1 if input is invalid
    """
    # Your code here
    img = np.asarray(image)

    dim = img.ndim

    if direction == "horizontal":
        filpped_img = img[:, ::-1, ...]
    elif direction == "vertical":
        filpped_img = img[::-1, :, ...]
    else:
        return -1
    
    return filpped_img
    