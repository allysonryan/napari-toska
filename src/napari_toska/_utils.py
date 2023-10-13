def get_neighborhood(neighborhood: str):
    from skimage.morphology import disk, ball, square
    import numpy as np
    if neighborhood == "n4":
        structure = disk(1)
    elif neighborhood == "n8":
        structure = np.ones((3, 3))
    elif neighborhood == "n6":
        structure = ball(1)
    elif neighborhood == "n18":
        structure = np.stack([disk(1), square(1), disk(1)])
    elif neighborhood == "n26":
        structure = np.ones((3, 3, 3))
    
    return structure