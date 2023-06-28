import numpy as np

_2D_NEIGHBORHOODS = ["n4", "n8"]
_3D_NEIGHBORHOODS = ["n6", "n18", "n26"]

def generate_labeled_skeletonization(label_image:'napari.types.LabelsData')->'napari.types.LabelsData':
    
    '''
    Skeletonize a label image and relabels the skeleton to match input labels.
    
    The used skeletonization algorithm is the scikit-image implementation of Lee, 94.
    
    Parameters:
    -----------
    
    label_image: napari.types.LabelsData
        Input napari version of numpy.ndarray containing integer labels of an instance segmentation.
    
    Returns:
    --------
    
    labeled_skeletons: napari.types.LabelsData
        A skeleton image multiplied by the respective label of the object of origin in the input label image.
    '''
    
    from skimage.morphology import skeletonize
    
    binary_skeleton = skeletonize(label_image.astype(bool)).astype(int)
    binary_skeleton = binary_skeleton / np.amax(binary_skeleton)
    labeled_skeletons = binary_skeleton * label_image
    
    return labeled_skeletons.astype(int)


def parse_skeleton(skel: "napari.types.LabelsData",
                   neighborhood: str = "n4") -> "napari.types.LabelsData":
    """
    Label the skeleton of a 2D or 3D object with 4-, 6-, 8-, 18-, or 26-connectivity.

    Parameters
    ----------
    skel : napari.types.LabelsData
        A skeletonized image
    neighborhood : str
        The neighborhood connectivity of the skeleton. Can be "n4", "n6", "n8", "n18", or "n26".
    z_dir : int, optional
        The direction of the z-axis, by default 0
    y_dir : int, optional
        The direction of the y-axis, by default 1
    x_dir : int, optional
        The direction of the x-axis, by default 2
        
    Returns
    -------
    skeleton_labels : napari.types.LabelsData
        A labeled image with the same shape as `skel` where each pixel is
        labeled as either an end point (1), a branch point (2), or a
        skeleton pixel (3).
    """
    from ._backend_toska_functions import (n4_parse_skel_2d,
                                           n8_parse_skel_2d,
                                           n6_parse_skel_3d,
                                           n18_parse_skel_3d,
                                           n26_parse_skel_3d)

    if len(skel.shape) == 2:
        x_dir = 1
        y_dir = 0
        if neighborhood == "n4":
            _, e_pts, b_pts, brnch, _, _ = n4_parse_skel_2d(skel, y_dir, x_dir)
        elif neighborhood == "n8":
            _, e_pts, b_pts, brnch, _, _ = n8_parse_skel_2d(skel, y_dir, x_dir)

    elif len(skel.shape) == 3:
        x_dir = 2
        y_dir = 1
        z_dir = 0
        if neighborhood == "n6":
            _, e_pts, b_pts, brnch, _, _ = n6_parse_skel_3d(skel, z_dir, y_dir, x_dir)
        elif neighborhood == "n18":
            _, e_pts, b_pts, brnch, _, _ = n18_parse_skel_3d(skel, z_dir, y_dir, x_dir)
        elif neighborhood == "n26":
            _, e_pts, b_pts, brnch, _, _ = n26_parse_skel_3d(skel, z_dir, y_dir, x_dir)

    skeleton_labels = np.zeros_like(skel, dtype=int)
    skeleton_labels[brnch > 0] = 2

    if len(skel.shape) == 2:
        for pt in b_pts:
            skeleton_labels[pt[0], pt[1]] = 3

        for pt in e_pts:
            skeleton_labels[pt[0], pt[1]] = 1

    elif len(skel.shape) == 3:
        for pt in b_pts:
            skeleton_labels[pt[0], pt[1], pt[2]] = 3

        for pt in e_pts:
            skeleton_labels[pt[0], pt[1], pt[2]] = 1

    return skeleton_labels
