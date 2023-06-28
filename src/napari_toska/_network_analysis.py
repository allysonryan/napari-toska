import numpy as np

def create_adjacency_matrix(labelled_skeletons: "napari.types.LabelsData",
                            neighborhood: str = "n4") -> np.ndarray:
    from scipy import ndimage
    from skimage.morphology import disk, ball, square
    from ._backend_toska_functions import generate_adjacency_matrix

    if len(labelled_skeletons.shape) == 2:
        if neighborhood == "n4":
            structure = disk(1)
        elif neighborhood == "n8":
            structure = np.ones((3, 3))

    elif len(labelled_skeletons.shape) == 3:
        if neighborhood == "n6":
            structure = ball(1)
        elif neighborhood == "n18":
            structure = np.stack([disk(1), square(1), disk(1)])
        elif neighborhood == "n26":
            structure = np.ones((3, 3, 3))

    # Retrieve branch points
    branch_points = labelled_skeletons == 3
    branch_points, _ = ndimage.label(
        branch_points, structure=structure)

    # Retrieve end points
    end_points = np.asarray(np.where(labelled_skeletons == 1)).T

    # Retrieve branches image
    branches, _ = ndimage.label(labelled_skeletons == 2, structure=structure)

    # generate adjacency matrix
    M = generate_adjacency_matrix(end_points, branch_points, branches, structure)

    return M
