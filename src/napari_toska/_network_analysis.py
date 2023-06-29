import numpy as np


def calculate_all_adjancency_matrices(
        labelled_skeletons: "napari.types.LabelsData",
        parsed_skeletons: "napari.types.LabelsData",
        neighborhood: str
):
    """
    Calculate all adjacency matrices for a given skeleton image.

    Parameters:
    -----------
    labelled_skeletons: napari.types.LabelsData
        A skeleton image with each skeleton carrying a unique label.
    parsed_skeletons: napari.types.LabelsData
        A skeleton image where each pixel is labelled according to the
        point type which can be either a terminal point (1), a branching
        point (3), or a chain point (2).
    neighborhood: str
        The neighborhood connectivity of the skeleton. Can be "n4",
        "n6", "n8", "n18", or "n26".

    Returns:
    --------
    adjacency_matrices: list
        A list of adjacency matrices for each skeleton in the image.
    """
    skeleton_ids = np.unique(labelled_skeletons)[1:]
    adjacency_matrices = [0] * len(skeleton_ids)

    skeleton_counter = 0
    for skeleton_id in skeleton_ids:
        # create a sub-skeleton image with only one skeleton with values
        # 0 (background) and 1 (skeleton end point) etc.
        sub_skeleton = parsed_skeletons * (labelled_skeletons == skeleton_id)
        adjacency_matrices[skeleton_counter] = create_adjacency_matrix(sub_skeleton, neighborhood)
        skeleton_counter += 1

    return adjacency_matrices


def create_adjacency_matrix(labelled_skeletons: "napari.types.LabelsData",
                            neighborhood: str = "n4") -> np.ndarray:
    from scipy import ndimage
    from skimage.morphology import disk, ball, square
    from ._backend_toska_functions import _generate_adjacency_matrix

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
