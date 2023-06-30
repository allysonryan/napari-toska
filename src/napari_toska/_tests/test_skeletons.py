def test_skeletonization():
    import napari_toska as nts
    import numpy as np
    from skimage.data import binary_blobs
    from skimage.measure import label

    labels = label(binary_blobs(seed=0))
    skeleton = nts.generate_labeled_skeletonization(labels)

    assert skeleton.max() == 15


def test_skeleton_parsing():
    import napari_toska as nts
    import numpy as np
    from skimage.data import binary_blobs
    from skimage.measure import label

    # check if the skeleton is parsed correctly
    # there should only be labels 1, 2, and 3
    labels = label(binary_blobs(seed=0))
    skeleton = nts.generate_labeled_skeletonization(labels)
    parsed_skeleton = nts.parse_single_skeleton(skeleton,
                                                label=2,
                                                neighborhood="n8",
                                                )
    for i in [1, 2, 3]:
        assert i in np.unique(parsed_skeleton)

    parsed_skeleton = nts.parse_all_skeletons(skeleton,
                                              neighborhood="n8")
    for i in [1, 2, 3]:
        assert i in np.unique(parsed_skeleton)