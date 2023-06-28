def test_skeletonization():
    import napari_toska as nts
    import numpy as np
    from skimage.data import binary_blobs
    from skimage.measure import label

    labels = label(binary_blobs(seed=0))
    skeleton = nts.generate_labeled_skeletonization(labels)

    assert skeleton.max() == 15