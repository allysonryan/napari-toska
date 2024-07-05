
def test_adjacency_matrix():
    import napari_toska as nts

    from skimage.data import binary_blobs
    from skimage.measure import label

    labels = label(binary_blobs(rng=0))
    labeled_skeletons = nts.generate_labeled_skeletonization(labels)
    parsed_skeletons_single = nts.parse_single_skeleton(labeled_skeletons, label=2, neighborhood='n8')

    adjacency_matrix = nts.create_adjacency_matrix(parsed_skeletons_single, neighborhood='n8')

    assert all(adjacency_matrix.sum(axis=0) == 2)
