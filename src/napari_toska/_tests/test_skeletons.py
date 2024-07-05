def test_skeletonization():
    import napari_toska as nts
    import numpy as np
    from skimage.data import binary_blobs
    from skimage.measure import label

    labels = label(binary_blobs(rng=0))
    skeleton = nts.generate_labeled_skeletonization(labels)

    assert skeleton.max() == 15

def test_simple_skeleton():
    import napari_toska as nts
    from skimage.morphology import skeletonize
    from skimage.data import binary_blobs
    from skimage.measure import label
    import numpy as np

    labels = label(binary_blobs(rng=0))
    labeled_skeletons = nts.generate_labeled_skeletonization(labels)

    # spine length (in image), number of pixels in spine
    parsed_skeletons_single = nts.parse_single_skeleton(labeled_skeletons, label=14, neighborhood='n8')
    branch_labels = nts.label_branches(parsed_skeletons_single, 1*(parsed_skeletons_single > 0),
                                        neighborhood='n8')

    adjacency_matrix = nts.create_adjacency_matrix(parsed_skeletons_single, neighborhood='n8')
    spine_image = nts.create_spine_image(adjacency_matrix, branch_labels)


def test_skeleton_parsing():

    # TODO: use simpler skeleton for this test
    import napari_toska as nts
    import numpy as np
    from skimage.data import binary_blobs
    from skimage.measure import label

    # check if the skeleton is parsed correctly
    # there should only be labels 1, 2, and 3
    labels = label(binary_blobs(rng=0))
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


def test_measurements():
    import numpy as np
    import napari_toska as nts

    mock_skeleton = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0,],
        [0, 1, 0, 0, 0, 0, 0, 0, 0,],
        [0, 0, 1, 0, 0, 0, 1, 0, 0,],
        [0, 0, 0, 1, 0, 1, 0, 0, 0,],
        [0, 0, 0, 0, 1, 0, 0, 0, 0,],
        [0, 0, 0, 0, 1, 0, 0, 0, 0,],
        [0, 0, 0, 0, 1, 0, 0, 0, 0,],
        [0, 0, 0, 0, 1, 0, 0, 0, 0,],
        [0, 0, 0, 1, 0, 1, 0, 0, 0,],
        [0, 0, 1, 0, 0, 0, 1, 0, 0,],
        [0, 0, 0, 0, 0, 0, 0, 1, 0,],
        [0, 0, 0, 0, 0, 0, 0, 0, 0,],
    ])

    # test parsing
    parsed_skeleton = nts.parse_all_skeletons(
        mock_skeleton, neighborhood='n8')
    for i in [1, 2, 3]:
        assert i in np.unique(parsed_skeleton)

    # test Intermediates
    adjacency_matrix = nts.create_adjacency_matrix(parsed_skeleton,
                                                   neighborhood='n8')
    graph = nts.convert_adjacency_matrix_to_graph(adjacency_matrix)
    assert adjacency_matrix.shape[0] == 6
    assert adjacency_matrix.shape[1] == 5

    labeled_branches_single = nts.label_branches(parsed_skeleton,
                                                 mock_skeleton,
                                                 neighborhood='n8')
    assert len(np.unique(labeled_branches_single) == 5)
    spine = nts.create_spine_image(adjacency_matrix=adjacency_matrix,
                                   labeled_branches=labeled_branches_single)
    assert len(np.unique(spine)[1:]) == 3

    # Measurements: Coarse
    features = nts.analyze_skeletons(
        labeled_skeletons=mock_skeleton,
        parsed_skeletons=parsed_skeleton)

    assert features.shape[0] == 1
    assert features.loc[0]['n_cycle_basis'] == 0
    assert features.loc[0]['n_branch_points'] == 2
    assert features.loc[0]['n_endpoints'] == 4
    assert features.loc[0]['n_nodes'] == 6
    assert features.loc[0]['n_branches'] == 5
    assert features.loc[0]['skeleton_id'] == 1
    # assert features.loc[0]['spine_length'] == 3

    # Measurements: Fine
    features_fine = nts.analyze_single_skeleton_network(parsed_skeleton,
                                                        neighborhood='n8')
    assert features_fine.shape[0] == 11
    assert np.array_equal(features_fine['degree'].dropna().unique(), [3, 1])
    assert np.array_equal(features_fine['component_type'].unique(), ['node', 'edge'])


def test_measurement_3d():

    # import 3d binary blobs
    import numpy as np
    import napari_toska as nts
    from skimage import data, measure

    image = data.binary_blobs(length=64, n_dim=3, rng=0, blob_size_fraction=0.3)
    labels = measure.label(image)
    skeletons = nts.generate_labeled_skeletonization(labels)
    parsed_skeleton = nts.parse_all_skeletons(skeletons, neighborhood='n26')

    parsed_skeleton_single = parsed_skeleton * (skeletons == 2)

    features_single = nts.analyze_single_skeleton(parsed_skeleton_single, neighborhood='n26')

    # analyze all skeletons
    features = nts.analyze_skeletons(
        labeled_skeletons=skeletons,
        parsed_skeletons=parsed_skeleton,
        neighborhood='n26')
