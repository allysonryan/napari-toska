import pandas as pd


def analyze_skeletons(
        labeled_skeletons: "napari.types.LabelsData",
        parsed_skeletons: "napari.types.LabelsData",
        neighborhood: str = "n8",
        viewer: "napari.Viewer" = None
        ) -> pd.DataFrame:
    import numpy as np

    for label in np.unique(labeled_skeletons)[1:]:
        parsed_skeletons_single = parsed_skeletons * (labeled_skeletons == label)
        df = analyze_single_skeleton(
            parsed_skeletons_single, neighborhood=neighborhood)
        df["skeleton_id"] = label
        if label == 1:
            df_all = df
        else:
            df_all = pd.concat([df_all, df], axis=0)

    # move skeleton id to first column by its name
    col = df_all.pop('skeleton_id')  # Remove column 'B' and store it in col
    df_all.insert(0, 'skeleton_id', col)

    # add label column
    df_all["label"] = df_all["skeleton_id"].values

    if viewer is not None:
        from napari_workflows._workflow import _get_layer_from_data
        from napari_skimage_regionprops import add_table
        skeleton_layer = _get_layer_from_data(viewer, labeled_skeletons)
        skeleton_layer.features = df_all
        add_table(skeleton_layer, viewer)

    return df_all


def analyze_single_skeleton(
        parsed_labeled_skeleton: "napari.types.LabelsData",
        neighborhood: str = "n8"
        ) -> pd.DataFrame:
    import networkx as nx
    import numpy as np
    from ._network_analysis import (
        create_adjacency_matrix,
        convert_adjacency_matrix_to_graph)
    from ._backend_toska_functions import skeleton_spine_search

    # create an adjacency matrix for the skeleton
    adjacency_matrix = create_adjacency_matrix(parsed_labeled_skeleton,
                                               neighborhood=neighborhood)
    graph = convert_adjacency_matrix_to_graph(adjacency_matrix)

    # number of end points
    n_endpoints = sum(adjacency_matrix.sum(axis=1) == 1)

    # number of branch points
    n_branch_points = sum(adjacency_matrix.sum(axis=1) > 1)

    # number of nodes
    n_nodes = n_endpoints + n_branch_points

    # number of branches
    n_branches = adjacency_matrix.shape[1]

    # spine length
    _, spine_paths_length = skeleton_spine_search(
        adjacency_matrix, graph)
    spine_length = np.nansum(spine_paths_length)

    # cycle basis
    directed_graph = graph.to_directed()
    possible_directed_cycles = list(nx.simple_cycles(directed_graph))

    n_cycle_basis = len(nx.cycle_basis(graph))
    n_possible_undirected_cycles = len(
        [x for x in possible_directed_cycles if len(x) > 2])//2

    df = pd.DataFrame(
        {
            "n_endpoints": [n_endpoints],
            "n_branch_points": [n_branch_points],
            "n_nodes": [n_nodes],
            "n_branches": [n_branches],
            "spine_length": [spine_length],
            "n_cycle_basis": [n_cycle_basis],
            "n_possible_undirected_cycles": [n_possible_undirected_cycles]
        }
    )

    return df
            
