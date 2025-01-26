from napari.layers import Labels
from napari_toska import parse_all_skeletons, generate_labeled_skeletonization
import networkx as nx
import numpy as np
import pandas as pd


class ToskaSkeleton(Labels):
    def __init__(self, labels_data: "napari.types.LabelsData", neighborhood, **kwargs):
        super().__init__(labels_data, **kwargs)

        self._neighborhood = neighborhood
        self.graph = nx.Graph()
        
        self._parse_skeleton(labels_data)
        self._build_nx_graph()

    @property
    def neighborhood(self):
        return self._neighborhood
    
    @property
    def graph(self) -> nx.Graph:
        return self.metadata.get('graph')
    
    @graph.setter
    def graph(self, graph: nx.Graph):
        self.metadata['graph'] = graph
    
    def _parse_skeleton(self, label_image: "napari.types.LabelsData") -> None:
        from skimage import measure

        labelled_skeletons = generate_labeled_skeletonization(label_image)
        parsed_skeletons = parse_all_skeletons(labelled_skeletons, neighborhood=self._neighborhood)
        end_points = measure.label(parsed_skeletons == 1)
        branches = measure.label(parsed_skeletons == 2)
        branch_points = measure.label(parsed_skeletons == 3)

        # create a labels image with unique labels for each object
        n_endpoints = end_points.max()
        n_branches = branches.max()
        branches[branches != 0] += n_endpoints
        branch_points[branch_points != 0] += n_endpoints + n_branches

        unique_labels = end_points + branches + branch_points

        # add object types to features (branch/end/chain)
        object_types = np.ones((unique_labels.max()), dtype=int)
        object_types[:n_endpoints] = 1
        object_types[n_endpoints:n_endpoints + n_branches] = 2
        object_types[n_endpoints + n_branches:] = 3

        self.data = unique_labels
        self.features = pd.DataFrame({
            'label': np.arange(1, unique_labels.max() + 1),
            'object_type': object_types,
        })

        # add skeleton ID to features
        for i, row in self.features.iterrows():
            self.features.loc[i, 'skeleton_id'] = labelled_skeletons[unique_labels == row['label']][0]

        return
    
    def _build_nx_graph(self):
        import tqdm

        # add all branch points and end points to Graph
        for _, row in self.features[self.features['object_type'] != 2].iterrows():
            self.graph.add_node(row['label'],
                                    object_type=row['object_type'],
                                    label=row['label'])
        

        df_branches = self.features[self.features['object_type'] == 2]

        # iterate over branches and find neighboring branch points or end points
        for i, row in tqdm.tqdm(df_branches.iterrows(), desc='Building Graph', total=len(df_branches)):
            connecting_labels = self._find_neighboring_labels(row['label'])

            # a branch should connect to exactly two other objects
            if len(connecting_labels) == 0:
                self.features.iloc[i, self.features.columns.get_loc('object_type')] = 1
                print('detected malformatted label: ', row['label'])
                continue

            self.graph.add_edge(connecting_labels[0], connecting_labels[1],
                                    label=row['label'])
            
        # check if there are any isolated nodes of type 1 (end points)
        isolated_nodes = [node for node in self.graph.nodes if self.graph.degree(node) == 0]
        if len(isolated_nodes) > 0:
            print('Found isolated nodes: ', isolated_nodes)

        # check for neighborhood around isolated nodes
        for node in isolated_nodes:
            connecting_labels = self._find_neighboring_labels(node)
            if len(connecting_labels) == 2:
                self.graph.add_edge(connecting_labels[0], connecting_labels[1],
                                    label=None)
            else:
                print('Could not connect isolated node: ', node)

    def _find_neighboring_labels(self, query_label: int):
        from skimage import morphology

        branch = self.data == query_label
        branch_point_coordinates = np.asarray(np.where(branch)).T

        # get bounding box around branch
        min_coords = branch_point_coordinates.min(axis=0) - 1
        max_coords = branch_point_coordinates.max(axis=0) + 2
        # Create a tuple of slices for each dimension
        slices = tuple(
            slice(min_coord, max_coord) for min_coord, max_coord in zip(min_coords, max_coords)
            )

        # Crop the image data using the slices,
        # then expand the branch points to overlap with the neighboring objects
        cropped_branch = branch[slices]
        cropped_branch = morphology.binary_dilation(cropped_branch, footprint=np.ones((3, 3)))
        cropped_data = self.data[slices]

        touching_labels = np.logical_xor(cropped_branch, cropped_data == query_label) * cropped_data
        connecting_labels = np.unique(touching_labels)
        connecting_labels = connecting_labels[connecting_labels != 0]

        return connecting_labels

