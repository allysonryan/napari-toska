from napari.layers import Labels
import napari_toska as nts
import networkx as nx
import numpy as np
import pandas as pd


class ToskaSkeleton(Labels):
    """
    A class to represent a skeleton image in napari.

    Parameters:
    -----------
    labels_data : napari.types.LabelsData
        The data to be processed - has to be a napari Label image.
    neighborhood : str
        The neighborhood connectivity of the skeleton. Can be "n4",
        "n6", "n8", "n18", or "n26". "n4" and "n8" apply only to 2D images.
        "n6", "n18", and "n26" apply to 3D images.
    **kwargs
        Additional keyword arguments to pass to the Labels layer.

    Attributes:
    -----------
    neighborhood : str
        The neighborhood connectivity of the skeleton.
    graph : nx.Graph
        A networkx graph representing the skeleton.
    """

    def __init__(self, labels_data: "napari.types.LabelsData", neighborhood: str, **kwargs):
        super().__init__(labels_data, **kwargs)

        self._neighborhood = neighborhood
        self.graph = nx.Graph()
        
    def analyze(self):
        """
        Analyze the skeleton data and build a networkx graph from it.

        Returns:
        --------
        None
        """
        self._parse_skeleton()
        self._build_nx_graph()
        self._measure_branch_length()

        # drop index column
        self.features = self.features.drop(columns='index')

    @property
    def neighborhood(self):
        return self._neighborhood
    
    @property
    def graph(self) -> nx.Graph:
        return self.metadata.get('graph')
    
    @graph.setter
    def graph(self, graph: nx.Graph):
        self.metadata['graph'] = graph
    
    def _parse_skeleton(self) -> None:
        from skimage import measure

        labelled_skeletons = nts.generate_labeled_skeletonization(self.data).astype(int)
        parsed_skeletons = nts.parse_all_skeletons(labelled_skeletons, neighborhood=self._neighborhood)
        end_points = measure.label(parsed_skeletons == 1)
        branches = measure.label(parsed_skeletons == 2)
        branch_points = measure.label(parsed_skeletons == 3)

        # create a labels image with unique labels for each object
        n_endpoints = end_points.max()
        n_branches = branches.max()
        branches[branches != 0] += n_endpoints
        branch_points[branch_points != 0] += n_endpoints + n_branches

        unique_labels = (end_points + branches + branch_points).astype(int)

        # add object types to features (branch/end/chain)
        object_types = np.ones((unique_labels.max()), dtype=int)
        object_types[:n_endpoints] = 1
        object_types[n_endpoints:n_endpoints + n_branches] = 2
        object_types[n_endpoints + n_branches:] = 3

        self.data = unique_labels
        self.features = pd.DataFrame({
            'label': np.arange(1, unique_labels.max() + 1).astype(int),
            'object_type': object_types,
        })

        # add skeleton ID to features
        for i, row in self.features.iterrows():
            self.features.loc[i, 'skeleton_id'] = labelled_skeletons[self.data == row['label']][0]

        # make column type int
        self.features['skeleton_id'] = self.features['skeleton_id'].astype(int)

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
                print('detected malformatted label: {}'.format(int(row['label'])),
                      'changed type from 2 (branch) -> 1 (end point)')
                continue

            self.graph.add_edge(connecting_labels[0], connecting_labels[1],
                                    label=row['label'])
            
        # check if there are any isolated nodes of type 1 (end points)
        isolated_nodes = np.array([node for node in self.graph.nodes if self.graph.degree(node) == 0], dtype=int)
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

    def _measure_branch_length(self):
        # create a label images with only branch labels, mute all the others

        LUT = np.asarray([0] + list(self.features['label']))
        object_type = np.asarray([0] + list(self.features['object_type']))

        # set entries to zero where the object type is 1 or 3
        LUT[object_type == 1] = 0
        LUT[object_type == 3] = 0
        branch_label_image = LUT[self.data]


        # measure the length of each branch
        branch_lengths = nts.calculate_branch_lengths(branch_label_image)
        
        # merge into features
        self.features = pd.merge(self.features, branch_lengths, on='label', how='left')
        self.features['branch_length'] = self.features['branch_length'].fillna(0)

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
