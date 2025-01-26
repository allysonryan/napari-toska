from napari.layers import Labels
from napari_toska import parse_all_skeletons, generate_labeled_skeletonization
import networkx as nx
import numpy as np
import pandas as pd


class ToskaSkeleton(Labels):
    def __init__(self, skeleton_data, neighborhood, **kwargs):
        super().__init__(skeleton_data, **kwargs)

        self._neighborhood = neighborhood
        self._graph = nx.Graph()
        
        self._parse_skeleton(skeleton_data)

    @property
    def neighborhood(self):
        return self._neighborhood
    
    def _parse_skeleton(self, skeleton):
        from skimage import measure

        labelled_skeletons = generate_labeled_skeletonization(skeleton)
        parsed_skeletons = parse_all_skeletons(labelled_skeletons, neighborhood=self._neighborhood)
        end_points = measure.label(parsed_skeletons == 1)
        branches = measure.label(parsed_skeletons == 2)
        branch_points = measure.label(parsed_skeletons == 3)

        branches[branches > 0] += end_points.max()
        branch_points[branch_points > 0] += end_points.max() + branches.max()

        unique_labels = end_points + branches + branch_points
        self.data = unique_labels

        # add object types to features (branch/end/chain)
        object_types = np.ones((unique_labels.max()), dtype=int)
        object_types[:end_points.max()] = 1
        object_types[end_points.max() + 1:branches.max() + end_points.max() + 1] = 2
        object_types[-branches.max():] = 3

        # infer skeleton id from labelled skeleton

        self.features = pd.DataFrame({
            'labels': np.arange(1, self.data.max() + 1),
            'object_type': object_types
        })


        
        
    
    def _build_nx_graph(self):
        from skimage import morphology
        import tqdm

        # add all branch points and end points to Graph
        for _, row in self.features[self.features['object_type'] != 2].iterrows():
            self.graph.add_node(row['label'],
                                    object_type=row['object_type'],
                                    label=row['label'])
        

        df_branches = self.features[self.features['object_type'] == 2]

        for i, row in tqdm.tqdm(df_branches.iterrows(), desc='Building Graph', total=len(df_branches)):
            branch = self.data == row['label']
            branch_point_coordinates = np.asarray(np.where(branch)).T

            # get bounding box around branch
            min_coords = branch_point_coordinates.min(axis=0) - 1
            max_coords = branch_point_coordinates.max(axis=0) + 2
            # Create a tuple of slices for each dimension
            slices = tuple(
                slice(min_coord, max_coord) for min_coord, max_coord in zip(min_coords, max_coords)
                )

            # Crop the image data using the slices
            cropped_branch = branch[slices]
            cropped_branch = morphology.binary_dilation(cropped_branch, footprint=np.ones((3, 3)))
            cropped_data = self.data[slices]

            touching_labels = np.logical_xor(cropped_branch, cropped_data == row['label']) * cropped_data
            connecting_labels = np.unique(touching_labels)
            connecting_labels = connecting_labels[connecting_labels != 0]

            if len(connecting_labels) == 0:
                self.features.iloc[i, self.features.columns.get_loc('object_type')] = 1
                print('detected malformatted label: ', row['label'])
                continue

            self.graph.add_edge(connecting_labels[0], connecting_labels[1],
                                    label=row['label'])

