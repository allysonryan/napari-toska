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


        
        
    
#register_layer_type(ToskaSkeleton, 'toska_skeleton')