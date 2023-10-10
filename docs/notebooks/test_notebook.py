# %%
import napari_toska as nts

# %%
from skimage.morphology import skeletonize
from skimage.data import binary_blobs
from skimage.measure import label
import numpy as np
import napari

# %%
labels = label(binary_blobs(seed=0))

# %%
labeled_skeletons = nts.generate_labeled_skeletonization(labels)
labeled_skeletons.max()

# %% [markdown]
# ## Parse skeletons

# %%
parsed_skeletons_single = nts.parse_single_skeleton(labeled_skeletons, label=2, neighborhood='n8')
parsed_skeletons_all = nts.parse_all_skeletons(labeled_skeletons, neighborhood='n8')

# %%
viewer = napari.Viewer()

# %%
viewer.add_labels(labels, name='labels')
viewer.add_labels(parsed_skeletons_all)
viewer.add_labels(parsed_skeletons_single)
viewer.add_labels(labeled_skeletons)

# %%
adjacency_matrix = nts.create_adjacency_matrix(parsed_skeletons_single, neighborhood='n8')
graph = nts.convert_adjacency_matrix_to_graph(adjacency_matrix)

# %%
labeled_branches = nts.label_branches(parsed_skeletons_single, labeled_skeletons, neighborhood='n8')
viewer.add_labels(labeled_branches)

# %%
spine = nts.create_spine_image(adjacency_matrix=adjacency_matrix,
                               labeled_branches=labeled_branches)
viewer.add_labels(spine)

# %% [markdown]
# ## Measurements

# %%
features = nts.analyze_skeletons(
    labeled_skeletons=labeled_skeletons,
    parsed_skeletons=parsed_skeletons_all)
features

# %%
parsed_skeletons_single = parsed_skeletons_all * (labeled_skeletons == 7)
nts.analyze_single_skeleton(
            parsed_skeletons_single, neighborhood='n8')

# %%



