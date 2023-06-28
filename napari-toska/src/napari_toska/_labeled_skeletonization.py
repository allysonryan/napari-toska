

def generate_labeled_skeletonization(label_image:'napari.types.LabelsData')->'napari.types.LabelsData':
    
    '''
    Skeletonize a label image and relabels the skeleton to match input labels.
    
    The used skeletonization algorithm is the scikit-image implementation of Lee, 94.
    
    Parameters:
    -----------
    
    label_image: napari.types.LabelsData
        Input napari version of numpy.ndarray containing integer labels of an instance segmentation.
    
    Returns:
    --------
    
    labeled_skeletons: napari.types.LabelsData
        A skeleton image multiplied by the respective label of the object of origin in the input label image.
    '''
    
    from skimage.morphology import skeletonize
    import numpy as np
    
    binary_skeleton = skeletonize(label_image.astype(bool)).astype(int)
    binary_skeleton = binary_skeleton / np.amax(binary_skeleton)
    labeled_skeletons = binary_skeleton * label_image
    
    return labeled_skeletons.astype(int)