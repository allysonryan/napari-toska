#!/usr/bin/env python
# coding: utf-8


from __future__ import division, print_function
import os
import numpy as np
import networkx as nx

from itertools import combinations
from scipy.ndimage import label
from scipy.ndimage.morphology import binary_dilation

###-----------------------------------------------

def extract_identities_all_timepoints(labeled_timelapse, background_idx = 0):
    
    ids_flat = np.unique(labeled_timelapse)
    ids_flat = ids_flat[ids_flat != background_idx]
    
    ids_shaped = np.zeros((labeled_timelapse.shape[0], len(ids_flat)), dtype = np.int64)
    for i in range(labeled_timelapse.shape[0]):
        ids_i = extract_identities_single_timepoint(labeled_timelapse, timepoint = i, 
                                                    background_idx = background_idx)
        
        ids_set = np.copy(ids_flat)
        for j in range(ids_set.shape[0]):
            
            if ids_set[j] in ids_i:
                continue
            else:
                ids_set[j] = background_idx
        ids_shaped[i,...] = ids_set
                
    return ids_shaped

###-----------------------------------------------

def skeleton_extraction_labeled_timelapse(labeled_timelapse, ids, background_idx = 0):
    
    skels = np.zeros_like(labeled_timelapse)
    for i in range(labeled_timelapse.shape[0]):
        i_tp = labeled_timelapse[i,...]
        i_ids = ids[i]
        
        i_skels = np.zeros_like(i_tp)
        for j in i_ids:
            if j == background_idx:
                continue
            else:
                j_skel = np.copy(i_tp)
                j_skel[j_skel != j] = 0
                j_skel = skeletonize(j_skel.astype(bool)).astype(int)
                j_skel[j_skel > 0] = j
                i_skels = i_skels + j_skel
                
        skels[i,...] = i_skels
    return skels

###-----------------------------------------------

def search_n4_neighbors(arr, y, x):
    
    '''arr must be a 2D array of binary image like data
    (y,x) are the scalar indices of a single element in arr'''
    
    n4 = (            A[y+1,x  ],             
          A[y  ,x-1],             A[y  ,x+1], 
                      A[y-1,x  ]            )
    
    return n4

###-----------------------------------------------

def search_n8_neighbors(arr, y, x):
    
    '''arr must be a 2D array of binary image like data
    (y,x) are the scalar indices of a single element in arr'''
    
    n8 = (arr[y+1,x-1], arr[y+1,x  ], arr[y+1,x+1], 
          arr[y  ,x-1]              , arr[y  ,x+1], 
          arr[y-1,x-1], arr[y-1,x  ], arr[y-1,x+1])
    
    return n8

###-----------------------------------------------

def search_n6_neighbors(arr, z, y, x):
    
    '''arr must be a 3D array of binary image like data
    (z,y,x) are the scalar indices of a single element in arr'''
    
    n6 = (                arr[z-1, y, x],
          
                          arr[z, y+1, x], 
          arr[z, y, x-1],                 arr[z, y, x+1], 
                          arr[z, y-1, x],
          
                          arr[z+1, y, x])
    
    return n6

###-----------------------------------------------

def search_n18_neighbors(arr, z, y, x):
    
    '''arr must be a 3D array of binary image like data
    (z,y,x) are the scalar indices of a single element in arr'''
    
    n18 = (                    arr[z-1, y+1, x  ],
           arr[z-1, y  , x-1], arr[z-1, y  , x  ], arr[z-1, y  , x+1],
                               arr[z-1, y-1, x  ],
           
           arr[z  , y+1, x-1], arr[z  , y+1, x  ], arr[z  , y+1, x+1], 
           arr[z  , y  , x-1],                     arr[z  , y  , x+1], 
           arr[z  , y-1, x-1], arr[z  , y-1, x  ], arr[z  , y-1, x+1],
           
                               arr[z+1, y+1, x  ], 
           arr[z+1, y  , x-1], arr[z+1, y  , x  ], arr[z+1,  y  , x+1], 
                               arr[z+1, y-1, x  ])
    
    return n18

###-----------------------------------------------

def search_n26_neighbors(arr, z, y, x):
    
    '''arr must be a 3D array of binary image like data
    (z,y,x) are the scalar indices of a single element in arr'''
    
    n26 = (arr[z-1,y+1,x-1], arr[z-1,y+1,x  ], arr[z-1,y+1,x+1], 
           arr[z-1,y  ,x-1], arr[z-1,y  ,x  ], arr[z-1,y  ,x+1], 
           arr[z-1,y-1,x-1], arr[z-1,y-1,x  ], arr[z-1,y-1,x+1], 
                     
           arr[z  ,y+1,x-1], arr[z  ,y+1,x  ], arr[z  ,y+1,x+1], 
           arr[z  ,y  ,x-1]                  , arr[z  ,y  ,x+1], 
           arr[z  ,y-1,x-1], arr[z  ,y-1,x  ], arr[z  ,y-1,x+1], 
                     
           arr[z+1,y+1,x-1], arr[z+1,y+1,x  ], arr[z+1,y+1,x+1], 
           arr[z+1,y  ,x-1], arr[z+1,y  ,x  ], arr[z+1,y  ,x+1], 
           arr[z+1,y-1,x-1], arr[z+1,y-1,x  ], arr[z+1,y-1,x+1])
    
    return n26

###-----------------------------------------------

def n4_pt_classification(skel, coords, y_dir, x_dir):
    
    '''2d partial connectivity element wise pt classification
    coords should be a 2d array with shape (n_px, 2) of positions of pixels in skel'''
    
    end_pts = [0]
    brnch_pts = [0]
    
    for i in range(coords.shape[0]):
        
        y, x = (coords[i, y_dir], coords[i, x_dir])
        px = skel[y, x]
        
        if px == 0:
            continue
        else:
            n4 = search_n4_neighbors(skel, y, x)
            sum_n4 = np.sum(n4)
            
            if sum_n4 == 2:
                continue
            elif sum_n4 > 2:
                brnch_pts.append((y,x))
            elif sum_n4 == 1:
                end_pts.append((y,x))
            else:
                continue
    
    end_pts = end_pts.pop(0)
    brnch_pts = brnch_pts.pop(0)
    
    return end_pts, brnch_pts

###-----------------------------------------------

def n8_pt_classification(skel, coords, y_dir, x_dir):
    
    '''2d full connectivity element wise pt classification
    coords should be a 2d array with shape (n_px, 2) of positions of pixels in skel'''
    
    end_pts = []
    brnch_pts = []
    
    for i in range(coords.shape[0]):
        
        y, x = (coords[i, y_dir], coords[i, x_dir])
        px = skel[y, x]
        
        if px == 0:
            continue
        else:
            n8 = search_n8_neighbors(skel, y, x)
            sum_n8 = np.sum(n8)
            
            if sum_n8 == 2:
                continue
            elif sum_n8 > 2:
                brnch_pts.append((y,x))
            elif sum_n8 == 1:
                end_pts.append((y,x))
            else:
                continue
    
    #end_pts = end_pts.pop(0)
    #brnch_pts = brnch_pts.pop(0)
    
    return end_pts, brnch_pts

###-----------------------------------------------

def n6_pt_classification(skel, coords, z_dir, y_dir, x_dir):
    
    '''3d partial connectivity element wise pt classification
    coords should be a 2d array with shape (n_px, 3) of positions of voxels in skel'''
    
    end_pts = [0]
    brnch_pts = [0]
    
    for i in range(coords.shape[0]):
        
        z, y, x = (coords[i, z_dir], coords[i, y_dir], coords[i, x_dir])
        vx = skel[z, y, x]
        
        if vx == 0:
            continue
        else:
            n6 = search_n6_neighbors(skel, z, y, x)
            sum_n6 = np.sum(n6)
            
            if sum_n6 == 2:
                continue
            elif sum_n6 > 2:
                brnch_pts.append((z,y,x))
            elif sum_n6 == 1:
                end_pts.append((z,y,x))
            else:
                continue
    
    end_pts = end_pts.pop(0)
    brnch_pts = brnch_pts.pop(0)
    
    return end_pts, brnch_pts

###-----------------------------------------------

def n18_pt_classification(skel, coords, z_dir, y_dir, x_dir):
    
    '''3d partial connectivity element wise pt classification
    coords should be a 2d array with shape (n_px, 3) of positions of pixels in skel'''
    
    end_pts = [0]
    brnch_pts = [0]
    
    for i in range(coords.shape[0]):
        
        z, y, x = (coords[i, z_dir], coords[i, y_dir], coords[i, x_dir])
        vx = skel[z, y, x]
        
        if vx == 0:
            continue
        else:
            n18 = search_n18_neighbors(skel, z, y, x)
            sum_n18 = np.sum(n18)
            
            if sum_n18 == 2:
                continue
            elif sum_n18 > 2:
                brnch_pts.append((z,y,x))
            elif sum_n18 == 1:
                end_pts.append((z,y,x))
            else:
                continue
    
    end_pts = end_pts.pop(0)
    brnch_pts = brnch_pts.pop(0)
    
    return end_pts, brnch_pts

###-----------------------------------------------

def n26_pt_classification(skel, coords, z_dir: int = 0, y_dir: int = 1, x_dir: int = 2):
    
    '''3d full connectivity element wise pt classification
    coords should be a 2d array with shape (n_px, 3) of positions of pixels in skel'''
    
    end_pts = []
    brnch_pts = []
    
    for i in range(coords.shape[0]):
        
        z, y, x = (coords[i, z_dir], coords[i, y_dir], coords[i, x_dir])
        vx = skel[z, y, x]
        
        if vx == 0:
            continue
        else:
            n26 = search_n26_neighbors(skel, z, y, x)
            sum_n26 = np.sum(n26)
            
            if sum_n26 == 2:
                continue
            elif sum_n26 > 2:
                brnch_pts.append((z,y,x))
            elif sum_n26 == 1:
                end_pts.append((z,y,x))
            else:
                continue
    
    #end_pts = end_pts.pop(0)
    #brnch_pts = brnch_pts.pop(0)
    
    return end_pts, brnch_pts

###-----------------------------------------------

def n4_parse_skel_2d(skel, y_dir, x_dir):
    
    '''Parameters:
       -----------
       
       skel:
       y_dir:
       x_dir:
       
       Returns:
       --------
       
       coords:
       e_pts:
       b_pts:
       brnch:
       brnch_ids:
       brnch_lengths:'''
    
    coords = np.asarray(np.where(skel)).T
    e_pts, b_pts = n8_pt_classification(skel, coords, y_dir, x_dir)
    
    brnch = np.copy(skel)
    for i in b_pts:
        brnch[i[0],i[1]] = 0
        
    se = np.array([[0,1,0],[1,1,1],[0,1,0]], dtype = int)
    
    brnch, brnch_ids = label(brnch, structure = se)
    brnch_ids = tuple(np.arange(1, brnch_ids+1))
    brnch_lengths = tuple([np.sum(brnch[brnch == i].astype(bool).astype(np.uint16)) for i in brnch_ids])
    
    return coords, e_pts, b_pts, brnch, brnch_ids, brnch_lengths

###-----------------------------------------------


def n8_parse_skel_2d(skel: "napari.types.LabelsData",
                     y_dir: int = 0,
                     x_dir: int = 1) -> "napari.types.LabelsData":
    """
    Label the skeleton of a 2D object with 8-connectivity.

    Parameters
    ----------
    skel : napari.types.LabelsData
        A 2D skeletonized image
    y_dir : int, optional
        The direction of the y-axis, by default 0
    x_dir : int, optional
        The direction of the x-axis, by default 1

    Returns
    -------
    skeleton_labels : napari.types.LabelsData
        A labeled image with the same shape as `skel` where each pixel is
        labeled as either an end point (1), a branch point (2), or a
        skeleton pixel (3).

    """

    coords = np.asarray(np.where(skel)).T
    e_pts, b_pts = n8_pt_classification(skel, coords, y_dir, x_dir)

    brnch = np.copy(skel)
    for i in b_pts:
        brnch[i[0], i[1]] = 0
    
    brnch, brnch_ids = label(
        brnch,
        structure=np.ones((3, 3), dtype=int)
        )
    brnch_ids = tuple(np.arange(1, brnch_ids+1))
    brnch_lengths = tuple([np.sum(brnch[brnch == i].astype(bool).astype(np.uint16)) for i in brnch_ids])

    return coords, e_pts, b_pts, brnch, brnch_ids, brnch_lengths

###-----------------------------------------------

def n6_parse_skel_3d(skel, z_dir, y_dir, x_dir):
    
    '''Parameters:
       -----------
       
       skel:
       z_dir:
       y_dir:
       x_dir:
       
       Returns:
       --------
       
       coords:
       e_pts:
       b_pts:
       brnch:
       brnch_ids:
       brnch_lengths:'''
    
    coords = np.asarray(np.where(skel)).T
    e_pts, b_pts = n6_pt_classification(skel, coords, z_dir, y_dir, x_dir)
    
    brnch = np.copy(skel)
    for i in b_pts:
        brnch[i[0],i[1],i[2]] = 0
    
    se = np.array([[[0,0,0],
                    [0,1,0],
                    [0,0,0]], 
                   
                   [[0,1,0],
                    [1,1,1],
                    [0,1,0]],
                   
                   [[0,0,0],
                    [0,1,0],
                    [0,0,0]]], dtype = int)
    
    brnch, brnch_ids = label(brnch, structure = se)
    brnch_ids = tuple(np.arange(1, brnch_ids+1))
    brnch_lengths = tuple([np.sum(brnch[brnch == i].astype(bool).astype(np.uint16)) for i in brnch_ids])
    
    return coords, e_pts, b_pts, brnch, brnch_ids, brnch_lengths

###-----------------------------------------------

def n18_parse_skel_3d(skel, z_dir, y_dir, x_dir):
    
    '''Parameters:
       -----------
       
       skel:
       z_dir:
       y_dir:
       x_dir:
       
       Returns:
       --------
       
       coords:
       e_pts:
       b_pts:
       brnch:
       brnch_ids:
       brnch_lengths:'''
    
    coords = np.asarray(np.where(skel)).T
    e_pts, b_pts = n18_pt_classification(skel, coords, z_dir, y_dir, x_dir)
    
    brnch = np.copy(skel)
    for i in b_pts:
        brnch[i[0],i[1],i[2]] = 0
    
    se = np.array([[[0,1,0],
                    [1,1,1],
                    [0,1,0]], 
                   
                   [[1,1,1],
                    [1,1,1],
                    [1,1,1]],
                   
                   [[0,1,0],
                    [1,1,1],
                    [0,1,0]]], dtype = int)
    
    brnch, brnch_ids = label(brnch, structure = se)
    brnch_ids = tuple(np.arange(1, brnch_ids+1))
    brnch_lengths = tuple([np.sum(brnch[brnch == i].astype(bool).astype(np.uint16)) for i in brnch_ids])
    
    return coords, e_pts, b_pts, brnch, brnch_ids, brnch_lengths

###-----------------------------------------------

def n26_parse_skel_3d(skel, z_dir: int = 0, y_dir: int = 1, x_dir: int = 2):
    
    '''Parameters:
       -----------
       
       skel:
       z_dir:
       y_dir:
       x_dir:
       
       Returns:
       --------
       
       coords:
       e_pts:
       b_pts:
       brnch:
       brnch_ids:
       brnch_lengths:'''
    
    coords = np.asarray(np.where(skel)).T
    e_pts, b_pts = n26_pt_classification(skel, coords, z_dir, y_dir, x_dir)
    
    brnch = np.copy(skel)
    for i in b_pts:
        brnch[i[0],i[1],i[2]] = 0
        
    brnch, brnch_ids = label(brnch, structure = np.ones((3,3,3), dtype = int))
    brnch_ids = tuple(np.arange(1, brnch_ids+1))
    brnch_lengths = tuple([np.sum(brnch[brnch == i].astype(bool).astype(np.uint16)) for i in brnch_ids])
    
    return coords, e_pts, b_pts, brnch, brnch_ids, brnch_lengths

###-----------------------------------------------

def n4_relabel_brnch_pts(branch_pts, branches_shape, branches_dtype):
    
    '''input:
       - branch_pts: list or tuple of tuples where each tuple is the coordinates of a branch point 
                     in the topological skeleton of an object
       - branches_shape: tuple containing shape of image (array) containing labeled topological skeleton branches
       - branches_dtype: image type (e.g. int32) of image (array) containing labeled topological skeleton branches
       
       output:
       - bp_img: labeled image (array) with shape branches_shape and branches_dtype containing skeleton branch points
       - n_bp: scalar reflecting how the maximum id assigned to labeled object in bp_img (should be range(1,n_bp+1))'''
    
    branch_pts = np.asarray(branch_pts)
    bp_img = np.zeros(branches_shape, dtype=branches_dtype)
    
    for pt in branch_pts:
        bp_img[pt[0],pt[1],pt[2]] = 1
        
    struct = np.array([[0,1,0],[1,1,1],[0,1,0]], dtype = int)
    bp_img, n_bp = label(bp_img, structure=struct, output=np.uint16)
    
    return bp_img, n_bp

###-----------------------------------------------

def n8_relabel_brnch_pts(branch_pts, branches_shape, branches_dtype):
    
    '''input:
       - branch_pts: list or tuple of tuples where each tuple is the coordinates of a branch point 
                     in the topological skeleton of an object
       - branches_shape: tuple containing shape of image (array) containing labeled topological skeleton branches
       - branches_dtype: image type (e.g. int32) of image (array) containing labeled topological skeleton branches
       
       output:
       - bp_img: labeled image (array) with shape branches_shape and branches_dtype containing skeleton branch points
       - n_bp: scalar reflecting how the maximum id assigned to labeled object in bp_img (should be range(1,n_bp+1))'''
    
    branch_pts = np.asarray(branch_pts)
    bp_img = np.zeros(branches_shape, dtype=branches_dtype)
    
    for pt in branch_pts:
        bp_img[pt[0],pt[1]] = 1
        
    struct = np.ones((3,3), dtype = int)
    bp_img, n_bp = label(bp_img, structure=struct, output=np.uint16)
    
    return bp_img, n_bp

###-----------------------------------------------

def n6_relabel_brnch_pts(branch_pts, branches_shape, branches_dtype):
    
    '''input:
       - branch_pts: list or tuple of tuples where each tuple is the coordinates of a branch point 
                     in the topological skeleton of an object
       - branches_shape: tuple containing shape of image (array) containing labeled topological skeleton branches
       - branches_dtype: image type (e.g. int32) of image (array) containing labeled topological skeleton branches
       
       output:
       - bp_img: labeled image (array) with shape branches_shape and branches_dtype containing skeleton branch points
       - n_bp: scalar reflecting how the maximum id assigned to labeled object in bp_img (should be range(1,n_bp+1))'''
    
    branch_pts = np.asarray(branch_pts)
    bp_img = np.zeros(branches_shape, dtype=branches_dtype)
    
    for pt in branch_pts:
        bp_img[pt[0],pt[1],pt[2]] = 1
        
    struct = np.array([[[0,0,0],
                        [0,1,0],
                        [0,0,0]], 
                   
                       [[0,1,0],
                        [1,1,1],
                        [0,1,0]],
                   
                       [[0,0,0],
                        [0,1,0],
                        [0,0,0]]], dtype = int)
    bp_img, n_bp = label(bp_img, structure=struct, output=np.uint16)
    
    return bp_img, n_bp

###-----------------------------------------------

def n18_relabel_brnch_pts(branch_pts, branches_shape, branches_dtype):
    
    '''input:
       - branch_pts: list or tuple of tuples where each tuple is the coordinates of a branch point 
                     in the topological skeleton of an object
       - branches_shape: tuple containing shape of image (array) containing labeled topological skeleton branches
       - branches_dtype: image type (e.g. int32) of image (array) containing labeled topological skeleton branches
       
       output:
       - bp_img: labeled image (array) with shape branches_shape and branches_dtype containing skeleton branch points
       - n_bp: scalar reflecting how the maximum id assigned to labeled object in bp_img (should be range(1,n_bp+1))'''
    
    branch_pts = np.asarray(branch_pts)
    bp_img = np.zeros(branches_shape, dtype=branches_dtype)
    
    for pt in branch_pts:
        bp_img[pt[0],pt[1],pt[2]] = 1
        
    struct = np.array([[[0,1,0],
                        [1,1,1],
                        [0,1,0]], 
                   
                       [[1,1,1],
                        [1,1,1],
                        [1,1,1]],
                   
                       [[0,1,0],
                        [1,1,1],
                        [0,1,0]]], dtype = int)
    bp_img, n_bp = label(bp_img, structure=struct, output=np.uint16)
    
    return bp_img, n_bp

###-----------------------------------------------

def n26_relabel_brnch_pts(branch_pts, branches_shape, branches_dtype):
    
    '''input:
       - branch_pts: list or tuple of tuples where each tuple is the coordinates of a branch point 
                     in the topological skeleton of an object
       - branches_shape: tuple containing shape of image (array) containing labeled topological skeleton branches
       - branches_dtype: image type (e.g. int32) of image (array) containing labeled topological skeleton branches
       
       output:
       - bp_img: labeled image (array) with shape branches_shape and branches_dtype containing skeleton branch points
       - n_bp: scalar reflecting how the maximum id assigned to labeled object in bp_img (should be range(1,n_bp+1))'''
    
    branch_pts = np.asarray(branch_pts)
    bp_img = np.zeros(branches_shape, dtype=branches_dtype)
    
    for pt in branch_pts:
        bp_img[pt[0],pt[1],pt[2]] = 1
        
    struct = np.ones((3,3,3), dtype = int)
    bp_img, n_bp = label(bp_img, structure=struct, output=np.uint16)
    
    return bp_img, n_bp

###-----------------------------------------------

def _generate_adjacency_matrix(e_pts, bp_img, branches, structure):
    """
    Generate an adjacency matrix for a skeletonized object.

    Parameters
    ----------
    e_pts : list or tuple of tuples
        Each tuple is the coordinates of an end point in the topological
        skeleton of an object.
    bp_img : numpy.ndarray
        A labeled image with the same shape as `branches` where each pixel is
        labeled as belonging to a specific branching point or brnaching cluster.
    branches : numpy.ndarray
        A labeled image with the same shape as `bp_img` where each pixel is
        labeled as belonging to a specific branch.
    structure : numpy.ndarray
        A topological-neighborhood-specific structuring element used to dilate the
        branching points in `bp_img` to identify which branches are connected to
        each branching point.

    Returns
    -------
    adj_mat : numpy.ndarray
        An adjacency matrix where each row represents a branching- or endpoint and
        each column represents a branch. A value of 1 indicates that the
        branching point or endpoint is connected to the branch.
    """
    # find number of branch points and branches
    n_bp = np.max(bp_img)
    m_branches = np.max(branches)

    # find the coordinates of each endpoint and number of endpoints
    # also give every end point a unique label
    e_pts = np.asarray(e_pts)
    n_ep = e_pts.shape[0]
    for i in range(n_ep):
        bp_img[tuple(e_pts[i])] = n_bp + 1 + i
        
    adjancency_matrix = np.zeros((n_bp + n_ep, m_branches), dtype = int)
    for i in range(n_bp + n_ep):
        
        mask = bp_img == i+1
        mask = binary_dilation(mask,structure=structure, iterations=1).astype(bool)
        coords = np.asarray(np.where(mask)).transpose()
        touches = np.zeros((coords.shape[0]), dtype = int)
        
        for j in range(coords.shape[0]):
            if len(branches.shape) == 2:
                y,x = coords[j]
                touches[j] = branches[y, x]
            elif len(branches.shape) == 3:
                z, y, x = coords[j]
                touches[j] = branches[z, y, x]
            
        
        touches = np.unique(touches)
        touches = touches[touches > 0]
        
        for k in touches:
            adjancency_matrix[i,k-1] = 1

    return adjancency_matrix        


def n4_adjacency_matrix(e_pts, bp_img, n_bp, branches, m_branches):
    
    '''...'''

    e_pts = np.argwhere(input_image == 1)
    
    adj_bps = np.zeros((n_bp, m_branches), dtype = int)
    
    struct = np.array([[0,1,0],[1,1,1],[0,1,0]], dtype = int)
    
    for i in range(n_bp):
        
        mask = bp_img == i+1
        mask = binary_dilation(mask,structure=struct, iterations=1).astype(bool)
        coords = np.asarray(np.where(mask)).transpose()
        touches = np.zeros((coords.shape[0]), dtype = int)
        
        for j in range(coords.shape[0]):
            y,x = coords[j]
            touches[j] = branches[y,x]
        
        touches = np.unique(touches)
        touches = touches[touches > 0]
        #print('bp', i, '-->', touches)
        
        for k in touches:
            adj_bps[i,k-1] = 1
        #print(np.sum(adj_bps[i,:]))
        
    e_pts = np.asarray(e_pts)
    n_ep = e_pts.shape[0]
        
    adj_eps = np.zeros((n_ep, m_branches), dtype = int)
    for i in range(n_ep):
        y,x = e_pts[i,...]
        branch_id = branches[y,x]
        adj_eps[i, branch_id-1] = 1
        
    adj_mat = np.concatenate((adj_bps, adj_eps), axis = 0)
    
    return adj_mat

###-----------------------------------------------

def n8_adjacency_matrix(e_pts, bp_img, n_bp, branches, m_branches):
    
    adj_bps = np.zeros((n_bp, m_branches), dtype = int)
    
    struct = np.ones((3,3), dtype = int)
    
    for i in range(n_bp):
        
        mask = bp_img == i+1
        mask = binary_dilation(mask,structure=struct, iterations=1).astype(bool)
        coords = np.asarray(np.where(mask)).transpose()
        touches = np.zeros((coords.shape[0]), dtype = int)
        
        for j in range(coords.shape[0]):
            y,x = coords[j]
            touches[j] = branches[y,x]
        
        touches = np.unique(touches)
        touches = touches[touches > 0]
        #print('bp', i, '-->', touches)
        
        for k in touches:
            adj_bps[i,k-1] = 1
        #print(np.sum(adj_bps[i,:]))
        
    e_pts = np.asarray(e_pts)
    n_ep = e_pts.shape[0]
        
    adj_eps = np.zeros((n_ep, m_branches), dtype = int)
    for i in range(n_ep):
        y,x = e_pts[i,...]
        branch_id = branches[y,x]
        adj_eps[i, branch_id-1] = 1
        
    adj_mat = np.concatenate((adj_bps, adj_eps), axis = 0)
    
    return adj_mat

###-----------------------------------------------

def n6_adjacency_matrix(e_pts, bp_img, n_bp, branches, m_branches):
    
    adj_bps = np.zeros((n_bp, m_branches), dtype = int)
    
    struct = np.array([[[0,0,0],
                        [0,1,0],
                        [0,0,0]], 
                   
                       [[0,1,0],
                        [1,1,1],
                        [0,1,0]],
                   
                       [[0,0,0],
                        [0,1,0],
                        [0,0,0]]], dtype = int)
    
    for i in range(n_bp):
        
        mask = bp_img == i+1
        mask = binary_dilation(mask,structure=struct, iterations=1).astype(bool)
        coords = np.asarray(np.where(mask)).transpose()
        touches = np.zeros((coords.shape[0]), dtype = int)
        
        for j in range(coords.shape[0]):
            z,y,x = coords[j]
            touches[j] = branches[z,y,x]
        
        touches = np.unique(touches)
        touches = touches[touches > 0]
        #print('bp', i, '-->', touches)
        
        for k in touches:
            adj_bps[i,k-1] = 1
        #print(np.sum(adj_bps[i,:]))
        
    e_pts = np.asarray(e_pts)
    n_ep = e_pts.shape[0]
        
    adj_eps = np.zeros((n_ep, m_branches), dtype = int)
    for i in range(n_ep):
        z,y,x = e_pts[i,...]
        branch_id = branches[z,y,x]
        adj_eps[i, branch_id-1] = 1
        
    adj_mat = np.concatenate((adj_bps, adj_eps), axis = 0)
    
    return adj_mat

###-----------------------------------------------

def n18_adjacency_matrix(e_pts, bp_img, n_bp, branches, m_branches):
    
    adj_bps = np.zeros((n_bp, m_branches), dtype = int)
    
    struct = np.array([[[0,1,0],
                        [1,1,1],
                        [0,1,0]], 
                   
                       [[1,1,1],
                        [1,1,1],
                        [1,1,1]],
                   
                       [[0,1,0],
                        [1,1,1],
                        [0,1,0]]], dtype = int)
    
    for i in range(n_bp):
        
        mask = bp_img == i+1
        mask = binary_dilation(mask,structure=struct, iterations=1).astype(bool)
        coords = np.asarray(np.where(mask)).transpose()
        touches = np.zeros((coords.shape[0]), dtype = int)
        
        for j in range(coords.shape[0]):
            z,y,x = coords[j]
            touches[j] = branches[z,y,x]
        
        touches = np.unique(touches)
        touches = touches[touches > 0]
        #print('bp', i, '-->', touches)
        
        for k in touches:
            adj_bps[i,k-1] = 1
        #print(np.sum(adj_bps[i,:]))
        
    e_pts = np.asarray(e_pts)
    n_ep = e_pts.shape[0]
        
    adj_eps = np.zeros((n_ep, m_branches), dtype = int)
    for i in range(n_ep):
        z,y,x = e_pts[i,...]
        branch_id = branches[z,y,x]
        adj_eps[i, branch_id-1] = 1
        
    adj_mat = np.concatenate((adj_bps, adj_eps), axis = 0)
    
    return adj_mat

###-----------------------------------------------

def n26_adjacency_matrix(e_pts, bp_img, n_bp, branches, m_branches):
    
    adj_bps = np.zeros((n_bp, m_branches), dtype = int)
    
    struct = np.ones((3,3,3), dtype = int)
    
    for i in range(n_bp):
        
        mask = bp_img == i+1
        mask = binary_dilation(mask,structure=struct, iterations=1).astype(bool)
        coords = np.asarray(np.where(mask)).transpose()
        touches = np.zeros((coords.shape[0]), dtype = int)
        
        for j in range(coords.shape[0]):
            z,y,x = coords[j]
            touches[j] = branches[z,y,x]
        
        touches = np.unique(touches)
        touches = touches[touches > 0]
        #print('bp', i, '-->', touches)
        
        for k in touches:
            adj_bps[i,k-1] = 1
        #print(np.sum(adj_bps[i,:]))
        
    e_pts = np.asarray(e_pts)
    n_ep = e_pts.shape[0]
        
    adj_eps = np.zeros((n_ep, m_branches), dtype = int)
    for i in range(n_ep):
        z,y,x = e_pts[i,...]
        branch_id = branches[z,y,x]
        adj_eps[i, branch_id-1] = 1
        
    adj_mat = np.concatenate((adj_bps, adj_eps), axis = 0)
    
    return adj_mat

###-----------------------------------------------

def skeleton_spine_search(nodes: np.ndarray, G: nx.Graph):
    """
    Search for the longest path in a skeleton graph that connects two
    end points.

    Parameters
    ----------
    nodes : numpy.ndarray
        A 2D adjacency matrix where each row represents a node and each column
        represents an edge. A value of 1 indicates that the node is connected
        to the edge.
    G : networkx.Graph
        A graph representation of the skeleton.

    Returns
    -------
    spine_path : tuple
        A tuple of node IDs that represent the longest path in the skeleton
        graph.
    spine_length : float
        The length of the longest path in the skeleton graph between two nodes
        of degree 1 (aka end points)
    """
    if len(list(G.edges())) == 0:
        return None, None
    
    node_degrees = np.sum(nodes, axis = 1)
    ep_pairs = tuple(combinations(tuple(np.where(node_degrees == 1)[0]), 2))
    
    ep_pair_paths = []
    path_weights = []
    
    for i in ep_pairs:
        ep_pair_paths.append(list(nx.all_simple_paths(G, source = i[0], target=i[1])))
        i_paths = list(nx.all_simple_paths(G, source = i[0], target=i[1]))
        
        i_weights = []
        for j in i_paths:
            i_weights.append(nx.path_weight(G, j, weight="weight"))
        path_weights.append(i_weights)
        
    path_weights = [max(x) for x in path_weights]
    path_loc = np.argmax(path_weights)
    spine_path = tuple(ep_pair_paths[path_loc])[0]
    spine_length = path_weights[path_loc]
    
    return spine_path, spine_length

###-----------------------------------------------

def find_spine_edges(spine_path: list):
    """
    Create a list of edges from a list of nodes.

    Parameters
    ----------
    spine_path : list
        A list of node IDs that represent a path in a skeleton graph.

    Returns
    -------
    spine_path_edges : list
        A list of edges that represent a path in a skeleton graph.
    """
    spine_path_edges = []
    
    for i in range(len(spine_path)-1):
        spine_path_edges.append((spine_path[i], spine_path[i+1]))
    
    return(tuple(spine_path_edges))

###-----------------------------------------------

def map_spine_edges_to_skeleton(
        spine_edges: list,
        adjancency_matrix: np.ndarray,
        branch_labels: "napari.types.LabelsData"):
    """
    Map the edges of a spine path to the branches in the skeleton image.

    Parameters
    ----------
    spine_edges : list
        A list of edges that represent a path in a skeleton graph.
    adjancency_matrix : numpy.ndarray
        A 2D adjacency matrix where each row represents a node and each column
        represents an edge. A value of 1 indicates that the node is connected
        to the edge.
    branch_labels : napari.types.LabelsData
        A labeled image with labelled skeleton branches.

    Returns
    -------
    spine_branch_labels : list
        A list of branch labels that represent a path in a skeleton graph.        
    """
    
    spine_branch_labels = []
    
    for i in spine_edges:
        pts_sum = adjancency_matrix[i[0],:] + adjancency_matrix[i[1],:]
        i_id = np.where(pts_sum == 2)[0][0]
        spine_branch_labels.append(branch_labels[i_id])
        
    return spine_branch_labels

