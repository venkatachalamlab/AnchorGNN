
import time
import tensorflow as tf
import numpy as np
import h5py
from scipy import ndimage
import matplotlib.pyplot as plt 
from pathlib import Path



def get_abs_pos(combined_df_t_idx,labels_shape):
    '''
    Obtain the absolutely coordinates annotations at the time t_idx with the segmented size of labels_shape from the normalized coordinates
    !!! Important notice: the absolute coordinates starting from 0, then it should be multiplied by (labels_shape-1)
    '''
    pos = np.array(combined_df_t_idx[['z','y','x']])
    abs_pos = (np.round(pos * (np.array(labels_shape)-1))).astype(int)
    return abs_pos


def get_volume_at_frame(file_name,t_idx):
    '''
    Get the 3D original volume at frame t_idx from the h5 file_name
    '''
    with h5py.File(file_name, 'r') as f:
        img_original = np.array(f['data'][t_idx:t_idx+1])
        mask = None
    f.close()
    return img_original,mask


def load_annotations_h5_t_idx(file_name,t_idx):
    '''
    Load annotation at all time 
    '''
    with h5py.File(file_name, 'r') as hdf: 
        dfs = []
        for key in hdf.keys():
            dataset = hdf[key]
            df = pd.DataFrame(dataset[:],columns=[key])
            dfs.append(df)

        combined_df = pd.concat(dfs, axis = 1)
    hdf.close()
    combined_df_t_idx = combined_df[combined_df['t_idx'] == t_idx].sort_values(by='worldline_id')
    return combined_df_t_idx


"""
Segmentation of nuclei in 3D volume using StarDist model
======================

This module is to segment the nuclei in 3D volume using StarDist model. 
The model is trained on 2D images, so the nuclei are segmented in each 2D slice.
The overlapping boundary in the nearby nuclei are removed.
Then using the annotation as markers in the watershed process. 
The analysis of the segmented nuclei properties is also included in this module.

.. tags:: segmentation, property analysis, StarDist
"""


import numpy as np  
from glob import glob
# from csbdeep.utils import Path, normalize
from csbdeep.io import save_tiff_imagej_compatible
import stardist
from stardist import random_label_cmap, _draw_polygons, export_imagej_rois
from stardist.models import StarDist2D
from stardist.plot import render_label

from scipy import ndimage
import skimage
import copy

from skimage.measure import regionprops_table
import pandas as pd


def normalize(x, pmin=3, pmax=99.8, axis=None, clip=False, eps=1e-20, dtype=np.float32):
    """Percentile-based image normalization."""

    mi = np.percentile(x,pmin,axis=axis,keepdims=True)
    ma = np.percentile(x,pmax,axis=axis,keepdims=True)
    if mi==0 and ma==0:
        return x
    else:
        return normalize_mi_ma(x, mi, ma, clip=clip, eps=eps, dtype=dtype)


def normalize_mi_ma(x, mi, ma, clip=False, eps=1e-20, dtype=np.float32):
    if dtype is not None:
        x   = x.astype(dtype,copy=False)
        mi  = dtype(mi) if np.isscalar(mi) else mi.astype(dtype,copy=False)
        ma  = dtype(ma) if np.isscalar(ma) else ma.astype(dtype,copy=False)
        eps = dtype(eps)

    try:
        import numexpr
        x = numexpr.evaluate("(x - mi) / ( ma - mi + eps )")
    except ImportError:
        x =                   (x - mi) / ( ma - mi + eps )

    if clip:
        x = np.clip(x,0,1)

    return x



class NucleiSegmentationAnnotation:
    def __init__(self, image, model, isotropy_scale,  normalize_lim, zoom_factor):
        """
        Segmentation of nuclei in 3D volume using StarDist model for each slice 
        Using the annotation as markers in the watershed algorithm to segment the nuclei in 3D volume

        input: image: the original 3D image array with the shape (depth, width, heigth)
               model: StarDist2D model
               isotropy_scale: tuple, the isotropy scale of the image
               normalize_lim: tuple, the normalization limit for the image
               zoom_factor:  the zoom factor for the image 
        output: labeled_neurons: (z, x, y) array, the labeled neurons
        """

        self.pad_size = [0,10,10] ### pad the image to avoid the boundary effect
        self.image = image
        self.model = model
        # self.abs_pos = abs_pos
        self.isotropy_scale = isotropy_scale
        self.normalize_lim = normalize_lim
        self.zoom_factor = zoom_factor
        self.slices = [
    slice(self.pad_size[0], -self.pad_size[0] if self.pad_size[0] > 0 else None),
    slice(self.pad_size[1], -self.pad_size[1] if self.pad_size[1] > 0 else None),
    slice(self.pad_size[2], -self.pad_size[2] if self.pad_size[2] > 0 else None)
]


    
    
    def normalize_volume(self, data, min_val, max_val):
        # return (data - np.min(data, axis=axis)) / (np.max(data, axis=axis) - np.min(data, axis=axis)) * (max_val - min_val) + min_val
        # return normalize(data, min_val, max_val, axis=(0,1,2))
        
        img_xy = np.zeros(data.shape)
        for i in range(data.shape[0]):
            img_xy[i] = normalize(data[i], min_val, max_val)
        return img_xy
    


    def segment_2D_nuclei(self, img_xy):
        """
        Segment nuclei in 2D image using StarDist model
        input: image: (z, x, y) array, 2D image
                model: StarDist2D model
        output: center: (n, 3) array, n number of nuclei's center coordinates, and the columns are z, x, y
                labels: (n,) array, the label of each nucleus, same shape as image

        """
       
        labels = np.zeros(img_xy.shape, dtype=np.int32)
        # labels = []
        center = []
        for z in range(0, img_xy.shape[0]):

            label_z, _ = self.model.predict_instances(img_xy[z])
            label_z[label_z>0] += np.max(labels) + 1
            labels[z] = label_z


        # self.abs_pos = (self.abs_pos/ (np.array([23,1024,1024]) -1) * (np.array(img_xy.shape) -1)).astype(np.int32)
        # self.abs_pos = self.update_abs_pos(self.abs_pos, labels) ## update the abs_pos based on the detected nuclei

 
        labels = self.labels_xy_annotation(labels, self.abs_pos)
            
        region_props_df = pd.DataFrame(regionprops_table(np.array(labels),  properties=['label', 'centroid']))
        center = np.round( region_props_df[['centroid-0','centroid-1','centroid-2']].values ).astype(int)

        return np.array(labels), np.vstack(center) 
    

    def labels_xy_annotation(self, labels_xy, abs_pos):
        ### if a single nucleus contains multiple annotations, then subdivide the nucleus into multiple nuclei

        values, counts = np.unique(labels_xy[tuple(abs_pos.T)], return_counts=True)
        merge_ID_list = values[counts>1]
        merge_ID_list = merge_ID_list[merge_ID_list>0]
        # for merge_ID  in values[counts>1]: 
        for merge_ID  in merge_ID_list: ## find non zero merge_ID
            index = labels_xy[tuple(abs_pos.T)] == merge_ID
            abs_pos_ind = abs_pos[index]

            bbox = ndimage.find_objects(labels_xy == merge_ID)[0]
            cropped_labels_xy = labels_xy[bbox] == merge_ID
            abs_pos_cropped = abs_pos_ind - np.array([s.start for s in bbox])



            distance = ndimage.distance_transform_edt(cropped_labels_xy)
            mask = np.zeros(distance.shape, dtype=bool)
            mask[tuple(abs_pos_cropped.T)] = True
            markers, _ = ndimage.label(mask)
            labels_cropped = skimage.segmentation.watershed(-distance, markers, mask=cropped_labels_xy)


            ## merge the cropped labels back to the original labels
            # labels_change = copy.deepcopy(labels_xy)
            # labels_xy[bbox][cropped_labels_xy>0] = labels_cropped[labels_cropped>0] + np.max(labels_xy) + 1
            labels_xy[bbox][labels_cropped>0] = labels_cropped[labels_cropped>0] + np.max(labels_xy) + 1 ## changed 10/23/24
            
            
        return labels_xy
    

    def get_merged_label(self,labels_xy_slice):
        properties = ['label', 'coords']

        region_props = regionprops_table(labels_xy_slice, properties=properties)
        region_props_df = pd.DataFrame(region_props)

        offsets = np.array([[-1, -1], [-1, 0], [-1, 1],
                            [0, -1],  [0, 0],   [0, 1],
                            [1, -1],  [1, 0],   [1, 1]])
        region_props_df['nearby_label'] = [None] * len(region_props_df)
        region_props_df['num_nearby_label'] = [None] * len(region_props_df)
        # image_shape = labels_xy_slice.shape
        for i in range(len(region_props_df)):
            coords = region_props_df['coords'][i]  # Shape (N, 2)
            connected_points = coords[:, np.newaxis, :] + offsets  # Shape (N, 9, 2) 
            ## since padded image, so the nearby boundary must be within the bound of the padded image
            connected_points = connected_points.reshape(-1, 2)
            connected_points = connected_points[(connected_points[:,0]>=0) &
                                                (connected_points[:,0]<labels_xy_slice.shape[0]) & 
                                                (connected_points[:,1]>=0) & 
                                                (connected_points[:,1]<labels_xy_slice.shape[1])]

            seg_ID = labels_xy_slice[connected_points[:,0],connected_points[:,1]]
            region_props_df.at[i, 'nearby_label'] = np.unique(seg_ID[seg_ID>0])
            region_props_df.at[i, 'num_nearby_label'] = len(np.unique(seg_ID[seg_ID>0]))


        nearby_label = region_props_df['label'][region_props_df['num_nearby_label']>1].values
        return nearby_label
    

    def find_overlapping_contours(self, labels_xy_slice):
        # unique_labels = np.unique(labels_xy_slice[labels_xy_slice>0])
        unique_labels = self.get_merged_label(labels_xy_slice)
        if len(unique_labels)>0:
            contours = [] 
            for label in unique_labels:
                contour = skimage.measure.find_contours((labels_xy_slice == label)*1, 0)[0]
                contours += np.unique(contour, axis=0).tolist()

            contours = np.array(contours).astype(int).reshape(-1,2)


            # Ensure contours is at least 2D before indexing
            # if len(contours.shape) == 1:
            #     raise ValueError("Contours array is 1-dimensional, expected 2-dimensional.")
            

            ind = (labels_xy_slice[contours[:, 0], contours[:, 1]]>0)
            nearby_points = contours[ind]
            return nearby_points
        else:
            return []

    def remove_overlapping_contours(self, labels_xy):
        labels_xy_isolated = copy.deepcopy(labels_xy)
        for z in range(labels_xy.shape[0]):
            labels_xy_slice = labels_xy[z]
            if not np.all(labels_xy[z] == 0):
                nearby_points = self.find_overlapping_contours(labels_xy_slice)
                if len(nearby_points)>0:
                    labels_xy_isolated[z, nearby_points[:, 0], nearby_points[:, 1]] = 0
        return labels_xy_isolated






    def find_first_not_in_abs_values(self,values, abs_values):
        for val in values:
            if abs(val) not in abs_values:
                return val
        return None

    def update_abs_pos(self, abs_pos, labels_xy):

        ## find the annotation that is not detected in the labels_xy and shift the abs_pos to the detected annotation
        abs_values = labels_xy[tuple(abs_pos.T)]
        undetected_ID_list = np.where(abs_values == 0)[0]
        img_shape = labels_xy.shape
 
        for ID in undetected_ID_list:
            abs_pos_ind = abs_pos[ID].reshape(-1,3)
            offsets = np.array([[dz, dy, dx] for dx in [-2, -1, 0, 1, 2] for dy in [-2, -1, 0, 1, 2] for dz in [-1,0,1]])
            expanded_pos = abs_pos_ind[:, None] + offsets
            expanded_pos[:, :, 0] = np.clip(expanded_pos[:, :, 0], 0, img_shape[0] - 1)
            expanded_pos[:, :, 1] = np.clip(expanded_pos[:, :, 1], 0, img_shape[1] - 1)
            expanded_pos[:, :, 2] = np.clip(expanded_pos[:, :, 2], 0, img_shape[2] - 1)
            expanded_pos_reshaped = expanded_pos.reshape(-1, expanded_pos.shape[-1])
            label_ID = labels_xy[tuple(expanded_pos_reshaped.T)]
            values, counts, indices = np.unique(label_ID[label_ID>0], return_counts=True, return_index=True)  
            values = values[counts.argsort()[::-1]]
            label_ID_ind = self.find_first_not_in_abs_values(values, abs_values)
            if label_ID_ind is not None:
                abs_pos[ID] = abs_pos[ID]+ offsets[label_ID == label_ID_ind][0]
        return abs_pos


    def get_markers_from_annotation(self, img_shape, abs_pos):  

        ## original method
        # annotation_markers = np.zeros(img.shape)    
        # annotation_markers[tuple(abs_pos.T)] = np.arange(1,abs_pos.shape[0]+1)
    
        annotation_markers = np.zeros(img_shape)
        offsets = np.array([[dz, dy, dx] for dx in [-2, -1, 0, 1, 2] for dy in [-2, -1, 0, 1, 2] for dz in [0]])
        expanded_pos = abs_pos[:, None] + offsets
        expanded_pos[:, :, 0] = np.clip(expanded_pos[:, :, 0], 0, annotation_markers.shape[0] - 1)
        expanded_pos[:, :, 1] = np.clip(expanded_pos[:, :, 1], 0, annotation_markers.shape[1] - 1)
        expanded_pos[:, :, 2] = np.clip(expanded_pos[:, :, 2], 0, annotation_markers.shape[2] - 1)
        expanded_pos_reshaped = expanded_pos.reshape(-1, expanded_pos.shape[-1])
        annotation_markers[tuple(expanded_pos_reshaped.T)] = np.repeat(np.arange(1, abs_pos.shape[0] + 1), offsets.shape[0])
        

        return annotation_markers



    def watershed_segmentation_3D(self, labels_xy, labels_xy_isolated):      

        
        self.abs_pos = self.update_abs_pos(self.abs_pos, labels_xy_isolated) 
        annotation_markers = self.get_markers_from_annotation(labels_xy.shape, self.abs_pos)

        binary_volume = labels_xy_isolated > 0
        distance = ndimage.distance_transform_edt(~binary_volume, sampling=self.isotropy_scale)
        distance[annotation_markers>0] = np.max(distance)

        labels = skimage.segmentation.watershed(-distance, 
                                                annotation_markers.astype(np.int32), 
                                                mask=labels_xy_isolated.astype(bool), 
                                                connectivity=3
                                                )
        
        return labels

    def map_labels_to_original(self, labels, labels_xy, center_xy):
        labels_xy_ID_list = labels_xy[tuple(center_xy.T)]
        labels_ID_list = labels[tuple(center_xy.T)]
        mapping = {key: value for key, value in zip(labels_xy_ID_list, labels_ID_list)}

        #### method 1 
        # labeled_neurons = np.vectorize(lambda x: mapping.get(x, 0))(labels_xy)
        
        #### method 2
        mapping_array = np.zeros(max(mapping.keys()) + 1, dtype=int)
        for k, v in mapping.items():
            mapping_array[k] = v
        labeled_neurons = np.take(mapping_array, labels_xy, mode='clip')
        return labeled_neurons

    def process_image(self):
        
        img_zoom = ndimage.zoom(self.image, (1,self.zoom_factor, self.zoom_factor) , order=0)


        
        return img_zoom




    def watershed_segmentation_2D(self, labels_xy):
        labels_xy_isolated = self.remove_overlapping_contours(labels_xy)
        labels = self.watershed_segmentation_3D(labels_xy, labels_xy_isolated)
        return labels



    def narrow_image_space(self,img_xy):
        img_proj = np.max(img_xy, axis=0)

        label_proj, _ = self.model.predict_instances(normalize(img_proj, self.normalize_lim[0], self.normalize_lim[1]))

        coords = np.array(np.where(label_proj>0)).T
        max_coords = np.max(coords,axis=0)
        min_coords = np.min(coords,axis=0)
        off_set = 10
        img_shape = label_proj.shape
        start = [np.max([min_coords[0]-off_set,0]), np.max([min_coords[1] - off_set,0])]   
        end = [np.min([max_coords[0]+off_set,img_shape[0]]), np.min([max_coords[1]+off_set,img_shape[1]])]  

        return start, end


    def run_segmentation_narrowdowm(self,img):
        labels_xy, center_xy = self.segment_2D_nuclei(img)
        labels = self.watershed_segmentation_2D(labels_xy)
        labeled_neurons_narrowdown = self.map_labels_to_original(labels, labels_xy, center_xy)
        return labeled_neurons_narrowdown
    


    def restrict_narradown_area(self, combined_df_t_idx, img_xy_shape, start, end):
        '''
        restrict the narrow down area based on the annotations, the prediction area should contain all the annotations
        '''
        
        abs_pos = get_abs_pos(combined_df_t_idx,np.array(img_xy_shape)) 
        abs_pos_min = np.min(abs_pos, axis=0)
        abs_pos_max = np.max(abs_pos, axis=0)
        start[0] = np.min([start[0], abs_pos_min[1]]) 
        start[1] = np.min([start[1], abs_pos_min[2]]) 
        end[0] = np.max([end[0], abs_pos_max[1]]) 
        end[1] = np.max([end[1], abs_pos_max[2]])
        return start, end  


    def run_segmentation(self,combined_df_t_idx):        
        img_xy = self.process_image() ## zoom 1024*1024

        img_xy_shape = img_xy.shape
        self.start, self.end = self.narrow_image_space(img_xy)
        self.start, self.end = self.restrict_narradown_area(combined_df_t_idx, img_xy_shape, self.start, self.end)
        
        

        self.abs_pos = get_abs_pos(combined_df_t_idx,np.array(img_xy_shape)) - np.array([0,self.start[0],self.start[1]]) + np.array(self.pad_size) 

        img_norm = self.normalize_volume(img_xy[:,self.start[0]:self.end[0],self.start[1]:self.end[1]], 
                                         self.normalize_lim[0], self.normalize_lim[1])
        # img_norm = self.normalize_volume(img_xy, self.normalize_lim[0], self.normalize_lim[1])

        
        img_norm_pad = np.pad(img_norm, ((self.pad_size[0], self.pad_size[0]), 
                                (self.pad_size[1], self.pad_size[1]), 
                                (self.pad_size[2], self.pad_size[2])), 
                                mode='constant', constant_values=0)
    
        labeled_neurons = np.zeros(img_xy_shape, dtype=np.int32)
    

        labeled_neurons[:,self.start[0]:self.end[0],self.start[1]:self.end[1]] = self.run_segmentation_narrowdowm(img_norm_pad)[tuple(self.slices)]
        # labeled_neurons= self.run_segmentation_narrowdowm(img_norm_pad)[tuple(self.slices)]

        self.abs_pos_new = (self.abs_pos - self.pad_size + np.array([0,self.start[0],self.start[1]])) * np.array([1,1/self.zoom_factor,1/self.zoom_factor])

        return ndimage.zoom(labeled_neurons, (1,1/self.zoom_factor, 1/self.zoom_factor) , order=0)
    








































