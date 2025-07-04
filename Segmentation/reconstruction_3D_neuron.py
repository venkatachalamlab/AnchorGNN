###############################################################################################################################################
# This notebook is to do prediction on the original volume at frame t_idx, loaded from the 'data.h5' file in the file_name.
# Load the file 'annotation.h5' which indicate there is a seperate nueron.
# Watershed the original neuron from the iou matrix of each nuclei with the existed previous neuron 
# Based on the annotation, find out the merged neuron with multiple annotations
# Segment the merged neuron by each annotation
###############################################################################################################################################

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import h5py
import scipy
import copy
import pandas as pd
from skimage import measure
import cv2
import os
import glob

import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
import tensorflow as tf
print(tf.test.is_built_with_cuda())
print(tf.__version__)
print(tf.config.list_physical_devices('GPU'))

from stardist.models import StarDist2D
from stardist.plot import render_label
from csbdeep.utils import normalize


################################################################ Visualization ################################################################


def generateCMAP(number):
    ##################### to assemble different color set from ########################
    if number<1:
            raise ValueError('number of colors Sz must be at least 1, but is: ',Sz)
    newcol = np.array([], dtype=np.int64).reshape(0,4)
    # color_set_list = ['gist_ncar','tab20','hsv','cividis','jet','Set2','Dark2','rainbow']
    color_set_list = ['gist_ncar','rainbow']
    times = int(number/len(color_set_list*5)+1)
    for color_set in color_set_list*times:
        if color_set == 'tab20':
            produce = 20
        elif color_set == 'gist_ncar' or color_set == 'hsv':
            produce = 10
        else:
            produce = 5
        cmp=plt.cm.get_cmap(color_set,produce)
        fillColor=np.array([1,1,1,1])
        idx=np.linspace(0, 1, produce)
        newcol = np.concatenate((newcol, cmp(idx)),axis = 0)

        if len(newcol) > number:
            break

    newcmp=ListedColormap(np.array(newcol))
    return newcmp


def apply_color(gray_image,label_img,color_map,transparency=0.5):
    ### this works with any dimension of gray images and apply color based on the labeled image #####
    dim = len(gray_image.shape)
    colored_img = color_map(label_img.astype(int))[...,:3]
    label_mask = np.expand_dims((label_img>0).astype('float'),axis=dim)*transparency    
    gray_image_3ch = np.repeat(np.expand_dims(gray_image,axis=dim),3,axis=dim).astype('float')/np.max(gray_image)
    overlay_img = (1-label_mask)*gray_image_3ch + label_mask*colored_img
    return overlay_img



def visualize_each_plane_seg(image,seg_volume_single,labels,color_map):
    for z in range(image.shape[0]):

        fig,(ax1,ax2) = plt.subplots(1,2,figsize = (15,15))

        # ax1.imshow(image[z], vmax = 0.1*np.max(image[z]))
        labels_z_rgb = apply_color(image[z],seg_volume_single[z],color_map,transparency=1)
        ax1.imshow(labels_z_rgb)
        ax1.set_title('Single Neuron at plane' +str(z))

        labels_z_rgb = apply_color(image[z],labels[z],color_map,transparency=0.5)
        ax2.imshow(labels_z_rgb)
        ax2.set_title('Instance Segmentation at plane ' +str(z))

        plt.show()
        

def visualize_annotation_watershed_seg_ID(image,labels,seg_volume_seg,abs_pos,unique_values,color_map,ID):
    '''
    Visualize how the Neuron 'ID' is watershed in the origianl segmentation and the watershed algorithm with the annotation
    '''
    labels_plot = (labels == ID)*1 
    ind = unique_values == ID
    abs_pos_plot = abs_pos[ind]

    ind = np.where(labels_plot>0)
    ylim = (np.min(ind[1])-10,np.max(ind[1])+10)
    xlim = (np.min(ind[2])-10,np.max(ind[2])+10)

    img_test = (image*labels_plot)

    z_plot = np.unique(np.where(labels_plot>0)[0])

    min_z = np.max([0,np.min(z_plot)-1])
    max_z = np.min([labels.shape[0],np.max(z_plot)+2])
    for z in range(min_z,max_z):
        if seg_volume_seg is not None:
            fig,(ax1,ax2,ax3) = plt.subplots(1,3,figsize=(15,15))
        else:
            fig,(ax1,ax2) = plt.subplots(1,2,figsize=(15,8))
        ax1.imshow(image[z],'gray',vmax= np.max(image[z]))
        ax1.set_title('original image at plane' +str(z))
        ax1.set_xlim(xlim)
        ax1.set_ylim(ylim)


        labels_z_rgb = apply_color(image[z],labels_plot[z],color_map,transparency=0.5)
        pos_z = abs_pos_plot[abs_pos_plot[:,0]==z]
        ax2.imshow(labels_z_rgb)
        if len(pos_z)>0:
            ax2.plot(pos_z[:,2],pos_z[:,1],'r*',markersize = 5)
        ax2.set_title('original segmentation Neuron at plane' +str(z))
        ax2.set_xlim(xlim)
        ax2.set_ylim(ylim)



        if seg_volume_seg is not None:

            labels_z_rgb = apply_color(image[z],seg_volume_seg[z],color_map,transparency=0.5)
            pos_z = abs_pos_plot[abs_pos_plot[:,0]==z]
            ax3.imshow(labels_z_rgb)
            if len(pos_z)>0:
                ax3.plot(pos_z[:,2],pos_z[:,1],'r*',markersize = 5)
            ax3.set_title('watershed Neuron at plane' +str(z))
            ax3.set_xlim(xlim)
            ax3.set_ylim(ylim)
        plt.show()
        plt.tight_layout()
        


def create_pngs(img,lbl,image_directory):
    '''
    To create a list of original and instacne segmentation pngs
    '''
    frames = []
    for z in range(len(img)):

        fig, (ai, al) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw=dict(width_ratios=(1, 0.8)))
        im = ai.imshow(img[z], cmap='gray', clim=(0, 0.5))
        ai.set_title("Fluorescent Image at Slice " +str(z))
        fig.colorbar(im, ax=ai)
        al.imshow(lbl[z], cmap=lbl_cmap)
        al.set_title("Instance Segmentation at Slice " +str(z))
        plt.tight_layout()
        frames.append(fig)
        fig.savefig(image_directory +str(z)+'.png')


def extract_number(filename):
    return int(filename.split('/')[-1].split('.png')[0])

def convert_png_mp4(image_directory,output_video_path,fps):
    '''
    Covvert the png into movie
    '''
    image_files = sorted(glob.glob(os.path.join(image_directory, '*.png')))
    image_files = sorted(image_files, key=extract_number)

  
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
    
    first_image = cv2.imread(image_files[0])
    height, width, layers = first_image.shape
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    for image_file in image_files:
        frame = cv2.imread(image_file)
        out.write(frame)

    out.release()

    print(f'Video saved as {output_video_path}')

    
###############################################################################################################################################################        
## Load and Annotate the h5 file     
## Assign the annotated neurons with the segmented ID       
###############################################################################################################################################################       
def load_annotations_h5(file_name):
    '''
    load all the annotations in the h5 file as pandas dataframe
    '''
    with h5py.File(file_name, 'r') as hdf: 
        dfs = []
        for key in hdf.keys():
            dataset = hdf[key]
            df = pd.DataFrame(dataset[:],columns=[key])
            dfs.append(df)

        combined_df = pd.concat(dfs, axis = 1)
    
        return combined_df




def get_abs_pos(path_annotations_h5,t_idx,labels_shape):
    '''
    Obtain the absolutely coordinates annotations at the time t_idx with the segmented size of labels_shape from the normalized coordinates
    !!! Important notice: the absolute coordinates starting from 0, then it should be multiplied by (labels_shape-1)
    '''

    combined_df = load_annotations_h5(path_annotations_h5)
    pos = np.array(combined_df[combined_df['t_idx'] == t_idx][['z','y','x']])
    abs_pos = (np.round(pos * (np.array(labels_shape)-1))).astype(int)
    return abs_pos



def get_unique_values(labels,abs_pos):
    '''
    Find the annotation's neuron ID in the segmented labels in a neaby 4*4 2d array since the segmentation is zoomed by the factor of 2
    '''
    unique_values = np.zeros(len(abs_pos))
    for i in range(len(abs_pos)):
        ID = labels[ abs_pos[i,0], abs_pos[i,1]-2:abs_pos[i,1]+2, abs_pos[i,2]-2:abs_pos[i,2]+2 ]
        values, counts = np.unique(ID[ID>0], return_counts = True)
        sorted_values = values[np.argsort(counts)[::-1]]
        ### sort the values??
        if len(sorted_values)>0:
            unique_values[i] = sorted_values[0]
    return unique_values


def get_most_frequent_value(arr):
    
    x = (arr).astype('int64').flatten()
    return np.bincount(x).argmax()


def assign_annotations_empty_label(unique_values,labels,abs_pos):
    '''
    If the annotation with no labels in the current plane,  assign the unique_values with this new ID by searching the nearby planes
    For annotation with no segmentation in the nearby planes for low intensity or wrong annotation, assign all zeros to this annaotation
    Modify the annotation if found in the nearby plane
    '''
    empty_pos_ind = np.where(unique_values==0)[0]
    for i in empty_pos_ind:

        if abs_pos[i,0]==0:
            z_new = 1
            ID = [labels[ z_new, abs_pos[i,1]-2:abs_pos[i,1]+2, abs_pos[i,2]-2:abs_pos[i,2]+2 ]]

            
        
        elif abs_pos[i,0]== labels.shape[0]:
            z_new = labels.shape[0]-1
            ID = [labels[ z_new, abs_pos[i,1]-2:abs_pos[i,1]+2, abs_pos[i,2]-2:abs_pos[i,2]+2 ]]

        elif abs_pos[i,0]< (labels.shape[0]-1):
            ID = [labels[ abs_pos[i,0]+1, abs_pos[i,1]-2:abs_pos[i,1]+2, abs_pos[i,2]-2:abs_pos[i,2]+2 ],
            labels[ abs_pos[i,0]-1, abs_pos[i,1]-2:abs_pos[i,1]+2, abs_pos[i,2]-2:abs_pos[i,2]+2 ]]

            
            

        # ID_list = np.unique(ID[ID>0]) #### there could be annotation with empty segmentation
        ID_list = np.array([get_most_frequent_value(arr) for arr in ID]) #### there could be annotation with empty segmentation


        if len(ID_list[ID_list>0])==1: 
            unique_values[i] = int((ID_list[ID_list>0]))
            
        # if len(ID_list)==1 or 
            # unique_values[i] = int(ID_list[0])
            # print(i,ID_list,unique_values[i])
        # elif len(ID_list)==0 or np.sum(ID_list)==0:
        elif np.sum(ID_list)==0:
            '''
            if there is no original segmentation in the nearby plane, this annaotation is wrong or in a relative low intensity
            '''
            print("warning:there is no original segmentation in the nearby plane at annotation: ", abs_pos[i])
            unique_values[i] = 0
            abs_pos[i] = 0
           
            # print(i,ID_list)
        elif len(ID_list[ID_list>0])==2:
            print(ID_list)
            ########
            '''
            Oct 8: modification
            '''
            z_max = np.min([abs_pos[i,0]+1, labels.shape[0]-1])
            # if np.sum(labels[z_max] == ID_list[0])==0:
            #     z_max = abs_pos[i,0]
                
            coords = np.array([np.array(measure.regionprops((labels[z_max] == ID_list[0])*1)[0].centroid),
                                np.array(measure.regionprops((labels[abs_pos[i,0]-1] == ID_list[1])*1)[0].centroid)])
            '''
            Oct 8: modification
            '''
            # coords = np.array([np.array(measure.regionprops((labels[abs_pos[i,0]-1] == ID_list[ind])*1)[0].centroid) for ind in range(len(ID_list))])
            ind = np.argmin(np.sum(coords - np.array([abs_pos[i,1],abs_pos[i,2]]),axis = 1))
            unique_values[i] = ID_list[ind]
            # print("final ID",ID_list[ind])
            
        if unique_values[i]>0 and unique_values[i]==ID_list[0]:
            abs_pos[i,0] = abs_pos[i,0] + 1
            
        elif unique_values[i]>0 and unique_values[i]==ID_list[1]:
            abs_pos[i,0] = abs_pos[i,0] - 1
            
            
            
    return unique_values,abs_pos


#########################################################################################################################################################
# Startdist_2D prediction
# initial instance segmentation 
#########################################################################################################################################################   
def get_volume_at_frame(file_name,t_idx):
    '''
    Get the 3D original volume at frame t_idx from the h5 file_name
    '''
    with h5py.File(file_name, 'r') as f:
        img_original = np.array(f['data'][t_idx:t_idx+1])
        mask = None
    f.close()
    return img_original,mask 
    
    
def instance_seg_slice(img_zoom,model):
    '''
    Use the stardist 2d model to segment each slice image in the 3d original volume zoomed by the factor of [1,zoom_factor,zoom_factor]
    '''
    
    ##### the following mode works for low resolution #####
    label_stars_zoom = np.empty(img_zoom.shape)
    for z in range(img_zoom.shape[0]):
        img_zoom_norm = normalize(img_zoom[z],3,99.8)
        if np.max(img_zoom_norm)>100:
            img_zoom_norm = normalize(img_zoom[z],3,100)
       
        label_stars_zoom[z],_ = model.predict_instances(img_zoom_norm)

#     #### the following mode works for high resolution #####
#     label_stars_zoom = np.empty(img_zoom.shape)
#     img_zoom_norm = normalize(img_zoom,pmin=1,pmax=99.8)
#     for z in range(img_zoom.shape[0]):       
#         label_stars_zoom[z],_ = model.predict_instances(img_zoom_norm[z])
    
    
    return label_stars_zoom


def update_seg_volume(img_zoom,seg_volume,abs_pos,unique_values,model):
    '''
    For the slice that doesn't have segmentation at certain annotation, change the threshold to do a second segmentation
    '''
    pos_unassigned = abs_pos[unique_values==0]
    for i in range(len(pos_unassigned )):
        [z,x,y] = pos_unassigned[i]
        img_zoom_norm = normalize(img_zoom[z],1,99.9)
        label_stars_zoom_test,_ = model.predict_instances(img_zoom_norm)
        a = label_stars_zoom_test[x,y]
        v = np.unique(a[a>0])
        if len(v)>0:
            '''
            if the new prediction has segmentation on this annotation, update the segmented volume
            '''
            v = int(np.unique(a[a>0]))
            ind = label_stars_zoom_test == v
            seg_volume[z][ind] = (np.max(seg_volume[z])+1)
    return seg_volume


######################################################################################################################################################
# Match nuclei funcs
# Get the 3D neuron from the nuclei intersection of union in nearby planes
######################################################################################################################################################

def update_new_neuron(Neuron_dict, nuclei_list, plane):

    neuron_ID_list  = list(Neuron_dict.keys())

    if len(neuron_ID_list)==0:
        max_value = 0
    else:
        max_value  = np.max(neuron_ID_list) 
        
    offset = max_value + 1
    Neuron_dict.update({offset + i: {plane: nuclei} for i, nuclei in enumerate(nuclei_list)})
    return Neuron_dict


def get_nuclei_ID_list_on_the_plane(img):
    return list(np.unique(img[img>0]).astype(int))


def get_neuron_ID_list_on_the_plane(Neuron_dict, plane):
    return [ID for ID in Neuron_dict if plane in Neuron_dict[ID]]


def keep_max_in_columns(arr_2d):
    max_indices = np.argmax(arr_2d, axis=0)
    result = np.zeros_like(arr_2d)
    result[max_indices, np.arange(arr_2d.shape[1])] = arr_2d[max_indices, np.arange(arr_2d.shape[1])]
    return result


def keep_max_in_rows(arr_2d):
    max_indices = np.argmax(arr_2d, axis=1)
    result = np.zeros_like(arr_2d)
    result[np.arange(arr_2d.shape[0]),max_indices] = arr_2d[np.arange(arr_2d.shape[0]),max_indices]
    return result


def pipeline_watershed(seg_volume):
    '''
    This pipeline aims to connecte nuclei in different planes by the iou matrix of existed neuron and neuclei in the next plane
    '''
    ################################  Initialize the dictionary of the neuron ################################
    ################################  Locate the first plane that has segmentation ################################
    Neuron_dict = {}
    for plane in range(seg_volume.shape[0]):
        img = seg_volume[plane]
        nuclei_list = get_nuclei_ID_list_on_the_plane(img)
        Neuron_dict = update_new_neuron(Neuron_dict, nuclei_list, plane)
        Neuron_ID_current_plane =  get_neuron_ID_list_on_the_plane(Neuron_dict, plane)
        if len(Neuron_dict)>0:
            break
        
        
    for plane in range(plane+1,seg_volume.shape[0]):
        ################################  Match nuclei in each plane starting from the second plane ################################
        img_current = seg_volume[plane]
        img_previous = seg_volume[plane-1]
        nuclei_list = get_nuclei_ID_list_on_the_plane(img_current)
        new_neuron_list = []
        matched_nuclei_list = []
        Neuron_ID_previous_plane = get_neuron_ID_list_on_the_plane(Neuron_dict, plane-1)


        if len(nuclei_list)>0 and len(Neuron_ID_previous_plane)>0:
            ################################  Compute the intersection_matrix ################################
            intersection_matrix = np.zeros( (len(Neuron_ID_previous_plane),len(nuclei_list)) )
            for Neuron_ID_ind, Neuron_ID in enumerate(Neuron_ID_previous_plane):
                nuclei_ID = Neuron_dict[Neuron_ID][plane-1]
                previous_ind = img_previous==nuclei_ID
                img_current_effective = img_current[previous_ind]
                values,counts = np.unique(img_current_effective[img_current_effective>0], return_counts=True)
                if len(values)>0:
                    for i,value in enumerate(values):
                        value_ind = nuclei_list.index(value)
                        intersection_matrix[Neuron_ID_ind,value_ind] = counts[i]           
            
            each_neuron = keep_max_in_rows(intersection_matrix) #### for each Neuron_ID there is only one maximum intersection !!!
            each_neuron = keep_max_in_columns(each_neuron) #### for each nuclei there is only one maximum intersection !!!

            ################################  Update the matached nuclei_ID in the exsited Neuron_ID ################################
            for i, Neuron_ID in enumerate(Neuron_ID_previous_plane):
                nuclei_ID_ind = (np.where(each_neuron[i]>0)[0])
                if len(nuclei_ID_ind)>0:        
                    nuclei_ID = nuclei_list[int(nuclei_ID_ind)]
                    Neuron_dict[Neuron_ID].update({plane:nuclei_ID})
                    matched_nuclei_list.append(nuclei_ID)

            ################################  Update the unmatached nuclei_ID as new Nueron ################################
            unmatched_neuclei_list = list(set(nuclei_list)-set(matched_nuclei_list))
            Neuron_dict = update_new_neuron(Neuron_dict, unmatched_neuclei_list, plane)        
            
    return Neuron_dict





############################################################################################################################################################### 
# Remove the Neuron that only exist on one plane
# Reorder the Nueron ID list 
######################################################################################################################################################
def get_Neuron_dict_3D(Neuron_dict):
    Neuron_dict_3D = {}
    # Neuron_single = {}
    i = 1
    for Neuron_ID in Neuron_dict.keys():
        if len(Neuron_dict[Neuron_ID])>1:
            Neuron_dict_3D.update({i:Neuron_dict[Neuron_ID]})
            i+=1
    return Neuron_dict_3D    
        
    
    

def get_seg_volume_single(seg_volume,Neuron_dict):
    #########  Remove the Neuron that only exist on one plane, Reorder the Nueron ID list ######
    Neuron_dict_3D = {}
    Neuron_single = {}
    i = 1
    for Neuron_ID in Neuron_dict.keys():
        if len(Neuron_dict[Neuron_ID])>1:
            Neuron_dict_3D.update({i:Neuron_dict[Neuron_ID]})
            i+=1
        else:
            Neuron_single.update({Neuron_ID:Neuron_dict[Neuron_ID]})

    ########## Get single neuron 3d array ################################

    seg_volume_single = np.zeros(seg_volume.shape)
    for Neuron_ID in Neuron_single.keys():
        for plane in Neuron_single[Neuron_ID].keys():
            nuclei_ID = Neuron_single[Neuron_ID][plane]
            ind = seg_volume[plane] == nuclei_ID
            seg_volume_single[plane][ind] = Neuron_ID
    return seg_volume_single
    # seg_volume_single = get_volume_dic(Neuron_single,seg_volume)
    # return seg_volume_single



# def get_volume_dic(Neuron_dict,seg_volume):
#     seg_volume_dict = np.zeros(seg_volume.shape)
#     for Neuron_ID in Neuron_dict.keys():
#         for plane in Neuron_dict[Neuron_ID].keys():
#             nuclei_ID = Neuron_dict[Neuron_ID][plane]
#             ind = seg_volume[plane] == nuclei_ID
#             seg_volume_dict[plane][ind] = Neuron_ID
#     return seg_volume_dict
    

################################  Convert the Neuron_dict into 3D array as watershed instance segmentation ################################
def get_seg_volume_watershed(Neuron_dict,seg_volume):
    '''
    convert the dictionary which stored where to find each neuron into 3d segmentation volume
    '''
    seg_volume_watershed = np.zeros(seg_volume.shape)
    for Neuron_ID in Neuron_dict.keys():
        for plane in Neuron_dict[Neuron_ID].keys():
            nuclei_ID = Neuron_dict[Neuron_ID][plane]
            ind = seg_volume[plane] == nuclei_ID
            seg_volume_watershed[plane][ind] = Neuron_ID
    return seg_volume_watershed





############################################################################################################################################################### 
# Match each Neuron with annotations
############################################################################################################################################################### 


def centroid_distance_matrix(df):
    '''
    compute the nuclei centroid distance in nearby planes 
    '''
    centroid_list = np.array(df['centroid'])
    centroid_dist = [np.sqrt(np.sum((np.array(centroid_list[i]) - np.array(centroid_list[i+1]))**2))   for i in range(len(centroid_list)-1)]
    
    '''
    Oct 8: list index out of range
    '''
    centroid_dist_next = copy.deepcopy(centroid_dist)
    if len(centroid_dist)>0:
        centroid_dist_next.append(centroid_dist[-1])
    else:
        centroid_dist_next.append(0)
 

    centroid_dist_up = copy.deepcopy(centroid_dist)
    if len(centroid_dist)>0:
        centroid_dist_up.insert(0,centroid_dist[0])
    else:
        centroid_dist_up.insert(0,0)
        
    '''
    Oct 8: list index out of range
    '''    
    return centroid_dist_next, centroid_dist_up


def find_local_minima(data):
    minima = None
    for i in range(1, len(data) - 1):  # We skip the first and the last elements
        if data[i] < data[i - 1] and data[i] < data[i + 1]:
            minima = data[i]
            break
    if minima is  None:
        minima = np.min(data)
    return data.index(minima), minima


def find_local_maxima(data):
    maxima = None
    for i in range(1, len(data) - 1):  # We skip the first and the last elements
        if data[i] > data[i - 1] and data[i] > data[i + 1]:
            maxima = data[i]
            break
    if maxima is  None:
        maxima = np.max(data)
    return data.index(maxima), maxima


def find_all_local_maxima(data):
    maxima = []
    for i in range(1, len(data) - 1):  # We skip the first and the last elements
        if data[i] > data[i - 1] and data[i] > data[i + 1]:
            maxima.append([int(i),data[i]])
    return maxima


def get_df_neuron_measurements(image,labels,ID):
    '''
    update each neuron's properties 
    Properties:[label, area, mean_intensity,centroid, centroid_dist_next,centroid_dist_up,centroid_dist_ratio]
    '''
    Neuron_ID_shape = {}

    for z in np.unique(np.where(labels==ID)[0]):
        ind = labels[z]==ID
        props = measure.regionprops(ind*1, intensity_image=(image[z]*ind))
        Neuron_ID_shape[z]= { 'label':z,'area': props[0].area, 'mean_intensity': props[0].mean_intensity, 'centroid': props[0].centroid}
    df = pd.DataFrame(Neuron_ID_shape).T
    centroid_dist_next, centroid_dist_up = centroid_distance_matrix(df)
    df['centroid_dist_next'] = centroid_dist_next
    df['centroid_dist_up'] = centroid_dist_up
    df['centroid_dist_ratio'] = np.array(centroid_dist_next)/np.array(centroid_dist_up)
    return df



def assign_new_group_segemented(labels,ID,neuron_seg_z):
    #### group the z plane from the neuron_seg_z, so that each group only contain one annotation
    all_planes = set(np.arange(np.min(np.unique(np.where(labels==ID)[0])),np.max( np.unique(np.where(labels==ID)[0]))+1))
    new_neuron = []
    new_seg_neuron = []
    i = 1
    for z in all_planes:
        new_neuron.append(z)
        if z in neuron_seg_z:
            new_seg_neuron.append(new_neuron)
            new_neuron = []
            i+=1
    new_seg_neuron.append(new_neuron) 

    return new_seg_neuron





def find_where_seg_area(labels,ID,df,z_list):
    '''
    From the existed annotations in the palne list 'z_list', 
    making use of the features in neuron 'ID' with the pandas dataframe 'df'
    sort mainly by the 'centroid_dist_next', then 'area' then 'centroid_dist_ratio'
    find the planes where to segment the merged neuron
    '''
    neuron_seg_z = [np.min(np.unique(np.where(labels==ID)[0])),np.max( np.unique(np.where(labels==ID)[0]))]
    neuron_seg_z = []
    group_z = None
    for i in range(len(z_list)-1):
        [s,e] = [z_list[i],z_list[i+1]]
        
        if e == np.max(df['label']):
            e = e-1
            
        if group_z == s-1 and s<e-1:
            s = s+1
        ind,coord_max = find_local_maxima(list(df.loc[s:e]['centroid_dist_next']))
        # ind,coord_max = find_local_maxima(list(df.loc[s:e]['centroid_dist_ratio']))
        group_z = list(df.loc[s:e]['label'])[ind]
        # print([s,e],":",group_z)
        if group_z < e:
        # and  df.loc[group_z]['centroid_dist_next'] > df.loc[group_z]['centroid_dist_up']:
            pass
        else:
            
            ind,coord_max = find_local_minima(list(df.loc[s:e]['area']))
            group_z = list(df.loc[s:e]['label'])[ind]
            if group_z < e:
                pass
            else:
                ind,coord_max = find_local_maxima(list(df.loc[s:e]['centroid_dist_ratio']))
                group_z = list(df.loc[s:e]['label'])[ind]
                
            
        neuron_seg_z.append(group_z)
        # print([s,e],":",group_z)
    neuron_seg_z = np.sort(neuron_seg_z)
    return neuron_seg_z 



def find_where_seg_meanint(labels,ID,df,z_list):
    '''
    From the existed annotations in the palne list 'z_list', 
    making use of the features in neuron 'ID' with the pandas dataframe 'df'
    sort mainly by the 'mean intensity', then 'area' then 'centroid_dist_ratio'
    find the planes where to segment the merged neuron
    '''
    #### The critera to find where to seg use the local minimum of mean intensity
    neuron_seg_z = [np.min(np.unique(np.where(labels==ID)[0])),np.max( np.unique(np.where(labels==ID)[0]))]
    neuron_seg_z = []
    group_z = None
    for i in range(len(z_list)-1):
        [s,e] = [z_list[i],z_list[i+1]]
        
        if e == np.max(df['label']):
            e = e-1
        if group_z == s-1 and s<e-1:
            s = s+1
        ind,coord_max = find_local_minima(list(df.loc[s:e]['mean_intensity']))
        # ind,coord_max = find_local_maxima(list(df.loc[s:e]['centroid_dist_ratio']))
        group_z = list(df.loc[s:e]['label'])[ind]
        # print([s,e],":",group_z)
        if group_z < e:
        # and  df.loc[group_z]['centroid_dist_next'] > df.loc[group_z]['centroid_dist_up']:
            pass
        else:
            
            ind,coord_max = find_local_maxima(list(df.loc[s:e]['centroid_dist_next']))
            group_z = list(df.loc[s:e]['label'])[ind]
            if group_z < e:
                pass
            else:
                ind,coord_max = find_local_maxima(list(df.loc[s:e]['centroid_dist_ratio']))
                group_z = list(df.loc[s:e]['label'])[ind]
                
        if group_z == np.max( np.unique(np.where(labels==ID)[0]))-1:
            group_z = group_z - 1
        neuron_seg_z.append(group_z)
        # print([s,e],":",group_z)
    neuron_seg_z = np.sort(neuron_seg_z)
    return neuron_seg_z 






def find_where_seg_centroid(df):
    '''
    From the existed annotations in the palne list 'z_list', 
    making use of the features in neuron 'ID' with the pandas dataframe 'df'
    sort only by the 'centroid_dist_next', 
    find the planes where to segment the merged neuron
    '''
    ind = find_all_local_maxima(list(df['centroid_dist_next']))
    
    if len(ind)>0:
        ind = (np.array(ind)[:,0]).astype(int)
        neuron_seg_z = np.array(df['label'])[ind]
        return neuron_seg_z
    else:
        return []


def if_well_seg(new_seg_neuron, z_list):
    '''
    Define is the merged in the z plane are well segemented or not by seeing if annotation exist in one neuron 
    '''
    criteria = 1
    if len(new_seg_neuron)<len(z_list):
        criteria = 0
    else:
        for i,z in enumerate(z_list): 
            if z in new_seg_neuron[i] and len(new_seg_neuron[i])>1:
                pass
            else:
                criteria = criteria * 0
    return bool(criteria)



def update_abs_pos(z_list, new_seg_neuron, abs_pos, unique_values, ID):
    '''
    There are situations that the annotations are not accurate in the z plane, then optimize the abs_pos
    '''
    assert len(new_seg_neuron)> len(z_list) or len(new_seg_neuron)== len(z_list), "new segmented neurons contain multiple annotations"
    abs_pos_new = copy.deepcopy(abs_pos)
    for i,z in enumerate(z_list):
        if len(new_seg_neuron[i])>0:
            if z in new_seg_neuron[i]:
                pass
            else:
                ind = np.argmin(abs(np.array(new_seg_neuron[i])- z))
                correct_z = new_seg_neuron[i][ind]
                ind = np.where(abs_pos_new[unique_values==ID][:,0] == z)
                ind3 = np.where(((unique_values==ID)*(abs_pos_new[:,0]==z))>0)[0][0]
                abs_pos_new[ind3,0] = correct_z
    return  abs_pos_new




def get_z_list(abs_pos,unique_values,ID):
    #### get the sorted unique z plane in the annotations, remove the rows that are all zeros
    arr = abs_pos[unique_values==ID]
    non_zero_rows = arr[~(arr == 0).all(axis=1)]
    z_list = np.sort(non_zero_rows[:,0])  
    return z_list


def check_single_nuclei_contain_multiple_annotations(abs_pos,unique_values,ID,df):
    #### if a single nuclei contain above 2 annotations, then the annotations that are farther will be replaced as all zeros
    z_list = get_z_list(abs_pos,unique_values,ID)
    z_uniuqe,z_counts = np.unique(z_list,return_counts = True)
    for z_repeat in z_uniuqe[z_counts>1]:

        coords_ID = abs_pos[unique_values==ID]
        coords = coords_ID[coords_ID[:,0]==z_repeat]
        dist = np.sqrt(np.sum(abs(coords[:,1:] - np.array(df.loc[z_repeat]['centroid']))**2,axis = 1))
        keep_ind = np.argmin(dist)
        remove_coords = coords[1 - keep_ind].reshape(-1,3)
        for remove_coord in remove_coords:
            ind = np.where((abs_pos == remove_coord).all(axis=1))[0]
            abs_pos[ind] = 0 
            
    z_list = get_z_list(abs_pos,unique_values,ID)
    return abs_pos,z_list


def remove_annotation_no_segmentation(abs_pos,unique_values,labels,ID):
    ### For a neuron that can't be segemented any more, remove the extra annotation
    all_planes = set(np.arange(np.min(np.unique(np.where(labels==ID)[0])),np.max( np.unique(np.where(labels==ID)[0]))+1))
    z_list = get_z_list(abs_pos,unique_values,ID)
    coords = abs_pos[unique_values==ID].reshape(-1,3)
    keep_ind = np.argmin(np.array(coords[:,0])-np.average(list(all_planes)))
    remove_coords = coords[1 - keep_ind].reshape(-1,3)
    for remove_coord in remove_coords:
        ind = np.where((abs_pos == remove_coord).all(axis=1))[0]
        abs_pos[ind] = 0 
    z_list = get_z_list(abs_pos,unique_values,ID)
    return abs_pos,z_list






def pip_regroup_merged_neuron_ID(image,labels,abs_pos,unique_values,ID):
    '''
    return the plane list to seg the merged neuron in the plane_z, switch the method to segment the merged neuron if needed
    '''
    df = get_df_neuron_measurements(image,labels,ID)
    abs_pos,z_list = check_single_nuclei_contain_multiple_annotations(abs_pos,unique_values,ID,df)


    neuron_seg_z  = find_where_seg_area(labels,ID,df,z_list)
    new_seg_neuron = assign_new_group_segemented(labels,ID,neuron_seg_z)

    if not if_well_seg(new_seg_neuron, z_list):
        neuron_seg_z  = find_where_seg_centroid(df)
        print("Use Centroid")
        if not len(neuron_seg_z) >0:
            abs_pos,z_list = remove_annotation_no_segmentation(abs_pos,unique_values,labels,ID)
            print("No seg",if_well_seg(assign_new_group_segemented(labels,ID,neuron_seg_z), z_list))
        new_seg_neuron = assign_new_group_segemented(labels,ID,neuron_seg_z)
       

        
    if len(neuron_seg_z) >0 and not if_well_seg(new_seg_neuron, z_list):

        if len(new_seg_neuron)==len(z_list):
            print("Optimize annotation")
            abs_pos_new = update_abs_pos(z_list, new_seg_neuron, abs_pos, unique_values, ID)
            z_list_new = get_z_list(abs_pos_new,unique_values,ID)
            print("new z_list",z_list_new)
            if not if_well_seg(new_seg_neuron, z_list_new):
                print("warning","new z_list",z_list_new)
            else:
                abs_pos = abs_pos_new
        else:
            neuron_seg_z  = find_where_seg_meanint(labels,ID,df,z_list)
            new_seg_neuron = assign_new_group_segemented(labels,ID,neuron_seg_z)
            print("use mean intensity",if_well_seg(new_seg_neuron, z_list))
            if not if_well_seg(new_seg_neuron, z_list):
                print("warning the annotation is not correct")
            
    
    ###########################################################################################################        
    ############################## The following is to find if this nuclei is acutally merged #################
    ############################## Must be the maximum area and intersection, the centroid must be in  ########
    ###########################################################################################################       
    merged_nuclei = {}
    
    for seg_z in neuron_seg_z:  
        
        if seg_z > np.min(df['label']) and seg_z < np.max(df['label']):    
            if seg_z > np.min(df['label']) and seg_z < np.max(df['label']):
                area_max = (df['area'][seg_z] > df['area'][seg_z-1]) and (df['area'][seg_z] > df['area'][seg_z+1]) 

            elif seg_z == np.min(df['label']) and seg_z < np.max(df['label']):
                area_max = df['area'][seg_z] > df['area'][seg_z+1]

            elif seg_z == np.max(df['label']) and seg_z < np.max(df['label']):
                area_max = df['area'][seg_z] > df['area'][seg_z-1]


            if area_max:

                ind1 = (labels[seg_z-1]==ID)
                ind2 = (labels[seg_z]==ID)
                ind3 = (labels[seg_z+1]==ID)
                [centroid_x1,centroid_y1] = np.array(measure.regionprops(ind1*1)[0].centroid).astype(int)
                [centroid_x3,centroid_y3] = np.array(measure.regionprops(ind3*1)[0].centroid).astype(int)
                centroid_in =  0 not in list(labels[seg_z][centroid_x1-1:centroid_x1+2,centroid_y1-1:centroid_y1+2].flatten()) and  0 not in list(
                    labels[seg_z][centroid_x3-1:centroid_x3+2,centroid_y3-1:centroid_y3+2].flatten())

                if centroid_in:
                    iou_max = np.sum((ind1+ind3)*ind2)>np.sum((ind1)*ind2) and np.sum((ind1+ind3)*ind2)>np.sum((ind3)*ind2)
                    if iou_max:
                        if len(merged_nuclei) == 0:
                            merged_nuclei.update({ID:{seg_z}})  
                        else:
                            merged_nuclei[ID].update({seg_z})
                    
    print(ID, new_seg_neuron,  z_list )
    return new_seg_neuron,abs_pos, merged_nuclei



def update_Neuron_dict_seg_merged_neuron(unique_values,image,labels,abs_pos,Neuron_dict):
    '''
    This is a loop for each neuron ID, segment the merged neuron along z direction, may modify the annotation of abs_pos
    '''
    values,counts = np.unique(unique_values,return_counts = True)
    ID_list = values[counts>1]
    filtered_IDs = ID_list[ID_list>0]
    merged_nuclei_all = {}
    for ID in filtered_IDs:
        new_seg_neuron,abs_pos, merged_nuclei = pip_regroup_merged_neuron_ID(image,labels,abs_pos,unique_values,ID)
        Neuron_dict = update_dict_regroup_neuron(Neuron_dict,ID,new_seg_neuron)
        merged_nuclei_all.update(merged_nuclei)
    return Neuron_dict,abs_pos,merged_nuclei_all


def update_dict_regroup_neuron(Neuron_dict,ID,new_seg_neuron):
    '''
    update the Neuron_dict with the new grouped neuron
    remove the merged neuron
    '''
    New_ID = np.max(list(Neuron_dict.keys()))+1
    for seg_neuron in new_seg_neuron:
        Neuron_dict[New_ID] = {k: Neuron_dict[ID][k] for k in seg_neuron if k in Neuron_dict[ID]}
        New_ID += 1
    del Neuron_dict[ID]       
    return Neuron_dict




