import numpy as np
import pandas as pd
import h5py
import os
import time
from load_func import *
# pd.set_option('display.max_columns', None)






def most_frequent_non_zero(arr):

    non_zero_values = arr[arr != 0]
    if len(non_zero_values) == 0:
        return 0  # Return 0 if no non-zero values exist

    v,c = np.unique(non_zero_values, return_counts=True)
    return v[np.argmax(c)]


def update_seg_ID(row, seg, seg_ref):

    ID = row['seg_ref_ID']
    ID_seg = row['seg_ID']
    if ID > 0:
        values = seg[seg_ref == ID]
        values = values[values > 0]
        # if len(values) > 0:
        #     v, c = np.unique(values, return_counts=True)
        #     vmax = v[np.argmax(c)]
        #     if ID_seg == 0 or ID_seg == vmax:
        #         return vmax, np.max(c), None
        #     else:
        #         return ID_seg, np.sum(values==ID_seg), None
        # else:
        #     return 0, 0, None 
        return ID_seg, np.sum(values==ID_seg), None
    
    return row['seg_ID'], 0, None



def get_most_frequent_ID(coords, offsets, seg):
    coords_nearby = coords[:, None, :] + offsets
    neighbor_values = np.array([seg[tuple(c.T)] for c in coords_nearby])
    most_frequent_values = np.apply_along_axis(most_frequent_non_zero, axis=1, arr=neighbor_values)
    return most_frequent_values



def filter_seg_id(group):
    # Calculate the condition: percent > 2 * rest and percent > 0.6
    max_percent = group['percent'].max()
    second_max_percent = group['percent'].nlargest(2).iloc[-1]



    if max_percent > 2 * second_max_percent :
        # Keep the seg_ID for rows meeting the condition, set others to 0
        group['seg_ID'] = group['seg_ID'].where(group['percent'] == max_percent, 0)
    else:
        group['seg_ID'] = 0

    return group



def annotation_seg_ID_seg_ref(annotation_path, t_idx, seg, seg_ref):

    # Load and prepare the combined_df_t_idx DataFrame
    combined_df_t_idx = load_annotations_h5_t_idx(annotation_path, t_idx)
    abs_pos = get_abs_pos(combined_df_t_idx, seg.shape) 
    combined_df_t_idx[['global_z', 'global_gy', 'global_gx']] = abs_pos


    coords = combined_df_t_idx[['global_z', 'global_gy', 'global_gx']].values.astype(int)
    offsets = np.array(np.meshgrid([-1,0,1], [-3, -2, -1, 0, 1, 2, 3], [-3, -2, -1, 0, 1, 2, 3])).T.reshape(-1, 3)
    most_frequent_values = get_most_frequent_ID(coords, offsets, seg)
    combined_df_t_idx.loc[:, 'seg_ID'] = most_frequent_values



    if seg_ref is None:
        combined_df_t_idx.loc[combined_df_t_idx.duplicated(subset='seg_ID', keep=False), 'seg_ID'] = 0
        combined_df_t_idx =combined_df_t_idx.sort_values(by='worldline_id').set_index('worldline_id').reset_index() ## reset the index based on the worldline_id
        return combined_df_t_idx
    


    ### 1. find the intersetion of seg_ref and seg
    coords = combined_df_t_idx[['global_z', 'global_gy', 'global_gx']].values.astype(int)
    combined_df_t_idx['seg_ref_ID'] = seg_ref[tuple(coords.T)]
    results = combined_df_t_idx.apply(lambda row: update_seg_ID(row, seg, seg_ref), axis=1)
    combined_df_t_idx['seg_ID'], combined_df_t_idx['intersection'], combined_df_t_idx['backup'] = zip(*results)




    ### 2. find the ungrouped IDs in seg that are not in seg_ID_list and Create un_seg with ungrouped IDs
    # Find ungrouped IDs in seg that are not in seg_ID_list and Create un_seg with ungrouped IDs
    seg_ID_list = combined_df_t_idx['seg_ID'].unique()
    ungrouped_ID_mask = np.isin(seg, seg_ID_list[seg_ID_list > 0], invert=True) & (seg > 0)
    un_seg = np.where(ungrouped_ID_mask, seg, 0)
    index_zero_seg_ID = combined_df_t_idx['seg_ID'] == 0
    coords = combined_df_t_idx[index_zero_seg_ID][['global_z', 'global_gy', 'global_gx']].values.astype(int)
    offsets = np.array(np.meshgrid([-1,0,1], [-3, -2, -1, 0, 1, 2, 3], [-3, -2, -1, 0, 1, 2, 3])).T.reshape(-1, 3)
    most_frequent_values = get_most_frequent_ID(coords, offsets, un_seg)
    combined_df_t_idx.loc[index_zero_seg_ID, 'seg_ID'] = most_frequent_values
    

    
    ### 3. filter the seg_ID based on the intersection and pixel_count
    neuron_ids, pixel_counts = np.unique(seg[seg>0], return_counts=True)
    neuron_pixel_counts_df = pd.DataFrame({'seg_ID': neuron_ids,'pixel_count': pixel_counts}) 
    combined_df_t_idx = pd.merge(combined_df_t_idx, neuron_pixel_counts_df, left_on='seg_ID', right_on='seg_ID', how='left')
    combined_df_t_idx['percent'] = combined_df_t_idx['intersection'] / combined_df_t_idx['pixel_count']
    selected_index = combined_df_t_idx['seg_ID'].duplicated(keep=False) & (combined_df_t_idx['seg_ID'] > 0)
    df = combined_df_t_idx[selected_index]
    df_filtered = df.groupby('seg_ID', group_keys=False).apply(filter_seg_id)
    combined_df_t_idx[selected_index] = df_filtered 



   
    combined_df_t_idx.loc[combined_df_t_idx.duplicated(subset='seg_ID', keep=False), 'seg_ID'] = 0
    combined_df_t_idx =combined_df_t_idx.sort_values(by='worldline_id').set_index('worldline_id').reset_index() ## reset the index based on the worldline_id
    return combined_df_t_idx






# t_idx = 800
# seg_path = '/Users/hangdeng/Documents/work/Neuron_tracking/seg/'+str(t_idx)+'.h5'
# seg = load_seg_h5(seg_path)
# seg_ref_path = '/Users/hangdeng/Documents/work/Neuron_tracking/seg_ref/'+str(t_idx)+'.h5'
# seg_ref = load_seg_h5(seg_ref_path)
# annotation_path = '/Users/hangdeng/Documents/work/Neuron_tracking/ZM9624/annotations.h5'
# combined_df_t_idx = annotation_seg_ID_seg_ref(annotation_path, t_idx, seg, seg_ref)
# combined_df_t_idx2 = annotation_seg_ID_seg_ref(annotation_path, t_idx, seg, None)
# combined_df_t_idx[combined_df_t_idx['seg_ID']>0].shape, combined_df_t_idx2[combined_df_t_idx2['seg_ID']>0].shape,  \
#     len(combined_df_t_idx[combined_df_t_idx['seg_ID'] != combined_df_t_idx2['seg_ID']]) 


