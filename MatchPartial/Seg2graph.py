##################################################################################### 
# Description: This script is used to convert the segmentation image to graph with nodes and edges features
#              The graph is saved as a dictionary
#              The key is the node index, the value is the list of connected nodes
#              The graph is undirected
#              The graph is saved as a pickle file
#####################################################################################



import torch
import h5py
import copy
import numpy as np
import pandas as pd
from pathlib import Path
from csbdeep.utils import normalize
from skimage.measure import regionprops_table
from scipy.spatial import KDTree
from torch_geometric.data import Data
from .load_func import *
# from relabel_graph_data import  annotation_seg_ID_seg_ref


class NeuronNodesFeatureExtractor:
    def __init__(self, img_h5_path, seg_h5_path, t_idx, ch):
        self.img_h5_path = img_h5_path
        self.seg_h5_path = seg_h5_path
        self.t_idx = t_idx
        self.ch = ch
        self.img_orig, self.seg = self.load_img_seg()

    def get_volume_at_frame(self, file_name, t_idx):
        '''
        Get the 3D original volume at frame t_idx from the h5 file_name
        '''
        with h5py.File(file_name, 'r') as f:
            img_original = np.array(f['data'][t_idx:t_idx + 1])
        return img_original, None

    def load_img_seg(self):
        if self.img_h5_path ==None or self.seg_h5_path == None:
            return None, None

        elif not Path(self.img_h5_path).exists() or not Path(self.seg_h5_path).exists():
            return None, None

        else:
            img_original, _ = self.get_volume_at_frame(self.img_h5_path, self.t_idx)
            img_orig = img_original[0, self.ch]

            with h5py.File(self.seg_h5_path, 'r') as f:
                seg = f['label'][:]
            return img_orig, seg

    def get_neuron_3D_features(self, seg, img_orig):
        props = regionprops_table(seg.astype(int), intensity_image=img_orig,
                                  properties=('centroid', 'area_filled', 'intensity_mean', 'coords', 'slice', 'label'))
        df_3D = pd.DataFrame(props)
        df_3D['slice_depth'] = df_3D['slice'].apply(lambda x: x[0].stop - x[0].start)
        df_3D = df_3D.drop(columns=['slice'])
        df_3D = df_3D.rename(columns={"area_filled": "volume", 'intensity_mean': 'intensity_mean_3D'})
        return df_3D

    def get_neuron_2D_features(self, seg, img_orig):
        '''
        For the maximum projection of each neuron, obtain the features
        '''
        neuron_ID_list = list(np.unique(seg).astype(int))
        neuron_ID_list.remove(0)
        props_df_list = []

        for neuron_ID in neuron_ID_list:
            neuron_ind = seg == neuron_ID
            seg_ID_2d = neuron_ID * np.max(neuron_ind, axis=0).astype(int)
            img_ID_2d = np.max(neuron_ind * img_orig, axis=0)

            props = regionprops_table(seg_ID_2d, intensity_image=img_ID_2d,
                                      properties=('orientation', 'axis_major_length', 'axis_minor_length',
                                                  'area_filled', 'eccentricity', 'perimeter', 'intensity_mean',
                                                  'label', 'centroid'))
            props_df = pd.DataFrame(props)
            props_df_list.append(props_df)

        df_2D = pd.concat(props_df_list, ignore_index=True)
        df_2D['axis_ratio'] = df_2D['axis_minor_length'] / df_2D['axis_major_length']
        df_2D = df_2D.rename(columns={"area_filled": "area_zmax", 'intensity_mean': 'intensity_mean_2D',
                                      'centroid-0': 'centroid_2d-0', 'centroid-1': 'centroid_2d-1'})
        return df_2D




    def extract_features(self):
        img_norm = self.normalize_img(self.img_orig, pmin=3, pmax=99.8)
        df_3D = self.get_neuron_3D_features(self.seg, img_norm)
        df_2D = self.get_neuron_2D_features(self.seg, self.img_orig)
        df_nodes_features = pd.merge(df_2D, df_3D, on='label', how='inner')


        
        return df_nodes_features
    

    
    def extract_features_from_seg(self, img_orig, seg):
        img_norm = self.normalize_img(img_orig, pmin=3, pmax=99.8)
        df_3D = self.get_neuron_3D_features(seg, img_norm)
        df_2D = self.get_neuron_2D_features(seg, img_orig)
        df_nodes_features = pd.merge(df_2D, df_3D, on='label', how='inner')
        return df_nodes_features

    def normalize_img(self, img, pmin, pmax):
        '''
        Normalizes the image using percentiles pmin and pmax
        '''
        img_min = np.percentile(img, pmin)
        img_max = np.percentile(img, pmax)
        img_norm = np.clip((img - img_min) / (img_max - img_min), 0, 1)
        img_norm = img_norm / np.max(img_norm)
        return img_norm


class NeuronLabelExtractor:
    def __init__(self, t_idx, annotation_path, seg_h5_path, seg_ref_h5_path=None):
        self.t_idx = t_idx
        self.seg_h5_path = seg_h5_path
        self.seg_ref_h5_path = seg_ref_h5_path
        self.annotation_path = annotation_path


    def most_frequent_non_zero(self,arr):

        non_zero_values = arr[arr != 0]
        if len(non_zero_values) == 0:
            return 0  # Return 0 if no non-zero values exist

        v,c = np.unique(non_zero_values, return_counts=True)
        return v[np.argmax(c)]


    def update_seg_ID(self, row, seg, seg_ref):

        ID = row['seg_ref_ID']
        ID_seg = row['seg_ID']
        if ID > 0:
            values = seg[seg_ref == ID]
            values = values[values > 0]
            return ID_seg, np.sum(values==ID_seg), None
        
        return row['seg_ID'], 0, None



    def get_most_frequent_ID(self, coords, offsets, seg):
        coords_nearby = coords[:, None, :] + offsets
        seg_shape = seg.shape
        coords_nearby[:, :, 0] = np.clip(coords_nearby[:, :, 0], 0, seg_shape[0]-1)
        coords_nearby[:, :, 1] = np.clip(coords_nearby[:, :, 1], 0, seg_shape[1]-1)
        coords_nearby[:, :, 2] = np.clip(coords_nearby[:, :, 2], 0, seg_shape[2]-1)
        neighbor_values = np.array([seg[tuple(c.T)] for c in coords_nearby])
        most_frequent_values = np.apply_along_axis(self.most_frequent_non_zero, axis=1, arr=neighbor_values)
        return most_frequent_values
 


    def filter_seg_id(self, group):
        # Calculate the condition: percent > 2 * rest and percent > 0.6
        max_percent = group['percent'].max()
        second_max_percent = group['percent'].nlargest(2).iloc[-1]



        if max_percent > 2 * second_max_percent :
            # Keep the seg_ID for rows meeting the condition, set others to 0
            group['seg_ID'] = group['seg_ID'].where(group['percent'] == max_percent, 0)
        else:
            group['seg_ID'] = 0

        return group



    def annotation_seg_ID_seg_ref(self, annotation_path, t_idx, seg, seg_ref):

        # Load and prepare the combined_df_t_idx DataFrame
        combined_df_t_idx = load_annotations_h5_t_idx(annotation_path, t_idx)
        abs_pos = get_abs_pos(combined_df_t_idx, seg.shape) 
        combined_df_t_idx[['global_z', 'global_gy', 'global_gx']] = abs_pos


        coords = combined_df_t_idx[['global_z', 'global_gy', 'global_gx']].values.astype(int)
        offsets = np.array(np.meshgrid([-1,0,1], [-3, -2, -1, 0, 1, 2, 3], [-3, -2, -1, 0, 1, 2, 3])).T.reshape(-1, 3)
        most_frequent_values = self.get_most_frequent_ID(coords, offsets, seg)
        combined_df_t_idx.loc[:, 'seg_ID'] = most_frequent_values



        if seg_ref is None:
            combined_df_t_idx.loc[combined_df_t_idx.duplicated(subset='seg_ID', keep=False), 'seg_ID'] = 0
            combined_df_t_idx =combined_df_t_idx.sort_values(by='worldline_id').set_index('worldline_id').reset_index() ## reset the index based on the worldline_id
            return combined_df_t_idx
        


        ### 1. find the intersetion of seg_ref and seg
        coords = combined_df_t_idx[['global_z', 'global_gy', 'global_gx']].values.astype(int)
        combined_df_t_idx['seg_ref_ID'] = seg_ref[tuple(coords.T)]
        results = combined_df_t_idx.apply(lambda row: self.update_seg_ID(row, seg, seg_ref), axis=1)
        combined_df_t_idx['seg_ID'], combined_df_t_idx['intersection'], combined_df_t_idx['backup'] = zip(*results)




        ### 2. find the ungrouped IDs in seg that are not in seg_ID_list and Create un_seg with ungrouped IDs
        # Find ungrouped IDs in seg that are not in seg_ID_list and Create un_seg with ungrouped IDs
        seg_ID_list = combined_df_t_idx['seg_ID'].unique()
        ungrouped_ID_mask = np.isin(seg, seg_ID_list[seg_ID_list > 0], invert=True) & (seg > 0)
        un_seg = np.where(ungrouped_ID_mask, seg, 0)
        index_zero_seg_ID = combined_df_t_idx['seg_ID'] == 0
        if np.sum(combined_df_t_idx['t_idx'] == 0) >0:
            coords = combined_df_t_idx[index_zero_seg_ID][['global_z', 'global_gy', 'global_gx']].values.astype(int)
            offsets = np.array(np.meshgrid([-1,0,1], [-3, -2, -1, 0, 1, 2, 3], [-3, -2, -1, 0, 1, 2, 3])).T.reshape(-1, 3)
            most_frequent_values = self.get_most_frequent_ID(coords, offsets, un_seg)
            combined_df_t_idx.loc[index_zero_seg_ID, 'seg_ID'] = most_frequent_values

        
        ### 3. filter the seg_ID based on the intersection and pixel_count
        neuron_ids, pixel_counts = np.unique(seg[seg>0], return_counts=True)
        neuron_pixel_counts_df = pd.DataFrame({'seg_ID': neuron_ids,'pixel_count': pixel_counts}) 
        combined_df_t_idx = pd.merge(combined_df_t_idx, neuron_pixel_counts_df, left_on='seg_ID', right_on='seg_ID', how='left')
        combined_df_t_idx['percent'] = combined_df_t_idx['intersection'] / combined_df_t_idx['pixel_count']
        selected_index = combined_df_t_idx['seg_ID'].duplicated(keep=False) & (combined_df_t_idx['seg_ID'] > 0)
        df = combined_df_t_idx[selected_index]
        df_filtered = df.groupby('seg_ID', group_keys=False).apply(self.filter_seg_id)
        combined_df_t_idx[selected_index] = df_filtered 



    
        combined_df_t_idx.loc[combined_df_t_idx.duplicated(subset='seg_ID', keep=False), 'seg_ID'] = 0
        combined_df_t_idx =combined_df_t_idx.sort_values(by='worldline_id').set_index('worldline_id').reset_index() ## reset the index based on the worldline_id
        return combined_df_t_idx
    

    def get_annotation_df(self):
        seg = load_seg_h5(self.seg_h5_path)
        if self.seg_ref_h5_path is not None:
            seg_ref = load_seg_h5(self.seg_ref_h5_path)
        else:
            seg_ref = None

        combined_df_t_idx = self.annotation_seg_ID_seg_ref(self.annotation_path, self.t_idx, seg, seg_ref)
        annotation_df = combined_df_t_idx[['worldline_id', 'seg_ID']]
        annotation_df = annotation_df.rename(columns={'seg_ID':'label'})

        annotation_df['worldline_id'] = annotation_df['worldline_id'] +1
        # annotation_df['worldline_id'] = (np.array(annotation_df['worldline_id']) * np.array(annotation_df['label']))
        return annotation_df[annotation_df['label']>0]




    def match_label_to_worldline_id(self, df_nodes_features):
        annotation_df = self.get_annotation_df()
        label_to_worldline_id = dict(zip(annotation_df['label'], annotation_df['worldline_id']))
        df_nodes_features['worldline_id'] = df_nodes_features['label'].map(label_to_worldline_id).fillna(0)
        return df_nodes_features
    


    




class GraphDataGenerator:
    def __init__(self, df_nodes_features, num_nearest, isotropic_voxel_size):
        self.df_nodes_features = df_nodes_features
        self.num_nearest = num_nearest
        self.isotropic_voxel_size = isotropic_voxel_size

    def produce_edges_features(self):
        self.df_nodes_features, index = self.search_nearby_nodes()
        pair_nodes = self.get_pair_nodes(index) 
        df_edges_features, nodes_start_indices, nodes_end_indices = self.get_edges_features(pair_nodes)
        return df_edges_features, nodes_start_indices, nodes_end_indices

    def get_pair_nodes(self, index):
        '''
        Return the edge with pair of nodes
        '''
        neuron_ID_arr = np.array(self.df_nodes_features['label'])
        repeated_array = np.repeat(neuron_ID_arr[:, np.newaxis], self.num_nearest, axis=1)
        arr = np.array([repeated_array, neuron_ID_arr[index]])
        pair_nodes = np.transpose(arr, (1, 2, 0)).reshape(-1, 2)
        return pair_nodes

    def search_nearby_nodes(self):
        '''
        Find the nearest neurons by distance.
        Return the indices based on the df_nodes_features['label'].
        Update the DataFrame with nearby neuron IDs.
        '''
        neuron_centroid = np.array(self.df_nodes_features[['centroid-0', 'centroid_2d-0', 'centroid_2d-1']]) * np.array(self.isotropic_voxel_size)
        neuron_ID_arr = np.array(self.df_nodes_features['label'])
        tree = KDTree(neuron_centroid)
        _, indices = tree.query(neuron_centroid, k=self.num_nearest + 1)
        index = indices[:, 1:]  # Exclude self
        self.df_nodes_features['neighbour'] = list(neuron_ID_arr[index])
        return self.df_nodes_features, index

    def get_edges_features(self, pair_nodes):
        '''
        Calculate the euclidean distance, euclidean vector, and angle between two neurons
        '''
        neuron_ID_arr = np.array(self.df_nodes_features['label'])
        neuron_centroid = np.array(self.df_nodes_features[['centroid-0', 'centroid_2d-0', 'centroid_2d-1']])

        nodes_start_indices = self.get_indices_arr(neuron_ID_arr, pair_nodes[:, 0])
        nodes_end_indices = self.get_indices_arr(neuron_ID_arr, pair_nodes[:, 1])
        coords_start = neuron_centroid[nodes_start_indices]
        coords_end = neuron_centroid[nodes_end_indices]
        euclidean_vector = coords_end - coords_start
        euclidean_distances = np.sqrt(np.sum(euclidean_vector * euclidean_vector, axis=1))

        nodes_start_ore = np.array(self.df_nodes_features['orientation'])[nodes_start_indices]
        nodes_end_ore = np.array(self.df_nodes_features['orientation'])[nodes_end_indices]
        pair_orientation = (np.sum(self.vector_orientation(nodes_start_ore) * self.vector_orientation(nodes_end_ore), axis=1))

        df_edges_features = pd.DataFrame(np.transpose(np.array([euclidean_vector[:, 0], euclidean_vector[:, 1], euclidean_vector[:, 2],
                                                                euclidean_distances, pair_orientation]), (1, 0)),
                                         columns=['euclidean_vector-0', 'euclidean_vector-1', 'euclidean_vector-2', 'euclidean_abs_dist', 'pair_orientation'])
        df_edges_features.insert(0, 'pair_nodes', [tuple(pair) for pair in pair_nodes])
        return df_edges_features, nodes_start_indices, nodes_end_indices

    def get_indices_arr(self, neuron_ID_arr, nodes_arr):
        '''
        Get indices for the neuron IDs.
        '''
        value_to_index = {value: idx for idx, value in enumerate(neuron_ID_arr)}
        indices = [value_to_index[node] for node in nodes_arr]
        return np.array(indices)

    def vector_orientation(self, orientation_arr):
        '''
        Convert the orientation (-pi, pi) into unit vectors
        '''
        arr = np.array([np.cos(orientation_arr), - np.sin(orientation_arr)])
        return np.transpose(arr, (1, 0))

    def produce_graph_data(self):
        '''
        Produce graph data for the nodes and edges.
        '''
        selected_nodes_features = [
            'centroid_2d-0', 'centroid_2d-1', 'centroid-0',
            'axis_major_length', 'axis_minor_length', 'axis_ratio',
            'eccentricity', 'centroid-1', 'centroid-2', 'slice_depth'
        ]

        selected_edges_features = ['euclidean_vector-0', 'euclidean_vector-1', 'euclidean_vector-2', 'euclidean_abs_dist', 'pair_orientation']
        
        df_edges_features, nodes_start_indices, nodes_end_indices = self.produce_edges_features()

        node_features = torch.tensor(np.array(self.df_nodes_features[selected_nodes_features].values), dtype=torch.float)


        # edge_index = torch.tensor([np.array(nodes_start_indices), np.array(nodes_end_indices)], dtype=torch.long) 
        edge_index = torch.tensor(np.array([nodes_start_indices, nodes_end_indices]), dtype=torch.long)

        edge_features = torch.tensor(np.array(df_edges_features[selected_edges_features].values), dtype=torch.float)
        
        data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_features)
        
        if 'worldline_id' in self.df_nodes_features.columns:
            data.y = torch.tensor(np.array(self.df_nodes_features['worldline_id']), dtype=torch.long)


        return data


    def save_graph2pt(self, data, t_idx, save_graph_path):
        if not save_graph_path.exists():
            save_graph_path.mkdir(exist_ok=True, parents=True)
        data.x = torch.nan_to_num(data.x, nan=0.0)
        torch.save(data, Path(save_graph_path)/(str(t_idx)+'.pt'))





def update_abs_pos_basedon_seg(seg, abs_pos, abs_pos_new):
    seg_new_list = acutual_seg_IDs(seg, abs_pos_new)
    seg_list = acutual_seg_IDs(seg, abs_pos)
    values_list, abs_pos_f = unique_merge_seg(seg_list, seg_new_list, abs_pos, abs_pos_new)
    return abs_pos_f


def acutual_seg_IDs(seg, abs_pos):
    unique_seg_actual = seg[tuple(abs_pos.T.astype(int))]
    all_seg_ID = np.unique(seg[seg>0])
    for i, v in enumerate(unique_seg_actual):
        if v != i+1:
            if i+1 in all_seg_ID:
                unique_seg_actual[i] = i+1
                abs_pos[i] = np.mean( np.argwhere(seg == i+1), axis = 0)
    return unique_seg_actual


def unique_merge_seg(seg_list, seg_new_list, abs_pos, abs_pos_new):
    values_list = []    
    abs_pos_f = copy.deepcopy(abs_pos)
    for i in range(len(seg_list)):
        if seg_list[i] == i+1:
            values_list.append(seg_list[i])
        elif seg_list[i] != i+1 and seg_new_list[i] == i+1:
            values_list.append(seg_new_list[i])
            abs_pos_f[i] = abs_pos_new[i]
        elif seg_list[i] ==0  and seg_new_list[i] !=0:
            values_list.append(seg_new_list[i])
            abs_pos_f[i] = abs_pos_new[i]
        elif seg_list[i] ==0  and seg_new_list[i] ==0:
            values_list.append(0)
        else:
            values_list.append(seg_list[i])
    values_list = np.array(values_list)
    return values_list, abs_pos_f



def index_if_unique_seg(seg, abs_pos_f, combined_df_t_idx):
    ## obtain all the segmentation IDs
    ind = np.unique(seg[seg>0])

    ## obtain the unique segmentation IDs from the updated annotations
    values_list = seg[tuple(abs_pos_f.T.astype(int))]
    v,c = np.unique(values_list[values_list>0], return_counts=True)
    v = v[c>1]

    ## for each segmentation ID, produce if they are unique or not
    ind_unique = np.ones_like(ind)
    ind_unique[np.where(np.isin(ind, v))[0]-1] = 0 

    return  torch.from_numpy(((combined_df_t_idx['worldline_id'].values)[ind-1] + 1) * ind_unique ).long()
    # return ind_unique







############# working example ################
# from load_func import *
# from Seg2graph import *
# import time
# num_nearest = 5 ## should I set as 5 or 10?
# isotropic_voxel_size = [1, 1, 1]
# [t_idx, ch] = [3, 1]
# train_val = 'train'
# folder_path = Path('/Users/hangdeng/Documents/work/Neuron_tracking/ZM9624/')
# img_h5_path = Path(folder_path)/('data.h5')
# seg_h5_path = Path('/Users/hangdeng/Documents/work/Neuron_tracking/seg/'+str(t_idx)+'.h5')
# annotation_path = Path(folder_path)/('annotations.h5')
# seg_ref_h5_path = None


# extractor = NeuronNodesFeatureExtractor(img_h5_path, seg_h5_path, t_idx, ch)
# df_nodes_features = extractor.extract_features()
# if train_val == 'train': 
#     label_extractor = NeuronLabelExtractor( t_idx, annotation_path, seg_h5_path, seg_ref_h5_path)
#     df_nodes_features = label_extractor.match_label_to_worldline_id(df_nodes_features)
# graph_generator = GraphDataGenerator(df_nodes_features, num_nearest, isotropic_voxel_size)
# data2 = graph_generator.produce_graph_data()

