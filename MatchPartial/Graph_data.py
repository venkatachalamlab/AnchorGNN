import h5py
import torch
import numpy as np
from load_func import *
from Build_graph_seg import *
# from GNN_model import *
# from model import *




def produce_graph_data(df_nodes_features, df_edges_features,nodes_start_indices,nodes_end_indices):
    # selected_nodes_features = ['orientation', 'axis_major_length', 'axis_minor_length',
    #        'area_zmax', 'eccentricity', 'perimeter', 'intensity_mean_2D',
    #        'centroid_2d-0', 'centroid_2d-1', 'axis_ratio', 'centroid-0',
    #        'centroid-1', 'centroid-2', 'volume', 'intensity_mean_3D', 'slice_depth']
    selected_nodes_features = [
            'centroid_2d-0', 'centroid_2d-1', 'centroid-0',
            'axis_major_length', 'axis_minor_length',
             'axis_ratio', 
           # 'area_zmax', 
        'eccentricity', 
        # 'perimeter',
           # 'intensity_mean_2D',
           'centroid-1', 'centroid-2', 
            # 'volume', 
            # 'intensity_mean_3D',
            'slice_depth',
            # 'orientation'
            ]
    selected_edges_features = ['euclidean_vector-0', 'euclidean_vector-1','euclidean_vector-2','euclidean_abs_dist', 'pair_orientation']
    node_features = torch.tensor(np.array(df_nodes_features[selected_nodes_features].values), dtype=torch.float)
    edge_index = torch.tensor([nodes_start_indices, nodes_end_indices], dtype=torch.long)
    edge_features = torch.tensor(np.array(df_edges_features[selected_edges_features].values), dtype=torch.float)
    
    # node_features[:,0:3] = node_features[:,0:3]/torch.tensor([512,512,23])
    # node_features[:,5:7] = node_features[:,7:8]/torch.tensor([512,512])
    # node_features[:,7:8] = node_features[:,7:8]/23
    
    
    data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_features)
    data.y = torch.tensor(np.array(df_nodes_features['worldline_id']))
    return data


def produce_data_graph(seg,img_orig,annotation_df,num_nearest):
    df_nodes_features = produce_nodes_features(seg,img_orig,annotation_df)
    df_edges_features,nodes_start_indices,nodes_end_indices = produce_edges_features(df_nodes_features, num_nearest)
    data = produce_graph_data(df_nodes_features, df_edges_features,nodes_start_indices,nodes_end_indices)
    return data



def get_data_generator(t_idx, ch, num_nearest):
    '''
    num_nearest: search for the neareast 5 points
    '''
    img_h5_path = '../Segmentation/Dataset/ZM9624/data.h5'
    img_original,_=get_volume_at_frame(img_h5_path,t_idx)
    img_orig = img_original[0,ch]

    seg_h5_path = 'seg/'+str(t_idx)+'.h5'
    f = h5py.File(seg_h5_path, 'r')
    seg = f['label'][:]
    f.close()

    folder_path = '../Segmentation/Dataset/ZM9624/'
    annotation_path = folder_path + 'annotations.h5'
    annotation_df = annotation_seg_ID(annotation_path,t_idx,seg)

    data = produce_data_graph(seg,img_orig,annotation_df,num_nearest)
    return data

