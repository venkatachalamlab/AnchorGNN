


########################################################################################
## This file contains the functions to evaluate the prediction results
# 1. Evaluate the prediction results
# 2. Plot the precision-recall curve
# 3. Visualize the false and true prediction
########################################################################################


import os
import sys
import torch
import numpy as np
import random
import scienceplots
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from tqdm import tqdm
import torch.nn.functional as F
from .utilize_func import get_pair_data, get_ground_truth, get_AM_mask, get_edge_label
from .load_func import get_volume_at_frame, load_annotations_h5_t_idx, load_annotations_h5
import pandas as pd
import h5py
from .mask_head import *
from .parameters import method, ch, extension_length
import time

def visualize_false_true_prediction_napari(data_path, graph_path, t1, t2, threshold, nearby_search_num, norm_scale, with_AM, device, model):

    
    data1, data2 = get_pair_data(graph_path,int(t1),int(t1+random.choice([-1,1])),device)
    coords1 = data1.x[:,0:3]
    coords2 = data2.x[:,0:3]
    edge_label1 = get_edge_label_gt(data1,data2,nearby_search_num,norm_scale,with_AM,device)
    all_match_scores_m, _ = model.forward_link_back(data1, data2, threshold)



    AM_mask1 = model.get_AM_mask(data1,data2,model.with_AM,2)
    AM_mask2 = model.get_AM_mask(data2,data1,model.with_AM,2)
    AM_mask = AM_mask1 * AM_mask2.T
    AM_mask[data1.y==0] = 0
    potential_edge = torch.transpose(torch.stack(torch.where(AM_mask>0)),1,0)


    false_index = (all_match_scores_m != edge_label1) & (all_match_scores_m == 1)
    selected_false =  (coords2[potential_edge[:,1][false_index]]) 

    true_index = (all_match_scores_m == edge_label1) & (all_match_scores_m == 1)
    selected_true =  coords2[potential_edge[:,1][true_index]]


    img_original, _ = get_volume_at_frame(data_path, t2)
    img_plot = img_original[0,1].max(0)

    # plt.figure(dpi=300)
    # plt.imshow(img_plot, 'gray', vmax= 0.1*img_plot.max())
    # plt.scatter(selected_false[:,1],selected_false[:,0],c = 'r',s = 1)
    # plt.scatter(selected_true[:,1],selected_true[:,0],c = 'b',s = 1)
    # plt.show()

    viewer = napari.Viewer()
    viewer.add_image(img_original[0,1])
    selected_index = np.array([2,0,1])

    viewer.add_points(selected_true[:,selected_index] , size=1, border_color='blue',face_color='blue',name = 'true')
    viewer.add_points(selected_false[:,selected_index] , size=1, border_color='red',face_color='red',name = 'false')
    napari.run()




def get_edge_label_gt(data1,data2,nearby_search_num,norm_scale,with_AM,device):
    AM = get_ground_truth(data1,data2)


    AM_mask1 = get_AM_mask(data1,data2,with_AM,nearby_search_num,norm_scale,device,2)
    AM_mask2 = get_AM_mask(data2,data1,with_AM,nearby_search_num,norm_scale,device,2)
    AM_mask = AM_mask1*AM_mask2.T



    
    AM_mask[data1.y==0] = 0


    if with_AM:
        AM_label = AM_mask + AM
        edge_label = AM_label[AM_mask>0]-1
    else:
        edge_label =  (AM*AM_mask)[AM_mask>0]
    return edge_label





def evaluate_prediction_results(t_ref, model, graph_path, method, t_max, nearby_search_num, norm_scale, with_AM, device, threshold, use_nms):
    '''
    method 1: for obtaining the precision - recall curve, to find the best thershold, output the intial score without threshold
    method 2: For acutally prediction used as anchors to put into zephir tracking
    '''
    # for threshold in  np.arange(0.6,1,0.05):
    idx = 0
    edge_label1_tot = torch.empty(0).to(device)
    all_match_scores_m_tot = torch.empty(0).to(device)



    num = 0
    for i in tqdm(range(0,t_max)):
        t1 = t_ref
        t2 = i
        data1, data2 = get_pair_data(graph_path,int(t1),int(t2),device)
        model.eval()
        

        if method == 1:
            edge_label1 = get_edge_label(data1,data2,nearby_search_num,norm_scale,with_AM,device)
            edge_label1_tot = torch.cat((edge_label1_tot,edge_label1))
            with torch.no_grad():
                all_match_scores1 = model(data1, data2)
            all_match_scores_m = F.softmax(all_match_scores1,dim = 1)[:,1]
            all_match_scores_m_tot = torch.cat((all_match_scores_m_tot,all_match_scores_m))


        if method == 2:
            edge_label1 = get_edge_label_gt(data1,data2,nearby_search_num,norm_scale,with_AM,device)
            edge_label1_tot = torch.cat((edge_label1_tot,edge_label1))
            with torch.no_grad():
                all_match_scores_m, _ = model.forward_link_back(data1, data2, threshold, use_nms)
            all_match_scores_m_tot = torch.cat((all_match_scores_m_tot,all_match_scores_m))

        num += 1


    return edge_label1_tot, all_match_scores_m_tot




def Plot_precision_recall(edge_label1_tot, all_match_scores_m_tot):
    
    # edge_label1_tot, all_match_scores_m_tot = evaluate_prediction_results(t_ref, model, graph_path, 1, t_max, nearby_search_num, norm_scale, with_AM, device, 0.5)
    


    y_true = edge_label1_tot.detach().cpu().numpy()
    y_scores = all_match_scores_m_tot.detach().cpu().numpy()


    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)

    # Compute F1 scores for each threshold to find the best
    f1_scores = 2 * (precision * recall) / (precision + recall)
    best_index = f1_scores.argmax()  # index of best F1 score

    # Extract best values
    best_threshold = thresholds[best_index]
    best_precision = precision[best_index]
    best_recall = recall[best_index]
    best_f1 = f1_scores[best_index]


    print(f"Best Threshold: {best_threshold}")
    print(f"Best Precision: {best_precision}")
    print(f"Best Recall: {best_recall}")
    print(f"Best F1 Score: {best_f1}")
    plt.style.use(['science', 'no-latex'])
    plt.figure(figsize=(4, 3), dpi = 300)
    plt.plot(thresholds, precision[:-1], label='precision')
    plt.plot(thresholds, recall[:-1], label='recall')
    # plt.plot(recall[:-1],precision[:-1])
    plt.xlabel('Threshold')
    plt.ylabel('Precision')
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.tight_layout()
    plt.title('Precision and Recall vs threshold') 
    plt.show()





def save_pandas_h5(h5_filename,pandas_df):
    with h5py.File(h5_filename, 'w') as h:
        for k, v in pandas_df.items():
            h.create_dataset(k, data=np.array(v.values))
    h.close()




def compare_df_match_orig(orig_annotation_path,df_matched,norm_scale):
    df_orig = load_annotations_h5(orig_annotation_path)
    df_matched_orig = df_orig.merge(df_matched[['t_idx', 'worldline_id']], on=['t_idx', 'worldline_id'])
    coords_diff = (df_matched[['z','y','x']].values - df_matched_orig[['z','y','x']].values) * norm_scale
    acc = np.sum(np.sum(coords_diff * coords_diff, axis = 1) < 4)/len(df_matched)
    return acc, len(df_matched)
    


def match_pair_df_reference(t_pairs, model, graph_path, method, nearby_search_num, norm_scale, with_AM, device, threshold, use_nms): 
    
    
    idx = 0
    edge_label1_tot = torch.empty(0).to(device)
    all_match_scores_m_tot = torch.empty(0).to(device)
    
    
    df_all =  pd.DataFrame()
    num = 0


    t_pairs = np.array(t_pairs)
    for i in tqdm(range(len(t_pairs))):

        t1, t2 = t_pairs[i]

        
        data1, data2 = get_pair_data(graph_path,int(t1),int(t2),device)
        model.eval()
        

        if method ==3:
            
            img_path = graph_path.parent/('data.h5')

            # model.mask1 = produce_head_mask(img_path, t1, ch)
            # model.mask2 = produce_head_mask(img_path, t2, ch)
            model.mask1 = extend_mask_along_central(img_path, t1, ch, centroid_ref, extension_length)    
            model.mask2 = extend_mask_along_central(img_path, t2, ch, centroid_ref, extension_length)
        with torch.no_grad():
            all_match_scores_m, score_matrix  = model.forward_link_back(data1, data2, threshold, use_nms)
        all_match_scores_m_tot = torch.cat((all_match_scores_m_tot,all_match_scores_m))
        
    
        selected_index = np.array([2,0,1])
        ind = torch.where(score_matrix==1)
        child_coords = data2.x[ind[1]][:,selected_index].detach().cpu().numpy() /norm_scale
        parent_coords = data1.x[ind[0]][:,selected_index].detach().cpu().numpy() /norm_scale
        
        df_t2 = {
                't_idx': t2,
                'z': child_coords[:,0],
                'y': child_coords[:,1],
                'x': child_coords[:,2],
    
            
                'parent_id': t1,
                'parent_coords_z': parent_coords[:,0],
                'parent_coords_y': parent_coords[:,1],
                'parent_coords_x': parent_coords[:,2],
                'worldline_id': data1.y[ind[0]].detach().cpu().numpy() - 1,
                'provenance': b'GNN'
                         }
    
        
        df = pd.DataFrame.from_dict(df_t2)
        df_all = pd.concat([df_all,df])
    
        num += 1
        
    return df_all



def match_pair_df_iou(t_ref_list, model, graph_path, method, t_max, nearby_search_num, norm_scale, with_AM, device, threshold, use_nms): 
    
    
    idx = 0
    edge_label1_tot = torch.empty(0).to(device)
    all_match_scores_m_tot = torch.empty(0).to(device)
    
    
    df_all =  pd.DataFrame()
    num = 0
    
    
    t_list = set(np.arange(0,t_max)) - set(t_ref_list)
    for i in tqdm(t_list):

        
        t2 = i
        df =  pd.DataFrame()
        for t1 in t_ref_list:
            data1, data2 = get_pair_data(graph_path,int(t1),int(t2),device)
            model.eval()
            

            if method ==3:
                img_path = graph_path.parent/('data.h5')

                model.mask1 = extend_mask_along_central(img_path, t1, ch, centroid_ref, extension_length)
                model.mask2 = extend_mask_along_central(img_path, t2, ch, centroid_ref, extension_length)
            
            with torch.no_grad():
                all_match_scores_m, score_matrix  = model.forward_link_back(data1, data2, threshold, use_nms)
        
            
            all_match_scores_m_tot = torch.cat((all_match_scores_m_tot,all_match_scores_m))
        
        
            
        
            
            selected_index = np.array([2,0,1])
            ind = torch.where(score_matrix==1)
            child_coords = data2.x[ind[1]][:,selected_index].detach().cpu().numpy() /norm_scale
            parent_coords = data1.x[ind[0]][:,selected_index].detach().cpu().numpy() /norm_scale
            
            df_t2 = {
                    't_idx': t2,
                    'z': child_coords[:,0],
                    'y': child_coords[:,1],
                    'x': child_coords[:,2],
        
                
                    'parent_id': t1,
                    'parent_coords_z': parent_coords[:,0],
                    'parent_coords_y': parent_coords[:,1],
                    'parent_coords_x': parent_coords[:,2],
                    'worldline_id': data1.y[ind[0]].detach().cpu().numpy() - 1,
                    'provenance': b'GNN'
                             }
        
            
            df_t2 = pd.DataFrame.from_dict(df_t2)
            df = pd.concat([df,df_t2])
    
        df_unique = df.drop_duplicates(subset=['t_idx', 'z', 'y', 'x', 'worldline_id'])
        # df_iou = df_unique[~df_unique['worldline_id'].duplicated(keep=False)]


        ### added by H 11-09-24 
        ### exclude the same predicted worldline_id with different coordinates
        ### exclude the same coordinates with different predicted worldline_id
        df_unique = df_unique[~df_unique['worldline_id'].duplicated(keep=False)]
        df_iou = df_unique[~df_unique[['z', 'y', 'x']].duplicated(keep=False)]

        

        df_all = pd.concat([df_all,df_iou ])
        num += 1
    return df_all


def exclude_nearby_points(data1, data2, selected_index, diff, device):
    
    
    coords = data1.x[:,selected_index]
    dist_matrix = torch.cdist(coords, coords, p=2) 
    nearby_points_mask = (dist_matrix < diff) & (dist_matrix != 0)
    close_point_rows = (nearby_points_mask.sum(dim=1) > 0).nonzero(as_tuple=True)[0]
    
    
    coords = data2.x[:,selected_index]
    dist_matrix = torch.cdist(coords, coords, p=2) 
    nearby_points_mask = (dist_matrix < diff) & (dist_matrix != 0)
    close_point_cols = (nearby_points_mask.sum(dim=1) > 0).nonzero(as_tuple=True)[0]
    
    mask = torch.ones((len(data1.x), len(data2.x)) )
    mask[close_point_rows,:] = 0
    mask[:, close_point_cols] = 0
    return mask.to(device)


def match_pair_df_iou_filter(t_ref_list, model, graph_path, method, t_max, nearby_search_num, norm_scale, with_AM, device, threshold, use_nms): 
    
     
    df_all =  pd.DataFrame()
    num = 0
    
    
    t_list = set(np.arange(0,t_max)) - set(t_ref_list)
    for i in tqdm(t_list):
    # for i in [0]:
        
        t2 = i
        df =  pd.DataFrame()
        
        for t1 in t_ref_list:
            data1, data2 = get_pair_data(graph_path,int(t1),int(t2),device)
            model.eval()
            
           

            if method ==3:
                img_path = graph_path.parent/('data.h5')
                # model.mask1 = produce_head_mask(img_path, t1, ch)
                # model.mask2 = produce_head_mask(img_path, t2, ch)
                model.mask1 = extend_mask_along_central(img_path, t1, ch, centroid_ref, extension_length)
                model.mask2 = extend_mask_along_central(img_path, t2, ch, centroid_ref, extension_length)
            
            with torch.no_grad():
                all_match_scores_m, score_matrix  = model.forward_link_back(data1, data2, threshold, use_nms)
        
            
            # all_match_scores_m_tot = torch.cat((all_match_scores_m_tot,all_match_scores_m))
      

            
            selected_index = np.array([2,0,1])
            mask = exclude_nearby_points(data1, data2, selected_index, 6, device)
            score_matrix = score_matrix * mask
            ind = torch.where(score_matrix==1)
            child_coords = data2.x[ind[1]][:,selected_index].detach().cpu().numpy() /norm_scale
            parent_coords = data1.x[ind[0]][:,selected_index].detach().cpu().numpy() /norm_scale
            
            df_t2 = {
                    't_idx': t2,
                    'z': child_coords[:,0],
                    'y': child_coords[:,1],
                    'x': child_coords[:,2],
        
                
                    'parent_id': t1,
                    'parent_coords_z': parent_coords[:,0],
                    'parent_coords_y': parent_coords[:,1],
                    'parent_coords_x': parent_coords[:,2],
                    'worldline_id': data1.y[ind[0]].detach().cpu().numpy() - 1,
                    'provenance': b'GNN'
                             }
        
            
            df_t2 = pd.DataFrame.from_dict(df_t2)
            df = pd.concat([df,df_t2])
    
        df_unique = df.drop_duplicates(subset=['t_idx', 'z', 'y', 'x', 'worldline_id'])
        # df_iou = df_unique[~df_unique['worldline_id'].duplicated(keep=False)]
        
        ### added by Hang 11-09-24 
        ### exclude the same predicted worldline_id with different coordinates
        ### exclude the same coordinates with different predicted worldline_id
        df_unique = df_unique[~df_unique['worldline_id'].duplicated(keep=False)]
        df_iou = df_unique[~df_unique[['z', 'y', 'x']].duplicated(keep=False)]
    
        df_all = pd.concat([df_all,df_iou ])
        num += 1

    
    return df_all
    

def match_pair_df_reference_filter(t_pairs, model, graph_path, method, nearby_search_num, norm_scale, with_AM, device, threshold, use_nms): 
    

    df_all =  pd.DataFrame()
    num = 0


    t_pairs = np.array(t_pairs)
    for i in tqdm(range(len(t_pairs))):
        t1 = t_pairs[i][0]
        t2 = t_pairs[i][1]

        
        data1, data2 = get_pair_data(graph_path,int(t1),int(t2),device)


        diff_len = abs(len(data2.x) -  len(data1.x))
        if  diff_len > 0 or diff_len==0 :
            model.eval()
            if method ==3:
                img_path = graph_path.parent/('data.h5')
                # model.mask1 = produce_head_mask(img_path, t1, ch)
                # model.mask2 = produce_head_mask(img_path, t2, ch)
                model.mask1 = extend_mask_along_central(img_path, t1, ch, centroid_ref, extension_length)
                model.mask2 = extend_mask_along_central(img_path, t2, ch, centroid_ref, extension_length)
          
            with torch.no_grad():
                all_match_scores_m, score_matrix  = model.forward_link_back(data1, data2, threshold, use_nms)
     
        
            selected_index = np.array([2,0,1])
    
  
            
            ind = torch.where(score_matrix==1)
            child_coords = data2.x[ind[1]][:,selected_index].detach().cpu().numpy() /norm_scale
            parent_coords = data1.x[ind[0]][:,selected_index].detach().cpu().numpy() /norm_scale
            
            df_t2 = {
                    't_idx': t2,
                    'z': child_coords[:,0],
                    'y': child_coords[:,1],
                    'x': child_coords[:,2],
        
                
                    'parent_id': t1,
                    'parent_coords_z': parent_coords[:,0],
                    'parent_coords_y': parent_coords[:,1],
                    'parent_coords_x': parent_coords[:,2],
                    'worldline_id': data1.y[ind[0]].detach().cpu().numpy() - 1,
                    'provenance': b'GNN'
                             }
    
        
            df = pd.DataFrame.from_dict(df_t2)
            df_all = pd.concat([df_all,df])
    
        num += 1
    
    
    return df_all





def matching_coordinates_from_grpound_truth(path_annotation_orig,df_all):
    """
    obatin the ground truth of the coordinates by matching worldline_id
    return matched_df, ground_truth_df
    """
    t_idx_list = np.unique(df_all['t_idx'].values)
    combined_df_match_all =  pd.DataFrame()
    for t_idx in tqdm(t_idx_list):
        df_t_idx = df_all[df_all['t_idx']==t_idx]
        worldline_id_list = df_t_idx['worldline_id'].values


        
        combined_df_t_idx = load_annotations_h5_t_idx(path_annotation_orig,t_idx)
        combined_df_match = combined_df_t_idx[combined_df_t_idx['worldline_id'].isin(worldline_id_list)]
        combined_df_match.loc[:, 'worldline_id'] = pd.Categorical(combined_df_match['worldline_id'], categories=worldline_id_list, ordered=True)
       
        combined_df_match_all = pd.concat([combined_df_match_all,combined_df_match])
    combined_df_match_all_sorted = combined_df_match_all.sort_values(['t_idx', 'worldline_id']).reset_index(drop=True)
    df_all_sorted = df_all.sort_values(['t_idx', 'worldline_id']).reset_index(drop=True)

    return df_all_sorted, combined_df_match_all_sorted



def compute_acc_Euclidean_distance(df_all, combined_df_match_all, norm_scale):
    """
    compute the Euclidean_distance difference from the matching GNN coordinateas and the ground truth
    """
    combined_df_match_all_sorted = combined_df_match_all.sort_values(['t_idx', 'worldline_id']).reset_index(drop=True)
    df_all_sorted = df_all.sort_values(['t_idx', 'worldline_id']).reset_index(drop=True)
    assert len(np.where(df_all_sorted['t_idx']!=(combined_df_match_all_sorted['t_idx'])))>0, print(" The matching has different frame index compared with the ground truth")
    assert len(np.where(df_all_sorted['worldline_id']!=(combined_df_match_all_sorted['worldline_id'])))>0 , print(" The matching has different worldline_id compared with the ground truth")
    coords_match = df_all_sorted[['z','y','x']] * norm_scale
    coords_gt = combined_df_match_all_sorted[['z','y','x']] * norm_scale
    coords_diff = np.sqrt(np.sum((coords_match - coords_gt) ** 2, axis = 1))
    return coords_diff








####################################################################################################################################################
#                                  Producing propagated chain to link multiple reference with one target frame                                     #
####################################################################################################################################################
# t_initial_list = [498, 444, 463]
# sim_tree = np.load('/work/venkatachalamlab/Hang/00Neuron_tracking_version2/01version/tracking/dev_zephir/ZM9624/pdcc_c1.npy')
# mask = np.eye(len(sim_tree))
# sim_tree[mask>0] = np.max(sim_tree)
# t_track = np.arange(t_max)
# df_all = match_chain_of_reference(model, t_track, graph_path, t_initial_list, sim_tree, threshold, use_nms, device)



def compute_link_matrix(model, t1, t2, graph_path, threshold, use_nms,  device):
    data1, data2 = get_pair_data(graph_path,int(t1),int(t2),device)
    model.eval()
    with torch.no_grad():
        all_match_scores_m, score_matrix  = model.forward_link_back(data1, data2, threshold, use_nms)
    return score_matrix

def compute_chain_link_matrix(model, t_chain_list, graph_path,threshold, use_nms, device):
    score_matrix_chain = None
    for i, t1 in  enumerate(t_chain_list[:-1]) : 
        t2 = t_chain_list[i+1]
        score_matrix_12 = compute_link_matrix(model, t1, t2, graph_path,threshold, use_nms,  device)
        if score_matrix_chain is None:
            score_matrix_chain = score_matrix_12.float()
        else:
            score_matrix_chain = score_matrix_chain.float() @ score_matrix_12.float()
    return score_matrix_chain

def propagate_chian_t_ref(model,sim_tree, t_ref, t_target, graph_path,threshold, use_nms, device):

    score_matrix_chain = 0

    t_chain_list0 = [t_ref, t_target]
    t_chain_list1 = [t_ref, np.argmin(sim_tree[t_ref]),t_target]
    t_chain_list2 = [t_ref, np.argmin(sim_tree[t_target]),t_target]
    t_chain_list_all = [ t_chain_list0, t_chain_list1, t_chain_list2]
    for t_chain_list in t_chain_list_all:            
        score_matrix_chain += compute_chain_link_matrix(model, t_chain_list, graph_path,threshold, use_nms, device)
            
    return score_matrix_chain


def match_chain_of_reference(model, t_track, graph_path, t_initial_list, sim_tree, threshold, use_nms, device):

    t_track = list(set(t_track) - set(t_initial_list))

    df_all =  pd.DataFrame()
    for t_target in tqdm(t_track):
        
        df_target =  pd.DataFrame()
        for t_ref in t_initial_list:

            data_ref, data_target= get_pair_data(graph_path,int(t_ref),int(t_target),device)
            score_matrix = propagate_chian_t_ref(model,sim_tree, t_ref, t_target, graph_path,threshold, use_nms, device)
            
            selected_index = np.array([2,0,1])
            ind = torch.where(score_matrix==1)

            child_coords = data_target.x[ind[1]][:,selected_index].detach().cpu().numpy() /norm_scale
            parent_coords = data_ref.x[ind[0]][:,selected_index].detach().cpu().numpy() /norm_scale
            
            
            df_t2 = {
                    't_idx': t_target,
                    'z': child_coords[:,0],
                    'y': child_coords[:,1],
                    'x': child_coords[:,2],
                    'ind':ind[1].detach().cpu().numpy() - 1,
        
                    'parent_id': t_ref,
                    'parent_coords_z': parent_coords[:,0],
                    'parent_coords_y': parent_coords[:,1],
                    'parent_coords_x': parent_coords[:,2],
                    'worldline_id': data_ref.y[ind[0]].detach().cpu().numpy() - 1,
                    'provenance': b'GNN'
                             }

            
            df = pd.DataFrame.from_dict(df_t2)
            df_target = pd.concat([df_target,df])
    
        df_target = df_target.drop_duplicates(subset=['ind', 'worldline_id'], keep='first')
        df_target = df_target[~df_target['worldline_id'].duplicated(keep=False) & ~df_target['ind'].duplicated(keep=False)]
    
        df_all = pd.concat([df_all,df_target])
    
    return df_all



####################################################################################################################################################
#     Compare the similarity betweent the centroid from the reference image descriptor and the one from target                                     #
####################################################################################################################################################


from zephir.models.zephir import ZephIR
from zephir.utils.utils import *

def corr_similarity(prediction, target):
    """Image registration loss, L_R.

    Normalized correlation loss between two lists of volumes (T, N, C, Z, Y, X).
    Loss is calculated over (C, Z, Y, X) axes and averaged over (N) axis. (T) axis
    is not reduced.

    :param prediction: child descriptors
    :param target: target descriptors
    :return: loss
    """

    vx = prediction - torch.mean(prediction, dim=[2, 3, 4, 5], keepdim=True)
    vy = target - torch.mean(target, dim=[2, 3, 4, 5], keepdim=True)
    # child descriptors can sometimes be empty and cause DivByZero error
    if torch.any(torch.std(prediction, dim=[2, 3, 4, 5]) == 0):
        sx = torch.std(prediction, dim=[2, 3, 4, 5])
        for t in range(sx.shape[0]):
            sx[t][sx[t] == 0] = 1
        sxy = torch.mul(torch.std(target, dim=[2, 3, 4, 5]),
                        torch.std(target, dim=[2, 3, 4, 5]))
    else:
        sxy = torch.mul(torch.std(prediction, dim=[2, 3, 4, 5]),
                        torch.std(target, dim=[2, 3, 4, 5]))
    cc = torch.div(torch.mean(torch.mul(vx, vy), dim=[2, 3, 4, 5]), sxy + 1e-5)
    return cc



def build_zephir(args, ZephIR, img_shape, shape_n, n_frame):
    allow_rotation = False
    dimmer_ratio = float(args['--dimmer_ratio'])
    grid_shape = (5,  2 * (int(args['--grid_shape']) // 2) + 1, 2 * (int(args['--grid_shape']) // 2) + 1)
    fovea_sigma = (1, float(args['--fovea_sigma']), float(args['--fovea_sigma']))
    n_chunks = int(args['--n_chunks'])
    n_frame = int(args['--n_frame'])
    grid_spacing = tuple(np.array(grid_shape) / np.array(img_shape))
    
    channel = int(args['--channel'])
    gamma = float(args['--gamma'])
    model_kwargs = {
        'allow_rotation': allow_rotation,
        'dimmer_ratio': dimmer_ratio,
        'fovea_sigma': fovea_sigma,
        'grid_shape': grid_shape,
        'grid_spacing': grid_spacing,
        'n_chunks': n_chunks,
        'n_frame': n_frame,
        'shape_n': shape_n,
        'ftr_ratio': 0.6,
        'ret_stride': 2,
    }
    
    zephir = ZephIR(**model_kwargs)
    return zephir




def extract_img_descriptor(dataset, zephir, xyz_norm, t_idx, gamma, channel, dev):

    data = get_data(dataset, t_idx, g=gamma, c=channel)
    vol = to_tensor(data, n_dim=5, grad=False, dev=dev)
    input_tensor = torch.stack([vol], dim=0)
    zephir.theta.zero_()
    zephir.rho = torch.nn.Parameter(torch.zeros_like(zephir.rho), requires_grad=True)
    
    xyz_anchor = to_tensor(xyz_norm,dev=dev) * 2 - 1
    rho_new = zephir.rho.clone().to(dev)
    rho_new[:1] = rho_new[:1] + xyz_anchor.detach().expand(1, -1, -1)
    zephir.rho = torch.nn.Parameter(rho_new, requires_grad=True)
    
    zephir.to(dev)
    pred = zephir(input_tensor)
    return pred



def compute_corr_sim(dataset, annotation, df_target_t_idx, args, ZephIR, img_shape, dev):
    channel = int(args['--channel'])
    gamma = float(args['--gamma'])
    t_target= np.unique(df_target_t_idx['t_idx'].values)[0]
    xyz_target = df_target_t_idx[['x','y','z']].values

    shape_n = len(xyz_target.reshape(-1,3))
    t_target_list = df_target_t_idx['t_idx'].values

    zephir_target = build_zephir(args, ZephIR, img_shape, shape_n, len(np.unique(t_target_list)))
    target = extract_img_descriptor(dataset, zephir_target, xyz_target, t_target, gamma, channel, dev)

    
    t_ref = np.unique(df_target_t_idx['parent_id'].values)[0]

    df_ref_t_idx = annotation[
        annotation[['t_idx', 'worldline_id']].apply(tuple, axis=1).isin(
            df_target_t_idx[['parent_id', 'worldline_id']].apply(tuple, axis=1)
            )
            ]

    
    xyz_ref = df_ref_t_idx[['x','y','z']].values

    shape_n = len(xyz_ref.reshape(-1,3))
    t_ref_list = df_ref_t_idx['t_idx'].values
    
    zephir_ref = build_zephir(args, ZephIR, img_shape, shape_n, len(np.unique(t_ref_list)))
    ref = extract_img_descriptor(dataset, zephir_ref, xyz_ref, t_ref, gamma, channel, dev)

    return corr_similarity(target, ref).detach().cpu().numpy()



