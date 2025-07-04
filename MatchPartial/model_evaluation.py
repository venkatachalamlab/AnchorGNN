

import sys
from pathlib import Path
import numpy as np
import time
import os
conda_env = os.environ.get('CONDA_DEFAULT_ENV')
print(f"Current conda environment: {conda_env}")
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import os
import shutil
import json
import csv
import pandas as pd
from scipy.io import loadmat
import cv2
import torch
import sys
torch.cuda.is_available()
from zephir.main import run_zephir
from zephir.utils.io import load_checkpoint, update_checkpoint
from zephir.methods.build_pdists import get_all_pdists
from zephir.__version__ import __version__
print("zephir version: ",__version__)
from datetime import datetime
from IPython.display import clear_output
from MatchPartial.parameters import *
from MatchPartial.load_func import *
from MatchPartial.utilize_func import *
from MatchPartial.model_sim_EGAT_v2_test import *
from MatchPartial.eval_prediction_func import *





def compute_link_matrix(model, t1, t2, graph_path, threshold, use_nms,  device):
    data1, data2 = get_pair_data(graph_path,int(t1),int(t2),device)
    model.eval()
    with torch.no_grad():
        all_match_scores_m, score_matrix = model.forward_link_back(data1, data2, threshold, use_nms)
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



def similarity_t_pairs(dataset,shape_t, channel,t_track,t_initial_list):
    d_full = get_all_pdists(dataset, shape_t, channel, pbar=True)
    t_track_actual = np.array(list(set(t_track) -set(t_initial_list)))
    a = np.array(t_initial_list)[np.argmin(d_full[:,np.array(t_initial_list)], axis = 1)]
    result = np.stack((a[np.array(t_track_actual)], np.array(t_track_actual)), axis=1)
    return result



def compute_acc_dist_num(dataset_orig, df_tracked, t_track, norm_scale):
    df_orig = get_annotation_file_df(dataset_orig,'annotations_orig.h5')
    df_tracked = df_tracked[df_tracked['t_idx'].isin(t_track)]
    df_tracked = df_tracked.sort_values(['t_idx', 'worldline_id']).reset_index(drop=True)
    df_result = df_orig[
    df_orig[['t_idx', 'worldline_id']].apply(tuple, axis=1).isin(
        df_tracked[['t_idx', 'worldline_id']].apply(tuple, axis=1)
        )
        ]
    df_matched_compare = df_result.sort_values(['t_idx', 'worldline_id']).reset_index(drop=True)    
    diff = (df_tracked[['z','y','x']].values - df_matched_compare[['z','y','x']].values) * norm_scale
    diff_scale = np.sqrt(np.sum(diff* diff, axis = 1))
    acc = round(np.sum(diff_scale < 4) / len(diff_scale), 3)
    num_matched = len(diff_scale)
    # print("accuracy", round(np.sum(diff_scale < 4) / len(diff_scale), 3), round(np.mean(diff_scale), 3), len(diff))
    return acc, num_matched


##########################################################################################################################
##                           Make initial annotations here                                                              ##
##########################################################################################################################

dataset = Path(dataset_path)
print("starting tracking", flush=True)
print(dataset_path, flush=True)
# with open(str(dataset / 'args.json')) as json_file:
#     args = json.load(json_file)
print("the args:", args, flush=True) 


##########################################################################################################################
##                           Make initial annotations here                                                              ##
##########################################################################################################################
    
t0 = time.time()
annotation_orig = get_annotation_file_df(folder_path, "annotations_orig.h5") 
annotation = pd.DataFrame()
for t_initial in t_initial_list:
    df = annotation_orig[annotation_orig['t_idx']==t_initial] 
    annotation = pd.concat([annotation,df], ignore_index=True)
key_list = annotation.keys().tolist()
key_list.remove('id')
print('initial annotated frames', t_initial_list)
##########################################################################################################################



##########################################################################################################################
##      If the partial annotations are saved as annotations.h5 already                                            ##
##########################################################################################################################
    
# annotation = get_annotation_df(dataset) 
##########################################################################################################################

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NodeLevelGNN(nearby_search_num, norm_scale, with_AM, device).to(device)
model.load_state_dict(torch.load(weights_path)['model'])
print(weights_path)





# ######################## Method 1: using all t_refs  to match all cross-link
t_pairs = similarity_t_pairs(dataset,shape_t, channel,t_track,t_initial_list)
print(t_pairs)
print("obtain the most similary reference frame")
match_acc_list = []
for threshold in tqdm(np.arange(0.5,1,0.05)):
# for threshold in [0.8]:
    
    df_all = match_pair_df_reference(t_pairs, model, graph_path, method, nearby_search_num, norm_scale, with_AM, device, threshold, use_nms)
    df_matched = df_all[['parent_id', 'provenance', 't_idx', 'worldline_id', 'x', 'y','z']]
    df_matched = df_matched.sort_values(['t_idx', 'worldline_id']).reset_index(drop=True)
    acc, num_matched = compute_acc_dist_num(dataset_path, df_matched, t_track, norm_scale)
    match_acc_list.append( [threshold, acc, num_matched] )
    print("acc, num_matched",acc, num_matched)
    print("match_acc_list",match_acc_list)
match_acc_list = np.array(match_acc_list)
df1 = pd.DataFrame(match_acc_list, columns=['threshold','acc','num_matched'])
df1.to_hdf(Path(dataset)/'match_acc_list1.h5', key='data', mode='w')
df1['matched_ave'] = df1['num_matched']/(len(t_pairs)-3)
print(df1)





######################## Method 2: using single referecen frame to match all 
print("must specify t_ref in the args as only one reference frame")
t_pairs = np.column_stack((np.full(t_max, 444 ), np.arange(t_max)))
match_acc_list = []
for threshold in tqdm(np.arange(0.5,1,0.05)):
    df_all = match_pair_df_reference(t_pairs, model, graph_path, method, nearby_search_num, norm_scale, with_AM, device, threshold, use_nms)
    df_matched = df_all[['parent_id', 'provenance', 't_idx', 'worldline_id', 'x', 'y','z']]
    df_matched = df_matched.sort_values(['t_idx', 'worldline_id']).reset_index(drop=True)
    acc, num_matched = compute_acc_dist_num(dataset_path, df_matched, t_track, norm_scale)
    match_acc_list.append( [threshold, acc, num_matched] )
    print("acc, num_matched",acc, num_matched)
    print("match_acc_list",match_acc_list)
match_acc_list = np.array(match_acc_list)
df2 = pd.DataFrame(match_acc_list, columns=['threshold','acc','num_matched'])
df2.to_hdf(Path(dataset)/'match_acc_list2.h5', key='data', mode='w')
df2['matched_ave'] = df2['num_matched']/(len(t_pairs)-3)
print(df2)




######################## Method 3: using all referecen frame to match all 
print("original annotated frames for link: ", t_initial_list )
match_acc_list = []
for threshold in tqdm(np.arange(0.5,1,0.05)):
    df_all = match_pair_df_iou(t_initial_list, model, graph_path, method, t_max, nearby_search_num, norm_scale, with_AM, device, threshold, use_nms)
    df_matched = df_all[['parent_id', 'provenance', 't_idx', 'worldline_id', 'x', 'y','z']]
    df_matched = df_matched.sort_values(['t_idx', 'worldline_id']).reset_index(drop=True)
    acc, num_matched = compute_acc_dist_num(dataset_path, df_matched, t_track, norm_scale)
    match_acc_list.append( [threshold, acc, num_matched] )
    print("acc, num_matched",acc, num_matched)
    print("match_acc_list",match_acc_list)
match_acc_list = np.array(match_acc_list)
df3 = pd.DataFrame(match_acc_list, columns=['threshold','acc','num_matched'])
df3.to_hdf(Path(dataset)/'match_acc_list3.h5', key='data', mode='w')
df3['matched_ave'] = df3['num_matched']/(len(t_pairs)-3)
print(df3)











































