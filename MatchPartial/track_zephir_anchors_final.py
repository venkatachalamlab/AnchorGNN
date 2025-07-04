

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
import copy
torch.cuda.is_available()
from zephir.main import run_zephir
from zephir.utils.io import load_checkpoint, update_checkpoint
from zephir.methods.build_pdists import get_all_pdists
from zephir.__version__ import __version__
print("zephir version: ",__version__)
from datetime import datetime
from IPython.display import clear_output
from MatchPartial.parameters import *
from MatchPartial.parameters import weights_path, shape_t, channel,t_track,t_initial_list,dataset_path
from MatchPartial.load_func import *
from MatchPartial.utilize_func import *
from MatchPartial.model_sim_EGAT_v2_h8 import *
from MatchPartial.eval_prediction_func import *


def similarity_t_pairs(dataset,shape_t, channel,t_track,t_initial_list):
    d_full = get_all_pdists(dataset, shape_t, channel, pbar=True)
    t_track_actual = np.array(list(set(t_track) -set(t_initial_list)))
    a = np.array(t_initial_list)[np.argmin(d_full[:,np.array(t_initial_list)], axis = 1)]
    result = np.stack((a[np.array(t_track_actual)], np.array(t_track_actual)), axis=1)
    return result


    
def compute_acc_dist(dataset_orig, df_tracked, t_track, norm_scale):
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
    print("accuracy", round(np.sum(diff_scale < 4) / len(diff_scale), 3), round(np.mean(diff_scale), 3), len(diff))




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
annotation_manual = copy.deepcopy(annotation)
key_list = annotation.keys().tolist()
key_list.remove('id')
print("save initial annotations in the current folder",Path(dataset_path)/file_name)
##########################################################################################################################



##########################################################################################################################
##      If the partial annotations are saved as annotations.h5 already                                            ##
##########################################################################################################################
    
# annotation = get_annotation_df(dataset) 
##########################################################################################################################





if match_method is not None:
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    model = NodeLevelGNN(nearby_search_num, norm_scale, with_AM, device).to(device)
    print("weights_path",weights_path)
    model.load_state_dict(torch.load(weights_path, map_location='cpu')['model']) 


    
    
    
    ######################## Method 1: using all t_refs  to match all 
    if match_method == [1] or match_method  == [1,4]:
        t_pairs = similarity_t_pairs(dataset,shape_t, channel,t_track,t_initial_list)
        print("obtain the most similary reference frame")
        df_all = match_pair_df_reference_filter(t_pairs, model, graph_path, method, nearby_search_num, norm_scale, with_AM, device, threshold, use_nms)
        df_matched = df_all[['parent_id', 'provenance', 't_idx', 'worldline_id', 'x', 'y','z']]
        df_matched = df_matched.sort_values(['t_idx', 'worldline_id']).reset_index(drop=True)
    
    
    
    ######################## Method 2: using single referecen frame to match all 
    if match_method == [2] or match_method  == [2,4]:
        print("must specify t_ref in the args as only one reference frame")
        t_pairs = np.column_stack((np.full(t_max, int(args['--t_ref'])), np.arange(t_max)))
        df_all = match_pair_df_reference(t_pairs, model, graph_path, method, nearby_search_num, norm_scale, with_AM, device, threshold, use_nms)
        df_matched = df_all[['parent_id', 'provenance', 't_idx', 'worldline_id', 'x', 'y','z']]
        df_matched = df_matched.sort_values(['t_idx', 'worldline_id']).reset_index(drop=True)
    
    
    
    if match_method == [3] or match_method  == [3,4]:
        print("original annotated frames for link: ", t_initial_list )
        df_all = match_pair_df_iou(t_initial_list, model, graph_path, method, t_max, nearby_search_num, norm_scale, with_AM, device, threshold, use_nms)
        df_matched = df_all[['parent_id', 'provenance', 't_idx', 'worldline_id', 'x', 'y','z']]
        df_matched = df_matched.sort_values(['t_idx', 'worldline_id']).reset_index(drop=True)
    
    
    
    
    if  4  in match_method :
        dataset_orig  = Path('/work/venkatachalamlab/Hang/00Neuron_tracking_version2/01version/dataset/ZM9624/')
        df_orig = get_annotation_file_df(dataset_orig,'annotations_orig.h5')
        df_result = df_orig[
        df_orig[['t_idx', 'worldline_id']].apply(tuple, axis=1).isin(
            df_matched[['t_idx', 'worldline_id']].apply(tuple, axis=1)
            )
            ]
        df_matched = df_result.sort_values(['t_idx', 'worldline_id']).reset_index(drop=True)


    if match_method == [0]:
        df_matched = pd.DataFrame()





print("annotation",annotation)
print("df_matched",df_matched)

if df_matched is not None and not df_matched.empty:
    annotation = pd.concat([annotation,df_matched ])
    save_pandas_h5(Path(dataset_path)/'anchor_GNN.h5', annotation)
    
    for t_idx in tqdm(t_track):
        for t_ref in t_initial_list:                                           
            df_target_t_idx = annotation[(annotation['t_idx']==t_idx) & (annotation['parent_id']==t_ref)]
            if len(df_target_t_idx) > 0:
                sim = compute_corr_sim(Path(dataset_path), annotation, df_target_t_idx, args, ZephIR, img_shape, device)
                mask = (annotation['t_idx']==t_idx) & (annotation['parent_id']==t_ref)
                annotation.loc[mask, 'corr_sim'] = sim.flatten()[:mask.sum()]
    selected_keys = ['id', 'parent_id', 'provenance', 't_idx', 'worldline_id', 'x', 'y', 'z']
    annotation_filtered = annotation[annotation['corr_sim']>sim_threshold][selected_keys] 
    annotation_filtered = pd.concat([annotation_filtered, annotation_manual])
    annotation_filtered = annotation_filtered.drop_duplicates()
    save_pandas_h5(Path(dataset_path)/file_name, annotation_filtered)
    print("save partial annotations in the current folder",Path(dataset_path)/file_name)
    print("annotation_filtered",len(annotation_filtered)/len(t_track))

    if (dataset_path / 'annotations_orig.h5').exists():
        compute_acc_dist(Path(dataset_path), annotation_filtered, t_track, norm_scale)







run_zephir(dataset=dataset, args=args)


df_tracked = load_annotations_h5(Path(dataset_path)/'coordinates.h5')
t_track = np.unique(df_tracked['t_idx'].values)
if (dataset_path / 'annotations_orig.h5').exists():
    compute_acc_dist(Path(dataset_path), df_tracked, [0], norm_scale)
    compute_acc_dist(Path(dataset_path), df_tracked, t_track, norm_scale)






