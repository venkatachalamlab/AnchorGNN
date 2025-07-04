
from pathlib import Path
import json
import os
import numpy as np
from pathlib import Path
import torch


# match_method = list(map(int, os.environ.get('MATCH_METHOD', '').split()))
# # match_method = list(map(int, os.environ.get('MATCH_METHOD', '').split(',')))
# folder_path = os.environ.get('FOLDER_PATH', '')
# folder_path = Path(folder_path)
# print("loading environment variables")
# print("match_method:", match_method, flush=True)
# print("folder_path:", folder_path, flush=True)


print("loading environment variables")
print("match_method:", match_method, flush=True)
print("folder_path:", folder_path, flush=True)
print("threshold",threshold)  



## segmentation parameters
train_val = 'train' ## if 'train',obtaine the neurons label 
seg_mask = True
gamma = None
seg_weights_filename = 'weights_best_42stacks_all.h5'
zoom_factor = 2
# centroid_ref = np.array([256, 256]) ### the centroid of the c.elegans head


num_nearest = 5 
isotropy_scale  = (5,1,1)
normalize_lim = (3,99.8) #(1,99.5)
zoom_factor = 2
isotropic_voxel_size = [1, 1, 1]






with open(Path(folder_path)/'metadata.json') as f:
    config = json.load(f)
t_max = config['shape_t']
norm_scale = np.array([config['shape_z'],config['shape_y'],config['shape_x']]) - 1



img_h5_path = Path(folder_path)/('data.h5')
if seg_mask:
    seg_path =  Path(folder_path)/('seg_mask')
else:
    seg_path =  Path(folder_path)/('seg')
save_graph_folder = Path(folder_path)/('graph')


##########################################################################################################################################
######################################################################## evaluation ######################################################
dataset_path = folder_path
file_name = 'annotations.h5' ## must be this can't change
t_ref_gnn = 444

f = open(folder_path/'args.json',) 
args = json.load(f) 
t_initial_list = list(map(int, args['--t_ref'].split(',')))
ch = int(args['--channel'])
use_GNN = True
use_nms = True


if use_GNN:
    args["--t_ref"] = None
    with open(folder_path / 'args.json', 'w') as f:
        json.dump(args, f, indent=4)
    





nearby_search_num = 5
with_AM = False
# graph_path = Path(folder_path)/('graph')


f = open(folder_path/'metadata.json',) 
metadata = json.load(f) 
norm_scale = np.array( [metadata['shape_z'], metadata['shape_y'], metadata['shape_x']]) - 1 ##'[z,y,x]'
method = 2 ## method of locating the neareast nodes
t_max = metadata['shape_t'] ## metadata['shape_t']
metadata['t_ref'] = t_initial_list

    
    


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')





    
    
    
    
    