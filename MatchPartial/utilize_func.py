import torch
import numpy as np
import random
from tqdm import tqdm
import torch.nn as nn
from sklearn.metrics import confusion_matrix, precision_recall_curve
import torch.nn.functional as F

from .model_sim_EGAT_v2_h8 import FocalLoss
from .load_func import *
from .torchcpd import DeformableRegistration
from .parameters import method, ch, centroid_ref, extension_length
from .mask_head import *

def deformable_registration(X,Y, tolerance, beta, device):
    reg_deformable = DeformableRegistration(X=X, Y=Y, max_iterations=500, tolerance = tolerance, beta = beta, device = device)
    TY, _ = reg_deformable.register()
    return TY



def random_t1_list():
    original_list = list(np.arange(0,1060,100))
    selected_numbers = random.sample(original_list, 7) 
    repeated_list = random.choices(selected_numbers, k=20)
    return np.array(repeated_list)



def get_ground_truth(data1,data2):
    eff_label = (data1.y.unsqueeze(1) + data2.y.unsqueeze(0)) > 0  ## to exclude label=0
    AM = ((data1.y.unsqueeze(1) - data2.y.unsqueeze(0)) == 0)*(eff_label)*1
    return AM

def search_nearby_indices(X,Y,nearby_search_num, tolerance, beta, norm_scale, device):
    norm_scale = torch.from_numpy(norm_scale).to(device)
    TY_normalized = deformable_registration(X/norm_scale ,Y/norm_scale, tolerance, beta, device)
    TY = (TY_normalized * norm_scale).to(torch.float32)

    distance_matrix = torch.cdist(X, TY)
    # distance_matrix = torch.cdist(X * torch.tensor([5,1,1]), TY* torch.tensor([5,1,1])) ### running test
     
    nearest_indices = torch.argsort(distance_matrix, dim=1)[:, :nearby_search_num]
    return nearest_indices, distance_matrix
    

def get_AM_mask(data1,data2,with_AM,nearby_search_num,norm_scale,device, method, mask1 = None, mask2 = None):
    
    ## Method 1: from nearby search
    if method == 1:
        coord = data1.x[:,0:3]
        coords2 = data2.x[:,0:3] 
        distance_matrix = torch.cdist(coord,coords2)
        nearest_indices = torch.argsort(distance_matrix, dim=1)[:, :nearby_search_num]

    
    #### Method 2: from deformation registration
    if  method == 2:
        [tolerance, beta] = [1e-5, 0.5]
        coord = data1.x[:,0:3][:,np.array([2,0,1])]
        coords2 = data2.x[:,0:3][:,np.array([2,0,1])]
        nearest_indices, distance_matrix = search_nearby_indices(coord,coords2,nearby_search_num, tolerance, beta, norm_scale, device)

    #### Method 2: Mask to registration on the head
    if method ==3:
        [tolerance, beta] = [1e-5, 0.5]
        X = data1.x[:,0:3][:,np.array([2,0,1])].to(device) 
        Y = data2.x[:,0:3][:,np.array([2,0,1])].to(device)
        ind1 = torch.from_numpy(mask1[tuple(X[:, 1:3].T.int().cpu().numpy())]).to(torch.bool)
        ind2 = torch.from_numpy(mask2[tuple(Y[:, 1:3].T.int().cpu().numpy())]).to(torch.bool)

        norm_scale = np.array([ 23 * 5, 512 * 1, 512 * 1])  - 1
        norm_scale = torch.from_numpy(norm_scale).to(device)
        reg = DeformableRegistration(X=X[ind1] / norm_scale, Y=Y[ind2] / norm_scale, beta=beta, tolerance=tolerance, device=device)
        maskY_normalized, a = reg.register()
        
        Y_norm = (Y/norm_scale).type(torch.float)  
        TY_normalized = reg.transform_point_cloud(Y_norm)
        TY = (TY_normalized * norm_scale).to(torch.float32)    
        distance_matrix = torch.cdist(X, TY)
        nearest_indices = torch.argsort(distance_matrix, dim=1)[:, :nearby_search_num]
    


    mask = torch.zeros_like(distance_matrix, dtype=bool).to(device)
    for row_idx, col_indices in enumerate(nearest_indices):
        mask[row_idx, col_indices] = 1

    if with_AM:
        AM = get_ground_truth(data1,data2)
        AM_mask = (mask + 1*(AM>0))>0
    else:
        # AM_mask = (torch.tensor(mask))>0
        AM_mask = mask > 0
        
    
    return AM_mask


def get_edge_label(data1,data2,nearby_search_num,norm_scale,with_AM,device, mask1=None, mask2=None):
    AM = get_ground_truth(data1,data2)



    if method ==3:
        
        AM_mask1 = get_AM_mask(data1,data2,with_AM,nearby_search_num,norm_scale,device,method,mask1= mask1,mask2 =mask2) ## only use method 2 for a single direction
        AM_mask2 = get_AM_mask(data2,data1,with_AM,nearby_search_num,norm_scale,device,method,mask1= mask1,mask2 =mask2)
    else:
        AM_mask1 = get_AM_mask(data1,data2,with_AM,nearby_search_num,norm_scale,device,method,mask1= None,mask2 =None) ## only use method 2 for a single direction
        AM_mask2 = get_AM_mask(data2,data1,with_AM,nearby_search_num,norm_scale,device,method,mask1= None,mask2 =None)
    AM_mask = AM_mask1 * AM_mask2.T


    
    AM_mask[data1.y==0] = 0

    # AM_mask[data1.y == 0][:, data2.y == 0] = 0


    if with_AM:
        AM_label = AM_mask + AM
        edge_label = AM_label[AM_mask>0]-1
    else:
        edge_label =  (AM*AM_mask)[AM_mask>0]
    return edge_label


def get_class_weights(edge_label):
    if torch.sum(edge_label).item()>1:
        class_ratio = (len(edge_label)-torch.sum(edge_label).item())/torch.sum(edge_label).item()
        return torch.tensor([1,class_ratio],dtype=torch.float)
    else:
        return torch.tensor([1,len(edge_label)],dtype=torch.float)



#### method 2: produce the pair of t1 and t2 with random t1
def generate_pair_t1_t2(t_end):  
    np.random.seed(42)
    t_list1 = np.arange(0, t_end, 1)
    np.random.shuffle(t_list1)
    np.random.seed(72)
    t_list2 = np.arange(0, t_end, 1)
    np.random.shuffle(t_list2)
    return np.transpose(np.array([t_list1, t_list2]),(1,0))


def compute_measure_matrix(tn, fp, fn, tp):
    acc = (tp+tn) /(tn+fp+fn+tp)
    recall = tp / (tp+fn)
    if (tp+fp)==0:
        precision = 0
    else:
        precision = tp / (tp+fp)
    FPR = fp / (tn+fp)
    if precision==0 and recall ==0:
        f_score = 0
    else:
        f_score = 2*precision*recall/(precision+recall)
    TPR = tp / (tp+fn)
    
    return [acc,precision,recall,FPR,f_score,TPR]


def get_pair_data(graph_path, t1,t2,device):
    
    data1 = torch.load(graph_path/ (str(t1)+'.pt'), weights_only=False)
    data2 = torch.load(graph_path/ (str(t2)+'.pt'), weights_only=False)

   
    return data1.to(device),data2.to(device)






def shuffle_2d_array(arr):
    # Flatten the array, shuffle it, then reshape back to original dimensions
    flattened = arr.flatten()
    np.random.shuffle(flattened)
    return flattened.reshape(arr.shape)







def generate_test_t1_t2(t_end,t_ref_list):
    result_array = np.empty((0, 2), int)
    for t_ref in t_ref_list:
        
        t_list = np.arange(0, t_end, t_end // 30)
        first_column = np.full_like(t_list, t_ref).reshape(-1, 1)  # Create a column of 444
        result = np.concatenate((first_column, t_list.reshape(-1, 1)), axis=1)
        result_array = np.concatenate((result_array, result), axis=0)
    return result_array




        
################################# evaluation ########################################

def eval_model(model,graph_path,test_t1_t2,device,nearby_search_num,norm_scale,with_AM,ratio):
    [loss_all, tn_all, fp_all, fn_all, tp_all] = [0, 0, 0, 0, 0]
    idx = 0
    for t1,t2 in test_t1_t2:
        data1, data2 = get_pair_data(graph_path,int(t1),int(t2),device)


        
        if method ==3:
            # mask1 = produce_head_mask(graph_path.parent/('data.h5'), t1, ch)
            # mask2 = produce_head_mask(graph_path.parent/('data.h5'), t2, ch)
            mask1 = extend_mask_along_central(graph_path.parent/('data.h5'), t1, ch, centroid_ref, extension_length)
            mask2 = extend_mask_along_central(graph_path.parent/('data.h5'), t2, ch, centroid_ref, extension_length)
            model.mask1 = mask1
            model.mask2 = mask2

        else:
            mask1 = None
            mask2 = None
        
        model.eval()
        with torch.no_grad():
            # all_match_scores1 = model.forward_train(data1, data2)
            all_match_scores1 = model(data1, data2)
            

        edge_label1 = get_edge_label(data1,data2,nearby_search_num,norm_scale,with_AM,device,mask1 = mask1, mask2 = mask2)
        class_weights1 = get_class_weights(edge_label1) 
        class_weights1 = (class_weights1 * torch.tensor([1,1])).to(device)
        criteria1 = nn.CrossEntropyLoss(weight = class_weights1,reduction='mean')
        # criteria1 = FocalLoss(alpha=class_weights1, gamma=0)

        loss1 = criteria1(all_match_scores1,edge_label1.to(torch.long))
        loss_all += loss1.item()
        
        all_match_scores_m = torch.argmax(all_match_scores1,dim = 1)
        if torch.sum(edge_label1) + torch.sum( all_match_scores_m) == 0:
            [tn, fp, fn, tp] = [edge_label1.shape[0], 0, 0, 0]
        else:
            tn, fp, fn, tp = confusion_matrix(edge_label1.detach().cpu(), all_match_scores_m.detach().cpu()).ravel()
        
        tn_all += tn
        fp_all += fp
        fn_all += fn 
        tp_all += tp
        idx +=1
    acc,precision,recall,FPR,f_score,TPR  = compute_measure_matrix(tn_all, fp_all, fn_all, tp_all)
    return [loss_all/idx,acc,precision,recall,FPR,f_score,TPR] 
  






