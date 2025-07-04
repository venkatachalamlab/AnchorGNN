import h5py 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import h5py
import numpy as np
from csbdeep.utils import normalize
import numpy as np
from .torchcpd import DeformableRegistration
from scipy.spatial import cKDTree


#####################################################################################################################
##                          Load the data and preprocess the data                                                 ###
#####################################################################################################################

def get_volume_at_frame_ch(file_name, t_idx, ch):
    '''
    Get the 3D original volume at frame t_idx from the h5 file_name
    '''
    with h5py.File(file_name, 'r') as f:
        img_original = np.array(f['data'][t_idx,ch])
    return img_original


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
    combined_df_t_idx = combined_df[combined_df['t_idx'] == t_idx]
    ## sort the dataframe by the the column of 'worldline_id'
    combined_df_t_idx = combined_df_t_idx.sort_values(by=['worldline_id'])
    return combined_df_t_idx
    

def get_abs_pos_t_idx(file_name,t_idx,labels_shape):
    '''
    Obtain the absolutely coordinates annotations at the time t_idx with the segmented size of labels_shape from the normalized coordinates
    !!! Important notice: the absolute coordinates starting from 0, then it should be multiplied by (labels_shape-1)
    '''
    combined_df_t_idx = load_annotations_h5_t_idx(file_name,t_idx)
    pos = np.array(combined_df_t_idx[['z','y','x']])
    abs_pos = (np.round(pos * (np.array(labels_shape)-1))).astype(int)
    return abs_pos



########################################################################################################################
##                                          Registration                                                             ###
########################################################################################################################

def visualize_registration(X,Y,TY,matched_pairs,Dist_diff,img1_orig,img2_orig):
    fig, ax = plt.subplots(1,5, figsize=(25,5))
    ax[0].imshow(img1_orig.max(0), vmax = 0.1*img1_orig.max(), cmap='gray')
    ax[0].scatter(Y[:,2],Y[:,1],c='b',s=1)
    # ax[0].scatter(TY[:,2],TY[:,1],c='r',s=1)
    ax[0].set_title("moving source Y")

    ax[1].imshow(img2_orig.max(0), vmax = 0.1*img2_orig.max(), cmap = 'gray')
    ax[1].scatter(X[:,2],X[:,1],c='g',s=1)
    ax[1].scatter(TY[:,2],TY[:,1],c='r',s=1)
    ax[1].set_title("alignment TY with fixed target X")

    idx = 1
    ax[2].imshow(img1_orig.max(0), vmax = 0.1*img1_orig.max(), cmap='gray')
    # ax[2].scatter(Y[:,2],Y[:,1], c='g', s = 0.2)
    # ax[2].scatter(TY[:,2],TY[:,1], c='b', s = 0.2)
    # for i,j in matched_pairs[idx:idx+1]:
    for i,j in matched_pairs:
        ax[2].plot([Y[j, 2], TY[i, 2]], [Y[j, 1], TY[i, 1]], 'r-', lw=1) 
        ax[2].plot( Y[j,2], Y[j,1], 'go', markersize = 1) 
        ax[2].plot( TY[i,2], TY[i,1], 'bo', markersize = 1) 


    ax[3].hist(Dist_diff, bins=100)
    ax[3].set_xlabel('Distance difference')
    ax[3].set_ylabel('Frequency')
    ax[3].set_title('Histogram of distance difference between TY and X')  

    ax[4].imshow(img1_orig.max(0), vmax = 0.1*img1_orig.max(), cmap='gray')
    ind = np.where(Dist_diff > 10)
    for i in ind[0]:
        j = matched_pairs[i][1]  
        ax[4].plot([Y[j, 2], TY[i, 2]], [Y[j, 1], TY[i, 1]], 'r-', lw=1) 
        ax[4].plot( Y[j,2], Y[j,1], 'go', markersize = 5) 
        ax[4].plot( TY[i,2], TY[i,1], 'bo', markersize = 5) 
    ax[4].set_title("Distance between matched pairs that are greater than 10 pixels")

    plt.show()


def deformable_registration(X,Y, tolerance, beta, device):
    reg_deformable = DeformableRegistration(X=X, Y=Y, max_iterations=500, tolerance = tolerance, beta = beta, device = device)
    TY, _ = reg_deformable.register()
    return TY


def match_pairs_index(X,TY,isotropic_scaling): 
    tree = cKDTree(X * isotropic_scaling) 
    distances, indices = tree.query(TY * isotropic_scaling)
    matched_pairs = list(zip(range(len(TY)), indices))
    return matched_pairs




