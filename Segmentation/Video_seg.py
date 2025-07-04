

#####################################################################################################################################################################
                                # Produce Segmentation for Jin's datasets # 
#####################################################################################################################################################################

from __future__ import print_function, unicode_literals, absolute_import, division
import sys
import numpy as np
import matplotlib
matplotlib.rcParams["image.interpolation"] = 'none'
import matplotlib.pyplot as plt
import time
import os
import re
import h5py
import copy
from glob import glob
from tqdm import tqdm
from tifffile import imread
from csbdeep.utils import Path, normalize
from stardist import fill_label_holes, random_label_cmap, calculate_extents, gputools_available
from stardist import Rays_GoldenSpiral
from stardist.matching import matching, matching_dataset
from stardist.models import Config3D, StarDist3D, StarDistData3D
from sklearn.cluster import DBSCAN
from .reconstruction_3D_neuron import *

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'MatchPartial')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from MatchPartial.parameters import seg_mask, folder_path, ch, gamma, seg_weights_filename, zoom_factor, train_val, seg_path, t_initial_list, t_max, t_track
from MatchPartial.mask_head import *

from multiprocessing import Pool
from functools import partial
from tqdm import tqdm
import concurrent.futures


np.random.seed(42)
lbl_cmap = random_label_cmap()

def save_var_h5(save_path_h5,var_list,name_list):
    assert len(var_list) > 0, 'variable list is empty'
    assert len(name_list) > 0, 'name list is empty'
    assert len(var_list) == len(name_list), 'variable and name list should has the same length'
    
    with h5py.File(save_path_h5, 'w') as hf:
        for i in range(len(var_list)):
            hf.create_dataset(name_list[i], data = var_list[i])
    hf.close()
    
def get_datasets_foldername(folder_path):
    folder_list = []
    for root, dirs, files in os.walk(folder_path):
        for dir in dirs:
            folder_list.append(os.path.join(root, dir) + '/')
    return folder_list


def plot_img_label(img, lbl, img_title="image (XY slice)", lbl_title="label (XY slice)", z=None, **kwargs):
    if z is None:
        z = img.shape[0] // 2    
    fig, (ai,al) = plt.subplots(1,2, figsize=(12,5), gridspec_kw=dict(width_ratios=(1.25,1)))
    im = ai.imshow(img[z], cmap='gray', clim=(0,1))
    ai.set_title(img_title)    
    fig.colorbar(im, ax=ai)
    al.imshow(lbl[z], cmap=lbl_cmap)
    al.set_title(lbl_title)
    plt.tight_layout()
    
    


def load_model_3D_and_2D(model_weights_path):
    # if X[0].ndim == 3 else X[0].shape[-1]
    n_channel = 1 
    # anisotropy=(5.333333333333333, 1.1428571428571428, 1.0)
    anisotropy=(5.0, 1.0, 1.0)
    n_rays = 96
    # Use OpenCL-based computations for data generator during training (requires 'gputools')
    use_gpu = False and gputools_available()
    use_gpu = True
    print("use_gpu: ", use_gpu)
    # Predict on subsampled grid for increased efficiency and larger field of view
    grid = tuple(1 if a > 1.5 else 2 for a in anisotropy)
    rays = Rays_GoldenSpiral(n_rays, anisotropy=anisotropy)

    conf = Config3D (
        rays             = rays,
        grid             = grid,
        anisotropy       = anisotropy,
        use_gpu          = use_gpu,
        n_channel_in     = n_channel,
        # adjust for your data below (make patch size as large as possible)
        train_patch_size = (8,128,128),
        train_batch_size = 6,
    )


    model = StarDist3D(conf, name='stardist', basedir='Segmentation/models')
    # model.load_weights('/work/venkatachalamlab/Hang/Segmentation/models/stardist/weights_best_15stack_ZM9624.h5')
    model.load_weights(model_weights_path)
    StarDist2D.from_pretrained()
    model_2D = StarDist2D.from_pretrained('2D_versatile_fluo')
    return model, model_2D


def get_mask_from_proj_pred(model_2D,img_zoom):
    '''
    must img_zoom is 3D image of (23*1024*1024)
    Get maximum projection of z on the zoomed imaged
    return the stardist 2d model prediction as mask to remove the nuts
    '''
    img_proj =  np.max(img_zoom,axis = 0)
    img_proj_norm = normalize(img_proj)
    img_proj_pred,_ = model_2D.predict_instances(img_proj_norm)
    return img_proj_pred







def get_frame_segmentation(model,model_2D,t_idx, ch, zoom_factor,folder_path,gamma, seg_mask):
    img_original,_ = get_volume_at_frame(os.path.join(folder_path,'data.h5'),t_idx)
    # if seg_mask:
    #     img_hist = np.sum(img_original[0, ch], axis=(0, 2))
    #     smoothed_hist = moving_average(img_hist, window_size=20)
    #     range_min, range_max = range_of_interest(img_hist, window_size=20, threshold=0.1)
    #     mask_tails = np.zeros(img_original[0, ch, 0].shape)
    #     range_max = min(range_max + 5, mask_tails.shape[0] - 1)
    #     range_min = max(range_min - 5, 0)
    #     mask_tails[range_min:range_max,:] = 1 
    #     img_original = img_original * (mask_tails[None,None,None])
    if seg_mask:
        # img_plot = np.max(img_original[0,ch],axis = (0))
        # # centroid_ref = np.array([256, 256])
        # centroid_ref = np.array(img_original[0,ch].shape[-2:])//2
        # coords, head, tail = find_head(img_plot,centroid_ref)
        # filled_mask = mask_head(head, img_original.shape[-2:])


        filled_mask = extend_mask_along_central(Path(folder_path)/('data.h5'), t_idx, ch, centroid_ref, extension_length=30)
            
        img_original = img_original * (filled_mask[None,None,None])
        
    
    
    img_zoom = scipy.ndimage.zoom(img_original[0,ch],[1,zoom_factor,zoom_factor],order = 0)
    # img_zoom = 255*img_zoom/np.max(img_zoom) ## commented for freely behaving worms
    if gamma is not None:
        img_zoom = 255 * (img_zoom.astype(np.float32)/255)**gamma
    img_norm = normalize(img_zoom, 1,99.8)
    img_proj_pred = get_mask_from_proj_pred(model_2D,img_zoom)
    pred, _ = model.predict_instances(img_norm*((img_proj_pred>0)), n_tiles=model._guess_n_tiles(img_norm), show_tile_progress=False)  ## for freely behaving worms
    # pred, _ = model.predict_instances(img_norm, n_tiles=model._guess_n_tiles(img_norm), show_tile_progress=False)
    label_z = scipy.ndimage.zoom(pred,[1,1/zoom_factor,1/zoom_factor],order = 0)
    # print("label_z",np.unique(label_z))
    return label_z



def get_img_shape(folder_path):
    hf = h5py.File(os.path.join(folder_path,'data.h5'), 'r')
    img_shape = hf['data'][:].shape
    hf.close()
    img_shape = np.array(img_shape)
    img_shape[1] = 1
    return img_shape 


def load_label_h5file(file_path):
    hf = h5py.File(file_path, 'r')
    label_z = hf['label'][:]
    hf.close()
    return label_z


def merge_each_frame_segmentation(folder_list):
    for folder_path in folder_list:
        img_shape = get_img_shape(folder_path)
        img_shape[1] = 0 ### only save for one channel
        label = np.zeros(img_shape)    
        # for t_idx in range(2):    
        for t_idx in tqdm(range(img_shape[0])):
            label_z = load_label_h5file('seg/' + str(t_idx)+'.h5')
            label[t_idx,0] = label_z
            
        save_var_h5('seg/'+'label.h5',[label],['label'])
        



################################################################################################   
                        #the following code works with no parallel computing #
################################################################################################  



current_dir = Path(__file__).parent.resolve()
model_weights_path = current_dir / 'model_weights' / seg_weights_filename
model, model_2D = load_model_3D_and_2D(model_weights_path)
print(folder_path)
img_shape = get_img_shape(folder_path)


print("ch, zoom_factor",ch, zoom_factor)
seg_path.mkdir(parents=True, exist_ok=True)  # Create the directory if it doesn't exist



# filenames = os.listdir(Path('/work/venkatachalamlab/Hang/00Neuron_tracking_version2/01version/dataset/anchor_eval/seg_mask'))
# numbers = []
# for filename in filenames:
#     numbers.extend(map(int, re.findall(r'\d+', filename)))
# t_list = list(set(np.arange(t_max)) - set(numbers))
# len(t_list)





    
################################################################################################   
                        #loop for each frame #
################################################################################################   


# for t_idx in tqdm(t_track):
#     print(t_idx)
#     label_z = get_frame_segmentation(model,model_2D,t_idx, ch, zoom_factor,folder_path,gamma,seg_mask)
#     # seg_path = folder_path / 'seg'
#     save_var_h5(seg_path / (str(t_idx)+'.h5') ,[label_z],['label'])
# print("finish the segmentation of movie in the folder path: ", folder_path)








################################################################################################   
    
    
    
    
################################################################################################   
                        #the following code works with parallel computing #
################################################################################################   





def process_iteration(t_idx,  model, model_2D, ch, zoom_factor, load_folder, save_folder, gamma, seg_mask, train_val):
    print(t_idx)
    # if train_val == 'train':
    #     label_z = get_frame_segmentation(model, model_2D, t_idx, ch, zoom_factor, load_folder, gamma, seg_mask)
    # else:
    label_z = get_frame_segmentation(model, model_2D, t_idx, ch, zoom_factor, load_folder, gamma, seg_mask)
    print(label_z.shape)

    save_var_h5(save_folder / (str(t_idx)+'.h5'), [label_z], ['label'])
    
    # save_var_h5(seg_path / (str(t_idx)+'.h5') ,[label_z],['label'])
    print("finish saving at "+str(t_idx))
    return t_idx  # Return t_idx to track progress






if __name__ == "__main__":

    img_shape = get_img_shape(folder_path)
    [ch, zoom_factor] = [int(ch),zoom_factor]
    
    # model_weights_path = os.path.abspath(os.path.join(os.getcwd(), 'model_weights', seg_weights_filename))
    current_dir = Path(__file__).parent.resolve()
    model_weights_path = current_dir / 'model_weights' / seg_weights_filename
    model, model_2D = load_model_3D_and_2D(model_weights_path)
    save_folder = Path(seg_path)
    save_folder.mkdir(parents=True, exist_ok=True) 

    
    filenames = os.listdir(Path(save_folder))
    numbers = []
    for filename in filenames:
        numbers.extend(map(int, re.findall(r'\d+', filename)))
    # t_list = list(set(np.arange(t_max)) - set(numbers))
    t_list = list(set(t_track) - set(numbers))

    
    chunk_size = 10
    chunks = [t_list[i:i + chunk_size] for i in range(0, len(t_list), chunk_size)]


    
    # for t_chunk in chunks:
    #     with concurrent.futures.ThreadPoolExecutor() as executor:
    #         process_partial = partial(process_iteration,  model=model, model_2D=model_2D, ch=ch, zoom_factor=zoom_factor, 
    #                                   load_folder=folder_path, save_folder=save_folder, gamma=gamma, seg_mask=seg_mask,train_val=train_val)
    #         # tqdm_parallel = tqdm(range(start_idx, end_idx, 1))
    #         tqdm_parallel = tqdm(t_chunk)
            
    #         results = list(executor.map(process_partial, tqdm_parallel))
    
    #     # Print the results (t_idx values)
    #     print("Processed t_idx values:", results)


    # for t_chunk in chunks:
    #     process_partial = partial(
    #         process_iteration,
    #         model=model,
    #         model_2D=model_2D,
    #         ch=ch,
    #         zoom_factor=zoom_factor,
    #         load_folder=folder_path,
    #         save_folder=save_folder,
    #         gamma=gamma,
    #         seg_mask=seg_mask,
    #         train_val=train_val
    #     )
    
    #     with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    #         futures = [executor.submit(process_partial, t_idx) for t_idx in t_chunk]
    #         results = []
    #         for f in tqdm(concurrent.futures.as_completed(futures), total=len(t_chunk)):
    #             results.append(f.result())
    
    #     print("Processed t_idx values:", results)




    for t_idx in tqdm(t_list):
        print(t_idx)
        label_z = get_frame_segmentation(model,model_2D,t_idx, ch, zoom_factor,folder_path,gamma,seg_mask)
        # seg_path = folder_path / 'seg'
        save_var_h5(seg_path / (str(t_idx)+'.h5') ,[label_z],['label'])
    print("finish the segmentation of movie in the folder path: ", folder_path)



        
###############################################################################################       
    
    
 
    
    
    
    
    
    
################################################################################################   
                #the following module is to visualize the segmentation in the paraview (.vtk)#
################################################################################################     
    

import h5py
import numpy as np
import vtk
from vtk.util import numpy_support
from scipy.ndimage import gaussian_filter



def conver_h5_vtk(hdf5_path,vtk_path,parameters):
    [dx, dy, dz, x0, y0, z0] = [1,1,3,0,0,0]
    file = h5py.File(hdf5_path, 'r')
    data = file['data'][...]
    label = file['label'][...]
    VTK_data = numpy_support.numpy_to_vtk(num_array=data.ravel(), deep=True, array_type=vtk.VTK_FLOAT)


    imgData = vtk.vtkImageData()
    imgData.SetDimensions(data.shape[-1], data.shape[-2], data.shape[-3])
    imgData.SetSpacing(dx, dy, dz)
    imgData.SetOrigin(x0, y0, z0)


    imgData.GetPointData().SetScalars(VTK_data)

    VTK_label = numpy_support.numpy_to_vtk(num_array=label.ravel(), deep=True, array_type=vtk.VTK_INT)
    imgData.GetPointData().AddArray(VTK_label)
    VTK_label.SetName("label")
    
    writer = vtk.vtkStructuredPointsWriter()
    writer.SetFileName(vtk_path)
    writer.SetInputData(imgData)
    writer.Write()

    print("done")
    
    
    
# save_var_h5(folder_path+'neuron20.h5',[img_norm, pred3],['data','label'])
# parameters = [1,1,3,0,0,0]
# conver_h5_vtk(folder_path+'neuron20.h5',folder_path+'neuron20.vtk',parameters)    
################################################################################################      
    
    
    
    
    
    
    
