


import numpy as np
import cv2
from sklearn.cluster import DBSCAN
from .load_func import get_volume_at_frame
from .parameters import centroid_ref, extension_length
from skimage import measure
from skimage.morphology import skeletonize, medial_axis, thin
from scipy.ndimage import distance_transform_edt
from skimage.feature import peak_local_max
from skimage.draw import polygon
import shutil
from pathlib import Path
import h5py
from tqdm import tqdm

def find_head(image, centroid_ref):
    image = (image/np.max(image) * 255).astype(np.uint8)
    coords = np.column_stack(np.where(image > 0.01*(image.max())))
    clustering = DBSCAN(eps=10, min_samples=10).fit(coords)

    labels = clustering.labels_
    unique_labels = set(labels)
    head, tail = None, None
    centroid = coords.mean(axis=0)

    closest_index = np.argmin(np.linalg.norm(coords - centroid, axis=1))
    if len(centroid) == 3:
        closest_index2 = np.argmin(np.linalg.norm(coords - np.array([centroid[0],centroid_ref[0],centroid_ref[1]]), axis=1))
    else:
        closest_index2 = np.argmin(np.linalg.norm(coords - np.array(centroid_ref),axis=1))
    head_label = labels[np.array([closest_index2, closest_index])]
    head = coords[np.isin(labels,head_label)]
    return coords, head, tail



def mask_head(head, img_plot_shape):
    mask = np.zeros((img_plot_shape), dtype=np.uint8)
    for y, x in head:
        cv2.circle(mask, (x, y), radius=10, color=1, thickness=-1)  # Filled circle
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled_mask = np.zeros_like(mask)
    cv2.drawContours(filled_mask, contours, -1, color=1, thickness=cv2.FILLED)
    return filled_mask


def produce_head_mask(img_path, t_idx, ch, centroid_ref):
    img_original,mask = get_volume_at_frame(img_path, t_idx)
    img_plot = np.max(img_original[0,ch],axis = (0))
    
    coords, head, tail = find_head(img_plot,centroid_ref)
    filled_mask = mask_head(head, img_original.shape[-2:])
    return filled_mask,img_plot




def mask_central_line(filled_mask, img_plot):
    distance = distance_transform_edt(filled_mask* img_plot) 
    central_line = peak_local_max(distance,
                                footprint=np.ones((3, 3)),
                                )
    coords = np.array(central_line)
    coords = coords[coords[:, 1].argsort()]
    x_coords = coords[:, 1]
    y_coords = coords[:, 0]
    polynomial = np.polyfit(x_coords, y_coords, 
                            deg=2
                            ) 
    fitted_curve = np.poly1d(polynomial)
    y_curve = fitted_curve(x_coords)
    return fitted_curve, x_coords



# def extend_mask_along_central(img_path, t_idx, ch, centroid_ref, extension_length):
#     filled_mask,img_plot = produce_head_mask(img_path, t_idx, ch, centroid_ref)
#     return filled_mask




def extend_mask_along_central(img_path, t_idx, ch, centroid_ref, extension_length):
    filled_mask,img_plot = produce_head_mask(img_path, t_idx, ch, centroid_ref)
    if extension_length ==0:
        return filled_mask
        
    fitted_curve, x_coords = mask_central_line(filled_mask, img_plot)
    x_min, x_max = min(x_coords), max(x_coords)

    x_extended = np.arange(x_min - extension_length, x_max + extension_length, 1)
    y_extended = fitted_curve(x_extended)
    y_extended = np.clip(y_extended, 0, filled_mask.shape[0] - 1)
    x_extended = np.clip(x_extended, 0, filled_mask.shape[1] - 1)


    half_widths = []
    for x, y in zip(x_coords, fitted_curve(x_coords)):
        # Get the mask values perpendicular to the central line
        y_range = np.arange(int(y) - 100, int(y) + 100)  # Assume max width is 100 pixels
        mask_values = filled_mask[np.clip(y_range, 0, filled_mask.shape[0] - 1).astype(int), int(x)]
        # Find the width
        indices = np.where(mask_values > 0)[0]
        if len(indices) > 0:
            half_width = (indices[-1] - indices[0]) // 2
        else:
            half_width = 0
        half_widths.append(half_width)

    half_widths = np.array(half_widths)



    extended_mask = np.zeros_like(filled_mask, dtype=np.uint8)

    for i, (x, y) in enumerate(zip(x_extended, y_extended)):
        if i < len(half_widths):  # Use the corresponding width from the original
            half_width = half_widths[i]
        else:  # For extrapolated points, use the last known width
            half_width = half_widths[-1]

        # Define the perpendicular region
        y_start = int(y - half_width)
        y_end = int(y + half_width)
        y_start = max(0, y_start)
        y_end = min(filled_mask.shape[0] - 1, y_end)

        # Fill the mask in the extended region
        extended_mask[y_start:y_end, int(x)] = 1
    final_mask = np.maximum(filled_mask, extended_mask)
    return final_mask



def mask_original_datasets_roi(img_path, ch, centroid_ref, extension_length):
    """
    copy the original datasets as 'data_orig.h5'
    apply the roi mask on the datasets, save as img_path to cover the original datasets
    """

    shutil.copy(Path(img_path) , img_path.parent / 'data_orig.h5' )
    with h5py.File(img_path, 'r+') as f:
        img = f['data']  
        times = np.array(f['times'])  # Load times if needed
        for t_idx in tqdm(range(len(times))):
            final_mask = extend_mask_along_central(img_path, t_idx, ch, centroid_ref, extension_length)
            img_original = img[t_idx:t_idx+1, ...]
            img_masked = img_original * final_mask[None, None, None, :]
            img[t_idx:t_idx+1, ...] = img_masked
    
        img_data = np.array(img)  
    
    with h5py.File(img_path.parent / 'data.h5', 'w') as f:
        
        f.create_dataset('data', data=img_data)
        f.create_dataset('times', data=times)
        # logging.debug("New HDF5 file created with 'data' and 'times' datasets.")
        print("Apply mask to exclude the extra tails. Done!")
    f.close()

