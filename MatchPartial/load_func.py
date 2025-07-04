
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
import os
import time
import re

def load_seg_h5(seg_path):
    with h5py.File(seg_path, 'r') as hdf:
        seg = hdf['label'][:]
    hdf.close()
    return seg


def get_volume_at_frame(file_name,t_idx):
    '''
    Get the 3D original volume at frame t_idx from the h5 file_name
    '''
    with h5py.File(file_name, 'r') as f:
        img_original = np.array(f['data'][t_idx:t_idx+1])
        mask = None
    f.close()
    return img_original,mask 


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
    return combined_df_t_idx


def load_annotations_h5(file_name):
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
    return combined_df
    

def get_abs_pos(combined_df_t_idx,labels_shape):
    '''
    Obtain the absolutely coordinates annotations at the time t_idx with the segmented size of labels_shape from the normalized coordinates
    !!! Important notice: the absolute coordinates starting from 0, then it should be multiplied by (labels_shape-1)
    '''
    pos = np.array(combined_df_t_idx[['z','y','x']])
    abs_pos = (np.round(pos * (np.array(labels_shape)-1))).astype(int)
    return abs_pos



def extract_time_paris(file_path):
    pattern = r'Frame #\d+\s+Parent #\d+\s+Reference #\d+\s+Distance to parent: d=\d+\.\d+'
    matching_lines = []
    with open(file_path, 'r') as file:
        for line in file:
            # Search for lines that match the pattern
            if re.search(pattern, line):
                matching_lines.append(line.strip())

    extracted_pairs = []

    # Regular expression to match frame and parent numbers
    pattern_t = r'Frame #(\d+)\s+Parent #(\d+)'


    for line in matching_lines:
        match = re.search(pattern_t, line)
        if match:
            # Append the extracted numbers as [parent, frame] to the list
            parent = int(match.group(2))  # Parent number
            frame = int(match.group(1))   # Frame number
            extracted_pairs.append([parent, frame])
    return extracted_pairs


def extract_time_paris_reference(file_path):
    pattern = r'Frame #\d+\s+Parent #\d+\s+Reference #\d+\s+Distance to parent: d=\d+\.\d+'
    matching_lines = []
    with open(file_path, 'r') as file:
        for line in file:
            # Search for lines that match the pattern
            if re.search(pattern, line):
                matching_lines.append(line.strip())

    extracted_pairs = []

    # Regular expression to match frame and parent numbers
    pattern_t = r'Frame #(\d+)\s+Parent #(\d+)\s+Reference #(\d+)'


    for line in matching_lines:
        match = re.search(pattern_t, line)
        if match:
            # Append the extracted numbers as [parent, frame] to the list
            parent = int(match.group(3))  # Parent number
            frame = int(match.group(1))   # Frame number
            extracted_pairs.append([parent, frame])
    return extracted_pairs  
    




def get_annotation_file_df(dataset: Path, file_name: str) -> pd.DataFrame:
    """Load and return annotations as an ordered pandas dataframe.
    This should contain the following:
    - t_idx: time index of each annotation
    - x: x-coordinate as a float between (0, 1)
    - y: y-coordinate as a float between (0, 1)
    - z: z-coordinate as a float between (0, 1)
    - worldline_id: track or worldline ID as an integer
    - provenance: scorer or creator of the annotation as a byte string
    """

    with h5py.File(Path(dataset)  / file_name, 'r') as f:
        data = pd.DataFrame()
        for k in f:
            data[k] = f[k]
    return data



def get_annotation_df(dataset: Path) -> pd.DataFrame:
    """Load and return annotations as an ordered pandas dataframe.
    This should contain the following:
    - t_idx: time index of each annotation
    - x: x-coordinate as a float between (0, 1)
    - y: y-coordinate as a float between (0, 1)
    - z: z-coordinate as a float between (0, 1)
    - worldline_id: track or worldline ID as an integer
    - provenance: scorer or creator of the annotation as a byte string
    """
    with h5py.File(dataset / 'annotations.h5', 'r') as f:
        data = pd.DataFrame()
        for k in f:
            data[k] = f[k]
    return data


def save_pandas_h5(save_h5_path, df):
    with h5py.File(save_h5_path, 'w') as hdf:
        for column in df.columns:
            data = df[column].to_numpy()
            if data.dtype == object:
                data = data.astype(h5py.string_dtype())
            hdf.create_dataset(column, data=data)
    hdf.close()

















