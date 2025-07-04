#! python
#
# Copyright 2021
# Author: Maedeh Seyedolmohadesin
from typing import Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib
from tqdm import tqdm

from pathlib import Path
import csv
import json
import pandas as pd
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Type, TypeVar
from dataclasses import dataclass, asdict, field
import pandas as pd
import numpy as np
import h5py
from math import floor
from matplotlib.backends.backend_pdf import PdfPages
# from annotations import load_annotations, save_annotations
from .io import load_flow_file, get_times, get_slice, AnnotationTable, WorldlineTable
# from transform import idx_from_coords


    

def load_annotations(dataset: Optional[Path] = None,
                     annotations_filename: str = "annotations.h5",
                     worldlines_filename: str = "worldlines.h5"
                    # ):
                     ) -> Tuple[AnnotationTable, WorldlineTable]:

    if dataset is None:
        dataset = Path(".")

    annotation_file = dataset / annotations_filename

    if annotation_file.exists():
        annotations = AnnotationTable.from_hdf(annotation_file)
    else:
        annotations = AnnotationTable()

    worldline_file = dataset / worldlines_filename
    if worldline_file.exists():
        worldlines = WorldlineTable.from_hdf(worldline_file)
    else:
        worldlines = WorldlineTable.from_annotations(annotations)

    return (annotations, worldlines)



def save_annotations(annotations: AnnotationTable,
                     worldlines: WorldlineTable,
                     dataset: Path = None,
                     annotations_filename: str = "annotations.h5",
                     worldlines_filename: str = "worldlines.h5") -> None:

    if dataset is None:
        dataset = Path(".")

    annotations.to_hdf(dataset / annotations_filename)
    worldlines.to_hdf(dataset / worldlines_filename)


def load_flow_file(flow_file_path):
    flow, index = [], []

    if os.path.exists(flow_file_path):
        with open(flow_file_path, "r") as f:
            data = csv.reader(f)
            for row in data:
                flow.append(row[1])
                index.append(int(row[0]))
    else:
        flow = ["none"]
        index = [0]

    return flow, index


def get_times() -> np.ndarray:
    """These data have two separate timestamps for red and green channels. We
    will return the *later* of the two, for causality purposes. This way, the
    data from time T was acquired strictly before timestamp T."""

    hdf1 = dataset_path / "data_camera1.h5"
    hdf2 = dataset_path / "data_camera2.h5"

    if os.path.isfile(hdf1) and os.path.isfile(hdf2):
        f1 = h5py.File(hdf1, 'r')
        f2 = h5py.File(hdf2, 'r')
        all_times = np.stack([f1["times"], f2["times"]])
        latest_times = all_times.max(axis=0)
        return latest_times

    elif os.path.isfile(hdf1):
        f1 = h5py.File(hdf1, 'r')
        return f1["times"]

    else:
        f2 = h5py.File(hdf2, 'r')
        return f2["times"]


def idx_from_coords(coords: tuple, shape: tuple) -> tuple:
    return tuple((_idx_from_coord(c, s) for (c, s) in zip(coords, shape)))


def extract_traces(dataset, 
                   annotation_path=None, 
                   provenance=None, 
                   t_ref=None, 
                   r=(1, 2, 2) , 
                   n_pixel=15, 
                   trace_filename=None, 
                   W_ids=None,
                   color=False,
                   high_res_color=False,
                   save_csv=False):
    
    # get some info about data and annotations
    if annotation_path is None:
        annotation_path = dataset
        
    if trace_filename is None:
        trace_filename="traces.npy"
        
    (A, W) = load_annotations(annotation_path)
    shape = get_slice(dataset, 1)[1].shape
    times = get_times(dataset)

    #save traces in folder traces
    trace_file_path = os.path.join(dataset, "traces")
    if not os.path.exists(trace_file_path):
        os.mkdir(trace_file_path)
    trace_file = os.path.join(trace_file_path, trace_filename)
    
    # filter by provenance
    if provenance is not None:
        A = A.filter(lambda x: x["provenance"] == provenance)
        
    # Construct W_ids list and neuron names
    if W_ids is None:
        W_ids = list(np.arange(W.df.shape[0]))
    names = list(W.df.loc[W_ids,:].name)
    names = [name.decode() for name in names]
    
    # extract traces for each timepoint and all the neurons
    if color:
        t_iter = range(1,len(times))
        if high_res_color:
            t_iter = range(2,len(times))

    else:
        t_iter = range(len(times))
       
    traces = np.zeros((len(W_ids), len(t_iter)))
    
    for t_idx, t in enumerate(tqdm(t_iter)):
        
        if t_ref is None:
            A_t = A.filter(lambda x: x["t_idx"] == t)
        else:
            A_t = A.filter(lambda x: x["t_idx"] == t_ref)
            
        volume = get_slice(dataset,t)[1]
        
        for idx, w_id in enumerate(W_ids):
            a = A_t.filter(lambda x: x["worldline_id"] == w_id).get_first()
            c = idx_from_coords((a.z, a.y, a.x), shape)
            
            lo = np.asarray([c[i]-r[i] for i in range(3)])
            hi = np.asarray([c[i]+r[i]+1 for i in range(3)])
            lo[lo<0] = 0
            
            v = volume[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
            traces[idx, t_idx] = np.mean(np.sort(v.flatten())[-n_pixel:])
            if np.isnan(traces[idx, t_idx]):
                print(idx, t)
    # save traces
    np.save(trace_file, traces)
    csv_traces = None
    if save_csv:
        traces_t = np.transpose(traces)
        times = np.expand_dims(times, axis=1)
        if color:
            times = times[1:] 
        times -= times[0,0]
        csv_traces = np.concatenate((times, traces_t), axis=1)
        header="time(s),"
        for name in names:
            header = header + name + ","
        np.savetxt(dataset/"traces/traces.csv", csv_traces, delimiter=",", header=header, comments='')
        
        
    return traces, csv_traces



def get_slice(t: int) -> np.ndarray:
    
    binary1 = dataset_path / "data_camera1"
    binary2 = dataset_path / "data_camera2"
    metadata_file = dataset_path/ "metadata.json"


    with open(metadata_file) as f:
        metadata = json.load(f)

    dtype = metadata["dtype"]
    z = metadata["shape_z"]
    y = metadata["shape_y"]
    x = metadata["shape_x"]
    
    if os.path.isfile(binary1) and os.path.isfile(binary2):
        img1 = np.reshape(np.fromfile(binary1, dtype=dtype, count=z*y*x,
                                      offset=t*z*y*x), (z,y,x), order='C')
        img2 = np.reshape(np.fromfile(binary2, dtype=dtype, count=z*y*x,
                                      offset=t*z*y*x), (z,y,x), order='C')
        return np.stack([img1, img2])
    elif os.path.isfile(binary1):
        img1 = np.reshape(np.fromfile(binary1, dtype=dtype, count=z*y*x,
                                  offset=t*z*y*x), (z,y,x), order='C')
        return img1
    else:
        img2 = np.reshape(np.fromfile(binary2, dtype=dtype, count=z*y*x,
                                  offset=t*z*y*x), (z,y,x), order='C')
        return img2

def _idx_from_coord(coord: float, shape: int) -> int:
    return max(floor(coord*shape - 1e-6), 0)
    
def idx_from_coords(coords: tuple, shape: tuple) -> tuple:
    return tuple((_idx_from_coord(c, s) for (c, s) in zip(coords, shape)))


def plot(dataset, 
         traces_file=None, 
         flow_file=None, 
         W_ids=None, 
         moving_average=False, 
         colors_dict=None, 
         sorted_names=False,
         n_per_panel=10,
         plot_name=None,
        plot_title=None,
        color=False):
    
    if plot_name is None:
        plot_name="traces.jpg"
    
    if traces_file is None:
        traces_file = dataset / "traces/traces.npy"
        
    if flow_file is None:
        flow_file = dataset/"flow_file.txt"
        
    if plot_title is None:
        plot_title = dataset
        
    plot_file = os.path.join(os.path.split(traces_file)[0], plot_name)

    _, W = load_annotations(dataset)
    if W_ids is None:
        W_ids = W.df.id
    names = list(W.df.loc[W_ids,:].name)
    names = [name.decode() for name in names]

    traces = np.load(traces_file)
    times = get_times(dataset)
    times -= times[0]
    if color:
        trace_times = times[1:]
        trace_times -= trace_times[0]
    else:
        trace_times = np.copy(times)
    flow, index = load_flow_file(flow_file)

    if colors_dict==None:
        colors_dict = {
            "Diacetyl": "#ABEBC6",
            "Butanone": "#F5B7B1",
            "200mM NaCl": "#FAD7A0",
            "Isoamyl alcohol": "#C39BD3",
            "0.01mM Fluorescein" : "#A9CCE3"
        }

    if sorted_names:
        sorted_names = sorted(names)
        sorted_ids = [W_ids[names.index(name)] for name in sorted_names]
        names = sorted_names
        W_ids = sorted_ids
        
    # constructing subplots
    n_panels = int(np.ceil(len(W_ids)/n_per_panel))
    plt.suptitle(plot_title, weight="bold", fontsize=18)
    xlim = len(times)
    plt.figure(figsize=(3*n_per_panel, 2*n_per_panel), dpi=500)
    plt.subplots_adjust(wspace =0.5, hspace=.75)
    for idx, W_id in enumerate(W_ids):
        plt.subplot(n_per_panel, n_panels, idx+1)
        trace = traces[W_id, :]
        ylim0 = np.min(trace)
        ylim1 = np.max(trace)
        
        for i in np.arange(2, len(index)-2, 3):
            plt.fill([times[index[i]],times[index[i+1]],times[index[i+1]],times[index[i]]],[ylim0,ylim0,ylim1,ylim1], 
                     colors_dict[flow[i]], label=flow[i])
        
        plt.plot(trace_times,trace, color='black', linewidth=.75)
        plt.title(names[idx])
        plt.xlabel("Time(s)")
        
        if idx == 0:
            plt.legend(fontsize="xx-large", loc='upper left', bbox_to_anchor=(-1, 1))
    
#     for n1 in range(n_panels):
#         plt.subplot(1,n_panels,n+1)

#         plt.yticks([])
#         plt.xticks([])
        
#         #fill horizontally 
#         for i in range(n_per_panel):
#             if i%2 ==0:
#                 plt.fill([0,xlim,xlim,0],[i,i,i+1,i+1], color= "#f2f2f2")
#             if i%2 ==1:
#                 plt.fill([0,xlim,xlim,0],[i,i,i+1,i+1], color= "#d9d9d9")
                
#         #fill vertically based on odors
#         for i in np.arange(2, len(index)-2, 3):
#             plt.fill([index[i],index[i+1],index[i+1],index[i]],[0,0,n_per_panel,n_per_panel], 
#                      colors_dict[flow[i]], label=flow[i])
            
#         if n == 0:
#             plt.legend(fontsize="xx-large", loc='upper center', bbox_to_anchor=(-.75, 1))

#         for idx, w_id in enumerate(w_ids[n*n_per_panel:(n+1)*n_per_panel]):
#             avg = traces[w_id, :] 

#             if moving_average:
#                 df = pd.DataFrame({'trace':avg})
#                 avg =  df["trace"].rolling(window=3).mean()

#             avg = np.asarray(avg)
#             avg = (avg -np.nanmin(avg))/(np.nanmax(avg) - np.nanmin(avg))
# #             avg = (avg - np.nanmean(avg))/(10*np.nanstd(avg))
# #             avg = avg/10
#             avg = avg  + n_per_panel - idx - 1
#             plt.plot(times, avg, color = 'black', linewidth = 1.5)
# #             plt.text(-100, n_per_panel-idx-1 , names[n*n_per_panel+idx], fontsize = 20, weight = 'bold', ha = "right")

#             plt.text(-100, n_per_panel-idx-1 , str(w_id)+"-"+names[n*n_per_panel+idx], fontsize = 20, weight = 'bold', ha = "right")
    plt.savefig(plot_file, bbox_inches = "tight", pad_inches = .5)
    plt.show()


def extract_traces_df(dataset, 
                annotation_path=None, 
                provenance=None,  
                r=(1, 2, 2) , 
                n_pixel=15, 
                trace_filename=None, 
                W_ids=None,
                color=False,
                high_res_color=False,
                t_iter=None):
        
    if trace_filename is None:
        trace_filename="traces.csv"
    
    print('Loading annotations ...')
    (A, W) = load_annotations(dataset, annotation_filename)
    for row in W:
        w = row
        if row.name.decode() == 'null':
            row.name = str(row.id).encode()
            W.df.loc[row.id, 'name'] = row.name
        
    # save_annotations(A, W, annotation_path)
    shape = get_slice_comb(dataset, 1)[1].shape
    times = get_times_comb(dataset)
    
    #save traces in folder traces
    trace_file_path = os.path.join(dataset, "traces")
    if not os.path.exists(trace_file_path):
        os.mkdir(trace_file_path)
    trace_file = os.path.join(trace_file_path, trace_filename)
    
    # filter by provenance
    if provenance is not None:
        A = A.filter(lambda x: x["provenance"] == provenance)
        
    # Construct W_ids list and neuron names
    if W_ids is None:
        W_ids = list(np.arange(W.df.shape[0]))
    names = list(W.df.loc[W_ids,:].name)
    # names = [name.decode() for name in names]
    names = [name for name in names]
    W_ids
    # extract traces for each timepoint and all the neurons
    
    if t_iter is None:
        if color:
            t_iter = range(1,len(times))
            if high_res_color:
                t_iter = range(2,len(times))
    
        else:
            t_iter = range(len(times))
    
    
    print('Extracting traces ...')    
    traces = np.zeros((len(W_ids), len(t_iter)))
    for t_idx, t in  enumerate(tqdm(t_iter)):
        time = times[t]
        A_t = A.filter(lambda x: x["t_idx"] == t)
    
        volume = get_slice_comb(dataset,t)[1]
    
        
        for idx, w_id in enumerate(W_ids):
            
            # a = A_t.filter(lambda x: x["worldline_id"] == w_id).get_first()
            filtered = A_t.df[A_t.df["worldline_id"].apply(lambda x: int(x) == int(w_id))]
            if len(filtered) == 0:
                print(f"worldline_id {w_id} not found")
                continue
            a = A_t.object_from_row(filtered.iloc[[0]])
            c = idx_from_coords((a.z, a.y, a.x), shape)
            
            lo = np.asarray([c[i]-r[i] for i in range(3)])
            hi = np.asarray([c[i]+r[i]+1 for i in range(3)])
            lo[lo<0] = 0
            
            v = volume[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
            traces[idx, t_idx] = np.mean(np.sort(v.flatten())[-n_pixel:])
            
            if np.isnan( traces[idx, t_idx]):
                print(idx, t)
    
    
    
    # save traces
    df = pd.DataFrame()
    df['worldline_id'] = W_ids
    df['name'] = names
    df = df.astype({'worldline_id': 'int32'}).copy()
    df_trace = pd.DataFrame(traces, columns=times[t_iter])
    df_final = pd.concat((df, df_trace), axis=1)
    df_final.to_csv(dataset/'traces'/trace_filename) 

    return df_final



################################################################################################################################
                                                    # Visualize
################################################################################################################################
def plot_from_df(dataset, 
    traces_file=None, 
    flow_file=None, 
    names=None, 
    update_names=False,
    moving_average=False, 
    colors_dict=None, 
    sorted_names=False,
    n_per_panel=10,
    plot_name=None,
    plot_title=None,
    save_obj=None):
    
    if plot_name is None:
        plot_name="traces.jpg"
    
    if traces_file is None:
        traces_file = dataset / "traces/traces.csv"
        
    if flow_file is None:
        flow_file = dataset/"flow_file.txt"
        
    if plot_title is None:
        plot_title = dataset
        
    plot_file = os.path.join(os.path.split(traces_file)[0], plot_name)

    print('Loading traces from {}'.format(traces_file))
    df = pd.read_csv(traces_file, index_col=0)

    #update track names
    if update_names:
        A, W = load_annotations(dataset)
        df_W_ids = df['worldline_id']
        # all_W_ids = list(np.arange(W.df.shape[0]))
        W_names = list(W.df.loc[df_W_ids,:].name)
        W_names = [name.decode() for name in W_names]

        df['name'] = W_names
        df.to_csv(traces_file)
    
    if sorted_names:
        df = df.sort_values(by='name')
        
    if names is None:
        names = np.array(df['name'])
    else:
        #only names exist is df
        names = [n for n in names if n in np.array(df['name'])]

    
    
    splitted_p = str(dataset).split('\\')
    if len(splitted_p) == 1:
        splitted_p = str(dataset).split('/')
    legend_title = '{}-{}-{}'.format(splitted_p[-3], splitted_p[-4], splitted_p[-2])
    

    time_cols = np.array(df.columns[2:])
    time_inds = np.arange(len(time_cols))
    t_inds = np.arange(0, len(time_cols), 200)
#     trace_times = times.astype('float64')
#     trace_times = trace_times/60 - trace_times[0]/60 #convert to min
#     trace_times = trace_times.astype('int')
    
    flow, index = load_flow_file(flow_file)
    index = np.array(index)-2 #trace starts at time 0

    if colors_dict==None:
        colors_dict = {
            "100mM NaCl": "#F5B7B1",
            "10mM CuSO4": "#D2B4DE",
            "1uM ascr#3": "#A9CCE3",
            "e-6 IAA": "#A3E4D7",
            "Fluorescein": "#ABEBC6",
            "OP50": "#FAD7A0",
            "450mM NaCl": "#EDBB99",
            "800mM Sorbitol": "#AED6F1",
            "Control": "#CCD1D1",
            "e-2 IAA": "#F9E79F"
        }

    # constructing subplots
    
    if n_per_panel == 'auto':
        n_per_panel = np.ceil(len(names)/2)
    n_panels = int(np.ceil(len(names)/n_per_panel))
    plt.figure(figsize=(10* n_panels, 2.5*n_per_panel), dpi=100)
    plt.subplots_adjust(wspace =0.2, hspace=.7)
#     plt.suptitle(plot_title, weight="bold", fontsize=18)
    for idx, name in enumerate(names):
#     for idx, row in df.iterrows():
        plt.subplot(int(n_per_panel), n_panels, idx+1)
        df_W  = df[df['name']==name]
        trace = np.array(df_W[time_cols])[0]

        ylim0 = np.nanmin(trace[30:])
        ylim1 = np.nanmax(trace[30:])
#         if ylim1 < 1:
#             ylim1 = 1
        
        for i in np.arange(2, len(index)-2, 3):
            plt.fill([time_inds[index[i]],time_inds[index[i+1]],time_inds[index[i+1]],time_inds[index[i]]],[ylim0,ylim0,ylim1,ylim1], 
                    colors_dict[flow[i]], label=flow[i])
        
        plt.plot(time_inds,trace, color='black', linewidth=1)
        plt.title(name, weight='bold')
        
        plt.xticks(time_inds[t_inds])
        plt.xlabel("frame")
        plt.xlim(0, len(time_cols))
        if not np.isnan(ylim0):
            plt.ylim(ylim0, ylim1)
        
        if idx == 0:
            plt.legend(fontsize="xx-large", loc='upper right', bbox_to_anchor=(-.2, 0),
                    title=legend_title, title_fontsize = "xx-large")
    
    if save_obj is None:
        plt.savefig(plot_file, bbox_inches = "tight", pad_inches = .5)
    else:
        save_obj.savefig(bbox_inches = "tight", pad_inches = .5)
    plt.show()
    plt.close()




def plot_from_multiple_dfs(
    dataset,
    traces_files=None,  # list of paths
    colors_list=None,   # list of color strings for each CSV
    names=None,
    update_names=False,
    moving_average=False,
    colors_dict=None,
    sorted_names=False,
    n_per_panel=10,
    plot_name=None,
    plot_title=None,
    save_obj=None):

    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    if traces_files is None:
        traces_files = [dataset / "traces/traces_orig.csv", dataset / "traces/traces_anchor.csv"]

    if colors_list is None:
        colors_list = ['black', 'blue']

    if plot_name is None:
        plot_name = "traces_compare.jpg"

    flow_file = dataset / "flow_file.txt"
    if plot_title is None:
        plot_title = str(dataset)

    plot_file = os.path.join(os.path.split(traces_files[0])[0], plot_name)

    dfs = []
    for tf in traces_files:
        print(f"Loading traces from {tf}")
        df = pd.read_csv(tf, index_col=0)
        if update_names:
            A, W = load_annotations(dataset)
            df_W_ids = df['worldline_id']
            W_names = list(W.df.loc[df_W_ids, :].name)
            W_names = [name.decode() for name in W_names]
            df['name'] = W_names
            df.to_csv(tf)
        dfs.append(df)

    if sorted_names:
        for i in range(len(dfs)):
            dfs[i] = dfs[i].sort_values(by='name')

    if names is None:
        names = np.array(dfs[0]['name'])
    else:
        names = [n for n in names if n in np.array(dfs[0]['name'])]

    splitted_p = str(dataset).replace("\\", "/").split('/')
    legend_title = '{}-{}-{}'.format(splitted_p[-3], splitted_p[-4], splitted_p[-2])

    time_cols = np.array(dfs[0].columns[2:])
    time_inds = np.arange(len(time_cols))
    t_inds = np.arange(0, len(time_cols), 200)

    flow, index = load_flow_file(flow_file)
    index = np.array(index) - 2

    if colors_dict is None:
        colors_dict = {
            "100mM NaCl": "#F5B7B1",
            "10mM CuSO4": "#D2B4DE",
            "1uM ascr#3": "#A9CCE3",
            "e-6 IAA": "#A3E4D7",
            "Fluorescein": "#ABEBC6",
            "OP50": "#FAD7A0",
            "450mM NaCl": "#EDBB99",
            "800mM Sorbitol": "#AED6F1",
            "Control": "#CCD1D1",
            "e-2 IAA": "#F9E79F"
        }

    if n_per_panel == 'auto':
        n_per_panel = np.ceil(len(names) / 2)
    n_panels = int(np.ceil(len(names) / n_per_panel))

    plt.figure(figsize=(10 * n_panels, 2.5 * n_per_panel), dpi=100)
    plt.subplots_adjust(wspace=0.2, hspace=.7)

    for idx, name in enumerate(names):
        plt.subplot(int(n_per_panel), n_panels, idx + 1)

        traces = []
        for df in dfs:
            df_W = df[df['name'] == name]
            trace = np.array(df_W[time_cols])[0] if len(df_W) else np.full(len(time_cols), np.nan)
            traces.append(trace)

        ylim0 = np.nanmin([t[30:] for t in traces])
        ylim1 = np.nanmax([t[30:] for t in traces])

        for i in np.arange(2, len(index) - 2, 3):
            plt.fill(
                [time_inds[index[i]], time_inds[index[i+1]], time_inds[index[i+1]], time_inds[index[i]]],
                [ylim0, ylim0, ylim1, ylim1],
                colors_dict[flow[i]], label=flow[i]
            )

        for t, color in zip(traces, colors_list):
            plt.plot(time_inds, t, color=color, linewidth=1)

        plt.title(name, weight='bold')
        plt.xticks(time_inds[t_inds])
        plt.xlabel("frame")
        plt.xlim(0, len(time_cols))
        if not np.isnan(ylim0):
            plt.ylim(ylim0, ylim1)

        if idx == 0:
            plt.legend(fontsize="xx-large", loc='upper right', bbox_to_anchor=(-.2, 0),
                       title=legend_title, title_fontsize="xx-large")

    if save_obj is None:
        plt.savefig(plot_file, bbox_inches="tight", pad_inches=.5)
    else:
        save_obj.savefig(bbox_inches="tight", pad_inches=.5)

    plt.show()
    plt.close()
    



def plot_traces_stacked_by_name_multipage(
    dataset,
    traces_files,
    colors_list,
    names=None,
    update_names=False,
    colors_dict=None,
    plot_name="traces_stacked.pdf",
    plot_title=None,
    names_per_page=10):


    flow_file = dataset / "flow_file.txt"
    if plot_title is None:
        plot_title = str(dataset)

    plot_file = os.path.join(os.path.split(traces_files[0])[0], plot_name)

    dfs = []
    for tf in traces_files:
        print(f"Loading traces from {tf}")
        df = pd.read_csv(tf, index_col=0)
        if update_names:
            A, W = load_annotations(dataset)
            df_W_ids = df['worldline_id']
            W_names = list(W.df.loc[df_W_ids, :].name)
            W_names = [name.decode() for name in W_names]
            df['name'] = W_names
            df.to_csv(tf)
        dfs.append(df)

    if names is None:
        names = np.array(dfs[0]['name'])
    else:
        names = [n for n in names if n in np.array(dfs[0]['name'])]

    time_cols = np.array(dfs[0].columns[2:])
    time_inds = np.arange(len(time_cols))
    t_inds = np.arange(0, len(time_cols), 200)

    flow, index = load_flow_file(flow_file)
    index = np.array(index) - 2

    if colors_dict is None:
        colors_dict = {
            "100mM NaCl": "#F5B7B1",
            "10mM CuSO4": "#D2B4DE",
            "1uM ascr#3": "#A9CCE3",
            "e-6 IAA": "#A3E4D7",
            "Fluorescein": "#ABEBC6",
            "OP50": "#FAD7A0",
            "450mM NaCl": "#EDBB99",
            "800mM Sorbitol": "#AED6F1",
            "Control": "#CCD1D1",
            "e-2 IAA": "#F9E79F"
        }

    n_rows = len(traces_files)

    with PdfPages(plot_file) as pdf:
        for chunk_start in range(0, len(names), names_per_page):
            chunk_names = names[chunk_start:chunk_start + names_per_page]
            n_cols = len(chunk_names)

            plt.figure(figsize=(4.5 * n_cols, 2.5 * n_rows), dpi=100)
            plt.subplots_adjust(wspace=0.3, hspace=0.6)

            for col_idx, name in enumerate(chunk_names):
                traces = []
                for df in dfs:
                    df_W = df[df['name'] == name]
                    trace = np.array(df_W[time_cols])[0] if len(df_W) else np.full(len(time_cols), np.nan)
                    traces.append(trace)

                ylim0 = np.nanmin([t[30:] for t in traces])
                ylim1 = np.nanmax([t[30:] for t in traces])
                if ylim0 == ylim1:
                    ylim1 += 1e-6  # avoid matplotlib warning

                for row_idx, (trace, color) in enumerate(zip(traces, colors_list)):
                    ax = plt.subplot(n_rows, n_cols, row_idx * n_cols + col_idx + 1)

                    for i in np.arange(2, len(index) - 2, 3):
                        ax.fill(
                            [time_inds[index[i]], time_inds[index[i + 1]], time_inds[index[i + 1]], time_inds[index[i]]],
                            [ylim0, ylim0, ylim1, ylim1],
                            colors_dict.get(flow[i], "#CCCCCC")
                        )

                    ax.plot(time_inds, trace, color=color, linewidth=1)
                    ax.set_xticks(time_inds[t_inds])
                    ax.set_xlim(0, len(time_cols))
                    ax.set_ylim(ylim0, ylim1)
                    if row_idx == n_rows - 1:
                        ax.set_xlabel("frame")
                    if col_idx == 0:
                        ax.set_ylabel(os.path.basename(traces_files[row_idx]))
                    if row_idx == 0:
                        ax.set_title(name, weight='bold')

            pdf.savefig(bbox_inches="tight", pad_inches=0.5)
            plt.close()

















































