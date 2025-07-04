"""
Fix annotation data using data.h5 and annotations.h5,
and save fixed annotations and segmentations for each time frame.
"""
import os
import json
from tqdm import tqdm
from .parameters_seg import *
from .Seg_pos import *
from .Seg2graph import *




def save_pandas_h5(save_h5_path: Path, df: pd.DataFrame):
    """Save a pandas DataFrame to an HDF5 file."""
    with h5py.File(save_h5_path, 'w') as hdf:
        for column in df.columns:
            data = df[column].to_numpy()
            if data.dtype == object:
                data = data.astype(h5py.string_dtype())
            hdf.create_dataset(column, data=data)


def save_seg_h5(seg: np.ndarray, t_idx: int, segmentation_path: Path):
    """Save a segmentation array to an HDF5 file for a specific time index."""
    with h5py.File(segmentation_path / f'{t_idx}.h5', 'w') as hdf:
        hdf.create_dataset('label', data=seg)


def fix_annotations():
    """Main pipeline for fixing annotations and segmentations."""
    with open(folder_path / 'metadata.json', 'r') as file:
        metadata = json.load(file)
    t_max = metadata['shape_t']

    
    segmentation_path = folder_path / 'segmentation_fixed'
    os.makedirs(segmentation_path, exist_ok=True)


    model = StarDist2D.from_pretrained('2D_versatile_fluo')
    t_idx_list = np.arange(t_max) if train_val == 'train' else t_initial_list
    fixed_annotation_df = pd.DataFrame()

    
    # Process each time frame
    for t_idx in tqdm(t_idx_list, desc="Fixing annotations"):
        # Load original image and mask
        img_original, _ = get_volume_at_frame(folder_path / 'data.h5', t_idx)

        # Load original annotations
        combined_df_t_idx = load_annotations_h5_t_idx(folder_path / 'annotations.h5', t_idx)

        # Compute absolute position of annotations
        abs_pos = get_abs_pos(combined_df_t_idx, img_original.shape[-3:])

        # Apply segmentation
        seg_annotator = NucleiSegmentationAnnotation(
            img_original[0, ch], model,
            isotropy_scale=isotropy_scale,
            normalize_lim=normalize_lim,
            zoom_factor=zoom_factor
        )
        seg = seg_annotator.run_segmentation(combined_df_t_idx)

        # Update positions based on segmentation
        abs_pos_updated = update_abs_pos_basedon_seg(seg, abs_pos, seg_annotator.abs_pos_new)

        # Update annotation dataframe
        annotation_df = copy.deepcopy(combined_df_t_idx)
        annotation_df[['z', 'y', 'x']] = abs_pos_updated / (np.array(seg.shape) - 1)

        # Append to global DataFrame
        fixed_annotation_df = pd.concat([fixed_annotation_df, annotation_df], axis=0)

        # Save outputs
        save_pandas_h5(folder_path / 'annotations_fixed.h5', fixed_annotation_df)
        save_seg_h5(seg, t_idx, segmentation_path)

    # Final sort and save
    df_sorted = fixed_annotation_df.sort_values(by=['t_idx', 'worldline_id'])
    save_pandas_h5(folder_path / 'annotations_fixed.h5', df_sorted)


# Entry point
if __name__ == "__main__":
    fix_annotations()