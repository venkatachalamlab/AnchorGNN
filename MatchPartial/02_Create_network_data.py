from .parameters import *
from .load_func import *
from .Seg2graph import *
import os
import json
from pathlib import Path
from tqdm import tqdm


def main():
    img_h5_path = Path(folder_path) / 'data.h5'

    save_graph_folder = Path(folder_path) / ('graph_mask' if seg_mask else 'graph')
    os.makedirs(save_graph_folder, exist_ok=True)

    for t_idx in tqdm(t_track, desc="Generating graph data"):
        seg_h5_path = seg_path / f"{t_idx}.h5"

        # Feature extraction
        extractor = NeuronNodesFeatureExtractor(img_h5_path, seg_h5_path, t_idx, ch)
        df_nodes_features = extractor.extract_features()

        # Label matching (only if in training or reference frames)
        if train_val == 'train' or t_idx in t_initial_list:
            seg_ref_h5_path = seg_path / f"{t_idx}.h5"
            annotation_path = Path(folder_path) / 'annotations_orig.h5'
            label_extractor = NeuronLabelExtractor(t_idx, annotation_path, seg_h5_path, seg_ref_h5_path)
            df_nodes_features = label_extractor.match_label_to_worldline_id(df_nodes_features)

        # Graph generation and saving
        graph_generator = GraphDataGenerator(df_nodes_features, num_nearest, isotropic_voxel_size)
        graph_data = graph_generator.produce_graph_data()
        graph_generator.save_graph2pt(graph_data, t_idx, save_graph_folder)


if __name__ == "__main__":
    main()