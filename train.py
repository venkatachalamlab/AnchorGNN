import argparse
import yaml
import subprocess
import os

def none_or_str(val):
    return None if val in [None, "None", "null"] else str(val)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to train.yaml")
    args = parser.parse_args()

    # Load YAML config
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    os.chdir(cfg["code_dir"])  # change directory

    args_list = [
        str(cfg["zoom_factor"]),
        str(cfg["match_method"]),
        cfg["folder_path"],
        cfg["save_folder"],
        str(cfg["threshold"]),
        str(cfg["sim_threshold"]) if cfg["sim_threshold"] is not None else "None",
        str(cfg["weights_path"]) if cfg["weights_path"] is not None else "None",
        cfg["train_val"],
        str(cfg["centroid_ref"]) if cfg["centroid_ref"] is not None else "None",
        str(cfg["extension_length"]) if cfg["extension_length"] is not None else "None",
        str(cfg["ratio"]),
        str(cfg["nearby_search_num"]),
        cfg["model_path"]
    ]

    cmds = [
        ["python3", "-m", "Segmentation.Video_seg"] + args_list,
        ["python3", "-m", "MatchPartial.02_Create_network_data"] + args_list,
        ["python3", "-m", "MatchPartial.06_Train_gnn_model_EGAT_v2"] + args_list,
    ]

    for cmd in cmds:
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()