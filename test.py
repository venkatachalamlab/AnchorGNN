import argparse
import yaml
import subprocess
import os

def none_or_str(x):
    return "None" if x is None else str(x)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to test.yaml")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    os.chdir(cfg["code_dir"])

    args_list = [
        str(cfg["zoom_factor"]),
        str(cfg["match_method"]),
        cfg["folder_path"],
        cfg["save_folder"],
        str(cfg["threshold"]),
        none_or_str(cfg["sim_threshold"]),
        none_or_str(cfg["weights_path"]),
        cfg["train_val"],
        none_or_str(cfg.get("centroid_ref")),
        none_or_str(cfg.get("extension_length")),
        none_or_str(cfg.get("ratio")),
        none_or_str(cfg.get("nearby_search_num")),
        none_or_str(cfg.get("model_path")),
    ]

    cmds = [
        ["python3", "-m", "Segmentation.Video_seg"] + args_list,
        ["python3", "-m", "MatchPartial.02_Create_network_data"] + args_list,
        ["python3", "-m", "MatchPartial.track_zephir_anchors_final"] + args_list,
    ]

    for cmd in cmds:
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()