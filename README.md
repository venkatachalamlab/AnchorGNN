# AnchorGNN: 3D Neuron Tracking in *C. elegans*

This repository implements a complete pipeline for 3D neuron tracking in *C. elegans* using automated segmentation, graph-based feature extraction, and a Graph Neural Network (GNN) for anchor neuron identification. The output anchors are integrated into the ZephIR tracking framework for enhanced non-rigid neuron tracking.

## 📜 Overview

Tracking neurons in deforming *C. elegans* brains is challenging due to nonlinear body motion, signal bleed-over, and imaging artifacts. This project addresses these issues by:

- Segmenting nuclei from 3D fluorescent microscopy images using StarDist.
- Constructing neuron graphs using node and edge features derived from the segmented volumes.
- Training a GNN model to match neurons across timepoints.
- Using high-confidence matched anchor neurons to guide ZephIR-based nonrigid tracking.



---

## 📁 Project Structure

```
Matching_anchor/
│
├── MatchPartial/                   # GNN training and testing
│   ├── parameters.py               # Configuration and constants
│   ├── model_sim_EGAT_v2_h8.py    # Edge-attention GNN architecture
│   ├── Seg2graph.py            # Feature extraction and graph construction
│   ├── eval_prediction_func.py    # Evaluation and matching functions
│   └── ...
│
├── Segmentation/                  # Segmentation pipeline with StarDist
│   └── Video_seg.py
│
├── Eval/                          # Downstream evaluation
│   └── traces.ipynb               # Extract neuronal activity from tracks
│
├── GUI/                           # Manual annotation interface https://github.com/venkatachalamlab/Segmentation_GUI
│   └── annotation_gui.py
│
├── train.py                       # Entry point to train GNN
├── test.py                        # Run inference on new data
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

---

## 🧠 Key Features

- **StarDist segmentation**: Fast and accurate detection of 2D/3D fluorescent nuclei. [StarDist](https://github.com/stardist/stardist)
- **Graph construction**: Nodes represent neurons, edges represent spatial proximity and orientation.
- **GNN matching**: Graph attention model trained to match neuron identities across frames.
- **Anchor-based ZephIR registration**: Guides deformable alignment with partial annotations.
- **Manual labeling GUI**: Simplifies annotation and review of neuron centroids.[Segmentation_GUI](https://github.com/venkatachalamlab/Segmentation_GUI)
- **Activity trace extraction**: Extracts calcium signal from tracked coordinates.

---

## 🖥️ Installation

```bash
git clone https://github.com/venkatachalamlab/AnchorGNN.git
cd AnchorGNN
pip install -r requirements.txt
```

---

## 🚀 Usage

### Train the GNN
```bash
python train.py --config train.yaml
```

### Test a pretrained model
```bash
python test.py --config test.yaml
```
### To produce neuronal activity traces, see [`Eval/Traces.ipynb`](./Eval/Traces.ipynb).
---

## 🔍 Evaluation Highlights
- Ablation study included for:
  - Feature sets (node vs node+edge)
  - Attention heads (1 vs 8)
- Activity traces from GNN-refined tracks match manual annotation more closely than baseline.

---

## 🔒 License

This project is licensed under the [MIT License](LICENSE).

---

## ✍️ Citation

If you use this code, please cite:

> Hang Deng, James Yu, Vivek Venkatachalam. *Neuron tracking in C. elegans through automated anchor neuron localization and segmentation.*
>
> [SPIE Conference Paper](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/12857/128570H/Neuron-tracking-in-C-elegans-through-automated-anchor-neuron-localization/10.1117/12.3001982.short)
