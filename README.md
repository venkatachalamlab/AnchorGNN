# Matching Anchor

This project implements a 3D neuron tracking pipeline for *C. elegans*, using image segmentation, feature extraction, and graph-based matching. It integrates deep learning with traditional optimization to align neurons across timepoints despite deformation, occlusion, or noise. Developed for high-resolution biological imaging tasks, it's adaptable to other segmentation-to-tracking workflows.

### Description

The pipeline begins with 3D segmentation using [StarDist](https://github.com/stardist/stardist), extracts node and edge features from identified neurons, and applies a [Graph Neural Network (GNN)](https://distill.pub/2021/gnn-intro/) to match neuron pairs across frames. A custom energy-minimization framework (ZephIR) then aligns neuron identities over time, using anchors to constrain optimization.

### Techniques of Interest

- 3D segmentation using StarDist-3D (radial polyhedron modeling)
- Neuron motion modeling with elastic spring systems and energy minimization
- Sparse field interpolation to propagate anchor constraints across a 3D volume
- YAML-based config parsing for training and evaluation workflows
- Manual labeling GUI see [Segmentation_GUI](https://github.com/venkatachalamlab/Segmentation_GUI) for implementation reference
- Graph Attentional construction and embedding for node matching (edge features)
- Training flow designed for small-scale annotated biomedical datasets

### Technologies of Note

- [StarDist 3D](https://github.com/stardist/stardist): star-convex polygon segmentation, optimized for nuclei
- [PyTorch](https://pytorch.org/): model training and inference
- [PyYAML](https://pyyaml.org/): structured configuration loading
- [Matplotlib](https://matplotlib.org/): visual analysis and plot generation
- [Tkinter](https://docs.python.org/3/library/tkinter.html): GUI for manual annotation
- [Jupyter](https://jupyter.org/): notebooks for evaluation and experimentation
- Fonts (used in plots): likely [DejaVu Sans](https://dejavu-fonts.github.io/), default in Matplotlib

### Project Structure

````markdown
.
├── train.py
├── test.py
├── train.yaml
├── test.yaml
├── MatchPartial/              # GNN model code for matching anchor neurons
│   ├── models/
│   ├── utils/
│   └── ...
├── Segmentation/              # StarDist-based segmentation pipeline and preprocessing
│   ├── data/
│   ├── training/
│   └── labeling_gui/
├── Eval/                      # Evaluation, visualization and traces notebooks
│   ├── paper_plots/
│   ├── __pycache__/
│   └── *.ipynb
└── README.md
````

#### Notable Directories

- `MatchPartial/`: Contains all GNN model architecture, feature extraction, and training logic for neuron linking.
- `Segmentation/`: Contains segmentation routines and manual labeling tools using StarDist and a custom Tkinter GUI.
- `Eval/`: Jupyter notebooks for training analysis, plotting accuracy metrics, and visualizing matching outputs.
  - To produce neuronal activity traces, see [`Eval/Traces.ipynb`](./Eval/Traces.ipynb).

### Files of Interest

- [`train.py`](./train.py): Trains the GNN model using a configuration file.  
  **Run with:**  
  ```bash
  python train.py --config train.yaml
  ```

- [`test.py`](./test.py): Tests trained model weights on a configured dataset.  
  **Run with:**  
  ```bash
  python test.py --config test.yaml
  ```

- [`train.yaml`](./train.yaml), [`test.yaml`](./test.yaml): Define all paths, hyperparameters, and model options for training and testing.
