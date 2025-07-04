# Matching Anchor

This project implements a 3D neuron tracking pipeline for *C. elegans*, using image segmentation, feature extraction, and graph-based matching. It integrates deep learning with traditional optimization to align neurons across timepoints despite deformation, occlusion, or noise. Developed for high-resolution biological imaging tasks, it's adaptable to other segmentation-to-tracking workflows.

### Description

The pipeline begins with 3D segmentation using [StarDist](https://github.com/stardist/stardist), extracts node and edge features from identified neurons, and applies a [Graph Neural Network (GNN)](https://distill.pub/2021/gnn-intro/) to match neuron pairs across frames. A custom energy-minimization framework (ZephIR) then aligns neuron identities over time, using anchors to constrain optimization.

### Techniques of Interest

- 3D segmentation using StarDist-3D (radial polyhedron modeling)
- Neuron motion modeling with elastic spring systems and energy minimization
- Sparse field interpolation to propagate anchor constraints across a 3D volume
- YAML-based config parsing for training and evaluation workflows
- Manual labeling GUI using [Tkinter](https://docs.python.org/3/library/tkinter.html)
- Graph construction and embedding for node matching (edge features + [node2vec](https://snap.stanford.edu/node2vec/))
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
├── Eval/
│   ├── paper_plots/
│   ├── __pycache__/
│   └── *.ipynb
└── README.md
````

#### Notable Directories

- `Eval/`: Contains Jupyter notebooks for training history, performance evaluation, and paper plots.
- `Eval/paper_plots/`: Includes evaluation images such as accuracy curves and training dynamics.

### Files of Interest

- [`train.py`](./train.py): Launches training of the GNN model with configurable parameters.
- [`test.py`](./test.py): Evaluates the model, generating matches between frames.
- [`train.yaml`](./train.yaml), [`test.yaml`](./test.yaml): Define hyperparameters, paths, and model settings.
