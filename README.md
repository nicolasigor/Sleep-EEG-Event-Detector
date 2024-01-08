# Sleep EEG Event Detector (SEED)

Repository for the code and pretrained weights of our deep-learning based detector (SEED) described in:

Tapia-Rivas, N.I., Estévez, P.A. & Cortes-Briones, J.A. A robust deep learning detector for sleep spindles and K-complexes: towards population norms. _Sci Rep_ **14**, 263 (2024).
https://doi.org/10.1038/s41598-023-50736-7

If you find this software useful, please consider citing our work.

## Roadmap

- [x] Paper officially published online (jan 2nd, 2024)
- [ ] Share a working (but messy) code and weights. The existing code uses tensorflow 1, and existing weights only work when using an NVIDIA GPU (due to the cudnn-accelerated LSTM implementation).
- [ ] Migrate from tensorflow 1 to tensorflow 2
- [ ] Generate checkpoints for both CPU and GPU


## Setup

SEED is implemented using tensorFlow in python.

For a safe installation, create a virtual environment with `python` 3.10. For example, if you use `conda`:
```bash
conda create -n seed python=3.10
conda activate seed
```

Inside the environment, install dependencies running `pip install -r requirements.txt`

**Note on Apple Silicon:** If you have an apple-silicon mac, you can accelerate tensorflow with `pip install tensorflow-metal` ([ref](https://developer.apple.com/metal/tensorflow-plugin/)).