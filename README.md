# Sleep EEG Event Detector (SEED)

Repository for the code and pretrained weights of our deep-learning based detector (SEED) described in:

Tapia-Rivas, N.I., Estévez, P.A. & Cortes-Briones, J.A. A robust deep learning detector for sleep spindles and K-complexes: towards population norms. _Sci Rep_ **14**, 263 (2024).
https://doi.org/10.1038/s41598-023-50736-7

If you find this software useful, please consider citing our work.

## Roadmap

- [x] Paper officially published online. (jan 2nd, 2024)
- [x] Share a working (but messy) code. The existing code uses tensorflow 1, which is deprecated. As a temporary fix, tensorflow 1 behaviour is requested to tensorflow 2 at import time. (jan 8th, 2024)
- [ ] Clean, update and simplify. Migrate from tensorflow 1 to tensorflow 2.
- [ ] Generate and share working checkpoints. 


**Note on existing pre-trained weights:** Existing checkpoints require a deprecated implementation of LSTM layers (`CuDNNLSTM` in `tf.contrib`), that was removed in TF2 and does not have an exact equivalent (so tensors won't match).



## Getting started

For now, your simplest entrypoint is `/scripts/`.
- `train.py`: Trains SEED, and generates predictions of the final model.
- `crossval_performance.py`: For a given training run, it fits the detection threshold of SEED and reports the cross-validation performance of that optimal threshold.


## Setup

SEED is implemented using tensorFlow in python.

For a safe installation, create a virtual environment with `python` 3.10. For example, if you use `conda`:
```bash
conda create -n seed python=3.10
conda activate seed
```

Inside the environment, install dependencies running `pip install -r requirements.txt`

**Note on Apple Silicon:** If you have an apple-silicon mac, you can accelerate tensorflow with `pip install tensorflow-metal` ([ref](https://developer.apple.com/metal/tensorflow-plugin/)).
