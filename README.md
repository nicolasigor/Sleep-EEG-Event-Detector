# Sleep EEG Event Detector (SEED)

Repository for the code and pretrained weights of our deep-learning based detector (SEED) described in:

Tapia-Rivas, N.I., Estévez, P.A. & Cortes-Briones, J.A. A robust deep learning detector for sleep spindles and K-complexes: towards population norms. _Sci Rep_ **14**, 263 (2024).
https://doi.org/10.1038/s41598-023-50736-7

If you find this software useful, please consider citing our work.

## Setup

SEED is implemented using TensorFlow 1 in python 3.9.

For a safe installation, create a virtual environment with `python` 3.9. For example, if you use `conda`:
```bash
conda create -n seed python=3.9
conda activate seed
```

Inside the environment, install dependencies running `pip install -r requirements.txt`



## Getting started

In the current state of the code, your simplest entrypoint is `/scripts/`.
- `train.py`: Trains SEED, and generates predictions of the final model.
- `crossval_performance.py`: For a given training run, it fits the detection threshold of SEED and reports the cross-validation performance of that optimal threshold.
- `nsrr_inference.py`: Script that uses an ensemble of trained SEED models to predict sleep spindles on the NSRR dataset.

Inside the scripts you will find further documentation.

## How to load data

The code loads a dataset with the function `load_dataset` defined in `sleeprnn/helpers/reader.py`. For example, to load the MODA dataset, the function loads the class `ModaSS` defined in `sleeprnn/data/moda_ss.py`. All of these datasets are sub-classes of the `Dataset` base class that is defined in `sleeprnn/data/dataset.py`.

### Use a dataset used by our research

If you want to use MASS data ([MASS paper](https://pubmed.ncbi.nlm.nih.gov/24909981/), [MODA paper](https://www.nature.com/articles/s41597-020-0533-4)), the easiest way is to organize your data files so that one of the following classes (defined in `sleeprnn/data/`) can be instantiated, according to the expected directory tree documented in each class definition:
- `MassSS`: Class to instantiate the MASS-SS2 dataset considering their sleep spindle annotations, from both experts.
- `MassKC`: Class to instantiate the MASS-SS2 dataset considering their K-complexes annotations.
- `ModaSS`: Class to instantiate the MODA dataset with its sleep spindle annotations. The MODA dataset is composed of signal segments extracted from the full MASS dataset (that is, from its five subsets, not only MASS-SS2), that were annotated for sleep spindles by a consensus of experts. To load this class, first run the scripts located in the `moda/` directory. Inside each script you will find further information.

### Use a dataset of your own

On the other hand, if you want to use your own dataset, the easiest way in the current state of the code is to create your own subclass of the `Dataset` base class in the `sleeprnn/data/` package.

You can use the existing subclasses as examples for the implemenation. The base class has documentation for the expected arguments in its constructor. Besides giving these arguments, you must implement the `_load_from_source` method, that is in charge of reading raw files and return the data dictionary, as illustrated by the template implementation in `Dataset`, or as you can also see on actual implementations for MASS-SS2 and MODA.

Once you create your own subclass of `Dataset`, you can add it as another option in the function `load_dataset` defined in `sleeprnn/helpers/reader.py`, so that it can be easily loaded by the training and fitting scripts by name.

As a future improvement, I would like to offer a more flexible option. For now, this is the way I implemented datasets to handle various steps that I needed for my experiments.


## Pending tasks

With the goal of sharing working code as fast as possible, I decided to share my original research code directly as a first step.

Therefore, it's quite messy. It contains many outdated packages (whose versions are specified in `requirements.txt`) and many pieces of code that are not used by the final published model. My plan is to clean and refactor the codebase so that only the published model is present and it's easy to use by users to either train it on your own data, or use checkpoints to predict on it.

These are some known pending tasks for the future:

- [ ] Upload and share existing model checkpoints that were used for the paper (TensorFlow 1).
- [ ] Improve code: clean unused model variants, migrate to TensorFlow 2, improve documentation, simplify process to use custom data, add example notebooks.
- [ ] Generate and share new checkpoints following the improved code.