# A Deep Ritz method for training neural operators via Energy Splitting scheme

This repository contains the python codes to generate train and evaluate the neural operators in the paper [DeepRitzSplit Neural Operator for Phase-field Models via Energy Splitting](https://arxiv.org/abs/2604.18261)

## Installation

- create a virtual environement with conda or venv: 
```
conda create --name ml python=3.13
```

- Activate your virtual environement and update pip :

```
conda activate ml
pip install -U pip wheel 
```

- then install packages with:

```
pip install -U torch torchvision --no-cache-dir
pip install -U "jax[cuda12]==0.8.2" equinox h5py jaxtyping matplotlib optax tqdm ipykernel jupyterlab grain --no-cache-dir
```

- Launch Jupyter Lab with 
```
jupyter lab 
```
or 
```
jupyter lab --no-browser --port xxxx
```
for customized port forwarding or launching notebook at remote server.

- For more information on jax installation (using GPU, Mac GPU, etc), see [Jax Installation](https://jax.readthedocs.io/en/latest/installation.html)

## Example of Use

Under construction
