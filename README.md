# DeFINO

This repository is for implementing Derivative-based Fisher-score Informed Neural Operator.

## Setup

To create conda environment from the terminal,

```
conda env create -f environment.yml
```

## Generating Joint Samples

To create joint samples in 2D, first create two empty folders

```
data_generation/src/img_{num of eigenvectors}/
data_generation/src/num_ev_{num of eigenvectors}/
```

To create joint samples in 3D, first create two empty folders

```
data_generation/src/3D/img_{num of eigenvectors}/
data_generation/src/3D/num_ev_{num of eigenvectors}/
```

Then in REPL,

```
cd(data_generation/src)
include(data_2D_memory_opt.jl)
```

This file will generate the followings in the two empty folders we created above:

- saturation field
- eigenvectors of Fisher Information Matrix (FIM)
- vector Jacobian product when $v$ = eigenvectors of FIM