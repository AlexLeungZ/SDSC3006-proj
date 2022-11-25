# SDSC3006 proj

 SDSC3006 group project

## Setting up running environment (with conda or mamba)

### Install conda or mamba

### Set up for conda-forge

```bash
conda config --add channels conda-forge
```

### Set up conda environment

```bash
conda create -n r_env # with environment name r_env as example
conda activate r_env
```

### Required package

- Jupyter Notebook
  1. Jupyter
  2. r-irkernel

- Base package
  1. r-base
  2. r-recommended
  3. r-languageserver
  4. r-renv

- Required Library
  1. r-ggplot2
  2. r-reshape2
  3. r-proc
  4. r-caret
  5. r-gbm
  6. r-kernlab
  7. r-randomforest

### Install package (mamba is recommended)

```bash
conda install -c conda-forge Jupyter r-base r-irkernel
conda install -c conda-forge r-recommended r-languageserver r-renv
conda install -c conda-forge r-ggplot2 r-reshape2 r-proc
conda install -c conda-forge r-caret r-gbm r-randomforest r-kernlab
```

## Assign Radian as the default R (optional)

```bash
alias r="radian"
```

## Setup the R kernel

### Setup the R kernel for Jupyter

```bash
radian # r if you did not install radian
```

```R
IRkernel::installspec()
quit()
```

### Choosing R as Jupyter kernel

## Credits

[Conda Docs](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html "Conda Docs")

[Conda-forge](https://conda-forge.org/docs/user/introduction.html "Conda-forge")

[Radian](https://github.com/randy3k/radian "Radian")
