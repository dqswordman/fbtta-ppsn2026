# Data Access

This public release does **not** redistribute datasets, benchmark archives, or raw dataset files.

## Required datasets

The codebase expects the following datasets for full end-to-end reruns:

1. `CIFAR-10`
2. `CIFAR-100`
3. `CIFAR-10-C`
4. `CIFAR-100-C`
5. `SVHN`

## Expected local directory names

The current loaders accept either of the following names when placed under a local `data/` directory:

```text
data/
  cifar10/ or cifar-10-python/
  cifar100/ or cifar-100-python/
  cifar10_c/ or CIFAR-10-C/
  cifar100_c/ or CIFAR-100-C/
  svhn/ or SVHN/
```

## Where to obtain the datasets

- CIFAR-10 / CIFAR-100:
  - official source: https://www.cs.toronto.edu/~kriz/cifar.html
  - the training and test sets can also be prepared through `torchvision`, but this release does not auto-download them for you

- CIFAR-10-C:
  - official release page: https://zenodo.org/records/2535967
  - expected contents: extracted corruption `.npy` files and `labels.npy`

- CIFAR-100-C:
  - official release page: https://zenodo.org/records/3555552
  - expected contents: extracted corruption `.npy` files and `labels.npy`

- SVHN:
  - official source: https://ufldl.stanford.edu/housenumbers/
  - this repository's helper expects an extracted image layout under `data/svhn/` or `data/SVHN/`

## Redistribution policy

- Dataset files are excluded from this public repository.
- Please review the original dataset licenses or terms of use before downloading or redistributing any dataset yourself.
- Benchmark archives such as `CIFAR-10-C.tar` and `CIFAR-100-C.tar` should remain outside the public code repository.

## What is still possible without the datasets

Without local dataset files, this public snapshot still lets you:

- inspect the method implementations
- inspect the processed result tables used in the paper
- inspect the final paper and supplement PDFs
- rerun the lightweight table or figure builders on the included processed outputs
