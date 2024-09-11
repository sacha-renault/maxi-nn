# Maxi nn

## Intro

- This can sound bad "maxi-nn" but it's just to contrast to my previous project [mini-nn](https://github.com/sacha-renault/mini-nn/).
- This project is still about learning.
- Optimization have been done since mini-nn.
  - Nodes of computation graph are tensors instead of being values.
  - Allowing vectorized computation.

## Installation

- Install [vcpkg](https://github.com/microsoft/vcpkg).
- Install [xtensor](https://github.com/xtensor-stack/xtensor) and [xtensor-blas](https://github.com/xtensor-stack/xtensor-blas).

```sh
./vcpkg install xtensor
./vcpkg install xtensor-blas
```

- Install blas deps :

```sh
  sudo apt-get install libopenblas-dev
```
