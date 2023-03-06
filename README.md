# EQS: RL in equality saturation

`omelette-original`: deep RL for E-graph construction

`pes/`: MCTS-GEB: (parallel) MCTS for E-graph construction

## Omelette

see: https://www.cl.cam.ac.uk/~ey204/pubs/MPHIL_P3/2022_Zak.pdf

## MCTS-GEB

### Preparation

Langage:

- rust (install and learn: https://www.rust-lang.org/)
- python (3.9 for now)

It is highly recommended to build/install in a virtual environment: https://docs.python.org/3/tutorial/venv.html

`python3.9 -m venv venv`

Rust-Python ffi:

Just install as a python package (https://github.com/PyO3/pyo3)

Packages:

- Deep learning framework: Pytorch or JAX

### To build/install

1. activation your virtual env `source ...`
2. install all python packages `pip install ...`
3. build rust-python ffi (must install as in https://pyo3.rs/v0.16.4/)

```
    cd rust-lib
    maturin develop
    cd -
```

4. install `pip install -e pes/`
5. (optional) run examples in examples/
