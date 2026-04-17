
Relaxation Neural Networks(RelaxNN)
=============
A deep learning framework for solving the nonlinear hyperbolic systems.  
![image](https://github.com/mazhengcn/relaxation-nn/blob/main/fig/relaxation.png)

This repository contains the official implementation for the paper: [Capturing Shock Waves by Relaxation Neural Networks](https://arxiv.org/abs/2404.01163)

Table of Contents
-----------------

-   [Install & Setup](#install-&-setup)
-   [Quickstart](#quickstart)
-   [Citation](#citation)
-   [Authors](#authors)
-   [License](#license)


Install & Setup
---------------

This code has been tested and confirmed to work with the following versions:
* PYTHON 3.12+
* PYTORCH 2.11.0
* NUMPY 2.4.4

Install RelaxNN with the following commands:

`git clone git@github.com:mazhengcn/relaxation-nn.git `

We recommend using `uv` to manage the experiment environment:

```bash
git clone git@github.com:mazhengcn/relaxation-nn.git
cd relaxation-nn
uv python install 3.12
uv sync
```

The default `uv sync` command installs the base RelaxNN environment only.

If you also need the optional `pinns_jax` stack, install the JAX extra with:

```bash
uv sync --extra jax
```

If your machine needs a specific CUDA build of PyTorch, keep the base `uv` workflow above and replace the default `torch` installation inside the environment with the wheel recommended by the official PyTorch selector.

If you want to run the JAX-specific tests, use:

```bash
uv run pytest pinns_jax
```

Quickstart
-----

To train our model, run:

```bash
uv run bash relaxnn/run_main.sh
```

You can also launch `main.py` directly with `uv run` and pass the config flags manually.

Finally, to evaluate the model's performance, you can use `evaluate.py`.

Our reference data are obtained by Clawpack: <https://www.clawpack.org/>, the information about the reference data are described in `data information.txt` in `data` folder.

Citation
-------

```
@misc{zhou2024capturing,
      title={Capturing Shock Waves by Relaxation Neural Networks}, 
      author={Nan Zhou and Zheng Ma},
      year={2024},
      eprint={2404.01163},
      archivePrefix={arXiv},
      primaryClass={math.NA}
}
```


Authors
-------

* Nan Zhou | [@nan](https://github.com/zhounan-sjtu)
* Zheng Ma | [@mazheng](https://github.com/mazhengcn)


License
-------

[Apache License 2.0](LICENSE)
