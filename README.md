
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
* PYTHON 3.10.12
* PYTORCH 2.3.0
* NUMPY 1.24.4

Install RelaxNN with the following commands:

`git clone git@github.com:mazhengcn/relaxation-nn.git `

Quickstart
-----

First, down to the `relaxnn` folder:

`cd relaxnn`

Then, to train our model, run the following command to pass the hyperparams and execute the `main.py`:

`bash run_main.sh`

Finally, to evaluate the model's performance, you can use `evaluate.py`.

Our reference data are obtained by Clawpack: <https://www.clawpack.org/>, the information about the reference data are described in `data information.txt` in `data` folder.


Authors
-------

* Nan Zhou | [@nan](https://github.com/zhounan-sjtu)
* Zheng Ma | [@mazheng](https://github.com/mazhengcn)


License
-------

[Apache License 2.0](LICENSE)