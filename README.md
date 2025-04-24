
# Data-Driven Stochastic Closure Modeling via Conditional Diffusion Model and Neural Operator

This repo contains the official implementation for the paper [Data-Driven Stochastic Closure Modeling via Conditional Diffusion Model and Neural Operator](https://www.sciencedirect.com/science/article/pii/S0021999125002888?casa_token=zNIDEGUOV1oAAAAA:s_9VgksmArtnTri2Z0y9mLVRhjiw0iA97t101PxJ0W9bKotxcsJSJjUaykHPRoFawFyU7O-QvQ).

by [Xinghao Dong](https://xdong99.github.io/), [Chuanqi Chen](https://github.com/ChuanqiChenCC), and [Jin-Long Wu](https://www.jinlongwu.org/).

--------------------

In this work, we present a data-driven modeling framework to
build stochastic and non-local closure models based on the conditional diffusion model and
neural operator. More specifically, the Fourier neural operator is used to approximate the
score function for a score-based generative diffusion model, which captures the conditional
probability distribution of the unknown closure term given some dependent information,
e.g., the numerically resolved scales of the true system, sparse experimental measurements
of the true closure term, and estimation of the closure term from existing physics-based
models. Fast sampling algorithms are also investigated to ensure the efficiency of the proposed framework. 

![schematic](Assets/Schematic.jpg)

A comprehensive study is performed on the 2-D Navier–Stokes equation, for which the stochastic viscous diffusion 
term is assumed to be unknown. The proposed methodology provides a systematic approach via generative machine learning 
techniques to construct data-driven stochastic closure models for multiscale dynamical systems with 
continuous spatiotemporal fields.

## Citations
```
@article{dong2025data,
  title={Data-driven stochastic closure modeling via conditional diffusion model and neural operator},
  author={Dong, Xinghao and Chen, Chuanqi and Wu, Jin-Long},
  journal={Journal of Computational Physics},
  pages={114005},
  year={2025},
  publisher={Elsevier}
}

```
