# collaborative-filtering-matrix-factorization
# Matrix Factorization Recommender System – Optimization Comparison

This repository presents an educational implementation of matrix factorization for recommender systems, focusing on optimization behavior and reconstruction performance under different experimental conditions. The project factorizes a user–item interaction matrix into latent representations and compares alternating closed-form optimization with gradient descent while analyzing convergence stability, latent dimensionality effects, and robustness to noisy observations.

## Overview

Matrix factorization is a core technique used in collaborative filtering–based recommender systems. Given a user–item interaction matrix \(X\), the objective is to approximate it using low-rank latent matrices:

X ≈ C · Pᵀ

where \(C\) represents item latent factors and \(P\) represents user latent factors. This project investigates how optimization strategy and hyperparameter choices affect reconstruction accuracy and convergence behavior.

## Objectives

- Implement matrix factorization from scratch using NumPy  
- Compare alternating closed-form updates (RMR / ALS-style) with batch gradient descent  
- Analyze convergence behavior under varying learning rates  
- Study reconstruction error as a function of latent dimension \(k\)  
- Evaluate robustness by introducing controlled noise into interaction data  
- Visualize reconstruction quality using matrix heatmaps  

## Features

- Matrix factorization without high-level ML frameworks  
- Optimization method comparison  
- Learning-rate stability experiments  
- Latent dimension sensitivity analysis  
- Noise robustness evaluation  
- Visualization-driven diagnostics  

## Repository Structure


matrix-factorization-recommender/
│
├── notebooks/
│ └── matrix_factorization_recommender.ipynb
├── README.md
├── requirements.txt
└── LICENSE


## Methodology

### Matrix Factorization

The optimization objective minimizes reconstruction error:

minimize || X − (C · Pᵀ) ||²

allowing discovery of hidden relationships between users and items through learned latent embeddings.

### Alternating Updates (RMR / ALS-style)

One factor matrix is fixed while solving analytically for the other, alternating updates until convergence. This approach typically provides stable convergence with minimal hyperparameter tuning.

### Gradient Descent Optimization

Latent factors are iteratively updated using gradient-based optimization:

θ = θ − η ∇L

Experiments demonstrate sensitivity to learning rate selection, convergence speed differences, and instability when step sizes are poorly chosen.

### Experimental Analysis

The notebook evaluates:
- Reconstruction error versus latent dimension
- Learning-rate impact on convergence
- Optimization stability comparison
- Performance degradation under injected noise
- Visual inspection through matrix heatmaps

## Requirements

Python 3.9 or higher.

Install dependencies:

```bash
pip install numpy matplotlib jupyter
```

Running the Project
```bash
jupyter notebook
```
Open the notebook:

notebooks/matrix_factorization_recommender.ipynb

Run all cells sequentially to reproduce experiments and visualizations.

Results

Experiments demonstrate that alternating closed-form updates converge reliably with stable error reduction, while gradient descent requires careful learning-rate tuning to avoid divergence. Increasing latent dimensionality improves reconstruction performance up to a saturation point, and injected noise produces predictable degradation in approximation quality. Optimization strategy significantly influences convergence behavior and numerical stability.
