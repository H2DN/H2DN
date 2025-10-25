# H²DN: Heterogeneity-Homogeneity Discovery Network  
### Code for the paper *“Machine learning based homogeneity–heterogeneity coupled complex network dynamics discovery”*

---

## 📘 Overview
This repository contains a **primitive implementation** of the paper *“Machine learning based homogeneity–heterogeneity coupled complex network dynamics discovery”*.  
We propose the **Heterogeneity-Homogeneity Discovery Network (H²DN)** — a two-stage framework for discovering the **governing equations of complex networks** directly from observational data.

H²DN simultaneously captures:
- **Group-level homogeneous dynamics**, shared among similar nodes.  
- **Individual-level heterogeneous dynamics**, unique to each node.  

This enables **interpretable**, **robust**, and **generalizable** discovery of network dynamics across various real-world systems.

> A more mature and fully documented version of H²DN will be publicly available on GitHub **after the paper is accepted**.

---

## 🚀 Key Features
- **Differentiable L₀ regularization**: Enables sparse and interpretable discovery of governing equations through continuous relaxation of discrete term selection. 
- **Sparse regression formulation**: Employs differentiable regularization for optimal balance between complexity and accuracy.  
- **Robustness to missing data**: Performs reliably under incomplete or noisy observations.  
- **Cross-domain generalization**: Validated on both synthetic and real-world networks, including:
  - Urban mobility systems  
  - Power grids  
  - Epidemic spreading models  
  - Climate dynamics  

---

## 🧩 Code Description

This demo includes three representative experiments:

1. **Synthetic Network Example**  
   This includes both *single-dynamics* and *multi-dynamics* complex networks.  
   It illustrates how **H²DN** identifies group-level homogeneous and individual-level heterogeneous dynamics in a controlled synthetic environment.

2. **Real-world Network Example**  
   This experiment involves multiple real-world systems, including the **global climate network**, **European power grid**, **monkeypox transmission network**, **urban rail transit network**, and **urban road traffic network**.  
   It demonstrates how H²DN infers governing laws and predicts node states from empirical observational data.

3. **Incomplete Data Experiment**  
   This test evaluates the robustness of H²DN under incomplete and sparse observations, showing its ability to maintain reliable performance even with missing data.

> Currently, we release only the **synthetic dynamics** experiment code.  
> The full implementation, including all the above experiments, will be made publicly available **after the paper is accepted**.

Each example follows the two-stage inference process:

- **Stage 1** – Infer group-level homogeneous dynamics.  
- **Stage 2** – Identify node-level heterogeneous dynamics.

---

## ⚙️ Requirements
This code was tested under the following environment:  
Python == 3.10.11  
PyTorch == 2.1.1 (GPU version)  
NumPy == 1.26.4  
SciPy == 1.11.4  
Matplotlib == 3.8.2  
pandas == 2.1.3  
scikit-learn == 1.3.2  
