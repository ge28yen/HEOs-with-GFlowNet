# GFlowNet for Catalyst Design: High Entropy Oxides (HEOs)

This repository contains an implementation of GFlowNet for tackling the catalyst design problem, with a focus on sampling diverse **High Entropy Oxides (HEOs)** that exhibit low overpotential values.

This proof-of-concept implementation showcases the integration of the **GFlowNet Framework** with a custom environment and proxy model to explore the design space of HEOs effectively.

This work is based on the original [GFlowNet repository](https://github.com/alexhernandezgarcia/gflownet) and extends it to address the HEO catalyst design problem.

---

## Overview

### **Key Features**
- **Data-Driven Approach:** Leverages data from ~200 lab experiments for training and testing.
- **Custom Environment:** Implements a tree-like assembly process for HEOs.
- **Proxy Model:** Uses a trained Multi-Layer Perceptron (MLP) as a reward model to guide the GFlowNet.
- **Exploration Focus:** Enables the discovery of diverse, promising HEO candidates.

---

## Repository Structure

### **Core Files**
- **`data.csv`**  
  Contains experimental data from ~200 lab experiments, which is used for training and testing models.

- **`regression_heo.ipynb`**  
  Jupyter notebook containing:
  - Data exploration and preprocessing.
  - Regression experiments using classical machine learning models and simple deep learning approaches.
  - Final trained model selection (an MLP).

- **`gflownet/envs/heo.py`**  
  Implements the environment for assembling HEOs in a tree-like structure. This environment defines the state and action space for the GFlowNet.

- **`gflownet/proxy/heo.py`**  
  Contains the proxy model implementation (the MLP selected from `regression_heo.ipynb`). This proxy serves as the reward function for the GFlowNet during training.

---

## Running the Code

To train and run the GFlowNet on the HEO catalyst design problem, use the following command:

```bash
python main.py env=heo proxy=heo