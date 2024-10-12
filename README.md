# Nursery Dataset and Mamba Model: Inference Attack Study

## Overview

This repository involves training a **Mamba model** on the **Nursery dataset** and then performing an **inference attack** on the trained model. This project is divided into two main phases:

1. **Phase 1: Training a Mamba Model on the Nursery Dataset**  
   In this phase, I will train a Mamba model, which is a machine learning model which becomes more and more relavant due to the paper [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752). It will be trained on the Nursery dataset, which is a classic dataset for evaluating models in classification tasks, using privacy sensitive data which will be interesting for phase 2.

2. **Phase 2: Performing an Inference Attack**  
   In the second phase, I will attempt an inference attack on the trained Mamba model. The purpose of this phase is to evaluate the model's vulnerability to privacy attacks, where an adversary tries to infer sensitive data used during training.

## Contents

This repository contains the following:

- **`data/`**: Contains the Nursery dataset and any pre-processing scripts.
- **`models/`**: Scripts and configurations for training the Mamba model.
- **`attacks/`**: Code to carry out the inference attack on the trained model.
- **`notebooks/`**: Jupyter notebooks documenting the model training, evaluation, and attack experiments.
- **`results/`**: Results from the model's performance and the success of the inference attack.


## Installation and Setup

To run this project locally, clone this repository and install the required dependencies. The project uses Python with the following main libraries:

- ...

You can install the required dependencies with:

```bash
pip install -r requirements.txt
