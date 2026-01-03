# ReserveCapacityPrediction_DeepONet

This repository implements a **Deep Operator Network (DeepONet)** model to predict the **reserve capacity of steel columns** based on their **geometrical properties** and **deformed shape**.  
The model learns an operator mapping structural input functions to a scalar reserve capacity output, serving as a fast surrogate for computationally expensive numerical simulations.
Addidtionally, we predict the **resisting moment of steel columns** bsed on the same informations.

A **Graph Neural Network (GNN)** is used as the branch network to naturally handle geometry and connectivity information.

---

## Project Overview

DeepONet is designed to approximate operators between function spaces rather than simple input–output mappings.  
It consists of:

- **Branch network**: Encodes functional inputs (geometry, deformed shapes, graph data)
- **Trunk network**: Encodes conditioning variables or evaluation coordinates
- **Output layer**: Combines both embeddings to predict reserve capacity or residual moment

The project is separated in **4 main parts**:
  1. _Data preparation_ (extraction and processing of required data for modelling)
  2. _Residual Capacity model training_ (objective: predicting residual capacity of columns)
  3. _Resisting Moment model training_ (objective: predicting base column resisting moment)
  4. _Model Analysis_ (latent space analysis (t-sne, pca), node importance (gradient, SHAP))
 
---

## Repository Structure
_NB_: RC refers to _Residual Capacity_ and RM refers to _Resisting Moment_.

  ├── config/                        # Configuration files for models and training
  ├── figures/                      # Model analysis results
  ├── src/
  │   ├── data.py                   # Data loading and preprocessing
  │   ├── dataset.py                # PyTorch dataset class returning graphs + trunk inputs
  │   ├── models.py                 # GNN and FNN branch + DeepONet model definitions
  │   └── utils.py                  # Ensemble of tool functions
  │ 
  ├── mainRC.py                     # Training loop for residual capacity prediction
  ├── mainRM.py                     # Training loop for reserve moment prediction
  │ 
  ├── model_testRC.py               # Evaluation scripts for reserve capacity models
  ├── model_testRM.py               # Evaluation scripts for resisting moment models
  │ 
  ├── latent_space_tsne.py         # Latent space analysis using t-sne (either RC or RM)
  ├── latent_space_pca.py          # Latent space analysis and pc correlation using pca (either RC or RM)
  │ 
  ├── node_importance.py            # Node importance analysis using gradient (either RC or RM)
  ├── SHAP_values_nodes.py          # Node importance analysis using SHAP values (either RC or RM)
  │ 
  ├── data_processing.ipynb         # Features creation pipeline
  │ 
  ├── .gitignore
  ├── README.md                    # This document
  └── requirements.txt             # Python dependencies

