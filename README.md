# PBSHM Coding Challenge

**Chair of Structural Mechanics and Monitoring | ETH Zurich**

## Overview

This project addresses a **Population-Based Structural Health Monitoring (PBSHM)** problem.
The goal is to detect structural damage across a population of shear-frame structures with varying geometry (4–8 storeys), using measurement-like features and graph representations.

The workflow progresses from simple baselines to more structured models:

1. Exploratory analysis
2. Supervised damage detection
3. Unsupervised anomaly detection
4. Graph-based extension



## How to Run

### Requirements

* MATLAB (tested with Statistics and Machine Learning Toolbox)

### Steps

1. Load the dataset:

   * `structures_measurements.json`
   * `structure_labels.csv`
   * `population_edges_geometry.csv`

2. Run scripts in order:

   * `task1_exploration.m` → generates plots and dataset insights
   * `task2_svm.m` → supervised damage detection
   * `task3_pca.m` → unsupervised anomaly detection
   * `task4_graph.m` (optional) → graph-based modeling

3. Results (figures and metrics) are automatically generated and saved.

---

## Key Design Decisions

### 1. Feature Engineering

* Used **modal frequencies** as primary features
* Derived **statistical summaries** (mean, std, min, max) to handle variable-size structures
* Rationale: modal frequencies are **physically linked to stiffness**, hence damage-sensitive

---

### 2. Handling Variable-Size Structures

* Converted node-level data into **fixed-length feature vectors**
* Enables use of classical ML models such as SVM

---

### 3. Supervised Learning (Task 2)

* Implemented **Support Vector Machine (SVM)** with hyperparameter optimization
* Used cross-validation for evaluation
* Addressed class imbalance using **SMOTE**

---

### 4. Unsupervised Learning (Task 3)

* Applied **PCA-based anomaly detection**
* Damage identified as deviation from healthy baseline
* No label information used during training

---

### 5. Population Graph (Task 1 & 5)

* Constructed using **geometry-based similarity (cosine similarity)**
* Each structure connected to its 5 nearest neighbors
* Used to explore structural similarity and potential transfer learning

---

## Main Insights

* **Feature choice is critical**: modal frequencies significantly improve detection
* **Class imbalance strongly affects performance**, mitigated via SMOTE
* **Unsupervised methods can detect damage**, but are less reliable than supervised ones
* **Geometry variability introduces challenges**, motivating graph-based approaches

---

## Conclusion

A simple, physically-informed feature design combined with classical ML models provides a strong baseline for PBSHM.
More advanced methods (e.g., graph-based learning) are promising for handling variability across structural populations.

---
