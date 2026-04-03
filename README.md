# PBSHM Coding Challenge

**DIAMOND A-DC1 Position | ETH Zurich**

## Overview

This repository contains an explanation on the solution of the coding challenge of the **Population-Based Structural Health Monitoring (PBSHM)** problem.
The main goal addressed is the damage etection on a population of 50 shear-frame structures with a variable number of storeys (4-8), using the provided measurement-like node features.

The workflow is the following:

1. Exploratory analysis of the population
2. Supervised damage detection using the SVM classifier
3. Unsupervised damage detection using PCA

## How to Run

### Requirements

* The coding was implemented in MATLAB

### Steps

1. Load the dataset:

   * `structures_measurements.json`
   * `structure_labels.csv`
   * `population_edges_geometry.csv`
   * `population_edge_weights_geometry.csv`

2. Run scripts in order:

   * `Task1.m` → Generates results of Task 1
   * `Task2.m` → Generates results of Task 2
   * `Task3.m` → Generates results of Task 3
   
    Ensure that the following function files are in the same folder with the runnable scripts. 

   * `SVM_case1.m`: SVM classifier of case 1 
   * `SVM_case2.m`: SVM classifier of case 2
   * `SVM_case3A.m`: SVM classifier of case 3A 
   * `SVM_case3B.m`: SVM classifier of case 3B 
   * `SVM_case3C.m`: SVM classifier of case 3C  
   * `SVM_case3D.m`: SVM classifier of case 3D
   * `SVM_case3E.m`: SVM classifier of case 3E 
   * `SMOTE.m`: Application of SMOTE
   * `ROC_CURVES.m`: ROC curve data
   * `baseline_U_PCA.m`: Baseline phase of PCA
   * `inspection_U_PCA.m`: Inspection phase of PCA

   
## Task 1 – Exploratory Analysis

This task focuses on exploring the population of 50 shear-frame structures to understand the variability in geometry and dynamic properties.


### Key Steps

- Extracted structural features such as number of storeys, total height, and storey/height statistical quantities
- Computed statistical quantities of modal frequencies (mean, standard deviation, skewness)  
- Built a simple population graph for visualization purposes based on the cosine similarity of geometric features   

### Visualizations

- Bar plots of storeys, heights, and modal frequencies  
- Histograms to assess distribution of structural properties  
- Frequency boxplots comparing healthy vs damaged structures (damage-sensitive patterns)  
- Graph visualization highlighting structural similarity and damage labels  

### Outcome

The analysis highlights variability across the population and reveals trends in modal frequencies that can serve as potential damage-sensitive features (DSFs).

## Task 2 – Supervised Damage Detection

This task develops a supervised baseline for structural damage detection using Support Vector Machines (SVMs).

### Key Steps

- Extracted fixed-length feature representations from structures with variable number of storeys  
- Considered three feature configurations based on modal frequencies and geometric features  
- Built balanced healthy/damaged datasets for classification  
- Applied SMOTE to generate synthetic damaged samples and mitigate class imbalance in storey-specific cases  
- Trained SVM classifiers and evaluated their performance using 5-fold cross-validation  

### Cases Studied

- **Case 1** → Dominant modal frequencies of the first 4 storeys  
- **Case 2** → Statistical summaries of modal frequencies and geometric features  
- **Case 3** → Separate feature matrices for structures with 4 to 8 storeys, with SMOTE augmentation  

### Evaluation

- Accuracy metric
- F1-score  
- ROC curves and AUC values 

### Outcomes

The supervised SVM approach provides a simple baseline for damage detection. Results of Cases 1 and 2 suggest that overall performance is moderate due to the fact that data scarcity limits detection accuracy. In Case 3 SMOTE enhances damage detection performance significantly, resolving class imbalance between the healthy and damaged members of the population.


## Task 3 – Unsupervised Damage Detection

This task investigates an unsupervised PCA-based approach for damage detection using healthy structures as a baseline.

### Key Steps

- Constructed two feature matrices based on modal frequencies and statistical/geometric summaries 
- The detection process is structured into the baseline phase (healthy only) and the inspection phase (healthy and damaged)
- Applied centered PCA to extract the main directions of variability from the healthy baseline  
- Computed Mahalanobis-distance-based damage indicators for inspection samples in the reduced PCA space  
- Evaluated detection performance using ROC curves and AUC values  

### Cases Studied

- **Case 1** → dominant modal frequencies of the first 4 storeys  
- **Case 2** → statistical summaries of modal frequencies and geometric features  

### Visualizations

- Eigenvalues of the covariance matrix  
- Explained variance and cumulative explained variance per principal component  
- Mahalanobis distance metric plots for healthy and damaged inspection samples  
- ROC curves and AUC comparison between the two cases  

### Outcome

The PCA-based method provides an unsupervised baseline for damage detection by identifying deviations from the healthy population, although its performance is generally less robust than the supervised SVM approach.


## Main Insights

* **Feature choice is critical**: Modal frequencies can be used as Damage Sensitive Features (DSFs)
* **Class imbalance strongly affects performance**, mitigated via SMOTE
* **Unsupervised methods can detect damage**, but are less reliable than supervised ones
* **Geometry variability introduces significant challenges**, motivating graph-based approaches

## Conclusion

A simple, physically-informed feature design combined with classical ML models provides a moderate baseline for PBSHM due to the limited amount of data available. More advanced methods (e.g., graph-based learning) are promising for handling variability across structural populations.


