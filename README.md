# Celestial Object Classification

DAT200 CA4 assignment developed for Kongsberg Satellite Services (KSAT) to classify celestial objects using machine learning techniques.

## Overview

This project was developed for Kongsberg Satellite Services (KSAT) to support their expansion into deep-space monitoring capabilities. The goal is to classify astronomical objects into three categories:
- Galaxies (class 0)
- Quasars/QSO (class 1) 
- Stars (class 2)

The dataset contains 100,000 observations with photometric measurements across five filter bands (u, g, r, i, z), positional coordinates, redshift measurements, and various metadata from advanced space monitoring systems.

## Competition Results

- **2nd place out of 59 teams** on Kaggle
- Best model F1-macro score: 97.6% (Random Forest)
- Competition: [DAT 200 CA 4 2025](https://www.kaggle.com/competitions/dat-200-ca-4-2025/overview)
- Evaluation metric: F1-macro score

## Models Implemented

1. **Random Forest Classifier** - Best performing model
2. **Logistic Regression** - Fast baseline approach  
3. **Support Vector Machine** - With PCA dimensionality reduction

## Key Features

- Feature engineering with astronomical color indices
- Redshift transformations and interactions
- Comprehensive data cleaning and preprocessing
- Hyperparameter optimization with cross-validation
- Stratified sampling for class imbalance

## Dataset

- Total observations: 100,000 (from KSAT's space monitoring systems)
- Training set: ~80,000 observations
- Features: 5 photometric bands (u, g, r, i, z), redshift, positional coordinates, and metadata
- Class distribution: 60% galaxies, 25% quasars, 15% stars
- Target encoding: GALAXY (0), QSO (1), STAR (2)

## Files

- `CA4_Astronomy_Classification.py` - Main analysis and model training
- `assets/` - Training and test data
- `results/` - Model predictions and submissions

## Team

Group 37:
- Jannicke Ådalen
- Marcus Dalaker Figenschou
- Rikke Sellevold Vegstein

## Usage

Run the main Python file to reproduce the analysis:

```python
python CA4_Astronomy_Classification.py
```

The script includes data preprocessing, exploratory data analysis, feature engineering, model training, and prediction generation.