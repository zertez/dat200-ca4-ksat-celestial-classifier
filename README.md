# DAT200 CA4: Celestial Object Classification

**Group 37**
- Jannicke Ådalen
- Marcus Dalaker Figenschou
- Rikke Sellevold Vegstein

## Project Overview

This project implements machine learning models for multi-class classification of astronomical objects using data provided by Kongsberg Satellite Services (KSAT). The solution classifies celestial objects into three categories (Galaxies, Quasars, and Stars) using photometric measurements and applies advanced feature engineering techniques with ensemble methods.

**Final Result:** 2nd place out of 59 teams on Kaggle

## Technical Approach

- **Models**: Random Forest Classifier (best), Support Vector Machine, Logistic Regression
- **Feature Engineering**: Astronomical color indices, redshift transformations, interaction terms
- **Data Processing**: Comprehensive cleaning, stratified sampling for class imbalance
- **Optimization**: Hyperparameter tuning with cross-validation, PCA dimensionality reduction
- **Evaluation**: F1-macro score optimization for multi-class performance

## Key Features

- Advanced feature engineering with astronomical domain knowledge
- Color index calculations from photometric bands (u, g, r, i, z)
- Redshift-based feature transformations and interactions
- Ensemble methods with Random Forest achieving best performance
- Comprehensive preprocessing pipeline for large-scale astronomical data

## Results

- **2nd place out of 59 teams** in DAT200 CA4 2025 Kaggle Competition
- **Best F1-macro score: 0.976** (Random Forest)
- Successful multi-class classification with high accuracy across all celestial object types
- Demonstrated effective handling of imbalanced astronomical dataset

## Files Structure

- `CA4_Astronomy_Classification.py` - Main analysis and model training script
- `assets/` - Training and test astronomical datasets
- `results/` - Model predictions and Kaggle submissions
- `models_testing/` - Rapid model evaluation and feature engineering experiments
- `pyproject.toml` - Project dependencies

## Requirements

See `pyproject.toml` for dependencies. Key requirements include scikit-learn, pandas, numpy, and matplotlib for astronomical data processing and machine learning pipeline implementation.

## Usage

1. Install dependencies: `uv sync` or `pip install -e .`
2. Place dataset files in `assets/` directory
3. Run the main script: `python CA4_Astronomy_Classification.py`
4. Check `results/` directory for generated predictions and submissions

## Dataset

- **Source**: Astronomical observations from Kongsberg Satellite Services (KSAT) for educational use
- **Size**: 100,000 total observations (80,000 training, 20,000 testing)
- **Target**: Multi-class classification (GALAXY=0, QSO=1, STAR=2)
- **Features**: 5 photometric bands (u,g,r,i,z), redshift, positional coordinates, metadata
- **Distribution**: 60% galaxies, 25% quasars, 15% stars (handled via stratified sampling)

## Competition

[Official Kaggle Competition: DAT 200 CA 4 2025](https://www.kaggle.com/competitions/dat-200-ca-4-2025/overview)