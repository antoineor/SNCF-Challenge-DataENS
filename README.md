# SNCF Delay Prediction Challenge (WIP)

## Overview
This repository contains my approach to the SNCF Data Challenge, which aims to predict real-time train delays between two stations based on historical and contextual data. The dataset exhibits strong temporal dependencies and a highly imbalanced distribution, with most samples showing no delay and a long negative tail with extreme outliers (trains delayed by up to 3 hours).

## Methodology

### Feature Engineering
- **Categorical Variable Treatment:** The station name ("gare") was encoded using a Generalized Linear Model (GLM), assigning a weight ("gare quality") to each station based on its impact on delays.
- **Stacked Model Approach:** The final model stacks a deep learning neural network on top of an XGBoost model, leveraging both architectures' strengths.
- **Temporal Features:** Extracted day-of-week and day-of-month features to capture periodic delay patterns.

### Modeling Approach
- **XGBoost Model:**
  - Objective: Mean Squared Error (MSE), as XGBoost handles outliers effectively through robust tree-based learning.
  - Hyperparameter tuning: Used gamma, colsample_bytree, subsample, and max_depth tuning to optimize performance.
- **Neural Network (TensorFlow/Keras):**
  - Loss Function: **Huber Loss** (chosen for its robustness to outliers, balancing MAE and MSE properties).
  - Architecture: Batch Normalization, Early Stopping (to avoid unnecessary overfitting).
  - Custom Metric: Used the Mean Absolute Error (MAE) of inverse-transformed predictions (using the quantile transformer) to better interpret performance in the original scale.

## Current Results
- **MAE on last submission:** **0.6751**
- **Public Leaderboard Ranking:** **Top 30%**
- **Academic Ranking:** **Top 8**

### Next Steps
- Further feature engineering (e.g., lag-based features, time series decomposition).
- Experiment with other stacking techniques (e.g., LGBM + DL, Meta-Learner).
- Tune hyperparameters more extensively for the DL model.

Work in progress 🚀
