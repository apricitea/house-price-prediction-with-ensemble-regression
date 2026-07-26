# House Price Prediction with Ensemble Regression

Kaggle's [House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
competition — predicting `SalePrice` from ~80 property features.

## Approach

- Missing-value imputation (e.g. `LotFrontage` predicted via regression on related features
  rather than dropped or mean-filled)
- Outlier removal via z-score filtering
- Feature engineering: dummy encoding of categoricals, sqrt-transform on skewed numeric
  features
- Compared six regressors: RandomForest, GradientBoosting, XGBoost, SVR, ExtraTrees, Lasso
- Final model: a stacking ensemble (GradientBoosting + RandomForest as base learners)

## Result

Stacked ensemble validation RMSE: **0.0995** (on log-transformed `SalePrice`, matching the
competition's evaluation metric).

## Contents

- `regression.ipynb` — full pipeline: EDA, imputation, feature engineering, model
  comparison, stacking
- `train.csv` / `test.csv` — competition data
- `Imputed_data_all.csv` — intermediate output after imputation
