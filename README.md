# Predicting Household Energy Consumption

**Linear regression outperformed polynomial regression on unseen data (R² 0.9936 vs 0.8992), showing that added model complexity introduced overfitting rather than improving predictive power.**

This project develops and compares linear and polynomial regression models to predict household energy demand, using a cleaned, real-world dataset of over 50,000 observations and 30 initial features *(source: name it here)*.

## Data Preparation

- Removed invalid values (NaN, Inf) and dropped corrupted or redundant features
- Correlation-based feature selection, reducing to 27 retained input features
- Z-score normalisation for consistent feature scaling

## Results

| Model | Training R² | Testing R² | Testing RMSE |
|-------|-------------|------------|----------------|
| Linear Regression | 0.9935 | 0.9936 | 0.0860 |
| Polynomial Regression (deg. 2) | 0.9935 | 0.8992 | 0.3400 |

The close agreement between linear regression's training and testing metrics indicates strong generalisation. The polynomial model performed comparably on training data but degraded sharply on the test set, a clear sign of overfitting driven by added multicollinearity.

## Conclusion

Linear regression was selected as the preferred model for its strong test accuracy, stability, and interpretability. This project underscores a common lesson in applied ML: greater model complexity doesn't guarantee better generalisation, and simpler models are often the right choice when they perform competitively.
