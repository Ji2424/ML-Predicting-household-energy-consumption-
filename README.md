# Project Summary

This project develops and compares linear and polynomial regression models to predict household energy demand using a cleaned and preprocessed real-world dataset containing over 50,000 observations and 30 initial features.

The data preparation stage involved removing invalid values (NaN and Inf), dropping corrupted or redundant features, performing correlation-based feature selection, and applying z-score normalization to ensure consistent feature scaling. After cleaning, 27 input features were retained for modeling.


## Linear Regression Performance

A linear regression model was trained using a 70:30 hold-out validation split. 

Training R²: 0.9935

Testing R²: 0.9936

Testing RMSE: 0.0860

The close agreement between training and testing metrics indicates strong generalization and minimal overfitting.

## Second-degree Polynomial Regression Performance

To evaluate whether additional complexity would improve performance, a second-degree polynomial regression model was implemented. While it captured nonlinear relationships, it introduced multicollinearity and increased model complexity. 

Training R²: 0.9935

Testing R²: 0.8992

Testing RMSE: 0.3400

## Conclusion
Although the polynomial model performed well on training data, its reduced testing performance indicates minor overfitting and weaker generalization compared to the linear model.

Overall, linear regression was selected as the preferred model due to its strong predictive accuracy on unseen data, stability, simplicity, and interpretability. This project demonstrates the importance of balancing model complexity with generalization performance, particularly in real-world energy forecasting applications.
