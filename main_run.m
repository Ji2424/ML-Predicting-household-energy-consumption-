%% task 1
% Loading the data
data = readtable('household_energy_data.csv');

% Check the size of the dataset
[numRows, numCols] = size(data);

% Missing values check
missing_values = sum(ismissing(data));
disp('Missing values per column:');
disp(missing_values);

% Drop Weather and Kettle Features
data.WeatherIcon = [];
data.Kettle_kW_ = [];

% Delet rows with any missing or infinite values
data = rmmissing(data);
data = data(~any(isinf(table2array(data)), 2), :);

% Convert table to matrix
dataMatrix = table2array(data);

% Compute correlation matrix
corrMatrix = corr(dataMatrix, 'Rows', 'pairwise');

% Get variable names
varNames = data.Properties.VariableNames;

% Get correlation vector with target
targetCorr = corrMatrix(1, 2:end);
featureNames = varNames(2:end);  

% Arrange by correlation value
[sortedCorr, sortIdx] = sort(abs(targetCorr), 'descend');
sortedFeatures = featureNames(sortIdx);
signedCorr = targetCorr(sortIdx); 

% Display all features and their correlation
fprintf('Feature\t\t\tCorrelation\n');
fprintf('----------------------------------------------\n');
for i = 1:length(sortedFeatures)
    fprintf('%-25s %.4f\n', sortedFeatures{i}, signedCorr(i));
end

% Separate target and features
y = data.EnergyRequestedFromGrid_kW_;
data.EnergyRequestedFromGrid_kW_ = []; 

% Normalise using z-score
X_normalised = varfun(@zscore, data);  

% Combine back with target
X_normalised.EnergyRequestedFromGrid_kW_ = y;

%% task 2
% Extract target variable and predictors
y = X_normalised.EnergyRequestedFromGrid_kW_;
X_normalised.EnergyRequestedFromGrid_kW_ = [];  % Remove target from predictors

% Convert predictors to matrix
X = table2array(X_normalised);  

% Set random for reproducibility
rng(1);

% Split the data: 70% training, 30% testing
cv = cvpartition(length(y), 'HoldOut', 0.3);
X_train = X(training(cv), :);
y_train = y(training(cv));
X_test  = X(test(cv), :);
y_test  = y(test(cv));

% Fit linear regression model
mdl = fitlm(X_train, y_train);

% Display model summary
disp(mdl);

% Predict on training and testing sets
y_pred_train = predict(mdl, X_train);
y_pred_test = predict(mdl, X_test);

% Mean Squared Error (MSE)
mse_train = mean((y_train - y_pred_train).^2);
mse_test = mean((y_test - y_pred_test).^2);

% R-squared
r2_train = 1 - sum((y_train - y_pred_train).^2) / sum((y_train - mean(y_train)).^2);
r2_test = 1 - sum((y_test - y_pred_test).^2) / sum((y_test - mean(y_test)).^2);

% Predict target values using the trained model
y_pred = predict(mdl, X_test);

% Display metrics
fprintf('Training MSE: %.4f\n', mse_train);
fprintf('Testing MSE: %.4f\n', mse_test);
fprintf('Training R^2: %.4f\n', r2_train);
fprintf('Testing R^2: %.4f\n', r2_test);

% Plot actual vs. predicted values
figure;
scatter(y_test, y_pred, 10, 'filled');
xlabel('Actual Energy Requested from Grid (kW)');
ylabel('Predicted Energy Requested from Grid (kW)');
title('Actual vs. Predicted Energy Consumption (Testing Set)');
grid on;
refline(1, 0);
legend('Predictions', 'Perfect Prediction Line', 'Location', 'southeast');

%% task 3
X_train_poly = [X_train(:, 1:3), X_train(:, 1:3).^2];
X_test_poly = [X_test(:, 1:3), X_test(:, 1:3).^2];

% Train polynomial regression model
polyModel = fitlm(X_train_poly, y_train);

% Predict on training and testing sets
y_train_pred = predict(polyModel, X_train_poly);
y_test_pred = predict(polyModel, X_test_poly);

% Calculate MSE, RMSE, R-square 
mse_train_poly = mean((y_train - y_train_pred).^2);
mse_test_poly = mean((y_test - y_test_pred).^2);

rmse_train_poly = sqrt(mse_train_poly);
rmse_test_poly = sqrt(mse_test_poly);

r2_train_poly = 1 - sum((y_train - y_train_pred).^2) / sum((y_train - mean(y_train)).^2);
r2_test_poly = 1 - sum((y_test - y_test_pred).^2) / sum((y_test - mean(y_test)).^2);

% Polynomial regression: Plot actual vs. predicted values
figure;
scatter(y_test, y_test_pred, 10, 'filled');
xlabel('Actual Energy Requested from Grid (kW)');
ylabel('Predicted Energy Requested from Grid (kW)');
title('Polynomial Regression: Actual vs. Predicted Energy Consumption (Testing Set)');
refline(1, 0);
grid on;
hold on;
refline(1, 0);
legend('Predictions', 'Perfect Prediction Line', 'Location', 'southeast');

% Display metrics
fprintf("Polynomial Regression Results:\n");
fprintf('Training MSE: %.4f\n', mse_train_poly);
fprintf('Training R^2: %.4f\n', r2_train_poly);
fprintf('Training RMSE: %.4f\n', rmse_train_poly);
fprintf('Testing MSE: %.4f\n', mse_test_poly);
fprintf('Testing R^2: %.4f\n', r2_test_poly);
fprintf('Testing RMSE: %.4f\n', rmse_test_poly);
