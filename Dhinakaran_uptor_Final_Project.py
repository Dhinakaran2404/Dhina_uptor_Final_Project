import pandas as pd
#from sklearn.datasets import load_boston  # Sample dataset (Boston Housing)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load Boston Housing dataset
boston = pd.read_csv("boston_house_prices.csv")
print(boston.columns)
# X = pd.DataFrame(boston.data, columns=boston.feature_names)
# y = boston.target  # Target variable
#
# # Split data into train and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # Standardize the data
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)
#
# # Apply PCA to reduce dimensions (keeping 2 principal components)
# pca = PCA(n_components=2)
# X_train_pca = pca.fit_transform(X_train_scaled)
# X_test_pca = pca.transform(X_test_scaled)
#
# # Train Linear Regression Model
# model = LinearRegression()
# model.fit(X_train_pca, y_train)
#
# # Make Predictions
# y_pred = model.predict(X_test_pca)
#
# # Model Evaluation
# mse = mean_squared_error(y_test, y_pred)
#
# print(f"Mean Squared Error: {mse:.2f}")
