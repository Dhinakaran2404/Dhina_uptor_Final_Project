import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load the dataset

df = pd.read_csv("Dhinakaran_Uptor_Final_Project.csv")

# Supervised Learning - Predicting Revenue
features = ["Subscribers", "Ads Revenue ($)", "Engagement (hrs)"]
target = "Revenue ($)"

# Split data into input features (X) and target variable (y)
X = df[features]
y = df[target]

# Split dataset into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict revenue on test data
y_pred = model.predict(X_test)

# Evaluate model performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Supervised Learning - Revenue Prediction:")
print(f"Mean Squared Error: {mse}")
print(f"R-squared Score: {r2}")

# Visualizing actual vs predicted revenue
plt.figure()
plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
plt.xlabel("Actual Revenue ($)")
plt.ylabel("Predicted Revenue ($)")
plt.title("Actual vs Predicted Revenue")
plt.show()

# Unsupervised Learning - Clustering Regions Based on Sales Trends
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[["Subscribers", "Revenue ($)", "Ads Revenue ($)", "Engagement (hrs)"]])

# Finding optimal clusters using the Elbow Method
inertia = []
k_range = range(1, 10)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plot the Elbow Method
plt.figure()
plt.plot(k_range, inertia, marker='o')
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal k")
plt.show()

# Apply K-Means Clustering with optimal k (e.g., 3 clusters)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)

# Visualizing Clusters
plt.figure()
plt.scatter(df["Subscribers"], df["Revenue ($)"], c=df["Cluster"], cmap='viridis', alpha=0.5)
plt.xlabel("Subscribers")
plt.ylabel("Revenue ($)")
plt.title("Clusters of Regions Based on Sales Trends")
plt.colorbar(label="Cluster")
plt.show()