import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import numpy as np

# Load the dataset
file_path = 'Mall_Customers.csv'  # Replace with your file path
data = pd.read_csv(file_path)

# Convert Gender to numerical values (Female = 0, Male = 1)
data['Gender'] = data['Gender'].apply(lambda x: 0 if x == 'Female' else 1)

# Select numerical features for clustering and visualization, excluding 'CustomerID'
numerical_data = data.drop(columns=['CustomerID']).select_dtypes(include=["float64", "int64"])

# Standardize the numerical features
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numerical_data)

# Perform K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(scaled_data)

# Add the cluster labels to the dataset
data["Cluster"] = clusters

# Scatter plots for clustering
features = list(numerical_data.columns)
pair_combinations = [(feature, "Spending Score (1-100)") for feature in features if feature != "Spending Score (1-100)"]

# Generate scatter plots and linear regression graphs
for feature_x, feature_y in pair_combinations:
    # Clustering scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(data[feature_x], data[feature_y], c=data["Cluster"], cmap="viridis", alpha=0.6)
    plt.title(f"Clustering: {feature_x} vs {feature_y}")
    plt.xlabel(feature_x)
    plt.ylabel(feature_y)
    plt.colorbar(label="Cluster")
    plt.savefig(f"clustering_{feature_x}_vs_{feature_y}.png")
    plt.close()

    # Linear regression
    X = data[[feature_x]].values  # Feature
    y = data[feature_y].values    # Spending Score
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    # Plot regression
    plt.figure(figsize=(8, 6))
    plt.scatter(X, y, alpha=0.6, label="Data")
    plt.plot(X, y_pred, color="red", label=f"Regression Line (R² = {model.score(X, y):.2f})")
    plt.title(f"Linear Regression: {feature_x} vs {feature_y}")
    plt.xlabel(feature_x)
    plt.ylabel(feature_y)
    plt.legend()
    plt.savefig(f"linear_regression_{feature_x}_vs_{feature_y}.png")
    plt.close()

print("Scatter plots and regression graphs saved!")
