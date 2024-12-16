import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load the dataset
file_path = 'Mall_Customers.csv'  # Replace with your file path
data = pd.read_csv(file_path)

# Convert Gender to numerical values (Female = 0, Male = 1)
data['Gender'] = data['Gender'].apply(lambda x: 0 if x == 'Female' else 1)

# Select numerical features
numerical_features = data.select_dtypes(include=["float64", "int64"]).drop(columns=["CustomerID"])
target_feature = "Spending Score (1-100)"

# Calculate R^2 values for each feature against 'Spending Score'
r_squared_values = {}
for feature in numerical_features.columns:
    if feature != target_feature:
        X = data[[feature]]  # Predictor
        y = data[target_feature]  # Target
        model = LinearRegression()
        model.fit(X, y)
        r_squared = model.score(X, y)  # R^2 score
        r_squared_values[feature] = r_squared

# Plot R^2 values as a bar graph
plt.figure(figsize=(8, 6))
plt.bar(r_squared_values.keys(), r_squared_values.values(), color='skyblue')
plt.title('R² Strength of Linear Regression with Spending Score')
plt.xlabel('Features')
plt.ylabel('R² Value')
plt.ylim(0, 1)  # R^2 ranges from 0 to 1
plt.xticks(rotation=45)
plt.tight_layout()

# Save the plot
output_path = "linear_regression_strengths.png"
plt.savefig(output_path)
plt.show()

print(f"Bar graph saved as: {output_path}")