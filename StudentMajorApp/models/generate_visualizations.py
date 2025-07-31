import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore

# Load dataset
data = pd.read_csv("/Users/talalalqahtani/Desktop/StudentMajorApp/models/Australian_Student_PerformanceData (ASPD24).csv")

# Sample dataset for visualization (optional)
data_sampled = data.sample(frac=0.1, random_state=42)

# Before Cleaning: Missing Values Heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(data_sampled.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Values (Before Cleaning)")
plt.savefig("/Users/talalalqahtani/Desktop/StudentMajorApp/models/before_cleaning_missing_values.png")
plt.show()

# Before Cleaning: Boxplot
plt.figure(figsize=(12, 6))
sns.boxplot(data=data_sampled.select_dtypes(include='number'))
plt.title("Boxplot of Numerical Features (Before Cleaning)")
plt.savefig("/Users/talalalqahtani/Desktop/StudentMajorApp/models/before_cleaning_boxplot.png")
plt.show()

# Cleaning Process
# Fill missing values for numeric and categorical separately
for column in data_sampled.columns:
    if data_sampled[column].dtype in ['float64', 'int64']:
        data_sampled[column] = data_sampled[column].fillna(data_sampled[column].mean())
    else:
        data_sampled[column] = data_sampled[column].fillna(data_sampled[column].mode()[0])

# Remove outliers using Z-score
numerical_columns = data_sampled.select_dtypes(include='number').columns
data_cleaned = data_sampled[(zscore(data_sampled[numerical_columns]) < 3).all(axis=1)]

# Scale numerical features
scaler = StandardScaler()
data_cleaned[numerical_columns] = scaler.fit_transform(data_cleaned[numerical_columns])

# After Cleaning: Missing Values Heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(data_cleaned.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Values (After Cleaning)")
plt.savefig("/Users/talalalqahtani/Desktop/StudentMajorApp/models/after_cleaning_missing_values.png")
plt.show()

# After Cleaning: Boxplot
plt.figure(figsize=(12, 6))
sns.boxplot(data=data_cleaned[numerical_columns])
plt.title("Boxplot of Numerical Features (After Cleaning)")
plt.savefig("/Users/talalalqahtani/Desktop/StudentMajorApp/models/after_cleaning_boxplot.png")
plt.show()