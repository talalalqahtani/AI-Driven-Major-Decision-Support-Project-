import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

# Load dataset
data = pd.read_csv("/Users/talalalqahtani/Desktop/StudentMajorApp/models/Australian_Student_PerformanceData (ASPD24).csv")

# Data Cleaning
data = data.fillna(data.mean())  # Fill missing numerical values
for col in ["Gender", "Learning Style", "Study Environment"]:
    data[col] = data[col].fillna(data[col].mode()[0])  # Fill categorical columns with mode

data = data.drop_duplicates()  # Remove duplicate rows

# Feature selection
features = [
    "High School GPA",
    "Entrance Exam Score",
    "Gender",
    "Learning Style",
    "Study Environment"
]
target = "Major"

X = data[features]
y = data[target]

# Encode categorical features
label_encoders = {}
for col in ["Gender", "Learning Style", "Study Environment"]:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Encode the target variable
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y)

# Scale numerical features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model and preprocessing tools
joblib.dump(model, "models/student_major_predictor_model.pkl")
joblib.dump(scaler, "models/student_major_scaler.pkl")
joblib.dump(target_encoder, "models/student_major_target_encoder.pkl")

print("Model training complete. Files saved in 'models' directory.")