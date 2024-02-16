from joblib import dump
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

# Load the training dataset
df = pd.read_csv('training_data.csv')

# Specify feature columns and target column
feature_columns = ['Avg Policy Size', 'Years Positive', 'Operating Acreage', 'Rain Index Variance', 'Distance to Nearest Customer (Miles)']
target_column = 'Converted?'

# Preprocess the dataset
numeric_features = feature_columns

numeric_transformer = SimpleImputer(strategy='mean')

preprocessor = Pipeline(steps=[
    ('imputer', numeric_transformer)
])

# Define the model pipeline
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=1000, random_state=42))
])

# Split the dataset into training and testing sets
X = df[feature_columns]
y = df[target_column].astype(int)  # Convert target column to integer
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest model
rf_pipeline.fit(X_train, y_train)

# Evaluate the model
predictions = rf_pipeline.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f'Model Accuracy: {accuracy:.4f}')

# Save the pipeline to a file
dump(rf_pipeline, 'model.joblib')
