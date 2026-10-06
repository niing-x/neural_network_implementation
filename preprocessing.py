import pickle

import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ==============================
# 1. Load Adult Dataset
# ==============================

adult = fetch_ucirepo(id=2)

X = adult.data.features.copy()
y = adult.data.targets.copy()

print("Original X shape:", X.shape)
print("Original y shape:", y.shape)


# ==============================
# 2. Clean Data
# ==============================

# Replace '?' with missing values
X = X.replace('?', np.nan)

# Convert target to Series
y = y.iloc[:, 0]

# Clean target labels
y = y.astype(str).str.strip().str.replace('.', '', regex=False)

# Convert target to binary
y = y.map({
    '<=50K': 0,
    '>50K': 1
})

print("\nTarget distribution:")
print(y.value_counts())

print("\nMissing values:")
print(X.isnull().sum())


# ==============================
# 3. Identify Feature Types
# ==============================

numerical_features = X.select_dtypes(
    include=['int64', 'float64']
).columns.tolist()

categorical_features = X.select_dtypes(
    include=['object', 'category']
).columns.tolist()

print("\nNumerical features:")
print(numerical_features)

print("\nCategorical features:")
print(categorical_features)


# ==============================
# 4. Split Dataset: 70/15/15
# ==============================

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

print("\nDataset split:")
print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)


# ==============================
# 5. Create Preprocessing Pipeline
# ==============================

numerical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numerical_transformer, numerical_features),
    ('cat', categorical_transformer, categorical_features)
])


# ==============================
# 6. Apply Preprocessing
# ==============================

X_train_processed = preprocessor.fit_transform(X_train)

X_val_processed = preprocessor.transform(X_val)

X_val_processed = preprocessor.transform(X_val)
X_test_processed = preprocessor.transform(X_test)

processed_data = {
    "train_features": X_train_processed,
    "train_labels": y_train.to_numpy(),
    "val_features": X_val_processed,
    "val_labels": y_val.to_numpy(),
    "test_features": X_test_processed,
    "test_labels": y_test.to_numpy(),
}

with open("processed_data.pkl", "wb") as file:
    pickle.dump(processed_data, file)


# ==============================
# 7. Check Results
# ==============================

print("\nAfter preprocessing:")
print("Training:", X_train_processed.shape)
print("Validation:", X_val_processed.shape)
print("Test:", X_test_processed.shape)
print("Saved processed dataset to: processed_data.pkl")

print("\nPreprocessing completed successfully!")