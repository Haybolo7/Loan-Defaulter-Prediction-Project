import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

def train_model(data_path="loan_dataset.csv", model_path="model/loan_model.joblib"):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file '{data_path}' not found.")

    df = pd.read_csv(data_path)

    # Derived Financial Features
    df['LOAN_TO_VALUE_RATIO'] = df['LOAN'] / (df['VALUE'] + 1e-5)
    df['NET_EQUITY'] = df['VALUE'] - df['MORTDUE']

    X = df.drop('BAD', axis=1)
    y = df['BAD']

    num_cols = ['LOAN', 'MORTDUE', 'VALUE', 'YOJ', 'DEROG', 'DELINQ', 
                'CLAGE', 'NINQ', 'CLNO', 'DEBTINC', 'LOAN_TO_VALUE_RATIO', 'NET_EQUITY']
    cat_cols = ['REASON', 'JOB']

    num_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', num_transformer, num_cols),
        ('cat', cat_transformer, cat_cols)
    ])

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pipeline.fit(X_train, y_train)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(pipeline, model_path)
    print(f"Model successfully saved to {model_path}")

if __name__ == "__main__":
    train_model()