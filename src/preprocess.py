import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def main():
    # ---------------------------------------------------------
    # 1. Define Paths & Initialize Directories
    # ---------------------------------------------------------
    input_path = "../data/movielens_merged.csv"
    metadata_dir = "../data/metadata"
    processed_dir = "../data/processed"

    os.makedirs(metadata_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    # ---------------------------------------------------------
    # 2. Load Dataset
    # ---------------------------------------------------------
    print(f"Loading dataset from {input_path}...")
    # Load as string to prevent mixed-type issues before encoding
    df = pd.read_csv(input_path, dtype=str)
    print(f"Dataset loaded. Shape: {df.shape}")

    # ---------------------------------------------------------
    # 3. Label Encoding
    # ---------------------------------------------------------
    print("Applying Label Encoding...")
    label_encoders = {}
    
    for col in df.columns:
        if col != 'label':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le

    df['label'] = df['label'].astype(np.float32)

    # Save encoders for inference (e.g., Streamlit app)
    encoders_path = os.path.join(metadata_dir, "label_encoders.pkl")
    joblib.dump(label_encoders, encoders_path)
    print(f"Label encoders saved to: {encoders_path}")

    # ---------------------------------------------------------
    # 4. Train-Test Split & Save
    # ---------------------------------------------------------
    print("Splitting dataset into train and test sets...")
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    train_path = os.path.join(processed_dir, "train.csv")
    test_path = os.path.join(processed_dir, "test.csv")
    
    # Save datasets without the index column
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    print("Train and test datasets saved to processed directory.")

    # ---------------------------------------------------------
    # 5. Calculate & Save Field Dimensions
    # ---------------------------------------------------------
    print("Calculating field dimensions for embedding layers...")
    u_i_features = ['user_id', 'movie_id']
    meta_features = [
        'movie_decade', 'movie_year', 'rating_year', 'rating_month', 
        'rating_decade', 'genre1', 'genre2', 'genre3', 'gender', 'age', 
        'occupation', 'zip'
    ]

    # Calculate max index + 1 for each categorical feature
    field_dims = np.max(df[u_i_features + meta_features].astype(np.int64).values, axis=0) + 1

    field_dims_path = os.path.join(metadata_dir, "field_dims.npy")
    np.save(field_dims_path, field_dims)
    print(f"Field dimensions saved to: {field_dims_path}")
    print("Data preprocessing completed successfully.")

if __name__ == "__main__":
    main()