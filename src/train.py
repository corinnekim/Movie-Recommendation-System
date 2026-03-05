import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Import custom model and utilities from model.py
from model import AutoIntPlus, test_model, get_NDCG, get_hit_rate

def main():
    # =========================================================
    # 1. Load Preprocessed Data & Meta-info
    # =========================================================
    print("Loading processed datasets...")
    # These paths assume you run the script from the 'src' directory
    train_df = pd.read_csv('../data/processed/train.csv')
    test_df = pd.read_csv('../data/processed/test.csv')
    
    # Load field dimensions for embedding layer configuration
    field_dims = np.load('../data/metadata/field_dims.npy')
    
    # Define feature groups and target label
    u_i_feature = ['user_id', 'movie_id']
    meta_features = [
        'movie_decade', 'movie_year', 'rating_year', 'rating_month', 
        'rating_decade', 'genre1', 'genre2', 'genre3', 'gender', 'age', 
        'occupation', 'zip'
    ]
    label = 'label'

    # Cast labels to float32 for TensorFlow optimization
    train_df[label] = train_df[label].astype(np.float32)
    test_df[label] = test_df[label].astype(np.float32)

    # =========================================================
    # 2. Hyperparameters Setup
    # =========================================================
    epochs = 5
    learning_rate = 0.0001
    dropout_rate = 0.4
    batch_size = 2048
    embed_dim = 16

    # =========================================================
    # 3. Initialize and Compile Model
    # =========================================================
    print("Initializing AutoInt+ Model...")
    model = AutoIntPlus(
        field_dims=field_dims, 
        embed_dim=embed_dim, 
        att_layers=3, 
        num_heads=2, 
        mlp_hidden_units=(32, 32), 
        dropout=dropout_rate
    )

    # Compile with Adam optimizer and Binary Crossentropy loss
    optimizer = Adam(learning_rate=learning_rate)
    loss_fn = BinaryCrossentropy(from_logits=False) 
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['binary_crossentropy'])

    # =========================================================
    # 4. Model Training
    # =========================================================
    print(f"Starting Training for {epochs} epochs...")
    train_features = train_df[u_i_feature + meta_features].values
    train_labels = train_labels = train_df[label].values.reshape(-1, 1)

    model.fit(
        x=train_features, 
        y=train_labels, 
        epochs=epochs, 
        batch_size=batch_size, 
        validation_split=0.1
    )

    # =========================================================
    # 5. Evaluation (Top-K Recommendation Metrics)
    # =========================================================
    print("Evaluating Model on Test Data...")
    top_k = 10
    user_pred_info = {}
    
    # Generate batch predictions for all test samples
    raw_predictions = test_model(model, test_df)

    # Sort and extract Top-K items per user
    for user, data_info in tqdm(raw_predictions.items(), desc="Processing Top-K"):
        ranklist = sorted(data_info, key=lambda s: s[1], reverse=True)[:top_k]
        # Store only item IDs for evaluation
        user_pred_info[str(user)] = [item[0] for item in ranklist]

    # Group ground truth (labels=1) by user
    test_truth = test_df[test_df['label'] == 1].groupby('user_id')['movie_id'].apply(list)

    ndcg_scores = []
    hr_scores = []

    for user, true_items in tqdm(test_truth.items(), desc="Calculating Metrics"):
        predicted = user_pred_info.get(str(user), [])
        true_set = list(set(np.array(true_items).astype(int)))
        
        ndcg_scores.append(get_NDCG(predicted, true_set))
        hr_scores.append(get_hit_rate(predicted, true_set))

    print(f"\n[Final Results]")
    print(f"Mean NDCG@{top_k}: {np.mean(ndcg_scores):.5f}")
    print(f"Mean HitRate@{top_k}: {np.mean(hr_scores):.5f}")

    # =========================================================
    # 6. Save Model Weights
    # =========================================================
    os.makedirs('../models', exist_ok=True)
    save_path = '../models/autoInt_model_weights.weights.h5'
    model.save_weights(save_path)
    print(f"Training Complete. Weights saved to: {save_path}")

if __name__ == "__main__":
    main()