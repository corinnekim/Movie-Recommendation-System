import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Dense, Flatten, Dropout, BatchNormalization, Embedding

# ==============================================================================
# 1. Feature Embedding Layer
# ==============================================================================
class FeatureEmbedding(Layer):
    """
    Translates sparse categorical features into dense embedding vectors.
    """
    def __init__(self, field_dims, embed_dim, **kwargs):
        super(FeatureEmbedding, self).__init__(**kwargs)
        self.total_features = sum(field_dims)
        self.embed_dim = embed_dim
        # Calculate offsets to map categorical values to a single continuous space
        self.offsets = tf.constant(
            np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.int32), dtype=tf.int32
        )
        self.embedding = Embedding(input_dim=self.total_features, output_dim=self.embed_dim)

    def call(self, inputs):
        # inputs shape: (batch_size, num_fields)
        # Shift indices using offsets to prevent overlap between different fields
        shifted_inputs = tf.cast(inputs, tf.int32) + self.offsets
        return self.embedding(shifted_inputs) # Output shape: (batch_size, num_fields, embed_dim)


# ==============================================================================
# 2. Deep Neural Network (MLP)
# ==============================================================================
class MultiLayerPerceptron(Layer):
    """
    Standard Feedforward Neural Network (DNN track) to learn implicit high-order interactions.
    """
    def __init__(self, hidden_units, dropout_rate=0.2, use_bn=False, **kwargs):
        super(MultiLayerPerceptron, self).__init__(**kwargs)
        self.use_bn = use_bn
        self.dense_layers = []
        self.bn_layers = []
        self.dropouts = []

        # Construct stacked Dense layers
        for units in hidden_units:
            self.dense_layers.append(Dense(units, activation='relu'))
            if self.use_bn:
                self.bn_layers.append(BatchNormalization())
            self.dropouts.append(Dropout(dropout_rate))
            
        # Final output layer for MLP track (outputs a single scalar score)
        self.output_layer = Dense(1, activation=None)

    def call(self, inputs, training=False):
        x = inputs
        for i in range(len(self.dense_layers)):
            x = self.dense_layers[i](x)
            if self.use_bn:
                x = self.bn_layers[i](x, training=training)
            x = self.dropouts[i](x, training=training)
        return self.output_layer(x)


# ==============================================================================
# 3. Multi-Head Self-Attention (The core of AutoInt)
# ==============================================================================
class InteractingLayer(Layer):
    """
    Multi-Head Self-Attention layer based on the AutoInt paper.
    Learns explicit combinations of different features.
    """
    def __init__(self, embed_dim, num_heads, use_residual=True, **kwargs):
        super(InteractingLayer, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.use_residual = use_residual
        
        # Dimensions for each attention head
        self.head_dim = embed_dim // num_heads 
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        # Keras built-in MultiHeadAttention is highly optimized and standard for this operation
        self.attention = tf.keras.layers.MultiHeadAttention(
            num_heads=self.num_heads, 
            key_dim=self.head_dim
        )
        
        # Residual connection projection (if needed to match dimensions)
        if self.use_residual:
            self.res_dense = Dense(embed_dim, use_bias=False)

    def call(self, inputs, training=False):
        # Apply Self-Attention: Query, Key, and Value are derived from the same inputs
        attended_features = self.attention(
            query=inputs, value=inputs, key=inputs, training=training
        )
        
        # Apply residual connection if enabled
        if self.use_residual:
            residual = self.res_dense(inputs)
            attended_features = attended_features + residual
            
        return tf.nn.relu(attended_features)


# ==============================================================================
# 4. Joint Model: AutoInt+ (Combining Attention and MLP)
# ==============================================================================
class AutoIntPlus(Model):
    """
    The complete AutoInt+ Model connecting the Embedding, Attention, and DNN tracks.
    """
    def __init__(self, field_dims, embed_dim=16, att_layers=3, num_heads=2, 
                 mlp_hidden_units=(32, 32), dropout=0.2, **kwargs):
        super(AutoIntPlus, self).__init__(**kwargs)
        
        self.num_fields = len(field_dims)
        self.embed_dim = embed_dim
        
        # 1. Embedding Component
        self.embedding = FeatureEmbedding(field_dims, embed_dim)
        
        # 2. Attention Component (AutoInt track)
        self.attention_layers = [InteractingLayer(embed_dim, num_heads) for _ in range(att_layers)]
        self.attention_output = Dense(1, activation=None) # Projects attention output to scalar score
        
        # 3. DNN Component (MLP track)
        self.mlp = MultiLayerPerceptron(mlp_hidden_units, dropout_rate=dropout)

    def call(self, inputs, training=False):
        # Step 1: Embed the sparse categorical inputs
        emb_features = self.embedding(inputs)
        
        # Step 2: Pass through the Attention Track (AutoInt)
        att_x = emb_features
        for layer in self.attention_layers:
            att_x = layer(att_x, training=training) 
            
        att_x = Flatten()(att_x)
        att_score = self.attention_output(att_x)
        
        # Step 3: Pass through the DNN Track (MLP)
        mlp_input = Flatten()(emb_features)
        mlp_score = self.mlp(mlp_input, training=training)
        
        # Step 4: Combine logits and apply Sigmoid for the final prediction
        final_logits = att_score + mlp_score
        y_pred = tf.nn.sigmoid(final_logits)
        
        return y_pred
    
# ==============================================================================
# 5. Evaluation Utilities
# ==============================================================================
def get_hit_rate(ranklist, y_true):
    """
    Calculates Hit Rate@K using set intersection for O(1) lookups.
    """
    hits = set(ranklist).intersection(set(y_true))
    return len(hits) / len(y_true) if len(y_true) > 0 else 0.0

def get_NDCG(ranklist, y_true):
    """
    Calculates Normalized Discounted Cumulative Gain (NDCG@K) using standard log2.
    """
    dcg = 0.0
    idcg = 0.0
    
    # Calculate DCG (Discounted Cumulative Gain)
    for i, item in enumerate(ranklist):
        if item in y_true:
            dcg += 1.0 / np.log2(i + 2)
            
    # Calculate IDCG (Ideal DCG)
    for i in range(min(len(y_true), len(ranklist))):
        idcg += 1.0 / np.log2(i + 2)
        
    return round((dcg / idcg), 5) if idcg > 0 else 0.0

def test_model(model, test_df, batch_size=2048):
    """
    Generates predictions natively using Keras batching and Pandas groupby.
    """
    # 1. Extract features and IDs
    features = test_df.iloc[:, :-1].values
    user_ids = test_df['user_id'].astype(int).values
    item_ids = test_df['movie_id'].astype(int).values

    # 2. Native Keras batched prediction 
    preds = model.predict(features, batch_size=batch_size, verbose=0).flatten()

    # 3. Create a DataFrame for fast operations
    results_df = pd.DataFrame({
        'user_id': user_ids,
        'movie_id': item_ids,
        'pred': preds
    })

    # 4. Group by user_id efficiently using Pandas C-engine
    user_pred_info = results_df.groupby('user_id').apply(
        lambda x: list(zip(x['movie_id'], x['pred']))
    ).to_dict()

    return user_pred_info