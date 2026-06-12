import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Dense, Flatten, Dropout, BatchNormalization, Embedding


# Feature Embedding Layer
class FeatureEmbedding(Layer):
    """sparse categorical feature를 dense embedding vector로 변환한다."""
    def __init__(self, field_dims, embed_dim, **kwargs):
        super(FeatureEmbedding, self).__init__(**kwargs)
        self.total_features = sum(field_dims)
        self.embed_dim = embed_dim
        # categorical 값을 하나의 연속된 index 공간으로 매핑하기 위한 offset
        self.offsets = tf.constant(
            np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.int32), dtype=tf.int32
        )
        self.embedding = Embedding(input_dim=self.total_features, output_dim=self.embed_dim)

    def call(self, inputs):
        # inputs shape: (batch_size, num_fields)
        # offset을 더해 field 간 index가 겹치지 않게 이동시킨다
        shifted_inputs = tf.cast(inputs, tf.int32) + self.offsets
        return self.embedding(shifted_inputs)  # 출력 shape: (batch_size, num_fields, embed_dim)


# Deep Neural Network (MLP)
class MultiLayerPerceptron(Layer):
    """고차 implicit interaction을 학습하는 DNN track."""
    def __init__(self, hidden_units, dropout_rate=0.2, use_bn=False, **kwargs):
        super(MultiLayerPerceptron, self).__init__(**kwargs)
        self.use_bn = use_bn
        self.dense_layers = []
        self.bn_layers = []
        self.dropouts = []

        # Dense layer를 쌓아 구성
        for units in hidden_units:
            self.dense_layers.append(Dense(units, activation='relu'))
            if self.use_bn:
                self.bn_layers.append(BatchNormalization())
            self.dropouts.append(Dropout(dropout_rate))

        # MLP track의 최종 출력층 (scalar score 하나를 출력)
        self.output_layer = Dense(1, activation=None)

    def call(self, inputs, training=False):
        x = inputs
        for i in range(len(self.dense_layers)):
            x = self.dense_layers[i](x)
            if self.use_bn:
                x = self.bn_layers[i](x, training=training)
            x = self.dropouts[i](x, training=training)
        return self.output_layer(x)


# Multi-Head Self-Attention (AutoInt의 핵심)
class InteractingLayer(Layer):
    """AutoInt 논문 기반 Multi-Head Self-Attention layer. feature 간 explicit 조합을 학습한다."""
    def __init__(self, embed_dim, num_heads, use_residual=True, **kwargs):
        super(InteractingLayer, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.use_residual = use_residual

        # attention head 하나당 차원
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        # Keras 내장 MultiHeadAttention 사용
        self.attention = tf.keras.layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.head_dim
        )

        # 차원을 맞추기 위한 residual connection projection
        if self.use_residual:
            self.res_dense = Dense(embed_dim, use_bias=False)

    def call(self, inputs, training=False):
        # query, key, value를 같은 input에서 뽑아 self-attention 적용
        attended_features = self.attention(
            query=inputs, value=inputs, key=inputs, training=training
        )

        # residual connection 적용
        if self.use_residual:
            residual = self.res_dense(inputs)
            attended_features = attended_features + residual

        return tf.nn.relu(attended_features)


# Joint Model: AutoInt+ (Attention과 MLP 결합)
class AutoIntPlus(Model):
    """Embedding, Attention, DNN track을 연결한 전체 AutoInt+ 모델."""
    def __init__(self, field_dims, embed_dim=16, att_layers=3, num_heads=2,
                 mlp_hidden_units=(32, 32), dropout=0.2, **kwargs):
        super(AutoIntPlus, self).__init__(**kwargs)

        self.num_fields = len(field_dims)
        self.embed_dim = embed_dim

        # embedding 구성 요소
        self.embedding = FeatureEmbedding(field_dims, embed_dim)

        # attention 구성 요소 (AutoInt track)
        self.attention_layers = [InteractingLayer(embed_dim, num_heads) for _ in range(att_layers)]
        self.attention_output = Dense(1, activation=None)  # attention 출력을 scalar score로 projection

        # DNN 구성 요소 (MLP track)
        self.mlp = MultiLayerPerceptron(mlp_hidden_units, dropout_rate=dropout)

    def call(self, inputs, training=False):
        # sparse categorical input을 embedding
        emb_features = self.embedding(inputs)

        # attention track(AutoInt) 통과
        att_x = emb_features
        for layer in self.attention_layers:
            att_x = layer(att_x, training=training)

        att_x = Flatten()(att_x)
        att_score = self.attention_output(att_x)

        # DNN track(MLP) 통과
        mlp_input = Flatten()(emb_features)
        mlp_score = self.mlp(mlp_input, training=training)

        # 두 logit을 합치고 sigmoid를 적용해 최종 예측
        final_logits = att_score + mlp_score
        y_pred = tf.nn.sigmoid(final_logits)

        return y_pred


# Evaluation Utilities
def get_hit_rate(ranklist, y_true):
    """set 교집합으로 Hit Rate@K를 계산한다."""
    hits = set(ranklist).intersection(set(y_true))
    return len(hits) / len(y_true) if len(y_true) > 0 else 0.0

def get_NDCG(ranklist, y_true):
    """log2 기반으로 NDCG@K(Normalized Discounted Cumulative Gain)를 계산한다."""
    dcg = 0.0
    idcg = 0.0

    # DCG(Discounted Cumulative Gain) 계산
    for i, item in enumerate(ranklist):
        if item in y_true:
            dcg += 1.0 / np.log2(i + 2)

    # IDCG(Ideal DCG) 계산
    for i in range(min(len(y_true), len(ranklist))):
        idcg += 1.0 / np.log2(i + 2)

    return round((dcg / idcg), 5) if idcg > 0 else 0.0

def test_model(model, test_df, batch_size=2048):
    """Keras batch 예측과 pandas groupby로 user별 예측 결과를 생성한다."""
    # feature와 ID 추출
    features = test_df.iloc[:, :-1].values
    user_ids = test_df['user_id'].astype(int).values
    item_ids = test_df['movie_id'].astype(int).values

    # Keras batch 예측
    preds = model.predict(features, batch_size=batch_size, verbose=0).flatten()

    # 빠른 연산을 위해 DataFrame 구성
    results_df = pd.DataFrame({
        'user_id': user_ids,
        'movie_id': item_ids,
        'pred': preds
    })

    # user_id 기준 groupby
    user_pred_info = results_df.groupby('user_id').apply(
        lambda x: list(zip(x['movie_id'], x['pred']))
    ).to_dict()

    return user_pred_info