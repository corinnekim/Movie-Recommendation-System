import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import joblib
from model import AutoIntPlus

# Mappings
AGE_MAP = {"1": "Under 18", "18": "18-24", "25": "25-34", "35": "35-44",
           "45": "45-49", "50": "50-55", "56": "56+"}

OCCUPATION_MAP = {0:"other", 1:"academic/educator", 2:"artist", 3:"clerical/admin",
                  4:"college/grad student", 5:"customer service", 6:"doctor/health care",
                  7:"executive/managerial", 8:"farmer", 9:"homemaker", 10:"K-12 student",
                  11:"lawyer", 12:"programmer", 13:"retired", 14:"sales/marketing",
                  15:"scientist", 16:"self-employed", 17:"technician/engineer",
                  18:"tradesman/craftsman", 19:"unemployed", 20:"writer"}

# Assets & Data Loading
@st.cache_resource
def load_assets():
    """모델 구조, 가중치, 미리 학습된 encoder를 로드한다."""
    metadata_dir = "../data/metadata"
    model_path = "../models/autoInt_model_weights.weights.h5"
    field_dims = np.load(os.path.join(metadata_dir, "field_dims.npy"))
    label_encoders = joblib.load(os.path.join(metadata_dir, "label_encoders.pkl"))

    model = AutoIntPlus(field_dims=field_dims, embed_dim=16)
    model(tf.zeros((1, len(field_dims))))
    model.load_weights(model_path)
    return model, label_encoders

@st.cache_data
def load_csv_data():
    """전처리된 데이터셋을 로드하고 컬럼명을 통일한다."""
    path = "../data/movielens"
    u_df = pd.read_csv(f"{path}/users_preprocessed.csv", dtype=str)
    m_df = pd.read_csv(f"{path}/movies_preprocessed.csv", dtype=str)
    r_df = pd.read_csv(f"{path}/ratings_preprocessed.csv", dtype=str)
    m_df = m_df.rename(columns={'year': 'movie_year', 'decade': 'movie_decade'})
    return u_df, m_df, r_df

model, label_encoders = load_assets()
users_df, movies_df, ratings_df = load_csv_data()

# UI Configuration
st.set_page_config(page_title="Movie Curator Pro", layout="wide")
st.title("🍿 Personalized Movie Curator")

st.sidebar.header("User Selection")
available_users = sorted(users_df['user_id'].unique().astype(int))
target_user_id = st.sidebar.selectbox("Choose User ID", available_users, key="unique_user_selectbox")

st.sidebar.subheader("Context Settings")
target_year = st.sidebar.slider("Simulation Year", 2000, 2003, 2000, key="unique_year_slider")
target_month = st.sidebar.slider("Simulation Month", 1, 12, 1, key="unique_month_slider")

# Main Dashboard Logic
if target_user_id:
    user_row = users_df[users_df['user_id'] == str(target_user_id)].iloc[0]

    # 프로필 섹션
    st.write(f"### 👤 User {target_user_id} Profile")
    st.markdown(f"**Gender:** {user_row['gender']} | **Age:** {AGE_MAP.get(user_row['age'], user_row['age'])} | **Occupation:** {OCCUPATION_MAP.get(int(user_row['occupation']), user_row['occupation'])}")

    # 히스토리 섹션
    st.write("### ✅ User's Past Top 10 Favorites")
    target_col = 'label' if 'label' in ratings_df.columns else 'rating'

    user_history = ratings_df[(ratings_df['user_id'] == str(target_user_id)) &
                              (ratings_df[target_col].isin(['1.0', '4', '5']))]

    merged_history = pd.merge(user_history, movies_df, on='movie_id', how='inner')
    history_display = merged_history.head(10).copy()

    if not history_display.empty:
        history_display.index = np.arange(1, len(history_display) + 1)
        st.table(history_display[['title', 'movie_year', 'genre1', 'genre2']])
    else:
        st.info("No favorite history found for this user.")

    st.divider()

    # 추천 엔진
    if st.sidebar.button("Show New Recommendations", key="unique_recommend_button"):
        with st.spinner("AI analyzing..."):

            # 유저 히스토리 분석 및 blacklist 생성
            target_col = 'label' if 'label' in ratings_df.columns else 'rating'
            user_log = ratings_df[ratings_df['user_id'] == str(target_user_id)].copy()
            user_log = pd.merge(user_log, movies_df, on='movie_id', how='inner')

            # 유저가 상호작용한 모든 고유 genre 추출
            all_interacted_genres = set(user_log['genre1'].dropna()) | \
                                    set(user_log['genre2'].dropna()) | \
                                    set(user_log['genre3'].dropna())

            blacklist = set()

            # 민감 genre는 무관용 처리
            sensitive_genres = ['Horror', "Children's"]
            for sg in sensitive_genres:
                if sg not in all_interacted_genres:
                    blacklist.add(sg)

            # 통계적 비선호 탐지 (3회 이상 시청 & 80% 이상 비선호)
            for g in all_interacted_genres:
                g_movies = user_log[(user_log['genre1'] == g) |
                                    (user_log['genre2'] == g) |
                                    (user_log['genre3'] == g)]
                total_views = len(g_movies)

                if total_views >= 3:
                    # 비선호(0.0 또는 2점 이하로 매핑) 개수 집계
                    dislike_count = len(g_movies[g_movies[target_col].isin(['0.0', '1', '2'])])
                    if (dislike_count / total_views) >= 0.8:
                        blacklist.add(g)

            # 순서 무관 combo 태깅 및 선호 추출
            def apply_unordered_combo(df):
                """genre 순서를 무시한 통합 combo 문자열을 만든다.
                   예: 'Romance'와 'Comedy'는 항상 'Comedy + Romance'가 된다."""
                combos = []
                for g1, g2 in zip(df['genre1'].fillna(""), df['genre2'].fillna("")):
                    genres = [g for g in (g1, g2) if g]
                    combos.append(" + ".join(sorted(genres)))  # 알파벳순 정렬
                return combos

            # 선호 정의를 위한 positive 히스토리 식별
            positive_log = user_log[user_log[target_col].isin(['1.0', '4', '5'])].copy()

            if not positive_log.empty:
                # Top 5 순서 무관 combo (Group A용)
                positive_log['combo'] = apply_unordered_combo(positive_log)
                user_top_5_combos = positive_log['combo'].value_counts().head(5).index.tolist()

                # Top 5 개별 genre (Group B, C의 anchor용)
                all_positive_genres = pd.concat([positive_log['genre1'],
                                                 positive_log['genre2'],
                                                 positive_log['genre3']]).dropna()
                user_top_5_genres = all_positive_genres.value_counts().head(5).index.tolist()
            else:
                user_top_5_combos, user_top_5_genres = [], []

            # Bridge/Explore 아이템이 진짜 새로운지 확인하기 위해 본 combo 전체 추적
            user_log['combo'] = apply_unordered_combo(user_log)
            all_seen_combos = set(user_log['combo'].dropna().unique())
            seen_movie_ids = user_log['movie_id'].unique()

            # 이미 본 영화 제거 후 후보 준비
            candidates = movies_df[~movies_df['movie_id'].isin(seen_movie_ids)].copy()
            candidates['combo'] = apply_unordered_combo(candidates)

            # blacklist 적용: blacklist genre를 포함한 후보 제거
            if blacklist:
                bl_mask = candidates['genre1'].isin(blacklist) | \
                          candidates['genre2'].isin(blacklist) | \
                          candidates['genre3'].isin(blacklist)
                candidates = candidates[~bl_mask].copy()

            # 계층 그룹화 (7:2:1 전략)
            # 안전한 범용/경험 genre 정의 (Tier 1 & 2)
            safe_tiers = ['Drama', 'Comedy', 'Romance', 'Adventure',
                          'Action', 'Sci-Fi', 'Fantasy', 'Animation', 'Musical']

            # Group A (Exploit: 7개)
            # Top 5 순서 무관 combo와 정확히 일치
            group_a_pool = candidates[candidates['combo'].isin(user_top_5_combos)].copy()

            # Group B (Safe Bridge: 2개)
            # G1은 유저 Top 5 genre에, G2는 Tier 1 또는 2 genre에 속해야 함
            bridge_mask = (candidates['genre1'].isin(user_top_5_genres)) & \
                          (candidates['genre2'].isin(safe_tiers)) & \
                          (~candidates['combo'].isin(all_seen_combos))
            group_b_pool = candidates[bridge_mask].copy()

            # Group C (Safety Discovery: 1개)
            # 본 적 없는 combo, Top 5 genre 또는 Tier 1/2 genre를 최소 하나 포함해야 함
            safety_anchors = set(user_top_5_genres + safe_tiers)
            discovery_mask = (~candidates['combo'].isin(all_seen_combos)) & \
                             (~candidates['movie_id'].isin(group_b_pool['movie_id'])) & \
                             (candidates['genre1'].isin(safety_anchors) | \
                              candidates['genre2'].isin(safety_anchors) | \
                              candidates['genre3'].isin(safety_anchors))
            group_c_pool = candidates[discovery_mask].copy()

            # 모델 inference 안전을 위해 학습에 없던 label 정리
            movie_cols = ['movie_id', 'movie_decade', 'movie_year', 'genre1', 'genre2', 'genre3']
            for col in movie_cols:
                le = label_encoders[col]
                group_a_pool = group_a_pool[group_a_pool[col].isin(le.classes_)]
                group_b_pool = group_b_pool[group_b_pool[col].isin(le.classes_)]
                group_c_pool = group_c_pool[group_c_pool[col].isin(le.classes_)]

            # Inference 및 최종 혼합
            def get_scores(model_instance, df_candidates, u_row, r_yr, r_mo, r_dec):
                n = len(df_candidates)
                if n == 0: return df_candidates

                inf_data = np.zeros((n, 14), dtype=np.int32)
                feature_order = ['user_id', 'movie_id', 'movie_decade', 'movie_year',
                                 'rating_year', 'rating_month', 'rating_decade',
                                 'genre1', 'genre2', 'genre3', 'gender', 'age', 'occupation', 'zip']

                for i, col in enumerate(feature_order):
                    le = label_encoders[col]
                    if col in df_candidates.columns:
                        inf_data[:, i] = le.transform(df_candidates[col])
                    else:
                        val = str(target_user_id) if col == 'user_id' else \
                              r_yr if col == 'rating_year' else \
                              r_mo if col == 'rating_month' else \
                              r_dec if col == 'rating_decade' else \
                              u_row[col]

                        inf_data[:, i] = le.transform([val])[0] if val in le.classes_ else 0

                tensor_features = tf.constant(inf_data, dtype=tf.int64)
                preds = model_instance(tensor_features, training=False)
                df_candidates['score'] = preds.numpy().flatten()
                return df_candidates

            r_year, r_month = str(target_year), str(target_month)
            r_decade = f"{target_year // 10 * 10}s"

            # 각 pool 채점
            res_a = get_scores(model, group_a_pool, user_row, r_year, r_month, r_decade)
            res_b = get_scores(model, group_b_pool, user_row, r_year, r_month, r_decade)
            res_c = get_scores(model, group_c_pool, user_row, r_year, r_month, r_decade)

            # 정렬 후 7:2:1로 선택, Group A는 다양성 제약 적용
            if not res_a.empty:
                sorted_a = res_a.sort_values(by='score', ascending=False)
                selected_a_rows = []
                combo_counts = {}

                # combo당 최대 3개 제약 적용
                for _, row in sorted_a.iterrows():
                    c = row['combo']
                    if combo_counts.get(c, 0) < 3:
                        selected_a_rows.append(row)
                        combo_counts[c] = combo_counts.get(c, 0) + 1

                    if len(selected_a_rows) == 7:
                        break
                top_a = pd.DataFrame(selected_a_rows)
            else:
                top_a = pd.DataFrame()

            top_b = res_b.sort_values(by='score', ascending=False).head(2) if not res_b.empty else pd.DataFrame()
            top_c = res_c.sort_values(by='score', ascending=False).head(1) if not res_c.empty else pd.DataFrame()

            final_recs = pd.concat([top_a, top_b, top_c])

            # Fallback: 정확히 10개를 보장
            # Fallback 1: B 또는 C가 너무 엄격했으면 남은 자리를 Group A로 채움
            if len(final_recs) < 10 and not res_a.empty:
                shortfall = 10 - len(final_recs)
                remaining_a = res_a[~res_a['movie_id'].isin(final_recs['movie_id'])].sort_values(by='score', ascending=False).head(shortfall)
                final_recs = pd.concat([final_recs, remaining_a])

            # 최종 fallback: 그래도 10개 미만이면 안전한 미선택 후보로 채움
            if len(final_recs) < 10:
                shortfall = 10 - len(final_recs)
                remaining_candidates = candidates[~candidates['movie_id'].isin(final_recs['movie_id'])].copy()

                # 모델 inference 안전을 위해 학습에 없던 label 정리
                for col in movie_cols:
                    le = label_encoders[col]
                    remaining_candidates = remaining_candidates[remaining_candidates[col].isin(le.classes_)]

                # 남은 pool 채점
                res_remaining = get_scores(model, remaining_candidates, user_row, r_year, r_month, r_decade)
                if not res_remaining.empty:
                    fillers = res_remaining.sort_values(by='score', ascending=False).head(shortfall)
                    final_recs = pd.concat([final_recs, fillers])

            # 최종 출력
            st.write(f"### ✨ Top 10 Recommendations")

            final_recs['Match %'] = final_recs['score'].apply(lambda x: f"{x:.1%}")
            final_recs.index = np.arange(1, len(final_recs) + 1)

            # 테이블 표시
            st.table(final_recs[['title', 'movie_year', 'genre1', 'genre2', 'Match %']])
