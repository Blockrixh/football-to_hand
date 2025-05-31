import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import xgboost as xgb
from catboost import CatBoostClassifier

# 페이지 설정
st.set_page_config(page_title="핸디캡 경기 예측기", layout="centered")
st.title("📊 핸디캡 경기 결과 예측기")

# 📂 파일 업로드
uploaded_file = st.file_uploader("📥 before.csv 파일을 업로드해주세요", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # 🧪 피처 불러오기
    with open("feature_columns_handicap.pkl", "rb") as f:
        feature_cols = pickle.load(f)

    # ✅ 모델 불러오기
    with open("model_hand_cat.pkl", "rb") as f:
        model_cat = pickle.load(f)

    model_xgb = xgb.Booster()
    model_xgb.load_model("model_hand_xgb.json")

    # 🎯 라벨 디코딩용 클래스 불러오기
    with open("label_encoder_hand.json", "r", encoding="utf-8") as f:
        label_map = json.load(f)
    index_to_label = {v: k for k, v in label_map.items()}

    # ✅ 입력값 체크
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        st.error(f"❌ 필요한 컬럼이 누락되었습니다: {missing_cols}")
    else:
        X_input = df[feature_cols]

        # 🎯 예측
        proba_cat = model_cat.predict_proba(X_input)
        dmatrix = xgb.DMatrix(X_input)
        proba_xgb = model_xgb.predict(dmatrix)

        proba_ensemble = (proba_cat + proba_xgb) / 2
        final_pred = np.argmax(proba_ensemble, axis=1)
        result_label = [index_to_label[i] for i in final_pred]

        # 🧾 결과 출력
        df_result = df.copy()
        df_result["예측결과"] = result_label
        df_result["무확률"] = proba_ensemble[:, 0].round(3)
        df_result["승확률"] = proba_ensemble[:, 1].round(3)
        df_result["패확률"] = proba_ensemble[:, 2].round(3)

        st.success("✅ 예측 완료!")
        st.dataframe(df_result)

        # 📥 결과 다운로드
        csv = df_result.to_csv(index=False).encode("utf-8-sig")
        st.download_button("📥 결과 다운로드", data=csv, file_name="prediction_result.csv", mime="text/csv")
