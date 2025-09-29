# streamlit_app.py
import streamlit as st
import pandas as pd
import joblib
import numpy as np
from utils_detect_target import detect_target
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import plotly.express as px

st.set_page_config(layout='wide', page_title='Claims Fraud Detector')

st.title("Insurance Claims Fraud Detector")
st.markdown("Upload dataset to train or load an existing trained pipeline (`models/fraud_model.joblib`).")

# Sidebar: load or train
option = st.sidebar.selectbox("Mode", ["Load Model", "Train Model (upload CSV)"])

if option == "Train Model (upload CSV)":
    uploaded_file = st.sidebar.file_uploader("Upload claims CSV", type=['csv'])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Preview data:", df.head())
        # attempt to detect target
        target = detect_target(df)
        st.write("Detected target column:", target)
        if st.sidebar.button("Train model now"):
            from train_pipeline import train_model
            pipeline, X_test, y_test = train_model(uploaded_file)
            joblib.dump(pipeline, 'models/fraud_model.joblib')
            st.success("Model trained and saved to models/fraud_model.joblib")
            # show simple evaluation
            y_pred = pipeline.predict(X_test)
            y_proba = pipeline.predict_proba(X_test)[:,1] if hasattr(pipeline, "predict_proba") else None
            st.subheader("Classification Report")
            st.text(classification_report(y_test, y_pred))
            if y_proba is not None:
                st.write("ROC AUC:", roc_auc_score(y_test, y_proba))

else:
    # Load existing model
    try:
        pipeline = joblib.load('models/fraud_model.joblib')
        st.success("Loaded models/fraud_model.joblib")
    except Exception as e:
        st.error("No pretrained model found at models/fraud_model.joblib. Train first or upload a model.")
        st.stop()

    st.header("Predict single claim")
    st.markdown("Enter claim features below. Leave blank to use example values.")

    # Build input form dynamically from pipeline preprocessor feature names if possible
    # Simpler: ask user to paste a CSV-row or JSON. We'll provide a small manual input interface.
    user_input = {}
    st.subheader("Manual entry")
    col1, col2 = st.columns(2)
    # A few sample fields — you should replace these with your dataset's real features
    with col1:
        user_input['claim_amount'] = st.number_input("claim_amount", min_value=0.0, value=1000.0)
        user_input['age'] = st.number_input("age", min_value=0, value=35)
        user_input['vehicle_year'] = st.number_input("vehicle_year", min_value=1900, max_value=2025, value=2015)
    with col2:
        user_input['policy_state'] = st.text_input("policy_state", value="NY")
        user_input['incident_type'] = st.text_input("incident_type", value="Collision")
        user_input['num_prior_claims'] = st.number_input("num_prior_claims", min_value=0, value=0)

    if st.button("Predict"):
        # Convert to DataFrame with single row
        x_new = pd.DataFrame([user_input])
        try:
            proba = pipeline.predict_proba(x_new)[:,1][0]
            pred = pipeline.predict(x_new)[0]
            st.write("Predicted label (1 = fraud):", int(pred))
            st.write("Fraud probability:", float(proba))
            if float(proba) > 0.5:
                st.warning("Potential Fraud — investigate further")
            else:
                st.success("Not likely fraud")
        except Exception as e:
            st.error("Error applying model to your input. You may need to provide feature names matching the training data. Error: " + str(e))

    st.header("Model insights")
    # show feature importances if available
    try:
        clf = pipeline.named_steps['clf']
        preproc = pipeline.named_steps['preprocessor']
        # attempt to get feature names
        feature_names = []
        try:
            # numeric names
            num = preproc.transformers_[0][2]
            # This part can be adjusted depending on ColumnTransformer internals
        except Exception:
            feature_names = None

        if hasattr(clf, 'feature_importances_'):
            st.subheader("Top feature importances")
            fi = clf.feature_importances_
            # best effort: show top 10 indices
            idx = np.argsort(fi)[-10:][::-1]
            for i in idx:
                st.write(f"Feature idx {i}, importance {fi[i]:.4f}")
    except Exception as e:
        st.info("Feature importances not available for the loaded pipeline.")

    st.header("Upload a CSV to score")
    score_file = st.file_uploader("Upload CSV to score", type=['csv'])
    if score_file is not None:
        df_score = pd.read_csv(score_file)
        try:
            preds = pipeline.predict(df_score)
            probs = pipeline.predict_proba(df_score)[:,1] if hasattr(pipeline, 'predict_proba') else None
            df_score['pred_fraud'] = preds
            if probs is not None:
                df_score['fraud_prob'] = probs
            st.write(df_score.head())
            st.download_button("Download scored CSV", df_score.to_csv(index=False), "scored_claims.csv")
        except Exception as e:
            st.error("Failed scoring: " + str(e))
