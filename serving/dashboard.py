"""
VoiceIntent Streamlit Dashboard
Member 4 - Ibrahim Noor

Interactive dashboard for predictions, analytics, and pipeline monitoring.
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from storage.db import get_session
from storage.models import Prediction, ModelRun, Call
from sqlalchemy import func, desc

st.set_page_config(
    page_title="VoiceIntent Dashboard",
    page_icon="🎤",
    layout="wide"
)

API_URL = "http://localhost:8000"

st.title("🎤 VoiceIntent Dashboard")
st.markdown("**Automated Voice-to-Intent Intelligence Pipeline**")
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["🎯 Predict", "📊 Analytics", "⚙️ Pipeline Status"])

with tab1:
    st.header("Intent Prediction")
    st.markdown("Upload an audio file to predict the banking intent")
    
    uploaded_file = st.file_uploader(
        "Choose an audio file (.mp3 or .wav)",
        type=['mp3', 'wav']
    )
    
    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/mp3')
        
        if st.button("Predict Intent", type="primary"):
            with st.spinner("Processing audio..."):
                try:
                    # Call API
                    files = {"file": uploaded_file}
                    response = requests.post(f"{API_URL}/predict", files=files)
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        with col1:
                            st.success("✅ Prediction Complete")
                            st.metric("Predicted Intent", result['intent'])
                            st.metric("Confidence", f"{result['confidence']:.2%}")
                            st.caption(f"Model: {result['model_version']}")
                        
                        with col2:
                            st.subheader("Transcript")
                            st.info(result['transcript'])
                            
                            if result['raw_transcript'] != result['transcript']:
                                with st.expander("View raw transcript"):
                                    st.text(result['raw_transcript'])
                        
                        st.subheader("Top 5 Intent Predictions")
                        top_5 = pd.DataFrame(result['top_5_intents'])
                        
                        fig = px.bar(
                            top_5,
                            x='confidence',
                            y='intent',
                            orientation='h',
                            title='Confidence Scores',
                            labels={'confidence': 'Confidence', 'intent': 'Intent'}
                        )
                        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                        st.plotly_chart(fig, use_container_width=True)
                    
                    else:
                        st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
                
                except requests.exceptions.ConnectionError:
                    st.error("⚠️ Cannot connect to API. Make sure the backend is running on port 8000.")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

with tab2:
    st.header("Analytics Dashboard")
    
    try:
        # Get metrics from API
        response = requests.get(f"{API_URL}/metrics")
        
        if response.status_code == 200:
            metrics = response.json()
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Predictions", metrics['total_predictions'])
            
            with col2:
                st.metric("Avg Confidence", f"{metrics['average_confidence']:.2%}")
            
            with col3:
                model_acc = metrics['current_model'].get('accuracy')
                st.metric("Model Accuracy", f"{model_acc:.2%}" if model_acc else "N/A")
            
            with col4:
                drift_score = metrics['current_model'].get('drift_score')
                drift_status = "🔴 Alert" if metrics.get('drift_alert') else "🟢 Normal"
                st.metric("Drift Status", drift_status)
            
            st.markdown("---")
            
            # Intent distribution
            if metrics['intent_distribution']:
                st.subheader("Intent Distribution")
                
                intent_df = pd.DataFrame([
                    {'Intent': intent, 'Count': count}
                    for intent, count in metrics['intent_distribution'].items()
                ]).sort_values('Count', ascending=False).head(20)
                
                fig = px.bar(
                    intent_df,
                    x='Count',
                    y='Intent',
                    orientation='h',
                    title='Top 20 Predicted Intents',
                    labels={'Count': 'Number of Predictions', 'Intent': 'Intent'}
                )
                fig.update_layout(
                    height=600,
                    yaxis={'categoryorder': 'total ascending'}
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No predictions yet. Upload audio in the Predict tab to get started.")
            
            st.markdown("---")
            st.subheader("Current Model")
            
            model_info = metrics['current_model']
            if model_info['version']:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write("**Version:**", model_info['version'])
                    st.write("**Accuracy:**", f"{model_info['accuracy']:.4f}" if model_info['accuracy'] else "N/A")
                
                with col2:
                    st.write("**F1 Score:**", f"{model_info['f1_score']:.4f}" if model_info['f1_score'] else "N/A")
                    st.write("**Drift Score:**", f"{model_info['drift_score']:.4f}" if model_info['drift_score'] else "N/A")
                
                with col3:
                    trained_at = datetime.fromisoformat(model_info['trained_at'])
                    st.write("**Trained:**", trained_at.strftime("%Y-%m-%d %H:%M"))
                    
                    if metrics.get('drift_alert'):
                        st.error("⚠️ Drift detected! Retrain recommended.")
            else:
                st.info("No model trained yet. Run the pipeline first.")
        
        else:
            st.error("Failed to fetch metrics from API")
    
    except requests.exceptions.ConnectionError:
        st.error("⚠️ Cannot connect to API. Make sure the backend is running on port 8000.")
    except Exception as e:
        st.error(f"Error: {str(e)}")

with tab3:
    st.header("Pipeline Status")
    
    # Pipeline status
    try:
        response = requests.get(f"{API_URL}/pipeline/status")
        
        if response.status_code == 200:
            status = response.json()
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                if status['status'] == 'COMPLETED':
                    st.success(f"✅ {status['status']}")
                elif status['status'] == 'ERROR':
                    st.error(f"❌ {status['status']}")
                elif status['status'] == 'NOT_RUN':
                    st.warning(f"⚠️ {status['status']}")
                else:
                    st.info(f"ℹ️ {status['status']}")
                
                if status['status'] != 'NOT_RUN':
                    st.write("**Last Run:**", status.get('last_run', 'N/A'))
                    st.write("**Model:**", status.get('model_version', 'N/A'))
            
            with col2:
                if status['status'] == 'COMPLETED':
                    st.write("**Accuracy:**", status.get('accuracy', 'N/A'))
                    st.write("**F1 Score:**", status.get('f1_score', 'N/A'))
                    st.write("**Training Samples:**", status.get('training_samples', 'N/A'))
    
    except requests.exceptions.ConnectionError:
        st.error("⚠️ Cannot connect to API")
    except Exception as e:
        st.error(f"Error: {str(e)}")
    
    st.markdown("---")
    
    st.subheader("Model Training History")
    
    try:
        with get_session() as session:
            runs = session.query(ModelRun).order_by(desc(ModelRun.trained_at)).limit(10).all()
            
            if runs:
                runs_data = []
                for run in runs:
                    runs_data.append({
                        'Version': run.model_version,
                        'Accuracy': f"{run.accuracy:.4f}" if run.accuracy else "N/A",
                        'F1 Score': f"{run.macro_f1:.4f}" if run.macro_f1 else "N/A",
                        'Samples': run.training_samples,
                        'Drift Score': f"{run.drift_score:.4f}" if run.drift_score else "N/A",
                        'Drift Alert': '🔴' if run.drift_score and run.drift_score > 0.15 else '🟢',
                        'Trained At': run.trained_at.strftime("%Y-%m-%d %H:%M:%S")
                    })
                
                df = pd.DataFrame(runs_data)
                
                def highlight_drift(row):
                    if row['Drift Alert'] == '🔴':
                        return ['background-color: #ffebee'] * len(row)
                    return [''] * len(row)
                
                st.dataframe(
                    df.style.apply(highlight_drift, axis=1),
                    use_container_width=True,
                    hide_index=True
                )
                
                st.subheader("Accuracy Trend")
                
                acc_data = []
                for run in reversed(runs):
                    if run.accuracy:
                        acc_data.append({
                            'Version': run.model_version,
                            'Accuracy': run.accuracy,
                            'Timestamp': run.trained_at
                        })
                
                if acc_data:
                    acc_df = pd.DataFrame(acc_data)
                    fig = px.line(
                        acc_df,
                        x='Timestamp',
                        y='Accuracy',
                        markers=True,
                        title='Model Accuracy Over Time'
                    )
                    fig.update_layout(yaxis_range=[0, 1])
                    st.plotly_chart(fig, use_container_width=True)
            
            else:
                st.info("No training runs yet. Run the pipeline to train the first model.")
    
    except Exception as e:
        st.error(f"Database error: {str(e)}")
    
    st.markdown("---")
    
    st.subheader("System Health")
    
    try:
        response = requests.get(f"{API_URL}/health")
        
        if response.status_code == 200:
            health = response.json()
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                status_icon = "🟢" if health['status'] == 'ok' else "🔴"
                st.metric("API Status", f"{status_icon} {health['status'].upper()}")
            
            with col2:
                db_icon = "🟢" if health['db_connected'] else "🔴"
                st.metric("Database", f"{db_icon} {'Connected' if health['db_connected'] else 'Disconnected'}")
            
            with col3:
                whisper_icon = "🟢" if health['whisper_loaded'] else "🔴"
                st.metric("Whisper Model", f"{whisper_icon} {'Loaded' if health['whisper_loaded'] else 'Not Loaded'}")
            
            with col4:
                classifier_icon = "🟢" if health['classifier_loaded'] else "🔴"
                st.metric("Classifier", f"{classifier_icon} {'Loaded' if health['classifier_loaded'] else 'Not Loaded'}")
            
            st.caption(f"Last checked: {health['timestamp']}")
    
    except requests.exceptions.ConnectionError:
        st.error("⚠️ Cannot connect to API")
    except Exception as e:
        st.error(f"Error: {str(e)}")

st.markdown("---")
st.caption("VoiceIntent · AI 620: Fundamentals of Data Engineering · LUMS SBASSE")
