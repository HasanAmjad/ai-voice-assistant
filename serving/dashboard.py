"""Authored by: Ibrahim Noor."""

import os
import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from streamlit_mic_recorder import mic_recorder
from datetime import datetime
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from storage.db import get_session
from storage.models import ModelRun, Prediction
from sqlalchemy import desc

st.set_page_config(
    page_title="VoiceIntent Dashboard",
    page_icon="🎤",
    layout="wide",
)

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.title("🎤 VoiceIntent Dashboard")
st.markdown("**Automated Voice-to-Intent Intelligence Pipeline**")
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["🎯 Predict", "📊 Analytics", "⚙️ Pipeline Status"])

with tab1:
    st.header("Intent Prediction")
    st.markdown("Record a question or upload an audio file to predict the banking intent.")

    st.session_state.setdefault("mic_counter", 0)
    st.session_state.setdefault("upload_counter", 0)

    record_col, upload_col = st.columns(2)

    with record_col:
        st.markdown("**🎙️ Record from microphone**")
        mic_key = f"mic_recorder_{st.session_state['mic_counter']}"
        mic_audio = mic_recorder(
            start_prompt="Click to record",
            stop_prompt="Stop recording",
            just_once=True,
            use_container_width=True,
            format="webm",
            key=mic_key,
        )
        if mic_audio and mic_audio.get("bytes"):
            st.session_state["recorded_audio"] = {
                "bytes": mic_audio["bytes"],
                "filename": "recording.webm",
                "mime": "audio/webm",
            }
            st.session_state["upload_counter"] += 1
            st.rerun()

        if st.session_state.get("recorded_audio"):
            if st.button("Clear recording", key="clear_recording"):
                st.session_state.pop("recorded_audio", None)
                st.session_state["mic_counter"] += 1
                st.rerun()

    with upload_col:
        st.markdown("**📁 Upload an audio file**")
        upload_key = f"file_uploader_{st.session_state['upload_counter']}"
        uploaded_file = st.file_uploader(
            "Choose an audio file (.mp3 or .wav)",
            type=["mp3", "wav"],
            label_visibility="collapsed",
            key=upload_key,
        )

    if uploaded_file is not None:
        upload_id = (uploaded_file.name, uploaded_file.size)
        if st.session_state.get("last_upload_id") != upload_id:
            st.session_state["last_upload_id"] = upload_id
            st.session_state.pop("recorded_audio", None)
    else:
        st.session_state.pop("last_upload_id", None)

    audio_payload = None
    audio_filename = None
    audio_mime = None
    if st.session_state.get("recorded_audio"):
        rec = st.session_state["recorded_audio"]
        audio_payload = rec["bytes"]
        audio_filename = rec["filename"]
        audio_mime = rec["mime"]
    elif uploaded_file is not None:
        audio_payload = uploaded_file.getvalue()
        audio_filename = uploaded_file.name
        audio_mime = uploaded_file.type or "audio/mp3"

    if audio_payload is not None and len(audio_payload) == 0:
        st.warning("⚠️ The audio is empty. Please record again or upload a non-empty file.")
        audio_payload = None

    if audio_payload is not None:
        st.audio(audio_payload, format=audio_mime)
        st.caption(f"📎 {audio_filename} · {len(audio_payload):,} bytes")

        if st.button("Predict Intent", type="primary"):
            with st.spinner("Processing audio..."):
                try:
                    files = {"file": (audio_filename, audio_payload, audio_mime)}
                    response = requests.post(f"{API_URL}/predict", files=files, timeout=120)

                    if response.status_code == 200:
                        result = response.json()

                        col1, col2 = st.columns(2)

                        with col1:
                            if result.get("escalated"):
                                st.warning("🟡 Low Confidence - Escalating to Agent")
                            else:
                                st.success("✅ Prediction Complete")
                            st.metric("Predicted Intent", result["intent"])
                            st.metric("Confidence", f"{result['confidence']:.2%}")
                            if "confidence_threshold" in result:
                                st.caption(f"Threshold: {result['confidence_threshold']:.0%}")
                            st.caption(f"Model: {result['model_version']}")

                        with col2:
                            st.subheader("Transcript")
                            st.info(result["transcript"])

                            if result["raw_transcript"] != result["transcript"]:
                                with st.expander("View raw transcript"):
                                    st.text(result["raw_transcript"])

                        if result.get("response_text"):
                            if result.get("escalated"):
                                st.subheader("📞 Agent Handoff")
                                st.warning(result["response_text"])
                            else:
                                st.subheader("🔊 Assistant Response")
                                st.success(result["response_text"])
                            try:
                                audio_resp = requests.get(
                                    f"{API_URL}{result['response_audio_url']}", timeout=60
                                )
                                if audio_resp.status_code == 200:
                                    st.audio(audio_resp.content, format="audio/mp3")
                                else:
                                    st.caption("(Audio response unavailable.)")
                            except requests.exceptions.RequestException as e:
                                st.caption(f"(Audio response unavailable: {e})")

                        st.subheader("Top 5 Intent Predictions")
                        top_5 = pd.DataFrame(result["top_5_intents"])
                        threshold = result.get("confidence_threshold", 0.4)
                        predicted = result["intent"]
                        bar_colors = [
                            "#27ae60" if (row["intent"] == predicted and not result.get("escalated"))
                            else "#e67e22" if (row["intent"] == predicted and result.get("escalated"))
                            else "#bdc3c7"
                            for _, row in top_5.iterrows()
                        ]
                        fig = go.Figure(go.Bar(
                            x=top_5["confidence"],
                            y=top_5["intent"],
                            orientation="h",
                            marker_color=bar_colors,
                            text=[f"{c:.1%}" for c in top_5["confidence"]],
                            textposition="outside",
                        ))
                        fig.add_vline(
                            x=threshold,
                            line_dash="dash",
                            line_color="red",
                            annotation_text=f"Threshold {threshold:.0%}",
                            annotation_position="top right",
                        )
                        fig.update_layout(
                            title="Confidence Scores",
                            xaxis_title="Confidence",
                            yaxis_title="Intent",
                            yaxis={"categoryorder": "total ascending"},
                            xaxis={"range": [0, max(1.0, top_5["confidence"].max() + 0.15)]},
                            showlegend=False,
                        )
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
        response = requests.get(f"{API_URL}/metrics")

        if response.status_code == 200:
            metrics = response.json()

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Predictions", metrics["total_predictions"])

            with col2:
                st.metric("Avg Confidence", f"{metrics['average_confidence']:.2%}")

            with col3:
                model_acc = metrics["current_model"].get("accuracy")
                st.metric("Model Accuracy", f"{model_acc:.2%}" if model_acc else "N/A")

            with col4:
                drift_status = "🔴 Alert" if metrics.get("drift_alert") else "🟢 Normal"
                st.metric("Drift Status", drift_status)

            st.markdown("---")

            st.subheader("📞 Recent Calls")
            try:
                threshold_val = 0.4
                rows = []
                with get_session() as session:
                    recent = (
                        session.query(
                            Prediction.predicted_at,
                            Prediction.cleaned_transcript,
                            Prediction.predicted_intent,
                            Prediction.confidence_score,
                        )
                        .order_by(desc(Prediction.predicted_at))
                        .limit(10)
                        .all()
                    )
                    for predicted_at, transcript, intent, confidence in recent:
                        escalated = bool(confidence is not None and confidence < threshold_val)
                        rows.append({
                            "Time": predicted_at.strftime("%H:%M:%S"),
                            "Transcript": (transcript or "")[:80],
                            "Intent": intent,
                            "Confidence": f"{confidence:.2%}" if confidence is not None else "—",
                            "Routed To": "🟡 Agent" if escalated else "✅ Auto",
                        })
                if rows:
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                else:
                    st.info("No predictions yet. Make one in the Predict tab.")
            except Exception as e:
                st.error(f"Could not load recent predictions: {e}")

            st.markdown("---")

            st.subheader("🌀 Live Distribution Drift")
            st.caption("Compares the most recent 100 predictions against the prior 100. Drift > 0.15 triggers an alert.")
            try:
                drift_resp = requests.get(f"{API_URL}/drift?window=100", timeout=10)
                if drift_resp.status_code == 200:
                    drift_data = drift_resp.json()
                    if not drift_data.get("ready"):
                        st.info(drift_data.get("message", "Not enough predictions yet."))
                    else:
                        c1, c2, c3 = st.columns(3)
                        with c1:
                            st.metric("Drift Score", f"{drift_data['drift_score']:.4f}")
                        with c2:
                            st.metric("Threshold", f"{drift_data['threshold']:.2f}")
                        with c3:
                            badge = "🔴 ALERT" if drift_data["drift_alert"] else "🟢 OK"
                            st.metric("Status", badge)
                        movers = drift_data.get("top_movers", [])
                        if movers:
                            mv_df = pd.DataFrame(movers)
                            mv_df["delta"] = mv_df["delta"].apply(lambda x: f"{x:+.2%}")
                            mv_df["recent_share"] = mv_df["recent_share"].apply(lambda x: f"{x:.2%}")
                            mv_df["prior_share"] = mv_df["prior_share"].apply(lambda x: f"{x:.2%}")
                            mv_df.columns = ["Intent", "Δ", "Recent share", "Prior share"]
                            st.write("**Top movers**")
                            st.dataframe(mv_df, use_container_width=True, hide_index=True)
                else:
                    st.warning("Drift endpoint returned an error.")
            except requests.exceptions.RequestException:
                st.caption("(Drift endpoint unavailable.)")

            st.markdown("---")

            if metrics["intent_distribution"]:
                st.subheader("Intent Distribution")

                intent_df = pd.DataFrame([
                    {"Intent": intent, "Count": count}
                    for intent, count in metrics["intent_distribution"].items()
                ]).sort_values("Count", ascending=False).head(20)

                fig = px.bar(
                    intent_df,
                    x="Count",
                    y="Intent",
                    orientation="h",
                    title="Top 20 Predicted Intents",
                    labels={"Count": "Number of Predictions", "Intent": "Intent"},
                )
                fig.update_layout(
                    height=600,
                    yaxis={"categoryorder": "total ascending"},
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No predictions yet. Upload audio in the Predict tab to get started.")

            st.markdown("---")
            st.subheader("Current Model")

            model_info = metrics["current_model"]
            if model_info["version"]:
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.write("**Version:**", model_info["version"])
                    st.write("**Accuracy:**", f"{model_info['accuracy']:.4f}" if model_info["accuracy"] else "N/A")

                with col2:
                    st.write("**F1 Score:**", f"{model_info['f1_score']:.4f}" if model_info["f1_score"] else "N/A")
                    st.write("**Drift Score:**", f"{model_info['drift_score']:.4f}" if model_info["drift_score"] else "N/A")

                with col3:
                    trained_at = datetime.fromisoformat(model_info["trained_at"])
                    st.write("**Trained:**", trained_at.strftime("%Y-%m-%d %H:%M"))

                    if metrics.get("drift_alert"):
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

    try:
        response = requests.get(f"{API_URL}/pipeline/status")

        if response.status_code == 200:
            status = response.json()

            col1, col2 = st.columns([1, 2])

            with col1:
                if status["status"] == "COMPLETED":
                    st.success(f"✅ {status['status']}")
                elif status["status"] == "ERROR":
                    st.error(f"❌ {status['status']}")
                elif status["status"] == "NOT_RUN":
                    st.warning(f"⚠️ {status['status']}")
                else:
                    st.info(f"ℹ️ {status['status']}")

                if status["status"] != "NOT_RUN":
                    st.write("**Last Run:**", status.get("last_run", "N/A"))
                    st.write("**Model:**", status.get("model_version", "N/A"))

            with col2:
                if status["status"] == "COMPLETED":
                    st.write("**Accuracy:**", status.get("accuracy", "N/A"))
                    st.write("**F1 Score:**", status.get("f1_score", "N/A"))
                    st.write("**Training Samples:**", status.get("training_samples", "N/A"))

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
                        "Version": run.model_version,
                        "Accuracy": f"{run.accuracy:.4f}" if run.accuracy else "N/A",
                        "F1 Score": f"{run.macro_f1:.4f}" if run.macro_f1 else "N/A",
                        "Samples": run.training_samples,
                        "Drift Score": f"{run.drift_score:.4f}" if run.drift_score else "N/A",
                        "Drift Alert": "🔴" if run.drift_score and run.drift_score > 0.15 else "🟢",
                        "Trained At": run.trained_at.strftime("%Y-%m-%d %H:%M:%S"),
                    })

                df = pd.DataFrame(runs_data)

                def highlight_drift(row):
                    """Tint rows that crossed the drift threshold."""
                    if row["Drift Alert"] == "🔴":
                        return ["background-color: #ffebee"] * len(row)
                    return [""] * len(row)

                st.dataframe(
                    df.style.apply(highlight_drift, axis=1),
                    use_container_width=True,
                    hide_index=True,
                )

                st.subheader("Accuracy Trend")

                acc_data = []
                for run in reversed(runs):
                    if run.accuracy:
                        acc_data.append({
                            "Version": run.model_version,
                            "Accuracy": run.accuracy,
                            "Timestamp": run.trained_at,
                        })

                if acc_data:
                    acc_df = pd.DataFrame(acc_data)
                    fig = px.line(
                        acc_df,
                        x="Timestamp",
                        y="Accuracy",
                        markers=True,
                        title="Model Accuracy Over Time",
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
                status_icon = "🟢" if health["status"] == "ok" else "🔴"
                st.metric("API Status", f"{status_icon} {health['status'].upper()}")

            with col2:
                db_icon = "🟢" if health["db_connected"] else "🔴"
                st.metric("Database", f"{db_icon} {'Connected' if health['db_connected'] else 'Disconnected'}")

            with col3:
                whisper_icon = "🟢" if health["whisper_loaded"] else "🔴"
                st.metric("Whisper Model", f"{whisper_icon} {'Loaded' if health['whisper_loaded'] else 'Not Loaded'}")

            with col4:
                classifier_icon = "🟢" if health["classifier_loaded"] else "🔴"
                st.metric("Classifier", f"{classifier_icon} {'Loaded' if health['classifier_loaded'] else 'Not Loaded'}")

            st.caption(f"Last checked: {health['timestamp']}")

    except requests.exceptions.ConnectionError:
        st.error("⚠️ Cannot connect to API")
    except Exception as e:
        st.error(f"Error: {str(e)}")

st.markdown("---")
st.caption("VoiceIntent · AI 620: Fundamentals of Data Engineering · LUMS SBASSE")
