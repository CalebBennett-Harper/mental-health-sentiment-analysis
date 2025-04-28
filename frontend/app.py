"""
Streamlit frontend for Mental Health Sentiment Analysis System.
Provides UI for text and speech emotion analysis and history tracking.
"""

import streamlit as st
import requests

# --- App Configuration ---
st.set_page_config(
    page_title="Mental Health Sentiment Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- API Connection Utility ---
def get_api_url():
    return st.session_state.get("api_url", "http://localhost:8000/api/v1")

def set_api_url():
    st.sidebar.text_input(
        "FastAPI Backend URL",
        value=get_api_url(),
        key="api_url",
        help="Set the URL of your FastAPI backend (e.g., http://localhost:8000/api/v1)"
    )

# --- Sidebar Navigation ---
set_api_url()
page = st.sidebar.radio(
    "Navigation",
    ["✍️ Text Analysis", "🎤 Speech Analysis", "📈 History"]
)

st.title("🧠 Mental Health Sentiment Analysis")
st.write("Analyze your emotions from text or speech in real time.")

# --- Page Routing ---
if page == "✍️ Text Analysis":
    st.header("Text Emotion Analysis")
    st.info("Enter your thoughts and get instant emotional feedback.")

    # Text input
    user_text = st.text_area("Share your thoughts here:", height=150)

    # Process button
    if st.button("Analyze Emotions"):
        if user_text:
            from datetime import datetime
            with st.spinner("Analyzing your text..."):
                try:
                    response = requests.post(
                        f"{get_api_url()}/analyze/text",
                        json={"text": user_text}
                    )
                    if response.status_code == 200:
                        result = response.json()
                        st.success(f"Analysis complete!")
                        st.subheader("Primary Emotion")
                        st.markdown(f"### {result['emotion'].capitalize()}")
                        st.progress(result['confidence'])
                        st.caption(f"Confidence: {result['confidence']:.2%}")
                        if result.get('explanation'):
                            st.markdown(f"**Explanation**: {result['explanation']}")
                        st.subheader("All Emotions")
                        emotions = list(result['scores'].keys())
                        scores = list(result['scores'].values())
                        st.bar_chart({e: s for e, s in zip(emotions, scores)})
                        if "history" not in st.session_state:
                            st.session_state.history = []
                        st.session_state.history.append({
                            "type": "text",
                            "input": user_text,
                            "result": result,
                            "timestamp": str(datetime.now())
                        })
                    else:
                        st.error(f"Error: {response.status_code} - {response.text}")
                except Exception as e:
                    st.error(f"Error connecting to API: {str(e)}")
        else:
            st.warning("Please enter some text to analyze.")
elif page == "🎤 Speech Analysis":
    st.header("Speech Emotion Analysis")
    st.info("Record or upload audio to analyze your emotions.")

    # File upload for audio
    audio_file = st.file_uploader(
        "Upload an audio file (WAV, MP3, M4A, etc.)",
        type=["wav", "mp3", "m4a", "ogg", "flac"],
        accept_multiple_files=False
    )

    # (Optional) In-browser audio recording (Streamlit doesn't natively support, but st_audiorec is a community component)
    try:
        import st_audiorec
        st.write("Or record audio directly:")
        audio_bytes = st_audiorec.st_audiorec()
    except ImportError:
        audio_bytes = None
        st.caption("Install st-audiorec for in-browser recording: pip install st-audiorec")

    # Process button
    if st.button("Analyze Speech"):
        if audio_file or audio_bytes:
            from datetime import datetime
            with st.spinner("Transcribing and analyzing your speech..."):
                try:
                    files = None
                    if audio_file:
                        files = {"file": (audio_file.name, audio_file, audio_file.type)}
                    elif audio_bytes:
                        files = {"file": ("recorded_audio.wav", audio_bytes, "audio/wav")}
                    response = requests.post(
                        f"{get_api_url()}/analyze/speech",
                        files=files
                    )
                    if response.status_code == 200:
                        result = response.json()
                        st.success("Speech analysis complete!")
                        st.subheader("Transcription")
                        st.markdown(f"> {result.get('transcribed_text', '(No transcription)')}")
                        st.subheader("Primary Emotion")
                        st.markdown(f"### {result['emotion'].capitalize()}")
                        st.progress(result['confidence'])
                        st.caption(f"Confidence: {result['confidence']:.2%}")
                        if result.get('explanation'):
                            st.markdown(f"**Explanation**: {result['explanation']}")
                        st.subheader("All Emotions")
                        emotions = list(result['scores'].keys())
                        scores = list(result['scores'].values())
                        st.bar_chart({e: s for e, s in zip(emotions, scores)})
                        if "history" not in st.session_state:
                            st.session_state.history = []
                        st.session_state.history.append({
                            "type": "speech",
                            "input": "uploaded" if audio_file else "recorded",
                            "result": result,
                            "timestamp": str(datetime.now())
                        })
                    else:
                        st.error(f"Error: {response.status_code} - {response.text}")
                except Exception as e:
                    st.error(f"Error connecting to API: {str(e)}")
        else:
            st.warning("Please upload or record audio to analyze.")
elif page == "📈 History":
    st.header("Emotional History")
    st.info("View your emotional timeline and export your data.")
    import pandas as pd
    from datetime import datetime
    import plotly.express as px
    from collections import Counter

    # Ensure history exists
    history = st.session_state.get("history", [])

    if not history:
        st.warning("No analysis history yet. Analyze some text or speech to see your emotional timeline.")
    else:
        # Create DataFrame for visualization
        df = pd.DataFrame(history)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp")
        # Expand all emotion scores into columns
        emotion_keys = set()
        for r in df["result"]:
            emotion_keys.update(r["scores"].keys())
        for key in emotion_keys:
            df[key] = df["result"].apply(lambda r: r["scores"].get(key, 0.0))
        df["primary_emotion"] = df["result"].apply(lambda r: r["emotion"])
        df["confidence"] = df["result"].apply(lambda r: r["confidence"])

        # --- Filtering ---
        min_date = df["timestamp"].min().date()
        max_date = df["timestamp"].max().date()
        date_range = st.date_input(
            "Select date range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        filtered_df = df[(df["timestamp"].dt.date >= date_range[0]) & (df["timestamp"].dt.date <= date_range[1])]
        emotion_filter = st.multiselect("Filter by emotion", sorted(list(emotion_keys)), default=list(emotion_keys))
        filtered_df = filtered_df[filtered_df["primary_emotion"].isin(emotion_filter)]
        input_filter = st.multiselect("Filter by input type", ["text", "speech"], default=["text", "speech"])
        filtered_df = filtered_df[filtered_df["type"].isin(input_filter)]

        # --- Multi-line chart for all emotions ---
        st.subheader("Timeline of Emotions (Confidence)")
        if not filtered_df.empty:
            fig = px.line(
                filtered_df,
                x="timestamp",
                y=list(emotion_keys),
                labels={"value": "Confidence", "timestamp": "Time"},
                title="Emotion Confidence Over Time"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data to display for selected filters.")

        # --- Pie chart for emotion distribution ---
        st.subheader("Emotion Distribution")
        emotion_counts = Counter(filtered_df["primary_emotion"])
        if emotion_counts:
            pie_fig = px.pie(
                names=list(emotion_counts.keys()),
                values=list(emotion_counts.values()),
                title="Distribution of Primary Emotions"
            )
            st.plotly_chart(pie_fig, use_container_width=True)
        else:
            st.info("No data to display for emotion distribution.")

        # --- Summary statistics ---
        st.subheader("Analysis Insights")
        if not filtered_df.empty:
            most_common = filtered_df["primary_emotion"].mode()[0]
            avg_conf = filtered_df["confidence"].mean()
            st.markdown(f"- **Most common emotion:** {most_common.capitalize()}")
            st.markdown(f"- **Average confidence:** {avg_conf:.2%}")
            # Trend: compare first and last confidence
            first_row = filtered_df.iloc[0]
            last_row = filtered_df.iloc[-1]
            trend = last_row["confidence"] - first_row["confidence"]
            trend_str = "increasing" if trend > 0 else ("decreasing" if trend < 0 else "stable")
            st.markdown(f"- **Confidence trend:** {trend_str} ({trend:+.2%})")
            # Highlight significant change
            if abs(trend) > 0.2:
                st.warning(f"Significant change detected: Confidence changed by {trend:+.2%} over the selected period.")
        else:
            st.info("No data for insights.")

        # --- Data Table ---
        st.dataframe(filtered_df[["timestamp", "type", "input", "primary_emotion", "confidence"]], use_container_width=True)
        # --- Export functionality ---
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="Export History as CSV",
            data=csv,
            file_name="sentiment_history.csv",
            mime="text/csv"
        )