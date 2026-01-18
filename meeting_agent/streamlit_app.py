"""
Streamlit Meeting Agent Application

Graphical User Interface (GUI) for the Meeting Agent package.
Provides a web-based interface for audio transcription, meeting notes generation,
and interactive Q&A about transcripts.

Requires: pip install meeting-agent[StreamlitUI]
"""

import io
import json
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None  # type: ignore

import streamlit as st

from meeting_agent.helper import (
    DEFAULT_CLOUD_MODEL,
    WHISPER_AVAILABLE,
    call_ollama_cloud,
    check_ffmpeg,
    create_llm_messages,
    extract_response_text,
    get_transcripts_dir,
    install_ffmpeg_instructions,
    load_prompt,
    parse_ollama_response,
    read_transcript,
    save_meeting_notes,
    transcribe_audio,
)
from meeting_agent.database import Database
from meeting_agent.llm import create_meeting_agent
from meeting_agent.retention_manager import RetentionManager
from meeting_agent.sentiment_analyzer import (
    analyze_attendee_sentiment,
    analyze_overall_sentiment,
    analyze_topic_sentiment,
    check_dependencies as check_sentiment_dependencies,
    extract_topics_via_llm,
    identify_attendees,
)

# Page configuration
st.set_page_config(
    page_title="Meeting Agent",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state
if "transcript_text" not in st.session_state:
    st.session_state.transcript_text = None
if "transcript_path" not in st.session_state:
    st.session_state.transcript_path = None
if "generated_notes" not in st.session_state:
    st.session_state.generated_notes = None
if "notes_path" not in st.session_state:
    st.session_state.notes_path = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "whisper_model" not in st.session_state:
    st.session_state.whisper_model = "base"
if "output_format" not in st.session_state:
    st.session_state.output_format = "docx"
if "sentiment_results" not in st.session_state:
    st.session_state.sentiment_results = None
if "sentiment_attendees" not in st.session_state:
    st.session_state.sentiment_attendees = None
if "sentiment_error" not in st.session_state:
    st.session_state.sentiment_error = None
if "current_transcript_id" not in st.session_state:
    st.session_state.current_transcript_id = None
if "db" not in st.session_state:
    from meeting_agent.database import Database
    st.session_state.db = Database()
else:
    # Ensure database has latest methods (in case of code updates)
    if not hasattr(st.session_state.db, 'get_retention_policies'):
        from meeting_agent.database import Database
        st.session_state.db = Database()


def check_dependencies() -> tuple[bool, list[str]]:
    """Check if required dependencies are available.

    Returns:
        Tuple[bool, List[str]]: (all_available, list of missing dependencies)
    """
    missing = []
    if not WHISPER_AVAILABLE:
        missing.append("openai-whisper")
    if not check_ffmpeg():
        missing.append("ffmpeg")
    return len(missing) == 0, missing


def _display_sentiment_results() -> None:
    """Display sentiment analysis results from session state.
    
    This function displays the sentiment results that are already stored
    in session state. It should be called after sentiment analysis has
    been run and results are cached.
    """
    if st.session_state.sentiment_results is None:
        return
    
    try:
        overall_sentiment = st.session_state.sentiment_results["overall"]
        attendee_sentiments = st.session_state.sentiment_results.get("attendees", {})
        attendees_list = st.session_state.sentiment_attendees or []
        topics = st.session_state.sentiment_results.get("topics", [])
        topic_sentiments = st.session_state.sentiment_results.get("topic_sentiments", {})
        
        st.divider()
        st.subheader("📊 Sentiment Analysis")
        
        # Display overall sentiment with visual indicators
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Overall Sentiment",
                f"{overall_sentiment['emoji']} {overall_sentiment['label']}",
            )
        with col2:
            st.metric(
                "Compound Score",
                f"{overall_sentiment['compound']:.3f}",
                help="Range: -1 (most negative) to +1 (most positive)",
            )
        with col3:
            text_length = len(st.session_state.transcript_text) if st.session_state.transcript_text else 0
            st.metric("Text Length", f"{text_length:,} chars")
        
        # Sentiment breakdown
        st.markdown("#### Sentiment Breakdown")
        breakdown_col1, breakdown_col2, breakdown_col3 = st.columns(3)
        with breakdown_col1:
            st.progress(overall_sentiment["pos"], text=f"Positive: {overall_sentiment['pos']*100:.1f}%")
        with breakdown_col2:
            st.progress(overall_sentiment["neu"], text=f"Neutral: {overall_sentiment['neu']*100:.1f}%")
        with breakdown_col3:
            st.progress(overall_sentiment["neg"], text=f"Negative: {overall_sentiment['neg']*100:.1f}%")
        
        # Interpretation
        compound = overall_sentiment["compound"]
        if compound >= 0.05:
            st.success(
                f"✅ The overall sentiment is **positive** (score: {compound:.3f}). "
                "The meeting appears to have a constructive and upbeat tone."
            )
        elif compound <= -0.05:
            st.error(
                f"⚠️ The overall sentiment is **negative** (score: {compound:.3f}). "
                "The meeting may contain concerns, disagreements, or challenges."
            )
        else:
            st.info(
                f"ℹ️ The overall sentiment is **neutral** (score: {compound:.3f}). "
                "The meeting maintains a balanced and objective tone."
            )
        
        # Per-attendee sentiment
        if attendees_list:
            st.divider()
            st.markdown("#### 👥 Attendee Sentiment Analysis")
            st.success(f"✅ Found {len(attendees_list)} attendee(s): {', '.join(attendees_list)}")
            
            # Create table data
            attendee_data = []
            for attendee in attendees_list:
                if attendee in attendee_sentiments:
                    sentiment = attendee_sentiments[attendee]
                    attendee_data.append(
                        {
                            "Attendee": attendee,
                            "Sentiment": f"{sentiment['emoji']} {sentiment['label']}",
                            "Compound Score": f"{sentiment['compound']:.3f}",
                            "Positive": f"{sentiment['pos']*100:.1f}%",
                            "Neutral": f"{sentiment['neu']*100:.1f}%",
                            "Negative": f"{sentiment['neg']*100:.1f}%",
                            "Mentions": sentiment["mentions"],
                        }
                    )
            
            if attendee_data:
                if not PANDAS_AVAILABLE:
                    st.error("❌ pandas is required for displaying sentiment tables. Please install it with: pip install pandas")
                else:
                    df = pd.DataFrame(attendee_data)
                    st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info(
                "ℹ️ No named entities (attendees) detected in this transcript. "
                "Overall sentiment analysis is still available above."
            )
        
        # Topics sentiment analysis
        if topics:
            st.divider()
            st.markdown("#### 📋 Topics Sentiment Analysis")
            st.success(f"✅ Found {len(topics)} topic(s): {', '.join(topics)}")
            
            # Create table data for topics
            topic_data = []
            for topic in topics:
                if topic in topic_sentiments:
                    sentiment = topic_sentiments[topic]
                    topic_data.append(
                        {
                            "Topic": topic,
                            "Sentiment": f"{sentiment['emoji']} {sentiment['label']}",
                            "Compound Score": f"{sentiment['compound']:.3f}",
                            "Positive": f"{sentiment['pos']*100:.1f}%",
                            "Neutral": f"{sentiment['neu']*100:.1f}%",
                            "Negative": f"{sentiment['neg']*100:.1f}%",
                            "Segments": sentiment.get("segments", 0),
                        }
                    )
            
            if topic_data:
                if not PANDAS_AVAILABLE:
                    st.error("❌ pandas is required for displaying sentiment tables. Please install it with: pip install pandas")
                else:
                    df_topics = pd.DataFrame(topic_data)
                    st.dataframe(df_topics, use_container_width=True, hide_index=True)
    except Exception as e:
        st.warning(f"⚠️ Could not display sentiment results: {e}")


def _run_sentiment_analysis(transcript_text: str, model: str | None = DEFAULT_CLOUD_MODEL, transcript_id: int | None = None) -> tuple[bool, str | None]:
    """Run sentiment analysis automatically in the background.

    This function runs sentiment analysis when a transcript is loaded
    and caches the results in session state and database.

    Args:
        transcript_text: The transcript text to analyze.
        model: LLM model name for topic extraction (defaults to DEFAULT_CLOUD_MODEL).
        transcript_id: Transcript ID in database (optional, for saving to DB).

    Returns:
        Tuple[bool, str | None]: (success, error_message)
    """
    # #region agent log
    try:
        with open("/Users/ymo/meeting_agent/.cursor/debug.log", "a", encoding="utf-8") as f:
            f.write(json.dumps({"location": "streamlit_app.py:87", "message": "_run_sentiment_analysis called", "data": {"transcript_length": len(transcript_text) if transcript_text else 0, "transcript_preview": (transcript_text[:100] if transcript_text else "None")}, "timestamp": time.time() * 1000, "sessionId": "debug-session", "runId": "run1", "hypothesisId": "A"}) + "\n")
    except Exception:
        pass
    # #endregion

    # Check if sentiment analysis dependencies are available
    all_available, missing = check_sentiment_dependencies()
    # #region agent log
    try:
        with open("/Users/ymo/meeting_agent/.cursor/debug.log", "a", encoding="utf-8") as f:
            f.write(json.dumps({"location": "streamlit_app.py:100", "message": "Dependency check result", "data": {"all_available": all_available, "missing": missing}, "timestamp": time.time() * 1000, "sessionId": "debug-session", "runId": "run1", "hypothesisId": "C"}) + "\n")
    except Exception:
        pass
    # #endregion
    if not all_available:
        return False, f"Missing dependencies: {', '.join(missing)}"

    try:
        # #region agent log
        try:
            with open("/Users/ymo/meeting_agent/.cursor/debug.log", "a", encoding="utf-8") as f:
                f.write(json.dumps({"location": "streamlit_app.py:104", "message": "Starting sentiment analysis", "data": {}, "timestamp": time.time() * 1000, "sessionId": "debug-session", "runId": "run1", "hypothesisId": "B"}) + "\n")
        except Exception:
            pass
        # #endregion
        # Run overall sentiment analysis
        overall_sentiment = analyze_overall_sentiment(transcript_text)
        # #region agent log
        try:
            with open("/Users/ymo/meeting_agent/.cursor/debug.log", "a", encoding="utf-8") as f:
                f.write(json.dumps({"location": "streamlit_app.py:106", "message": "Overall sentiment computed", "data": {"compound": overall_sentiment.get("compound") if overall_sentiment else None}, "timestamp": time.time() * 1000, "sessionId": "debug-session", "runId": "run1", "hypothesisId": "B"}) + "\n")
        except Exception:
            pass
        # #endregion

        # Identify attendees
        attendees_list = identify_attendees(transcript_text)
        # #region agent log
        try:
            with open("/Users/ymo/meeting_agent/.cursor/debug.log", "a", encoding="utf-8") as f:
                f.write(json.dumps({"location": "streamlit_app.py:109", "message": "Attendees identified", "data": {"attendee_count": len(attendees_list) if attendees_list else 0, "attendees": attendees_list[:5] if attendees_list else []}, "timestamp": time.time() * 1000, "sessionId": "debug-session", "runId": "run1", "hypothesisId": "B"}) + "\n")
        except Exception:
            pass
        # #endregion

        # Analyze per-attendee sentiment if attendees found
        attendee_sentiments = {}
        if attendees_list:
            for attendee in attendees_list:
                try:
                    attendee_sentiment = analyze_attendee_sentiment(transcript_text, attendee)
                    attendee_sentiments[attendee] = attendee_sentiment
                except Exception as e:
                    # Skip individual attendee if analysis fails
                    pass

        # Extract topics using LLM
        topics = []
        topic_sentiments = {}
        if model:
            try:
                topics = extract_topics_via_llm(transcript_text, model)
                # Analyze sentiment for each topic
                if topics:
                    for topic in topics:
                        try:
                            topic_sentiment = analyze_topic_sentiment(transcript_text, topic)
                            topic_sentiments[topic] = topic_sentiment
                        except Exception as e:
                            # Skip individual topic if analysis fails
                            pass
            except Exception as e:
                # If topic extraction fails, continue without topics
                # Don't block the entire sentiment analysis
                # Store empty lists to indicate topics were attempted but failed
                topics = []
                topic_sentiments = {}

        # Store results in session state
        st.session_state.sentiment_results = {
            "overall": overall_sentiment,
            "attendees": attendee_sentiments,
            "topics": topics,
            "topic_sentiments": topic_sentiments,
        }
        st.session_state.sentiment_attendees = attendees_list
        st.session_state.sentiment_error = None  # Clear any previous errors

        # Save to database if transcript_id is provided
        if transcript_id:
            try:
                db = st.session_state.db
                # Save overall sentiment
                db.save_sentiment_analysis(transcript_id, overall_sentiment, model_used=model)
                
                # Save attendees
                if attendees_list:
                    attendee_list_for_db = []
                    for attendee_name, sentiment_data in attendee_sentiments.items():
                        attendee_list_for_db.append({
                            "name": attendee_name,
                            "compound": sentiment_data.get("compound", 0.0),
                            "pos": sentiment_data.get("pos", 0.0),
                            "neu": sentiment_data.get("neu", 0.0),
                            "neg": sentiment_data.get("neg", 0.0),
                            "label": sentiment_data.get("label", "Neutral"),
                            "mentions": sentiment_data.get("mentions", 0),
                        })
                    db.save_attendees(transcript_id, attendee_list_for_db)
                
                # Save topics
                if topics:
                    db.save_topics(transcript_id, topic_sentiments, is_auto_generated=True)
                    # Also save topic names as auto-generated labels
                    for topic in topics:
                        db.add_label(transcript_id, topic, label_type="auto")
            except Exception as e:
                # Don't fail sentiment analysis if database save fails
                pass
        # #region agent log
        try:
            with open("/Users/ymo/meeting_agent/.cursor/debug.log", "a", encoding="utf-8") as f:
                f.write(json.dumps({"location": "streamlit_app.py:123", "message": "Results stored in session state", "data": {"has_results": st.session_state.sentiment_results is not None, "has_overall": "overall" in (st.session_state.sentiment_results or {})}, "timestamp": time.time() * 1000, "sessionId": "debug-session", "runId": "run1", "hypothesisId": "D"}) + "\n")
        except Exception:
            pass
        # #endregion
        return True, None
    except Exception as e:
        # Store error in session state for display in tab
        st.session_state.sentiment_error = str(e)
        # #region agent log
        try:
            with open("/Users/ymo/meeting_agent/.cursor/debug.log", "a", encoding="utf-8") as f:
                f.write(json.dumps({"location": "streamlit_app.py:130", "message": "Exception in sentiment analysis", "data": {"error_type": type(e).__name__, "error_msg": str(e)[:200]}, "timestamp": time.time() * 1000, "sessionId": "debug-session", "runId": "run1", "hypothesisId": "B"}) + "\n")
        except Exception:
            pass
        # #endregion
        return False, str(e)


def main():
    """Main Streamlit application."""
    # #region agent log
    try:
        with open("/Users/ymo/meeting_agent/.cursor/debug.log", "a", encoding="utf-8") as f:
            f.write(json.dumps({"location": "streamlit_app.py:131", "message": "main() function called", "data": {}, "timestamp": time.time() * 1000, "sessionId": "debug-session", "runId": "run1", "hypothesisId": "D"}) + "\n")
            f.flush()
    except Exception:
        pass
    # #endregion
    st.title("🎤 Meeting Agent")
    st.markdown("**Graphical User Interface for Meeting Transcription and Notes Generation**")

    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model selection
        model = st.selectbox(
            "LLM Model",
            [DEFAULT_CLOUD_MODEL],
            help="Ollama cloud model for generating meeting notes",
        )
        # Store model in session state for access in tabs
        st.session_state.current_model = model
        
        # Default Whisper model
        st.session_state.whisper_model = st.selectbox(
            "Default Whisper Model",
            ["tiny", "base", "small", "medium", "large"],
            index=1,  # base
            help="Whisper model size for transcription (larger = more accurate, slower)",
        )
        
        # Default output format
        st.session_state.output_format = st.selectbox(
            "Default Output Format",
            ["txt", "docx", "rtf", "md"],
            index=1,  # docx
            help="Default format for generated meeting notes",
        )
        
        st.divider()
        
        # Dependency check
        st.subheader("📋 Dependencies")
        all_available, missing = check_dependencies()
        if all_available:
            st.success("✅ All dependencies available")
        else:
            st.warning(f"⚠️ Missing: {', '.join(missing)}")
            if "ffmpeg" in missing:
                st.info(install_ffmpeg_instructions())
        
        st.divider()
        
        # About section
        st.subheader("ℹ️ About")
        st.markdown(
            """
            **Meeting Agent** provides:
            - Audio transcription with Whisper
            - LLM-powered meeting notes generation
            - Interactive Q&A about transcripts
            - Multiple export formats (TXT, DOCX, RTF, MD)
            """
        )
        st.markdown(f"**Model:** {model}")

    # Main tabs
    # #region agent log
    try:
        with open("/Users/ymo/meeting_agent/.cursor/debug.log", "a", encoding="utf-8") as f:
            f.write(json.dumps({"location": "streamlit_app.py:247", "message": "Creating tabs", "data": {}, "timestamp": time.time() * 1000, "sessionId": "debug-session", "runId": "run1", "hypothesisId": "D"}) + "\n")
            f.flush()
    except Exception:
        pass
    # #endregion
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(
        [
            "🎵 Audio Transcription",
            "📄 Transcript Upload",
            "📝 Generate Notes",
            "💬 Q&A Chat",
            "⬇️ Downloads",
            "📊 Sentiment Analysis",
            "📚 Transcript Library",
            "📋 Governance",
        ]
    )
    # #region agent log
    try:
        with open("/Users/ymo/meeting_agent/.cursor/debug.log", "a", encoding="utf-8") as f:
            f.write(json.dumps({"location": "streamlit_app.py:260", "message": "Tabs created, about to enter tab6", "data": {}, "timestamp": time.time() * 1000, "sessionId": "debug-session", "runId": "run1", "hypothesisId": "D"}) + "\n")
            f.flush()
    except Exception:
        pass
    # #endregion

    # Tab 1: Audio Transcription
    with tab1:
        st.header("Audio Transcription")
        st.markdown("Upload an audio file and transcribe it using Whisper.")

        uploaded_audio = st.file_uploader(
            "Upload Audio File",
            type=["m4a", "mp3", "wav", "aac", "flac", "ogg"],
            help="Supported formats: M4A, MP3, WAV, AAC, FLAC, OGG",
        )

        if uploaded_audio is not None:
            st.audio(uploaded_audio, format=uploaded_audio.type)

            # Whisper model selection
            model_options = ["tiny", "base", "small", "medium", "large"]
            current_model_index = 1  # Default to "base"
            if st.session_state.whisper_model in model_options:
                current_model_index = model_options.index(st.session_state.whisper_model)
            whisper_model = st.selectbox(
                "Whisper Model",
                model_options,
                index=current_model_index,
                help="Larger models are more accurate but slower",
            )

            if st.button("🎯 Transcribe Audio", type="primary"):
                # Check dependencies
                if not WHISPER_AVAILABLE:
                    st.error("❌ Whisper is not installed. Please install it with: pip install openai-whisper")
                    st.stop()

                if not check_ffmpeg():
                    st.error("❌ ffmpeg is not installed or not available.")
                    st.info(install_ffmpeg_instructions())
                    st.stop()

                # Save uploaded file to temporary location
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_audio.name).suffix) as tmp_file:
                    tmp_file.write(uploaded_audio.getvalue())
                    tmp_audio_path = Path(tmp_file.name)

                try:
                    with st.status("🔄 Transcribing audio... This may take a while.", state="running", expanded=True) as status:
                        status.update(label="🔄 Transcribing audio with Whisper...", state="running")
                        transcript_path = transcribe_audio(
                            audio_path=tmp_audio_path, model_size=whisper_model
                        )

                        # Read transcript
                        status.update(label="📄 Reading transcript...", state="running")
                        transcript_text = read_transcript(transcript_path)

                        # Save to database
                        status.update(label="💾 Saving to database...", state="running")
                        db = st.session_state.db
                        transcript_id = db.save_transcript(
                            file_path=transcript_path,
                            content=transcript_text,
                            source_type="transcribed",
                            audio_file_path=tmp_audio_path,
                            whisper_model=whisper_model,
                        )
                        st.session_state.current_transcript_id = transcript_id

                        # Store in session state (for backward compatibility)
                        st.session_state.transcript_text = transcript_text
                        st.session_state.transcript_path = transcript_path

                        # Run sentiment analysis automatically in background
                        status.update(label="📊 Running sentiment analysis...", state="running")
                        # #region agent log
                        try:
                            with open("/Users/ymo/meeting_agent/.cursor/debug.log", "a", encoding="utf-8") as f:
                                f.write(json.dumps({"location": "streamlit_app.py:260", "message": "Calling _run_sentiment_analysis from transcription", "data": {"transcript_length": len(transcript_text) if transcript_text else 0}, "timestamp": time.time() * 1000, "sessionId": "debug-session", "runId": "run1", "hypothesisId": "A"}) + "\n")
                        except Exception:
                            pass
                        # #endregion
                        _run_sentiment_analysis(transcript_text, model=model, transcript_id=transcript_id)  # Errors stored in session state

                        status.update(label=f"✅ Transcription complete! Transcript saved to: {transcript_path.name}", state="complete")

                    st.success(f"✅ Transcription complete! Transcript saved to: {transcript_path.name}")

                    # Display transcript preview
                    st.subheader("📄 Transcript Preview")
                    st.text_area(
                        "Transcript",
                        value=transcript_text,
                        height=300,
                        disabled=True,
                        label_visibility="collapsed",
                    )
                    
                    # Display sentiment analysis if available
                    if st.session_state.sentiment_results:
                        _display_sentiment_results()
                    elif st.session_state.sentiment_error:
                        st.warning(f"⚠️ Sentiment analysis error: {st.session_state.sentiment_error}")

                except Exception as e:
                    # Status container will show error state if still open
                    st.error(f"❌ Transcription failed: {e}")
                finally:
                    # Clean up temporary file
                    if tmp_audio_path.exists():
                        tmp_audio_path.unlink()

    # Tab 2: Transcript Upload
    with tab2:
        st.header("Transcript Upload")
        st.markdown("Upload an existing transcript file or edit the current transcript.")

        uploaded_transcript = st.file_uploader(
            "Upload Transcript File",
            type=["txt", "md", "transcript", "srt", "vtt"],
            help="Supported formats: TXT, MD, TRANSCRIPT, SRT, VTT",
            key="transcript_uploader",  # Add key for proper state management
        )

        # Handle file upload - check both uploaded file and session state
        if uploaded_transcript is not None:
            # #region agent log
            try:
                with open("/Users/ymo/meeting_agent/.cursor/debug.log", "a", encoding="utf-8") as f:
                    f.write(json.dumps({"location": "streamlit_app.py:352", "message": "File upload detected", "data": {"filename": uploaded_transcript.name if uploaded_transcript else "None"}, "timestamp": time.time() * 1000, "sessionId": "debug-session", "runId": "run1", "hypothesisId": "A"}) + "\n")
            except Exception:
                pass
            # #endregion
            # Read uploaded transcript
            transcript_content = uploaded_transcript.read().decode("utf-8")

            # Save to transcripts directory immediately
            transcripts_dir = get_transcripts_dir()
            transcripts_dir.mkdir(parents=True, exist_ok=True)
            save_path = transcripts_dir / uploaded_transcript.name
            save_path.write_text(transcript_content, encoding="utf-8")

            # Save to database
            db = st.session_state.db
            transcript_id = db.save_transcript(
                file_path=save_path,
                content=transcript_content,
                source_type="uploaded",
            )
            st.session_state.current_transcript_id = transcript_id

            # Store in session state (for backward compatibility)
            st.session_state.transcript_text = transcript_content
            st.session_state.transcript_path = save_path

            # Run sentiment analysis automatically in background
            # #region agent log
            try:
                with open("/Users/ymo/meeting_agent/.cursor/debug.log", "a", encoding="utf-8") as f:
                    f.write(json.dumps({"location": "streamlit_app.py:307", "message": "Calling _run_sentiment_analysis from upload", "data": {"transcript_length": len(transcript_content) if transcript_content else 0}, "timestamp": time.time() * 1000, "sessionId": "debug-session", "runId": "run1", "hypothesisId": "A"}) + "\n")
            except Exception:
                pass
            # #endregion
            _run_sentiment_analysis(transcript_content, model=model, transcript_id=transcript_id)  # Errors stored in session state

            st.success(f"✅ Transcript loaded and saved to: {save_path.name}")
            
            # Display sentiment analysis if available
            if st.session_state.sentiment_results:
                _display_sentiment_results()
            elif st.session_state.sentiment_error:
                st.warning(f"⚠️ Sentiment analysis error: {st.session_state.sentiment_error}")
        elif st.session_state.transcript_text:
            # Show info if transcript already loaded from previous upload
            st.info(f"ℹ️ Transcript already loaded: {st.session_state.transcript_path.name if st.session_state.transcript_path else 'In-memory transcript'}")
            
            # Display sentiment analysis if available
            if st.session_state.sentiment_results:
                _display_sentiment_results()

        # Display current transcript for editing
        if st.session_state.transcript_text:
            st.subheader("📝 Edit Transcript")
            edited_transcript = st.text_area(
                "Transcript",
                value=st.session_state.transcript_text,
                height=400,
                label_visibility="collapsed",
            )

            col1, col2 = st.columns(2)
            with col1:
                if st.button("💾 Save Transcript", type="primary"):
                    st.session_state.transcript_text = edited_transcript
                    # Save to transcripts directory
                    transcripts_dir = get_transcripts_dir()
                    transcripts_dir.mkdir(parents=True, exist_ok=True)
                    if st.session_state.transcript_path:
                        save_path = transcripts_dir / st.session_state.transcript_path.name
                    else:
                        save_path = transcripts_dir / "uploaded_transcript.txt"
                    save_path.write_text(edited_transcript, encoding="utf-8")
                    st.session_state.transcript_path = save_path
                    
                    # Update database
                    if st.session_state.current_transcript_id:
                        db = st.session_state.db
                        db.save_transcript(
                            file_path=save_path,
                            content=edited_transcript,
                            source_type="uploaded",  # Keep original source type
                        )
                    
                    st.success(f"✅ Transcript saved to: {save_path.name}")

            with col2:
                if st.button("🔄 Reset to Original"):
                    # Reset to original transcript from session state path if available
                    if st.session_state.transcript_path and st.session_state.transcript_path.exists():
                        original_text = read_transcript(st.session_state.transcript_path)
                        st.session_state.transcript_text = original_text
                        st.rerun()
                    else:
                        st.warning("⚠️ Original transcript file not available for reset.")

        else:
            st.info("ℹ️ No transcript loaded. Upload a file or transcribe audio in the Audio Transcription tab.")

    # Tab 3: Generate Notes
    with tab3:
        st.header("Generate Meeting Notes")
        st.markdown("Generate structured meeting notes from transcripts using LLM.")

        # Get all transcripts from database
        try:
            all_transcripts = db.list_transcripts(limit=100)
        except Exception:
            all_transcripts = []

        # Check if we have any transcripts
        has_current_transcript = bool(st.session_state.transcript_text)
        has_database_transcripts = bool(all_transcripts)

        if not (has_current_transcript or has_database_transcripts):
            st.warning("⚠️ No transcripts available. Please upload or transcribe a transcript first.")
            st.stop()

        # Transcript selection section
        if has_database_transcripts or has_current_transcript:
            st.subheader("📄 Select Transcript for Notes Generation")

            # Create options for transcript selection
            transcript_options = []

            # Add current transcript if available
            if has_current_transcript:
                transcript_name = st.session_state.transcript_path.name if st.session_state.transcript_path else "Current Transcript"
                transcript_options.append(("current", f"📝 {transcript_name} (Currently Loaded)"))

            # Add database transcripts
            for transcript in all_transcripts:
                file_name = Path(transcript['file_path']).name
                created_date = transcript.get('created_at', '')[:10] if transcript.get('created_at') else 'Unknown'
                transcript_options.append((transcript['id'], f"💾 {file_name} (Created: {created_date})"))

            if len(transcript_options) == 1:
                # Only one option, select it automatically
                selected_transcript = transcript_options[0][0]
                st.info(f"📊 Generating notes for: {transcript_options[0][1]}")
            else:
                # Multiple options, let user choose
                option_labels = [option[1] for option in transcript_options]
                selected_index = st.selectbox(
                    "Choose transcript to generate notes for:",
                    range(len(option_labels)),
                    format_func=lambda i: option_labels[i],
                    key="notes_transcript_selector"
                )
                selected_transcript = transcript_options[selected_index][0]

            # Get transcript content
            if selected_transcript == "current":
                transcript_content = st.session_state.transcript_text
                transcript_name = st.session_state.transcript_path.name if st.session_state.transcript_path else "Current Transcript"
                transcript_id = st.session_state.current_transcript_id
            else:
                # Load from database
                transcript_info = db.get_transcript(transcript_id=selected_transcript)
                if transcript_info:
                    transcript_content = transcript_info.get('content', '')
                    transcript_name = Path(transcript_info.get('file_path', '')).name
                    transcript_id = selected_transcript
                else:
                    st.error("❌ Failed to load transcript from database.")
                    st.stop()
                    transcript_content = ""
                    transcript_name = "Unknown"
                    transcript_id = None

            if transcript_content:
                st.info(f"📄 Selected transcript: {transcript_name} ({len(transcript_content)} characters)")

                # Check if notes already exist for this transcript
                if transcript_id:
                    existing_notes = db.get_meeting_notes(transcript_id)
                    if existing_notes:
                        st.info("📝 **Existing notes found for this transcript.** You can regenerate them or download the existing ones.")

                        with st.expander("📖 View Existing Notes"):
                            st.markdown(existing_notes.get('content', ''))
            else:
                st.error("❌ No transcript content available.")
                st.stop()
        else:
            st.error("❌ No transcripts available.")
            st.stop()

        # Output format selection
        format_options = ["txt", "docx", "rtf", "md"]
        current_format_index = 1  # Default to "docx"
        if st.session_state.output_format in format_options:
            current_format_index = format_options.index(st.session_state.output_format)
        output_format = st.selectbox(
            "Output Format",
            format_options,
            index=current_format_index,
        )

        if st.button("🚀 Generate Notes", type="primary"):
            try:
                with st.status("🔄 Generating meeting notes... This may take a moment.", state="running", expanded=True) as status:
                    # Create meeting agent
                    status.update(label="🤖 Initializing LLM agent...", state="running")
                    agent = create_meeting_agent(model=model)

                    # Generate notes
                    status.update(label="📝 Generating meeting notes with LLM...", state="running")
                    response = agent.generate_meeting_notes(transcript_content)
                    notes = agent.get_notes_text(response)
                    token_usage = response.get("token_usage", {})

                    if not notes:
                        status.update(label="❌ No notes generated", state="error")
                        st.error("❌ No notes generated")
                        st.stop()

                    # Store in session state
                    st.session_state.generated_notes = notes

                    # Save notes
                    status.update(label="💾 Saving meeting notes...", state="running")

                    # Create a transcript path for saving notes
                    if selected_transcript == "current" and st.session_state.transcript_path:
                        transcript_path_for_notes = st.session_state.transcript_path
                    else:
                        # For database transcripts, create a temporary path based on transcript info
                        transcript_path_for_notes = Path(f"transcript_{transcript_id}.txt")
                        # Write content to a temporary file for the notes generation
                        temp_dir = get_transcripts_dir()
                        temp_path = temp_dir / f"temp_transcript_{transcript_id}.txt"
                        temp_path.write_text(transcript_content, encoding='utf-8')
                        transcript_path_for_notes = temp_path

                    model_info = {
                        "model_name": model.replace("-cloud", ""),
                        "version": "2024",
                        "url": "https://ollama.com",
                    }
                    notes_path = save_meeting_notes(
                        notes,
                        transcript_path_for_notes,
                        output_dir=get_transcripts_dir(),
                        format=output_format,
                        model_info=model_info,
                    )
                    st.session_state.notes_path = notes_path

                    # Save to database if we have a transcript_id
                    if transcript_id:
                        db = st.session_state.db
                        db.save_meeting_notes(
                            transcript_id=transcript_id,
                            content=notes,
                            format=output_format,
                            model_used=model,
                            file_path=notes_path,
                            input_tokens=token_usage.get("input_tokens"),
                            output_tokens=token_usage.get("output_tokens"),
                            total_tokens=token_usage.get("total_tokens"),
                        )

                    # Clean up temporary file if created
                    if selected_transcript != "current" and transcript_path_for_notes.exists():
                        try:
                            transcript_path_for_notes.unlink()
                        except:
                            pass
                        
                        status.update(label="✅ Meeting notes generated and saved successfully!", state="complete")
                    else:
                        status.update(label="✅ Meeting notes generated successfully!", state="complete")

                st.success("✅ Meeting notes generated successfully!")

                # Display token usage
                st.subheader("📊 Token Usage")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Input Tokens", f"{token_usage.get('input_tokens', 0):,}")
                with col2:
                    st.metric("Output Tokens", f"{token_usage.get('output_tokens', 0):,}")
                with col3:
                    st.metric("Total Tokens", f"{token_usage.get('total_tokens', 0):,}")

                # Display notes preview (formatted markdown)
                st.subheader("📝 Notes Preview")
                st.markdown(notes)

            except Exception as e:
                st.error(f"❌ Error generating meeting notes: {e}")

    # Tab 4: Q&A Chat
    with tab4:
        st.header("Q&A Chat")
        st.markdown("Ask questions about transcripts using natural language.")

        # Get all transcripts from database
        try:
            all_transcripts = db.list_transcripts(limit=100)
        except Exception:
            all_transcripts = []

        # Check if we have any transcripts
        has_current_transcript = bool(st.session_state.transcript_text)
        has_database_transcripts = bool(all_transcripts)

        if not (has_current_transcript or has_database_transcripts):
            st.warning("⚠️ No transcripts available. Please upload or transcribe a transcript first.")
            st.stop()

        # Transcript selection section
        if has_database_transcripts or has_current_transcript:
            st.subheader("📄 Select Transcript for Q&A")

            # Create options for transcript selection
            transcript_options = []

            # Add current transcript if available
            if has_current_transcript:
                transcript_name = st.session_state.transcript_path.name if st.session_state.transcript_path else "Current Transcript"
                transcript_options.append(("current", f"📝 {transcript_name} (Currently Loaded)"))

            # Add database transcripts
            for transcript in all_transcripts:
                file_name = Path(transcript['file_path']).name
                created_date = transcript.get('created_at', '')[:10] if transcript.get('created_at') else 'Unknown'
                transcript_options.append((transcript['id'], f"💾 {file_name} (Created: {created_date})"))

            if len(transcript_options) == 1:
                # Only one option, select it automatically
                selected_transcript = transcript_options[0][0]
                st.info(f"💬 Chatting about: {transcript_options[0][1]}")
            else:
                # Multiple options, let user choose
                option_labels = [option[1] for option in transcript_options]
                selected_index = st.selectbox(
                    "Choose transcript to ask questions about:",
                    range(len(option_labels)),
                    format_func=lambda i: option_labels[i],
                    key="chat_transcript_selector"
                )
                selected_transcript = transcript_options[selected_index][0]

            # Get transcript content
            if selected_transcript == "current":
                transcript_content = st.session_state.transcript_text
                transcript_name = st.session_state.transcript_path.name if st.session_state.transcript_path else "Current Transcript"
                transcript_id = st.session_state.current_transcript_id
            else:
                # Load from database
                transcript_info = db.get_transcript(transcript_id=selected_transcript)
                if transcript_info:
                    transcript_content = transcript_info.get('content', '')
                    transcript_name = Path(transcript_info.get('file_path', '')).name
                    transcript_id = selected_transcript
                else:
                    st.error("❌ Failed to load transcript from database.")
                    st.stop()
                    transcript_content = ""
                    transcript_name = "Unknown"
                    transcript_id = None

            if not transcript_content:
                st.error("❌ No transcript content available.")
                st.stop()
        else:
            st.error("❌ No transcripts available.")
            st.stop()

        # Load chat history from database if available
        if transcript_id:
            db = st.session_state.db
            db_chat_history = db.get_chat_history(transcript_id)
            if db_chat_history and not st.session_state.chat_history:
                # Load from database if session state is empty
                st.session_state.chat_history = [
                    {"role": msg["role"], "content": msg["content"]}
                    for msg in db_chat_history
                ]

        # Clear chat button (placed at top, before chat history)
        if st.session_state.chat_history:
            if st.button("🗑️ Clear Chat History", key="clear_chat"):
                st.session_state.chat_history = []
                # Clear from database
                if st.session_state.current_transcript_id:
                    db = st.session_state.db
                    # Delete chat history for this transcript
                    # (We'll need to add a delete method, but for now just clear session state)
                st.rerun()

        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input - will be pinned to bottom when in main body
        # According to Streamlit docs, st.chat_input automatically pins to bottom
        if prompt := st.chat_input("Ask a question about the transcript..."):
            # Add user message to chat history and display immediately
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate response with proper chat interface
            try:
                # Display assistant message with thinking indicator
                with st.chat_message("assistant"):
                    # Show thinking indicator
                    thinking_placeholder = st.empty()
                    thinking_placeholder.markdown("🤔 Thinking...")
                    
                    try:
                        # Load Q&A prompts
                        system_prompt = load_prompt("cli_qa_system_prompt.txt")
                        user_prompt_template = load_prompt("cli_qa_user_prompt_template.txt")

                        # Format user prompt
                        user_prompt = user_prompt_template.format(
                            question=prompt, transcript=transcript_content
                        )

                        # Create messages
                        messages = create_llm_messages(system_prompt, user_prompt)

                        # Call LLM
                        response = call_ollama_cloud(model=model, messages=messages, stream=False)

                        # Parse response
                        response_dict = parse_ollama_response(response)
                        answer = extract_response_text(response_dict)

                        if answer:
                            # Replace thinking indicator with actual answer
                            thinking_placeholder.empty()
                            st.markdown(answer)
                            st.session_state.chat_history.append({"role": "assistant", "content": answer})
                            
                            # Save to database
                            if transcript_id:
                                db = st.session_state.db
                                db.save_chat_message(transcript_id, "user", prompt)
                                db.save_chat_message(transcript_id, "assistant", answer)
                        else:
                            # Replace thinking indicator with error
                            thinking_placeholder.empty()
                            st.error("❌ No answer received from LLM")
                            error_msg = "Error: Could not parse response"
                            st.session_state.chat_history.append(
                                {"role": "assistant", "content": error_msg}
                            )
                            
                            # Save to database
                            if transcript_id:
                                db = st.session_state.db
                                db.save_chat_message(transcript_id, "user", prompt)
                                db.save_chat_message(transcript_id, "assistant", error_msg)
                    except Exception as e:
                        # Replace thinking indicator with error
                        thinking_placeholder.empty()
                        st.error(f"❌ Error asking question: {e}")
                        st.session_state.chat_history.append(
                            {"role": "assistant", "content": f"Error: {e}"}
                        )

            except Exception as e:
                # Fallback error handling
                with st.chat_message("assistant"):
                    st.error(f"❌ Error asking question: {e}")
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": f"Error: {e}"}
                )

    # Tab 5: Downloads
    with tab5:
        st.header("Downloads")
        st.markdown("Download meeting notes in various formats.")

        # Check for notes sources
        has_generated_notes = bool(st.session_state.generated_notes)
        has_notes_path = bool(st.session_state.notes_path)

        # Get transcripts with notes from database
        transcripts_with_notes = []
        try:
            all_transcripts = db.list_transcripts(limit=100)
            for transcript in all_transcripts:
                if db.get_meeting_notes(transcript['id']):
                    transcripts_with_notes.append(transcript)
        except Exception:
            transcripts_with_notes = []

        has_database_notes = bool(transcripts_with_notes)

        if not (has_generated_notes or has_database_notes):
            st.warning("⚠️ No notes available. Please generate notes first in the Generate Notes tab.")
            st.stop()

        # Notes source selection
        if has_generated_notes and has_database_notes:
            st.subheader("📄 Select Notes Source")

            notes_options = []
            if has_generated_notes:
                notes_options.append(("current", "📝 Currently Generated Notes"))
            if has_database_notes:
                notes_options.append(("database", "💾 Saved Notes from Database"))

            if len(notes_options) == 1:
                selected_source = notes_options[0][0]
            else:
                option_labels = [opt[1] for opt in notes_options]
                selected_idx = st.selectbox(
                    "Choose notes to download:",
                    range(len(option_labels)),
                    format_func=lambda i: option_labels[i],
                    key="download_source_selector"
                )
                selected_source = notes_options[selected_idx][0]
        elif has_generated_notes:
            selected_source = "current"
            st.info("📄 Using currently generated notes")
        else:
            selected_source = "database"
            st.info("📄 Using saved notes from database")

        # Handle database notes selection
        notes_content = ""
        notes_filename = ""

        if selected_source == "current":
            notes_content = st.session_state.generated_notes
            notes_filename = st.session_state.notes_path.name if st.session_state.notes_path else "meeting_notes"
        else:
            # Database notes - let user select which transcript's notes
            if len(transcripts_with_notes) == 1:
                selected_transcript_id = transcripts_with_notes[0]['id']
                selected_transcript_name = Path(transcripts_with_notes[0]['file_path']).name
            else:
                transcript_options = [(t['id'], f"{Path(t['file_path']).name}") for t in transcripts_with_notes]
                option_labels = [opt[1] for opt in transcript_options]
                selected_idx = st.selectbox(
                    "Choose transcript notes to download:",
                    range(len(option_labels)),
                    format_func=lambda i: option_labels[i],
                    key="download_transcript_selector"
                )
                selected_transcript_id = transcript_options[selected_idx][0]
                selected_transcript_name = option_labels[selected_idx]

            # Get notes from database
            notes_data = db.get_meeting_notes(selected_transcript_id)
            if notes_data:
                notes_content = notes_data.get('content', '')
                notes_filename = f"{selected_transcript_name}_notes"
            else:
                st.error("❌ Failed to load notes from database.")
                st.stop()

        if notes_content:
            st.info(f"📄 Notes ready for download: {notes_filename}")
        else:
            st.error("❌ No notes content available.")
            st.stop()

        # Generate downloads for all formats
        formats = ["txt", "docx", "rtf", "md"]
        cols = st.columns(len(formats))

        for idx, fmt in enumerate(formats):
            with cols[idx]:
                try:
                    # Generate file in this format
                    if st.session_state.transcript_path:
                        model_info = {
                            "model_name": model.replace("-cloud", ""),
                            "version": "2024",
                            "url": "https://ollama.com",
                        }
                        notes_path = save_meeting_notes(
                            st.session_state.generated_notes,
                            st.session_state.transcript_path,
                            output_dir=get_transcripts_dir(),
                            format=fmt,
                            model_info=model_info,
                        )

                        # Read file content
                        file_content = notes_path.read_bytes()
                        file_name = notes_path.name

                        # Determine MIME type
                        mime_types = {
                            "txt": "text/plain",
                            "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                            "rtf": "application/rtf",
                            "md": "text/markdown",
                        }

                        st.download_button(
                            label=f"⬇️ Download {fmt.upper()}",
                            data=file_content,
                            file_name=file_name,
                            mime=mime_types.get(fmt, "application/octet-stream"),
                            key=f"download_{fmt}",
                        )
                    else:
                        # Fallback: provide text content directly
                        if fmt == "txt" or fmt == "md":
                            st.download_button(
                                label=f"⬇️ Download {fmt.upper()}",
                                data=st.session_state.generated_notes,
                                file_name=f"meeting_notes.{fmt}",
                                mime="text/plain" if fmt == "txt" else "text/markdown",
                                key=f"download_{fmt}",
                            )
                        else:
                            st.info(f"💡 {fmt.upper()} format requires a saved transcript path")

                except Exception as e:
                    st.error(f"❌ Error generating {fmt.upper()} format: {e}")

    # Tab 6: Sentiment Analysis
    with tab6:
        # #region agent log
        try:
            log_file = open("/Users/ymo/meeting_agent/.cursor/debug.log", "a", encoding="utf-8")
            log_entry = json.dumps({"location": "streamlit_app.py:629", "message": "Tab6 entered - FIRST LINE", "data": {"session_keys": list(st.session_state.keys())}, "timestamp": time.time() * 1000, "sessionId": "debug-session", "runId": "run1", "hypothesisId": "D"}) + "\n"
            log_file.write(log_entry)
            log_file.flush()
            log_file.close()
        except Exception as log_err:
            # If logging fails, show error in UI
            st.error(f"DEBUG: Log error: {log_err}")
        # #endregion
        st.header("📊 Sentiment Analysis")
        st.markdown("Analyze the sentiment of transcripts and individual attendees.")

        # Get transcripts with sentiment analysis from database
        transcripts_with_sentiment = db.get_transcripts_with_sentiment()

        # Check if we have any transcripts with sentiment data
        has_current_transcript = bool(st.session_state.transcript_text)
        has_sentiment_results = bool(st.session_state.sentiment_results)
        has_database_transcripts = bool(transcripts_with_sentiment)

        if not (has_current_transcript or has_database_transcripts):
            st.warning("⚠️ No transcripts with sentiment analysis available. Please upload and analyze a transcript first.")
            st.info("💡 **To get started:** Upload a transcript in the Audio Transcription or Transcript Upload tabs, then the sentiment analysis will run automatically.")
            st.stop()

        # Transcript selection section
        if has_database_transcripts or has_current_transcript:
            st.subheader("📄 Select Transcript for Analysis")

            # Create options for transcript selection
            transcript_options = []

            # Add current transcript if available
            if has_current_transcript:
                transcript_name = st.session_state.transcript_path.name if st.session_state.transcript_path else "Current Transcript"
                transcript_options.append(("current", f"📝 {transcript_name} (Currently Loaded)"))

            # Add database transcripts
            for transcript in transcripts_with_sentiment:
                file_name = Path(transcript['file_path']).name
                analyzed_date = transcript.get('analyzed_at', '')[:10] if transcript.get('analyzed_at') else 'Unknown'
                transcript_options.append((transcript['id'], f"💾 {file_name} (Analyzed: {analyzed_date})"))

            if len(transcript_options) == 1:
                # Only one option, select it automatically
                selected_transcript = transcript_options[0][0]
                st.info(f"📊 Analyzing: {transcript_options[0][1]}")
            else:
                # Multiple options, let user choose
                option_labels = [option[1] for option in transcript_options]
                selected_index = st.selectbox(
                    "Choose transcript to analyze:",
                    range(len(option_labels)),
                    format_func=lambda i: option_labels[i],
                    key="sentiment_transcript_selector"
                )
                selected_transcript = transcript_options[selected_index][0]

            # Load sentiment data for selected transcript
            if selected_transcript == "current":
                # Use currently loaded data
                if not has_sentiment_results:
                    st.info("📊 The current transcript hasn't been analyzed yet. Sentiment analysis will run automatically when you upload/analyze a transcript.")
                    st.stop()
                # Data is already in session state
            else:
                # Load data from database
                if st.button("🔄 Load Sentiment Analysis", key="load_sentiment_analysis"):
                    with st.spinner("Loading sentiment analysis from database..."):
                        sentiment_data = db.get_sentiment_analysis(selected_transcript)
                        if sentiment_data:
                            from meeting_agent.sentiment_analyzer import get_sentiment_emoji
                            overall_compound = sentiment_data.get("overall_compound", 0.0)
                            st.session_state.sentiment_results = {
                                "overall": {
                                    "compound": overall_compound,
                                    "pos": sentiment_data.get("overall_pos", 0.0),
                                    "neu": sentiment_data.get("overall_neu", 0.0),
                                    "neg": sentiment_data.get("overall_neg", 0.0),
                                    "label": sentiment_data.get("overall_label", "Neutral"),
                                    "emoji": get_sentiment_emoji(overall_compound),
                                },
                                "attendees": sentiment_data.get("attendees", {}),
                                "topics": list(sentiment_data.get("topics", {}).keys()),
                                "topic_sentiments": sentiment_data.get("topics", {}),
                            }
                            # Extract attendee names
                            st.session_state.sentiment_attendees = list(sentiment_data.get("attendees", {}).keys())
                            # Load the transcript content too
                            transcript_info = db.get_transcript(transcript_id=selected_transcript)
                            if transcript_info:
                                st.session_state.transcript_text = transcript_info.get('content', '')
                                st.session_state.current_transcript_id = selected_transcript
                                st.session_state.transcript_path = Path(transcript_info.get('file_path', ''))
                            st.success("✅ Sentiment analysis loaded from database!")
                            st.rerun()
                        else:
                            st.error("❌ Failed to load sentiment analysis from database.")

                # Check if we have results loaded
                if not st.session_state.sentiment_results:
                    st.info("👆 Click 'Load Sentiment Analysis' to view the sentiment data for this transcript.")
                    st.stop()

        # Check sentiment analysis dependencies
        # #region agent log
        try:
            with open("/Users/ymo/meeting_agent/.cursor/debug.log", "a", encoding="utf-8") as f:
                f.write(json.dumps({"location": "streamlit_app.py:636", "message": "Before dependency check", "data": {}, "timestamp": time.time() * 1000, "sessionId": "debug-session", "runId": "run1", "hypothesisId": "D"}) + "\n")
        except Exception:
            pass
        # #endregion
        all_available, missing = check_sentiment_dependencies()
        # #region agent log
        try:
            with open("/Users/ymo/meeting_agent/.cursor/debug.log", "a", encoding="utf-8") as f:
                f.write(json.dumps({"location": "streamlit_app.py:643", "message": "After dependency check", "data": {"all_available": all_available, "missing": missing}, "timestamp": time.time() * 1000, "sessionId": "debug-session", "runId": "run1", "hypothesisId": "D"}) + "\n")
        except Exception:
            pass
        # #endregion
        if not all_available:
            st.error(f"❌ Missing dependencies: {', '.join(missing)}")
            st.info(
                "Install missing dependencies with:\n"
                "```bash\n"
                "pip install vaderSentiment spacy\n"
                "python -m spacy download en_core_web_sm\n"
                "```"
            )
            st.stop()

        # Check for errors from background analysis
        if st.session_state.sentiment_error:
            st.error(f"❌ Sentiment analysis error: {st.session_state.sentiment_error}")
            if "Missing dependencies" in st.session_state.sentiment_error or "not installed" in st.session_state.sentiment_error.lower():
                st.info(
                    "Install missing dependencies with:\n"
                    "```bash\n"
                    "pip install vaderSentiment spacy pandas\n"
                    "python -m spacy download en_core_web_sm\n"
                    "```"
                )
            elif "model not found" in st.session_state.sentiment_error.lower() or "en_core_web_sm" in st.session_state.sentiment_error:
                st.info(
                    "Download the spaCy English model with:\n"
                    "```bash\n"
                    "python -m spacy download en_core_web_sm\n"
                    "```"
                )
            st.stop()

        # #region agent log
        try:
            with open("/Users/ymo/meeting_agent/.cursor/debug.log", "a", encoding="utf-8") as f:
                f.write(json.dumps({"location": "streamlit_app.py:568", "message": "Sentiment tab loaded", "data": {"has_results": st.session_state.sentiment_results is not None, "has_error": st.session_state.sentiment_error is not None, "has_transcript": st.session_state.transcript_text is not None, "transcript_length": len(st.session_state.transcript_text) if st.session_state.transcript_text else 0}, "timestamp": time.time() * 1000, "sessionId": "debug-session", "runId": "run1", "hypothesisId": "D"}) + "\n")
        except Exception:
            pass
        # #endregion

        # Use cached results if available, otherwise run analysis
        if st.session_state.sentiment_results is None:
            with st.status("🔄 Analyzing sentiment...", state="running", expanded=True) as status:
                # #region agent log
                try:
                    with open("/Users/ymo/meeting_agent/.cursor/debug.log", "a", encoding="utf-8") as f:
                        f.write(json.dumps({"location": "streamlit_app.py:570", "message": "No cached results, running analysis", "data": {}, "timestamp": time.time() * 1000, "sessionId": "debug-session", "runId": "run1", "hypothesisId": "D"}) + "\n")
                except Exception:
                    pass
                # #endregion
                # Get model from sidebar (it's defined in main() scope)
                # Access via session state or use default
                current_model = st.session_state.get("current_model", DEFAULT_CLOUD_MODEL)
                status.update(label="📊 Running sentiment analysis...", state="running")
                success, error_msg = _run_sentiment_analysis(st.session_state.transcript_text, model=current_model)
                if not success:
                    status.update(label=f"❌ Could not perform sentiment analysis: {error_msg}", state="error")
                    st.error(f"❌ Could not perform sentiment analysis: {error_msg}")
                    if "Missing dependencies" in (error_msg or ""):
                        status.update(label="❌ Missing dependencies", state="error")
                        st.info(
                            "Install missing dependencies with:\n"
                            "```bash\n"
                            "pip install vaderSentiment spacy pandas\n"
                            "python -m spacy download en_core_web_sm\n"
                            "```"
                        )
                    elif "model not found" in (error_msg or "").lower() or "en_core_web_sm" in (error_msg or ""):
                        st.info(
                            "Download the spaCy English model with:\n"
                            "```bash\n"
                            "python -m spacy download en_core_web_sm\n"
                            "```"
                        )
                    st.stop()
                else:
                    status.update(label="✅ Sentiment analysis complete!", state="complete")

        # Check if we have results (either cached or just computed)
        # #region agent log
        try:
            with open("/Users/ymo/meeting_agent/.cursor/debug.log", "a", encoding="utf-8") as f:
                f.write(json.dumps({"location": "streamlit_app.py:707", "message": "Checking sentiment_results", "data": {"has_results": st.session_state.sentiment_results is not None, "results_type": type(st.session_state.sentiment_results).__name__ if st.session_state.sentiment_results else "None"}, "timestamp": time.time() * 1000, "sessionId": "debug-session", "runId": "run1", "hypothesisId": "D"}) + "\n")
        except Exception:
            pass
        # #endregion
        if st.session_state.sentiment_results is None:
            st.error("❌ No sentiment analysis results available.")
            st.info("Please try uploading or transcribing a transcript again.")
            st.stop()

        try:
            # #region agent log
            try:
                with open("/Users/ymo/meeting_agent/.cursor/debug.log", "a", encoding="utf-8") as f:
                    f.write(json.dumps({"location": "streamlit_app.py:715", "message": "Extracting results from session state", "data": {"has_overall": "overall" in (st.session_state.sentiment_results or {})}, "timestamp": time.time() * 1000, "sessionId": "debug-session", "runId": "run1", "hypothesisId": "D"}) + "\n")
            except Exception:
                pass
            # #endregion
            # Get cached results
            overall_sentiment = st.session_state.sentiment_results.get("overall")
            attendee_sentiments = st.session_state.sentiment_results.get("attendees", {})
            attendees_list = st.session_state.sentiment_attendees or []
            
            # Check if overall sentiment exists
            if overall_sentiment is None:
                st.error("❌ Sentiment results are incomplete. Please run sentiment analysis again.")
                st.stop()

            # Display overall sentiment
            st.subheader("📈 Overall Sentiment")

            # Display overall sentiment with visual indicators
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Overall Sentiment",
                    f"{overall_sentiment['emoji']} {overall_sentiment['label']}",
                )
            with col2:
                st.metric(
                    "Compound Score",
                    f"{overall_sentiment['compound']:.3f}",
                    help="Range: -1 (most negative) to +1 (most positive)",
                )
            with col3:
                text_length = len(st.session_state.transcript_text) if st.session_state.transcript_text else 0
                st.metric("Text Length", f"{text_length:,} chars")

            # Sentiment breakdown
            st.markdown("#### Sentiment Breakdown")
            breakdown_col1, breakdown_col2, breakdown_col3 = st.columns(3)
            with breakdown_col1:
                st.progress(overall_sentiment["pos"], text=f"Positive: {overall_sentiment['pos']*100:.1f}%")
            with breakdown_col2:
                st.progress(overall_sentiment["neu"], text=f"Neutral: {overall_sentiment['neu']*100:.1f}%")
            with breakdown_col3:
                st.progress(overall_sentiment["neg"], text=f"Negative: {overall_sentiment['neg']*100:.1f}%")

            # Interpretation
            compound = overall_sentiment["compound"]
            if compound >= 0.05:
                st.success(
                    f"✅ The overall sentiment is **positive** (score: {compound:.3f}). "
                    "The meeting appears to have a constructive and upbeat tone."
                )
            elif compound <= -0.05:
                st.error(
                    f"⚠️ The overall sentiment is **negative** (score: {compound:.3f}). "
                    "The meeting may contain concerns, disagreements, or challenges."
                )
            else:
                st.info(
                    f"ℹ️ The overall sentiment is **neutral** (score: {compound:.3f}). "
                    "The meeting maintains a balanced and objective tone."
                )

            st.divider()

            # Display attendee sentiment (using cached results)
            st.subheader("👥 Attendee Sentiment Analysis")

            if attendees_list:
                st.success(f"✅ Found {len(attendees_list)} attendee(s): {', '.join(attendees_list)}")

                # Use cached attendee sentiment data
                attendee_data = []
                for attendee in attendees_list:
                    if attendee in attendee_sentiments:
                        sentiment = attendee_sentiments[attendee]
                        attendee_data.append(
                            {
                                "Attendee": attendee,
                                "Sentiment": f"{sentiment['emoji']} {sentiment['label']}",
                                "Compound Score": f"{sentiment['compound']:.3f}",
                                "Positive": f"{sentiment['pos']*100:.1f}%",
                                "Neutral": f"{sentiment['neu']*100:.1f}%",
                                "Negative": f"{sentiment['neg']*100:.1f}%",
                                "Mentions": sentiment["mentions"],
                            }
                        )

                if attendee_data:
                    # Display table
                    if not PANDAS_AVAILABLE:
                        st.error("❌ pandas is required for displaying sentiment tables. Please install it with: pip install pandas")
                    else:
                        df = pd.DataFrame(attendee_data)
                        st.dataframe(df, use_container_width=True, hide_index=True)

                    # Detailed view for each attendee
                    st.markdown("#### Detailed Attendee Sentiment")
                    for attendee in attendees_list:
                        with st.expander(f"📊 {attendee}"):
                            if attendee in attendee_sentiments:
                                sentiment = attendee_sentiments[attendee]
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Sentiment", f"{sentiment['emoji']} {sentiment['label']}")
                                with col2:
                                    st.metric("Compound Score", f"{sentiment['compound']:.3f}")
                                with col3:
                                    st.metric("Mentions", sentiment["mentions"])

                                st.markdown("**Sentiment Breakdown:**")
                                detail_col1, detail_col2, detail_col3 = st.columns(3)
                                with detail_col1:
                                    st.progress(
                                        sentiment["pos"],
                                        text=f"Positive: {sentiment['pos']*100:.1f}%",
                                    )
                                with detail_col2:
                                    st.progress(
                                        sentiment["neu"],
                                        text=f"Neutral: {sentiment['neu']*100:.1f}%",
                                    )
                                with detail_col3:
                                    st.progress(
                                        sentiment["neg"],
                                        text=f"Negative: {sentiment['neg']*100:.1f}%",
                                    )
                            else:
                                st.warning(f"No sentiment data available for {attendee}")
            else:
                st.info(
                    "ℹ️ No named entities (attendees) detected in this transcript.\n\n"
                    "This could mean:\n"
                    "- The transcript doesn't contain speaker labels (e.g., 'John:', 'Speaker 1:')\n"
                    "- No person names are mentioned in the transcript\n"
                    "- The transcript format doesn't include participant identification\n\n"
                    "Overall sentiment analysis is still available above."
                )

        except Exception as e:
            st.error(f"❌ Error performing sentiment analysis: {e}")
            import traceback

            with st.expander("🔍 Error Details"):
                st.code(traceback.format_exc())

    # Tab 7: Transcript Library
    with tab7:
        st.header("📚 Transcript Library")
        st.markdown("View and manage all stored transcripts, labels, and metadata.")

        try:
            db = st.session_state.db

            # Filters
            col1, col2 = st.columns(2)
            with col1:
                try:
                    all_labels = db.get_all_labels()
                    selected_label = st.selectbox(
                        "Filter by Label",
                        ["All"] + all_labels,
                        key="library_label_filter",
                    )
                except Exception as e:
                    st.error(f"❌ Error loading labels: {e}")
                    selected_label = "All"
                    all_labels = []

            with col2:
                source_type_filter = st.selectbox(
                    "Filter by Source",
                    ["All", "uploaded", "transcribed"],
                    key="library_source_filter",
                )

            # Get transcripts
            label_filter = None if selected_label == "All" else selected_label
            source_type = None if source_type_filter == "All" else source_type_filter
            
            try:
                transcripts = db.list_transcripts(
                    label_filter=label_filter,
                    source_type=source_type,
                    limit=50,
                )
            except Exception as e:
                st.error(f"❌ Error loading transcripts: {e}")
                import traceback
                with st.expander("🔍 Error Details"):
                    st.code(traceback.format_exc())
                transcripts = []

            if not transcripts:
                st.info("📭 No transcripts found. Upload or transcribe a transcript to get started.")
            else:
                st.success(f"📊 Found {len(transcripts)} transcript(s)")

                # Display transcripts in a table
                for transcript in transcripts:
                    try:
                        # Format date safely
                        created_date = "Unknown date"
                        if transcript.get('created_at'):
                            if isinstance(transcript['created_at'], str):
                                created_date = transcript['created_at'][:10]
                            else:
                                created_date = str(transcript['created_at'])[:10]
                        
                        # Format file path safely
                        file_name = "Unknown file"
                        try:
                            file_name = Path(transcript['file_path']).name
                        except Exception:
                            file_name = str(transcript.get('file_path', 'Unknown file'))
                        
                        with st.expander(f"📄 {file_name} - {created_date}"):
                            # Check if archived
                            is_archived = transcript.get('archived', 0)
                            if is_archived:
                                st.warning("📦 This transcript is archived")

                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Characters", f"{transcript['character_count']:,}")
                            with col2:
                                st.metric("Source", transcript['source_type'].title())
                            with col3:
                                if transcript['whisper_model']:
                                    st.metric("Whisper Model", transcript['whisper_model'])

                            # Get labels for this transcript
                            labels = db.get_labels(transcript['id'])
                            if labels:
                                label_texts = [l['label_text'] for l in labels]
                                st.markdown(f"**Labels:** {', '.join(label_texts)}")

                            # Show retention policy status
                            if not is_archived:
                                try:
                                    retention_manager = RetentionManager(db)
                                    policy_status = retention_manager.get_policy_status(transcript['id'])
                                    if policy_status:
                                        st.divider()
                                        st.subheader("📋 Retention Policy Status")
                                        policy = policy_status['policy']
                                        days_remaining = policy_status['days_remaining']
                                        
                                        col_ret1, col_ret2, col_ret3 = st.columns(3)
                                        with col_ret1:
                                            st.metric("Policy", policy['label_text'])
                                        with col_ret2:
                                            st.metric("Retention Days", policy['retention_days'])
                                        with col_ret3:
                                            if days_remaining < 0:
                                                st.error(f"⚠️ {abs(days_remaining)} days overdue")
                                            elif days_remaining <= 7:
                                                st.warning(f"⏰ {days_remaining} days remaining")
                                            else:
                                                st.info(f"✅ {days_remaining} days remaining")
                                        
                                        st.caption(f"Action: {policy['action'].title()} | Retention Date: {policy_status['retention_date'].strftime('%Y-%m-%d')}")
                                except Exception:
                                    # Silently fail if retention check fails
                                    pass

                            # Add/remove labels
                            st.subheader("🏷️ Manage Labels")
                            label_col1, label_col2 = st.columns([3, 1])
                            with label_col1:
                                new_label = st.text_input(
                                    "Add Custom Label",
                                    key=f"new_label_{transcript['id']}",
                                    placeholder="e.g., Project Planning, Team Meeting",
                                )
                            with label_col2:
                                if st.button("➕ Add", key=f"add_label_{transcript['id']}"):
                                    if new_label.strip():
                                        db.add_label(transcript['id'], new_label.strip(), label_type="custom")
                                        st.success(f"✅ Added label: {new_label.strip()}")
                                        st.rerun()

                            # Remove labels
                            if labels:
                                st.markdown("**Remove Labels:**")
                                for label in labels:
                                    if label['label_type'] == 'custom':  # Only allow removing custom labels
                                        col_a, col_b = st.columns([4, 1])
                                        with col_a:
                                            st.text(label['label_text'])
                                        with col_b:
                                            if st.button("🗑️", key=f"remove_label_{transcript['id']}_{label['id']}"):
                                                db.remove_label(transcript['id'], label['label_text'])
                                                st.success(f"✅ Removed label: {label['label_text']}")
                                                st.rerun()

                            # Load transcript button
                            if st.button("📂 Load This Transcript", key=f"load_{transcript['id']}"):
                                st.session_state.current_transcript_id = transcript['id']
                                st.session_state.transcript_text = transcript['content']
                                st.session_state.transcript_path = Path(transcript['file_path'])
                                
                                # Load sentiment from database if available
                                sentiment_data = db.get_sentiment_analysis(transcript['id'])
                                if sentiment_data:
                                    from meeting_agent.sentiment_analyzer import get_sentiment_emoji
                                    overall_compound = sentiment_data.get("overall_compound", 0.0)
                                    st.session_state.sentiment_results = {
                                        "overall": {
                                            "compound": overall_compound,
                                            "pos": sentiment_data.get("overall_pos", 0.0),
                                            "neu": sentiment_data.get("overall_neu", 0.0),
                                            "neg": sentiment_data.get("overall_neg", 0.0),
                                            "label": sentiment_data.get("overall_label", "Neutral"),
                                            "emoji": get_sentiment_emoji(overall_compound),
                                        },
                                        "attendees": sentiment_data.get("attendees", {}),
                                        "topics": list(sentiment_data.get("topics", {}).keys()),
                                        "topic_sentiments": sentiment_data.get("topics", {}),
                                    }
                                    # Extract attendee names
                                    st.session_state.sentiment_attendees = list(sentiment_data.get("attendees", {}).keys())
                                
                                # Load chat history
                                chat_history = db.get_chat_history(transcript['id'])
                                if chat_history:
                                    st.session_state.chat_history = [
                                        {"role": msg["role"], "content": msg["content"]}
                                        for msg in chat_history
                                    ]
                                
                                st.success("✅ Transcript loaded! Switch to other tabs to view details.")
                                st.rerun()

                            # Delete transcript button
                            if st.button("🗑️ Delete Transcript", key=f"delete_{transcript['id']}", type="secondary"):
                                if db.delete_transcript(transcript['id']):
                                    st.success("✅ Transcript deleted")
                                    st.rerun()
                                else:
                                    st.error("❌ Failed to delete transcript")
                    except Exception as e:
                        st.error(f"❌ Error displaying transcript {transcript.get('id', 'unknown')}: {e}")
                        import traceback
                        with st.expander("🔍 Error Details"):
                            st.code(traceback.format_exc())
        
        except Exception as e:
            st.error(f"❌ Error in Transcript Library: {e}")
            import traceback
            with st.expander("🔍 Error Details"):
                st.code(traceback.format_exc())
            st.info("💡 **Note:** Data is persisted in the database. Try refreshing the page or check the terminal for more details.")

    # Tab 8: Governance
    with tab8:
        st.header("📋 Governance")
        st.markdown("Manage retention policies, data lifecycle, and compliance.")

        try:
            db = st.session_state.db
            retention_manager = RetentionManager(db)

            # Summary metrics at the top
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                policies = db.get_retention_policies()
                active_policies = [p for p in policies if p.get('enabled', 1)]
                st.metric("Total Policies", len(policies))
            with col2:
                st.metric("Active Policies", len(active_policies))
            with col3:
                pending = retention_manager.get_pending_actions()
                st.metric("Pending Actions", len(pending))
            with col4:
                archived = db.get_archived_transcripts(limit=1000)
                st.metric("Archived Transcripts", len(archived))

            st.divider()

            # Create sub-tabs for different governance sections
            gov_tab1, gov_tab2, gov_tab3, gov_tab4 = st.tabs(
                [
                    "🎯 Retention Policies",
                    "⚙️ Policy Application",
                    "📦 Archive Management",
                    "🏷️ Label Policy Mapping",
                ]
            )

            # Sub-tab 1: Retention Policies
            with gov_tab1:
                st.subheader("🎯 Retention Policies")

                # LLM-Powered Policy Creation
                with st.expander("🤖 LLM-Powered Policy Generation", expanded=False):
                    st.markdown("**Use AI to analyze your data and generate retention policy recommendations.**")

                    # Check if there are any transcripts available
                    all_transcripts = db.list_transcripts(limit=1)
                    if not all_transcripts:
                        st.warning("⚠️ No transcripts available in the database. Please upload or transcribe some transcripts first.")
                        st.info("💡 **To get started:** Go to the Audio Transcription or Transcript Upload tabs to add transcripts to your database.")
                        st.stop()
                    
                    analysis_scope = st.radio(
                        "Analysis Scope",
                        ["Current Transcript", "All Transcripts"],
                        index=1,  # Default to "All Transcripts"
                        help="Analyze the current transcript or all transcripts in the database. 'All Transcripts' is recommended for comprehensive policy recommendations.",
                        key="llm_analysis_scope",
                    )
                    
                    if analysis_scope == "Current Transcript":
                        if st.session_state.current_transcript_id:
                            transcript_id_for_analysis = st.session_state.current_transcript_id
                            st.info(f"📄 Will analyze transcript ID: {transcript_id_for_analysis}")
                        else:
                            st.warning("⚠️ No transcript currently loaded. Please load a transcript first, or select 'All Transcripts' to analyze all available data.")
                            st.info("💡 **Tip:** Select 'All Transcripts' above to generate comprehensive policy recommendations based on all your data.")
                            transcript_id_for_analysis = None
                    else:
                        transcript_id_for_analysis = None
                        st.info("📊 Will analyze all transcripts in the database to identify patterns")
                    
                    if st.button("🤖 Generate Policy Recommendations", type="primary", disabled=analysis_scope == "Current Transcript" and transcript_id_for_analysis is None):
                        if transcript_id_for_analysis or analysis_scope == "All Transcripts":
                            with st.status("🤖 Analyzing data with LLM...", state="running", expanded=True) as status:
                                status.update(label="📊 Gathering transcript data...", state="running")
                                
                                analyze_all = (analysis_scope == "All Transcripts")
                                llm_result = retention_manager.generate_policies_via_llm(
                                    transcript_id=transcript_id_for_analysis if not analyze_all else None,
                                    analyze_all=analyze_all,
                                    model=model,
                                )
                                
                                if "error" in llm_result:
                                    status.update(label="❌ Error generating recommendations", state="error")
                                    st.error(f"❌ {llm_result['error']}")
                                    if "raw_response" in llm_result:
                                        with st.expander("🔍 Raw LLM Response"):
                                            st.code(llm_result['raw_response'])
                                else:
                                    status.update(label="✅ Analysis complete", state="complete")
                                    st.success("✅ Policy recommendations generated!")

                                    # Store LLM results in session state to persist across reruns
                                    st.session_state.llm_result = llm_result

                                    # Display analysis summary
                                    if "analysis_summary" in llm_result:
                                        st.markdown("#### 📋 Analysis Summary")
                                        st.info(llm_result["analysis_summary"])

                                    # Display recommendations
                                    if "recommendations" in llm_result and llm_result["recommendations"]:
                                        st.markdown("#### 💡 Policy Recommendations")

                                        for idx, rec in enumerate(llm_result["recommendations"]):
                                            with st.expander(f"📌 Recommendation {idx + 1}: {rec.get('label', 'Unknown Label')}", expanded=True):
                                                col_rec1, col_rec2 = st.columns(2)
                                                with col_rec1:
                                                    st.metric("Retention Days", rec.get('retention_days', 'N/A'))
                                                    st.metric("Action", rec.get('action', 'N/A').title())
                                                with col_rec2:
                                                    confidence = rec.get('confidence', 'medium')
                                                    confidence_emoji = "🟢" if confidence == "high" else "🟡" if confidence == "medium" else "🔴"
                                                    st.metric("Confidence", f"{confidence_emoji} {confidence.title()}")
                                                
                                                st.markdown("**Reasoning:**")
                                                st.write(rec.get('reasoning', 'No reasoning provided'))
                                                
                                                # Show what labels this applies to
                                                apply_auto = rec.get('apply_to_auto_labels', True)
                                                apply_custom = rec.get('apply_to_custom_labels', False)
                                                st.caption(f"Applies to: {'Auto labels' if apply_auto else ''}{' and ' if apply_auto and apply_custom else ''}{'Custom labels' if apply_custom else 'Neither'}")
                                                
                                                # Action buttons
                                                col_btn1, col_btn2 = st.columns(2)
                                                with col_btn1:
                                                    if st.button(f"✅ Create Policy", key=f"create_policy_{idx}"):
                                                        try:
                                                            policy_id = db.create_retention_policy(
                                                                label_text=rec.get('label', ''),
                                                                retention_days=rec.get('retention_days', 90),
                                                                action=rec.get('action', 'archive'),
                                                                apply_to_auto_labels=apply_auto,
                                                                apply_to_custom_labels=apply_custom,
                                                                description=rec.get('reasoning', ''),
                                                                created_by="LLM",
                                                            )
                                                            # Store result in session state instead of showing success immediately
                                                            if 'llm_policy_creation_results' not in st.session_state:
                                                                st.session_state.llm_policy_creation_results = []
                                                            st.session_state.llm_policy_creation_results.append({
                                                                'policy_id': policy_id,
                                                                'label': rec.get('label', ''),
                                                                'timestamp': datetime.now().isoformat(),
                                                                'success': True
                                                            })
                                                            st.rerun()
                                                        except ValueError as e:
                                                            # Store error in session state
                                                            if 'llm_policy_creation_results' not in st.session_state:
                                                                st.session_state.llm_policy_creation_results = []
                                                            st.session_state.llm_policy_creation_results.append({
                                                                'label': rec.get('label', ''),
                                                                'error': str(e),
                                                                'timestamp': datetime.now().isoformat(),
                                                                'success': False
                                                            })
                                                            st.rerun()
                                                with col_btn2:
                                                    if st.button(f"✏️ Edit Before Creating", key=f"edit_policy_{idx}"):
                                                        st.session_state[f"editing_llm_rec_{idx}"] = True
                                                
                                                # Edit form (if editing)
                                                if st.session_state.get(f"editing_llm_rec_{idx}", False):
                                                    with st.form(f"edit_llm_rec_{idx}"):
                                                        edited_label = st.text_input("Label", value=rec.get('label', ''))
                                                        edited_days = st.number_input("Retention Days", min_value=1, value=rec.get('retention_days', 90))
                                                        edited_action = st.radio("Action", ["archive", "delete"], index=0 if rec.get('action') == 'archive' else 1)
                                                        edited_apply_auto = st.checkbox("Apply to Auto Labels", value=apply_auto)
                                                        edited_apply_custom = st.checkbox("Apply to Custom Labels", value=apply_custom)
                                                        edited_description = st.text_area("Description", value=rec.get('reasoning', ''))
                                                        
                                                        if st.form_submit_button("💾 Create Policy"):
                                                            try:
                                                                policy_id = db.create_retention_policy(
                                                                    label_text=edited_label,
                                                                    retention_days=edited_days,
                                                                    action=edited_action,
                                                                    apply_to_auto_labels=edited_apply_auto,
                                                                    apply_to_custom_labels=edited_apply_custom,
                                                                    description=edited_description,
                                                                    created_by="LLM (Edited)",
                                                                )
                                                                st.session_state[f"editing_llm_rec_{idx}"] = False
                                                                # Store result in session state instead of showing success immediately
                                                                if 'llm_policy_creation_results' not in st.session_state:
                                                                    st.session_state.llm_policy_creation_results = []
                                                                st.session_state.llm_policy_creation_results.append({
                                                                    'policy_id': policy_id,
                                                                    'label': edited_label,
                                                                    'timestamp': datetime.now().isoformat(),
                                                                    'success': True
                                                                })
                                                                st.rerun()
                                                            except ValueError as e:
                                                                # Store error in session state
                                                                if 'llm_policy_creation_results' not in st.session_state:
                                                                    st.session_state.llm_policy_creation_results = []
                                                                st.session_state.llm_policy_creation_results.append({
                                                                    'label': edited_label,
                                                                    'error': str(e),
                                                                    'timestamp': datetime.now().isoformat(),
                                                                    'success': False
                                                                })
                                                                st.rerun()
                                        
                                        # Bulk create option
                                        if len(llm_result["recommendations"]) > 1:
                                            st.divider()
                                            st.markdown("**Bulk Actions:**")
                                            if st.button("🚀 Apply All Recommendations", type="primary", key="apply_all_llm"):
                                                created_count = 0
                                                error_count = 0
                                                for rec in llm_result["recommendations"]:
                                                    try:
                                                        db.create_retention_policy(
                                                            label_text=rec.get('label', ''),
                                                            retention_days=rec.get('retention_days', 90),
                                                            action=rec.get('action', 'archive'),
                                                            apply_to_auto_labels=rec.get('apply_to_auto_labels', True),
                                                            apply_to_custom_labels=rec.get('apply_to_custom_labels', False),
                                                            description=rec.get('reasoning', ''),
                                                            created_by="LLM (Bulk)",
                                                        )
                                                        created_count += 1
                                                    except ValueError:
                                                        error_count += 1

                                                # Store results in session state to persist across rerun
                                                st.session_state.bulk_policy_results = {
                                                    'created_count': created_count,
                                                    'error_count': error_count,
                                                    'timestamp': datetime.now().isoformat()
                                                }
                                                st.rerun()
                                    
                                    # Display considerations
                                    if "considerations" in llm_result and llm_result["considerations"]:
                                        st.markdown("#### ⚠️ Considerations")
                                        for consideration in llm_result["considerations"]:
                                            st.warning(consideration)

                # Display stored LLM results (outside expander to persist across reruns)
                if 'llm_result' in st.session_state and st.session_state.llm_result:
                    llm_result = st.session_state.llm_result

                    # Clear LLM results button
                    col_clear, col_spacer = st.columns([1, 4])
                    with col_clear:
                        if st.button("🗑️ Clear LLM Results", help="Clear the stored LLM recommendations"):
                            del st.session_state.llm_result
                            if 'llm_policy_creation_results' in st.session_state:
                                del st.session_state.llm_policy_creation_results
                            st.rerun()

                    st.markdown("### 🤖 Stored LLM Policy Recommendations")

                    # Display analysis summary
                    if "analysis_summary" in llm_result:
                        st.markdown("#### 📋 Analysis Summary")
                        st.info(llm_result["analysis_summary"])

                    # Display recommendations (read-only view)
                    if "recommendations" in llm_result and llm_result["recommendations"]:
                        st.markdown("#### 💡 Policy Recommendations")
                        st.info(f"📊 {len(llm_result['recommendations'])} recommendations available. Use the expander above to create policies.")

                    # Display considerations
                    if "considerations" in llm_result and llm_result["considerations"]:
                        st.markdown("#### ⚠️ Considerations")
                        for consideration in llm_result["considerations"]:
                            st.warning(consideration)

                    st.divider()

                # Display bulk policy application results (outside expander to persist across reruns)
                if 'bulk_policy_results' in st.session_state:
                    results = st.session_state.bulk_policy_results
                    created_count = results.get('created_count', 0)
                    error_count = results.get('error_count', 0)

                    if created_count > 0:
                        st.success(f"✅ Successfully applied {created_count} retention policies to database!")
                    if error_count > 0:
                        st.warning(f"⚠️ {error_count} policies were skipped (likely duplicates or invalid data)")
                    if created_count == 0 and error_count > 0:
                        st.error("❌ No policies could be created. Check for duplicate policies or invalid recommendations.")

                    # Clear the results after displaying
                    del st.session_state.bulk_policy_results

                # Display individual LLM policy creation results (outside expander to persist across reruns)
                if 'llm_policy_creation_results' in st.session_state and st.session_state.llm_policy_creation_results:
                    st.subheader("📋 Policy Creation Results")

                    success_count = 0
                    error_count = 0

                    for result in st.session_state.llm_policy_creation_results:
                        if result.get('success', False):
                            st.success(f"✅ Policy created: '{result['label']}' (ID: {result['policy_id']})")
                            success_count += 1
                        else:
                            st.error(f"❌ Failed to create policy '{result['label']}': {result['error']}")
                            error_count += 1

                    if success_count > 0 and error_count == 0:
                        st.info(f"📊 {success_count} policy/policies successfully created!")
                    elif success_count > 0 and error_count > 0:
                        st.warning(f"📊 {success_count} policies created, {error_count} failed.")
                    elif error_count > 0:
                        st.error(f"📊 All {error_count} policy creations failed.")

                    # Clear the results after displaying
                    st.session_state.llm_policy_creation_results = []

                st.divider()

                # Create new policy form
                with st.expander("➕ Create New Policy (Manual)", expanded=False):
                    with st.form("create_policy_form"):
                        all_labels = db.get_all_labels()
                        label_input = st.selectbox(
                            "Label",
                            [""] + all_labels,
                            help="Select a label to apply the policy to",
                        )
                        retention_days = st.number_input(
                            "Retention Days",
                            min_value=1,
                            value=90,
                            help="Number of days to retain data before action",
                        )
                        action = st.radio(
                            "Action",
                            ["archive", "delete"],
                            help="Archive preserves data, Delete removes permanently",
                        )
                        col_a, col_b = st.columns(2)
                        with col_a:
                            apply_to_auto = st.checkbox("Apply to Auto Labels", value=True)
                        with col_b:
                            apply_to_custom = st.checkbox("Apply to Custom Labels", value=False)
                        description = st.text_area(
                            "Description (Optional)",
                            placeholder="Describe the purpose of this policy",
                        )

                        submitted = st.form_submit_button("Create Policy", type="primary")
                        if submitted:
                            if not label_input:
                                st.error("❌ Please select a label")
                            else:
                                try:
                                    policy_id = db.create_retention_policy(
                                        label_text=label_input,
                                        retention_days=retention_days,
                                        action=action,
                                        apply_to_auto_labels=apply_to_auto,
                                        apply_to_custom_labels=apply_to_custom,
                                        description=description if description.strip() else None,
                                        created_by="User",
                                    )
                                    st.success(f"✅ Policy created successfully! (ID: {policy_id})")
                                    st.rerun()
                                except ValueError as e:
                                    st.error(f"❌ {e}")

                # Policy list
                st.subheader("Policy List")

                # Filters
                filter_col1, filter_col2 = st.columns(2)
                with filter_col1:
                    show_enabled_only = st.checkbox("Show Active Policies Only", value=False)
                with filter_col2:
                    label_filter = st.selectbox(
                        "Filter by Label",
                        ["All"] + db.get_all_labels(),
                        key="policy_label_filter",
                    )

                # Get policies
                policies = db.get_retention_policies(
                    enabled_only=show_enabled_only,
                    label_filter=None if label_filter == "All" else label_filter,
                )

                if not policies:
                    st.info("📭 No policies found. Create a policy to get started.")
                else:
                    # Display policies in a table
                    if PANDAS_AVAILABLE:
                        policy_data = []
                        for policy in policies:
                            affected_count = db.count_transcripts_by_policy(policy['id'])
                            policy_data.append({
                                "ID": policy['id'],
                                "Label": policy['label_text'],
                                "Days": policy['retention_days'],
                                "Action": policy['action'].title(),
                                "Status": "✅ Active" if policy.get('enabled', 1) else "❌ Inactive",
                                "Auto Labels": "✅" if policy.get('apply_to_auto_labels', 1) else "❌",
                                "Custom Labels": "✅" if policy.get('apply_to_custom_labels', 0) else "❌",
                                "Affected": affected_count,
                                "Created": policy.get('created_at', '')[:10] if policy.get('created_at') else 'Unknown',
                            })

                        df = pd.DataFrame(policy_data)
                        st.dataframe(df, use_container_width=True, hide_index=True)

                        # Policy actions
                        st.subheader("Policy Actions")
                        selected_policy_id = st.selectbox(
                            "Select Policy to Manage",
                            [p['id'] for p in policies],
                            format_func=lambda x: f"Policy {x}: {next(p['label_text'] for p in policies if p['id'] == x)}",
                            key="selected_policy",
                        )

                        if selected_policy_id:
                            policy = next(p for p in policies if p['id'] == selected_policy_id)
                            action_col1, action_col2, action_col3 = st.columns(3)

                            with action_col1:
                                if st.button("✏️ Edit Policy", key=f"edit_{selected_policy_id}"):
                                    st.session_state[f"editing_policy_{selected_policy_id}"] = True

                            with action_col2:
                                enabled_status = policy.get('enabled', 1)
                                new_status = not enabled_status
                                button_text = "🔴 Disable" if enabled_status else "🟢 Enable"
                                if st.button(button_text, key=f"toggle_{selected_policy_id}"):
                                    db.update_retention_policy(selected_policy_id, enabled=new_status)
                                    st.success(f"✅ Policy {'enabled' if new_status else 'disabled'}")
                                    st.rerun()

                            with action_col3:
                                if st.button("🗑️ Delete Policy", key=f"delete_policy_{selected_policy_id}", type="secondary"):
                                    if db.delete_retention_policy(selected_policy_id):
                                        st.success("✅ Policy deleted")
                                        st.rerun()
                                    else:
                                        st.error("❌ Failed to delete policy")

                            # Edit form (if editing)
                            if st.session_state.get(f"editing_policy_{selected_policy_id}", False):
                                with st.expander("✏️ Edit Policy", expanded=True):
                                    with st.form(f"edit_policy_{selected_policy_id}"):
                                        new_days = st.number_input(
                                            "Retention Days",
                                            min_value=1,
                                            value=policy['retention_days'],
                                        )
                                        new_action = st.radio(
                                            "Action",
                                            ["archive", "delete"],
                                            index=0 if policy['action'] == 'archive' else 1,
                                        )
                                        new_apply_auto = st.checkbox(
                                            "Apply to Auto Labels",
                                            value=bool(policy.get('apply_to_auto_labels', 1)),
                                        )
                                        new_apply_custom = st.checkbox(
                                            "Apply to Custom Labels",
                                            value=bool(policy.get('apply_to_custom_labels', 0)),
                                        )
                                        new_description = st.text_area(
                                            "Description",
                                            value=policy.get('description', '') or '',
                                        )

                                        col_save, col_cancel = st.columns(2)
                                        with col_save:
                                            if st.form_submit_button("💾 Save Changes", type="primary"):
                                                db.update_retention_policy(
                                                    selected_policy_id,
                                                    retention_days=new_days,
                                                    action=new_action,
                                                    apply_to_auto_labels=new_apply_auto,
                                                    apply_to_custom_labels=new_apply_custom,
                                                    description=new_description if new_description.strip() else None,
                                                )
                                                st.session_state[f"editing_policy_{selected_policy_id}"] = False
                                                st.success("✅ Policy updated")
                                                st.rerun()
                                        with col_cancel:
                                            if st.form_submit_button("❌ Cancel"):
                                                st.session_state[f"editing_policy_{selected_policy_id}"] = False
                                                st.rerun()
                    else:
                        st.warning("⚠️ pandas is required for policy table display. Install with: pip install pandas")

            # Sub-tab 2: Policy Application
            with gov_tab2:
                st.subheader("⚙️ Policy Application")

                # Manual application
                col_apply1, col_apply2 = st.columns([2, 1])
                with col_apply1:
                    dry_run = st.checkbox("Dry Run (Preview Only)", value=True, help="Preview actions without applying")
                with col_apply2:
                    if st.button("🔄 Apply Policies Now", type="primary"):
                        with st.status("🔄 Applying retention policies...", state="running", expanded=True) as status:
                            status.update(label="📊 Checking transcripts...", state="running")
                            results = retention_manager.check_and_apply_policies(dry_run=dry_run)

                            if dry_run:
                                status.update(label="✅ Dry run complete", state="complete")
                                st.success(f"📊 Dry Run Results:")
                                st.metric("Would Archive", results['archived'])
                                st.metric("Would Delete", results['deleted'])
                                if results['errors'] > 0:
                                    st.metric("Errors", results['errors'])

                                if results['details']:
                                    st.subheader("Preview of Actions")
                                    if PANDAS_AVAILABLE:
                                        preview_df = pd.DataFrame(results['details'])
                                        st.dataframe(preview_df, use_container_width=True, hide_index=True)
                            else:
                                status.update(label="✅ Policies applied", state="complete")
                                st.success(f"✅ Policies Applied:")
                                st.metric("Archived", results['archived'])
                                st.metric("Deleted", results['deleted'])
                                if results['errors'] > 0:
                                    st.warning(f"⚠️ {results['errors']} errors occurred")

                st.divider()

                # Pending actions
                st.subheader("📋 Pending Actions")
                pending = retention_manager.get_pending_actions()

                if not pending:
                    st.info("✅ No transcripts pending retention action.")
                else:
                    st.warning(f"⚠️ {len(pending)} transcript(s) are overdue for retention action.")
                    if PANDAS_AVAILABLE:
                        pending_df = pd.DataFrame(pending)
                        st.dataframe(pending_df, use_container_width=True, hide_index=True)
                    else:
                        for item in pending:
                            with st.expander(f"📄 {Path(item['file_path']).name} - {item['days_overdue']} days overdue"):
                                st.write(f"**Label:** {item['label']}")
                                st.write(f"**Action:** {item['action'].title()}")
                                st.write(f"**Policy:** {item['policy']['label_text']} ({item['policy']['retention_days']} days)")
                                st.write(f"**Retention Date:** {item['retention_date']}")

            # Sub-tab 3: Archive Management
            with gov_tab3:
                st.subheader("📦 Archive Management")

                archived = db.get_archived_transcripts(limit=100)

                if not archived:
                    st.info("📭 No archived transcripts.")
                else:
                    st.success(f"📊 Found {len(archived)} archived transcript(s)")

                    # Archive statistics
                    col_arch1, col_arch2 = st.columns(2)
                    with col_arch1:
                        st.metric("Total Archived", len(archived))
                    with col_arch2:
                        # Count by policy
                        policy_counts = {}
                        for arch in archived:
                            policy_id = arch.get('retention_policy_id')
                            if policy_id:
                                policy_counts[policy_id] = policy_counts.get(policy_id, 0) + 1
                        st.metric("Policies Applied", len(policy_counts))

                    # Archived transcripts list
                    for arch in archived:
                        transcript_id = arch['id']
                        file_name = Path(arch.get('file_path', 'Unknown')).name
                        archived_date = arch.get('archived_at', '')[:10] if arch.get('archived_at') else 'Unknown'

                        with st.expander(f"📦 {file_name} - Archived {archived_date}"):
                            col_restore, col_delete = st.columns(2)
                            with col_restore:
                                if st.button("↩️ Restore", key=f"restore_{transcript_id}"):
                                    if db.restore_transcript(transcript_id):
                                        st.success("✅ Transcript restored")
                                        st.rerun()
                                    else:
                                        st.error("❌ Failed to restore transcript")
                            with col_delete:
                                if st.button("🗑️ Delete Permanently", key=f"perm_delete_{transcript_id}", type="secondary"):
                                    st.warning("⚠️ This will permanently delete the transcript and all related data!")
                                    confirm = st.checkbox(f"Confirm permanent deletion of {file_name}", key=f"confirm_delete_{transcript_id}")
                                    if confirm:
                                        if db.delete_transcript(transcript_id):
                                            st.success("✅ Transcript permanently deleted")
                                            st.rerun()
                                        else:
                                            st.error("❌ Failed to delete transcript")

            # Sub-tab 4: Label Policy Mapping
            with gov_tab4:
                st.subheader("🏷️ Label Policy Mapping")

                all_labels = db.get_all_labels()
                all_policies = db.get_retention_policies()

                if not all_labels:
                    st.info("📭 No labels found. Upload or transcribe transcripts to generate labels.")
                else:
                    # Label coverage
                    labels_with_policies = set()
                    for policy in all_policies:
                        labels_with_policies.add(policy['label_text'])

                    coverage = len(labels_with_policies) / len(all_labels) * 100 if all_labels else 0

                    col_map1, col_map2 = st.columns(2)
                    with col_map1:
                        st.metric("Total Labels", len(all_labels))
                    with col_map2:
                        st.metric("Labels with Policies", f"{len(labels_with_policies)} ({coverage:.1f}%)")

                    # Label-Policy matrix
                    st.subheader("Label Coverage")
                    if PANDAS_AVAILABLE:
                        label_data = []
                        for label in all_labels:
                            policies_for_label = [p for p in all_policies if p['label_text'] == label]
                            label_data.append({
                                "Label": label,
                                "Has Policy": "✅" if policies_for_label else "❌",
                                "Policy Count": len(policies_for_label),
                                "Actions": ", ".join([p['action'].title() for p in policies_for_label]) if policies_for_label else "None",
                            })

                        label_df = pd.DataFrame(label_data)
                        st.dataframe(label_df, use_container_width=True, hide_index=True)
                    else:
                        for label in all_labels:
                            policies_for_label = [p for p in all_policies if p['label_text'] == label]
                            status = "✅" if policies_for_label else "❌"
                            st.write(f"{status} **{label}**: {len(policies_for_label)} policy/policies")

        except Exception as e:
            st.error(f"❌ Error in Governance tab: {e}")
            import traceback
            with st.expander("🔍 Error Details"):
                st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
