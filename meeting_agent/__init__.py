"""
Meeting Agent Package

A Python package for transcribing audio, processing meeting transcripts,
and generating structured meeting notes using LLM technology.
"""

__version__ = "1.0.0"

# Import commonly used functions for convenience
from meeting_agent.helper import (
    DEFAULT_CLOUD_MODEL,
    call_ollama_cloud,
    get_transcripts_dir,
    save_meeting_notes,
)
from meeting_agent.llm import MeetingAgent, create_meeting_agent

__all__ = [
    "DEFAULT_CLOUD_MODEL",
    "MeetingAgent",
    "call_ollama_cloud",
    "create_meeting_agent",
    "get_transcripts_dir",
    "save_meeting_notes",
]
