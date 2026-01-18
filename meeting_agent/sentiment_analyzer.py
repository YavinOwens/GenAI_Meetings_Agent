"""
Sentiment Analysis Module for Meeting Transcripts

Provides functionality to analyze sentiment of meeting transcripts,
including overall sentiment and per-attendee sentiment analysis.
"""

import json
import re
from typing import Any

from meeting_agent.helper import (
    call_ollama_cloud,
    create_llm_messages,
    extract_response_text,
    load_prompt,
)

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False

try:
    import spacy

    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False


def check_dependencies() -> tuple[bool, list[str]]:
    """Check if sentiment analysis dependencies are available.

    Returns:
        Tuple[bool, List[str]]: (all_available, list of missing dependencies)
    """
    missing = []
    if not VADER_AVAILABLE:
        missing.append("vaderSentiment")
    if not SPACY_AVAILABLE:
        missing.append("spacy")
    return len(missing) == 0, missing


def get_sentiment_label(compound_score: float) -> str:
    """Convert compound sentiment score to label.

    Args:
        compound_score: VADER compound score (-1 to 1).

    Returns:
        str: Sentiment label ('Positive', 'Neutral', or 'Negative').
    """
    if compound_score >= 0.05:
        return "Positive"
    elif compound_score <= -0.05:
        return "Negative"
    else:
        return "Neutral"


def get_sentiment_emoji(compound_score: float) -> str:
    """Get emoji for sentiment score.

    Args:
        compound_score: VADER compound score (-1 to 1).

    Returns:
        str: Emoji representing sentiment.
    """
    if compound_score >= 0.05:
        return "😊"
    elif compound_score <= -0.05:
        return "😞"
    else:
        return "😐"


def analyze_overall_sentiment(transcript: str) -> dict[str, Any]:
    """Analyze overall sentiment of the transcript.

    Args:
        transcript: The transcript text to analyze.

    Returns:
        dict: Dictionary containing sentiment scores and label.
            Keys: 'compound', 'pos', 'neu', 'neg', 'label', 'emoji'

    Raises:
        ImportError: If vaderSentiment is not installed.
    """
    if not VADER_AVAILABLE:
        raise ImportError(
            "vaderSentiment is not installed. "
            "Install it with: pip install vaderSentiment"
        )

    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(transcript)

    compound = scores["compound"]
    label = get_sentiment_label(compound)
    emoji = get_sentiment_emoji(compound)

    return {
        "compound": compound,
        "pos": scores["pos"],
        "neu": scores["neu"],
        "neg": scores["neg"],
        "label": label,
        "emoji": emoji,
    }


def identify_attendees_via_speaker_labels(transcript: str) -> list[str]:
    """Identify attendees by looking for speaker labels in transcript.

    Looks for patterns like "John:", "Speaker 1:", "Alice:", "[00:00] Sarah:", etc.

    Args:
        transcript: The transcript text.

    Returns:
        list[str]: List of unique attendee names found.
    """
    # Metadata fields that should be excluded from attendee identification
    metadata_blacklist = {
        "date",
        "duration",
        "time",
        "location",
        "format",
        "prepared",
        "distribution",
        "status",
        "next",
        "materials",
        "meeting",
        "transcript",
        "session",
    }

    # Pattern 1: Match timestamped speaker labels: "[00:00] Name:"
    timestamped_pattern = r"\[.*?\]\s*([A-Z][a-zA-Z]+):"
    # Pattern 2: Match non-timestamped speaker labels: "Name:" or "Speaker X:"
    # Matches: "John:", "Alice Smith:", "Speaker 1:", "Participant 2:"
    non_timestamped_pattern = r"^([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?|Speaker\s+\d+|Participant\s+\d+):"

    attendees = set()

    for line in transcript.split("\n"):
        line = line.strip()
        if not line:
            continue

        name = None

        # Try timestamped pattern first (more specific)
        match = re.search(timestamped_pattern, line)
        if match:
            name = match.group(1).strip()
        else:
            # Fall back to non-timestamped pattern
            match = re.match(non_timestamped_pattern, line)
            if match:
                name = match.group(1).strip()

        if name:
            # Filter out metadata fields and invalid names
            name_lower = name.lower()
            
            # Skip if in metadata blacklist
            if name_lower in metadata_blacklist:
                continue

            # Skip generic labels if we have specific names
            if name_lower in ["speaker", "participant"]:
                continue

            # Ensure name is reasonable length (2-30 characters)
            if not (2 <= len(name) <= 30):
                continue

            # Ensure name starts with capital letter (proper noun)
            if not name[0].isupper():
                continue

            # Skip if name contains common metadata keywords
            if any(keyword in name_lower for keyword in ["date", "time", "duration", "location"]):
                continue

            attendees.add(name)

    return sorted(list(attendees))


def identify_attendees_via_ner(transcript: str) -> list[str]:
    """Identify attendees using Named Entity Recognition (spaCy).

    Args:
        transcript: The transcript text.

    Returns:
        list[str]: List of unique person names found.

    Raises:
        ImportError: If spacy is not installed.
        OSError: If spaCy model is not downloaded.
    """
    if not SPACY_AVAILABLE:
        raise ImportError(
            "spacy is not installed. Install it with: pip install spacy\n"
            "Then download the English model: python -m spacy download en_core_web_sm"
        )

    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError as e:
        raise OSError(
            "spaCy English model not found. "
            "Download it with: python -m spacy download en_core_web_sm"
        ) from e

    doc = nlp(transcript)
    person_names = set()

    for ent in doc.ents:
        if ent.label_ == "PERSON":
            # Clean up the name (remove extra whitespace)
            name = " ".join(ent.text.split())
            if len(name) > 1:  # Filter out single characters
                person_names.add(name)

    return sorted(list(person_names))


def identify_attendees(transcript: str) -> list[str]:
    """Identify attendees using speaker labels first, then NER as fallback.

    Args:
        transcript: The transcript text.

    Returns:
        list[str]: List of unique attendee names found.
    """
    # Try speaker labels first
    attendees = identify_attendees_via_speaker_labels(transcript)

    # If no attendees found via speaker labels, try NER
    if not attendees and SPACY_AVAILABLE:
        try:
            attendees = identify_attendees_via_ner(transcript)
        except (ImportError, OSError):
            # If NER fails, return empty list
            pass

    return attendees


def extract_attendee_segments(transcript: str, attendee: str) -> list[str]:
    """Extract text segments mentioning a specific attendee.

    Args:
        transcript: The transcript text.
        attendee: Name of the attendee to find segments for.

    Returns:
        list[str]: List of text segments mentioning the attendee.
    """
    segments = []

    # Split by lines and look for lines containing the attendee name
    lines = transcript.split("\n")
    current_segment = []

    for line in lines:
        line_lower = line.lower()
        attendee_lower = attendee.lower()

        # Check if line contains attendee name or is a speaker label
        if attendee_lower in line_lower or line.strip().startswith(f"{attendee}:"):
            # If we have accumulated text, save it
            if current_segment:
                segments.append(" ".join(current_segment))
                current_segment = []

            # Add this line and following lines until next speaker or blank line
            current_segment.append(line)
        elif current_segment:
            # Continue accumulating if we're in a segment
            if line.strip():  # Non-empty line
                current_segment.append(line)
            else:
                # Blank line ends the segment
                if current_segment:
                    segments.append(" ".join(current_segment))
                    current_segment = []

    # Add any remaining segment
    if current_segment:
        segments.append(" ".join(current_segment))

    # If no segments found via speaker labels, look for mentions in text
    if not segments:
        # Find sentences containing the attendee name
        sentences = re.split(r"[.!?]+", transcript)
        for sentence in sentences:
            if attendee.lower() in sentence.lower():
                segments.append(sentence.strip())

    return segments


def analyze_attendee_sentiment(transcript: str, attendee: str) -> dict[str, Any]:
    """Analyze sentiment for a specific attendee.

    Args:
        transcript: The transcript text.
        attendee: Name of the attendee to analyze.

    Returns:
        dict: Dictionary containing sentiment scores, label, and mention count.
            Keys: 'compound', 'pos', 'neu', 'neg', 'label', 'emoji', 'mentions'

    Raises:
        ImportError: If vaderSentiment is not installed.
    """
    if not VADER_AVAILABLE:
        raise ImportError(
            "vaderSentiment is not installed. "
            "Install it with: pip install vaderSentiment"
        )

    # Extract segments mentioning this attendee
    segments = extract_attendee_segments(transcript, attendee)

    if not segments:
        # If no segments found, return neutral sentiment
        return {
            "compound": 0.0,
            "pos": 0.0,
            "neu": 1.0,
            "neg": 0.0,
            "label": "Neutral",
            "emoji": "😐",
            "mentions": 0,
        }

    # Combine all segments and analyze
    combined_text = " ".join(segments)
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(combined_text)

    compound = scores["compound"]
    label = get_sentiment_label(compound)
    emoji = get_sentiment_emoji(compound)

    return {
        "compound": compound,
        "pos": scores["pos"],
        "neu": scores["neu"],
        "neg": scores["neg"],
        "label": label,
        "emoji": emoji,
        "mentions": len(segments),
    }


def extract_topics_via_llm(transcript: str, model: str) -> list[str]:
    """Extract topics from transcript using LLM.

    Args:
        transcript: The transcript text.
        model: LLM model name to use (must include '-cloud' suffix).

    Returns:
        list[str]: List of extracted topics.

    Raises:
        Exception: If topic extraction fails.
    """
    try:
        # Load prompts
        system_prompt = load_prompt("topic_extraction_system_prompt.txt")
        user_prompt_template = load_prompt("topic_extraction_user_prompt_template.txt")
        user_prompt = user_prompt_template.format(transcript=transcript)

        # Create messages
        messages = create_llm_messages(system_prompt, user_prompt)

        # Call LLM
        response = call_ollama_cloud(model=model, messages=messages, stream=False)

        # Extract response text
        response_text = extract_response_text(response)

        # Parse JSON array from response
        # Try to find JSON array in the response (might have extra text)
        json_match = re.search(r'\[.*?\]', response_text, re.DOTALL)
        if json_match:
            topics_json = json_match.group(0)
            topics = json.loads(topics_json)
        else:
            # Fallback: try parsing the whole response
            topics = json.loads(response_text.strip())

        # Validate and clean topics
        if not isinstance(topics, list):
            raise ValueError(f"Expected list of topics, got {type(topics)}")

        # Filter out empty or invalid topics
        valid_topics = [str(topic).strip() for topic in topics if topic and str(topic).strip()]

        return valid_topics

    except Exception as e:
        # Return empty list on error (don't block sentiment analysis)
        return []


def extract_topic_segments(transcript: str, topic: str) -> list[str]:
    """Extract text segments relevant to a specific topic.

    Args:
        transcript: The transcript text.
        topic: The topic to find segments for.

    Returns:
        list[str]: List of text segments mentioning the topic.
    """
    segments = []
    topic_lower = topic.lower()
    topic_words = topic_lower.split()

    # Split transcript into lines
    lines = transcript.split("\n")
    current_segment = []

    for line in lines:
        line_lower = line.lower()

        # Check if line contains topic or any of its keywords
        if any(word in line_lower for word in topic_words if len(word) > 2):
            # If we have accumulated text, save it
            if current_segment:
                segments.append(" ".join(current_segment))
                current_segment = []

            # Add this line and following lines until blank line or next speaker
            current_segment.append(line)
        elif current_segment:
            # Continue accumulating if we're in a segment
            if line.strip():  # Non-empty line
                current_segment.append(line)
            else:
                # Blank line ends the segment
                if current_segment:
                    segments.append(" ".join(current_segment))
                    current_segment = []

    # Add any remaining segment
    if current_segment:
        segments.append(" ".join(current_segment))

    # If no segments found, try sentence-level matching
    if not segments:
        sentences = re.split(r"[.!?]+", transcript)
        for sentence in sentences:
            if any(word in sentence.lower() for word in topic_words if len(word) > 2):
                segments.append(sentence.strip())

    return segments


def analyze_topic_sentiment(transcript: str, topic: str) -> dict[str, Any]:
    """Analyze sentiment for a specific topic.

    Args:
        transcript: The transcript text.
        topic: The topic to analyze.

    Returns:
        dict: Dictionary containing sentiment scores, label, and segment count.
            Keys: 'compound', 'pos', 'neu', 'neg', 'label', 'emoji', 'segments'

    Raises:
        ImportError: If vaderSentiment is not installed.
    """
    if not VADER_AVAILABLE:
        raise ImportError(
            "vaderSentiment is not installed. "
            "Install it with: pip install vaderSentiment"
        )

    # Extract segments mentioning this topic
    segments = extract_topic_segments(transcript, topic)

    if not segments:
        # If no segments found, return neutral sentiment
        return {
            "compound": 0.0,
            "pos": 0.0,
            "neu": 1.0,
            "neg": 0.0,
            "label": "Neutral",
            "emoji": "😐",
            "segments": 0,
        }

    # Combine all segments and analyze
    combined_text = " ".join(segments)
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(combined_text)

    compound = scores["compound"]
    label = get_sentiment_label(compound)
    emoji = get_sentiment_emoji(compound)

    return {
        "compound": compound,
        "pos": scores["pos"],
        "neu": scores["neu"],
        "neg": scores["neg"],
        "label": label,
        "emoji": emoji,
        "segments": len(segments),
    }
