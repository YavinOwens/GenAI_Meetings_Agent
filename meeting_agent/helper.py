"""
Helper utilities for the Meeting Agent project.

Contains color formatting functions, file operations, and other utility functions.
"""

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import tiktoken

# Optional imports for audio transcription
try:
    import torch
    import whisper

    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    torch = None

# Ollama cloud integration
try:
    from ollama import Client

    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    Client = None


# ANSI color codes for terminal output
class Colors:
    """ANSI color codes for terminal output"""

    # Reset
    RESET = "\033[0m"

    # Text colors
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    # Bright colors
    BRIGHT_BLACK = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"

    # Text styles
    BOLD = "\033[1m"
    DIM = "\033[2m"
    ITALIC = "\033[3m"
    UNDERLINE = "\033[4m"


def color_text(text: str, color_code: str) -> str:
    """Apply a color code to text.

    Args:
        text: Text to colorize.
        color_code: ANSI color code from Colors class.

    Returns:
        str: Colorized text with reset code.
    """
    return f"{color_code}{text}{Colors.RESET}"


def color_answer(text: str) -> str:
    """Colorize LLM answer text (cyan).

    Args:
        text: Text to colorize.

    Returns:
        str: Colorized text in cyan.
    """
    return color_text(text, Colors.CYAN)


def color_answer_label(text: str) -> str:
    """Colorize answer label (bright cyan).

    Args:
        text: Text to colorize.

    Returns:
        str: Colorized text in bright cyan.
    """
    return color_text(text, Colors.BRIGHT_CYAN)


def color_header(text: str) -> str:
    """Colorize header text (bold).

    Args:
        text: Text to colorize.

    Returns:
        str: Colorized text in bold.
    """
    return color_text(text, Colors.BOLD)


def color_header_border(text: str) -> str:
    """Colorize header border (bright blue).

    Args:
        text: Text to colorize.

    Returns:
        str: Colorized text in bright blue.
    """
    return color_text(text, Colors.BRIGHT_BLUE)


def color_success(text: str) -> str:
    """Colorize success messages (bright green).

    Args:
        text: Text to colorize.

    Returns:
        str: Colorized text in bright green.
    """
    return color_text(text, Colors.BRIGHT_GREEN)


def color_error(text: str) -> str:
    """Colorize error messages (bright red).

    Args:
        text: Text to colorize.

    Returns:
        str: Colorized text in bright red.
    """
    return color_text(text, Colors.BRIGHT_RED)


def color_info(text: str) -> str:
    """Colorize info messages (cyan).

    Args:
        text: Text to colorize.

    Returns:
        str: Colorized text in cyan.
    """
    return color_text(text, Colors.CYAN)


def color_warning(text: str) -> str:
    """Colorize warning messages (yellow).

    Args:
        text: Text to colorize.

    Returns:
        str: Colorized text in yellow.
    """
    return color_text(text, Colors.YELLOW)


def color_command(text: str) -> str:
    """Colorize command examples (cyan).

    Args:
        text: Text to colorize.

    Returns:
        str: Colorized text in cyan.
    """
    return color_text(text, Colors.CYAN)


def color_label(text: str) -> str:
    """Colorize labels (yellow).

    Args:
        text: Text to colorize.

    Returns:
        str: Colorized text in yellow.
    """
    return color_text(text, Colors.YELLOW)


def color_dim(text: str) -> str:
    """Colorize dim/subtle text.

    Args:
        text: Text to colorize.

    Returns:
        str: Colorized text in dim style.
    """
    return color_text(text, Colors.DIM)


# File and prompt utilities


def load_prompt(prompt_filename: str) -> str:
    """Load a prompt from the prompt_gallery directory.

    Args:
        prompt_filename: Name of the prompt file (e.g., 'meeting_notes_system_prompt.txt').

    Returns:
        str: The prompt content.

    Raises:
        FileNotFoundError: If the prompt file doesn't exist.
    """
    prompt_gallery = Path(__file__).parent / "prompt_gallery"
    prompt_path = prompt_gallery / prompt_filename

    if not prompt_path.exists():
        raise FileNotFoundError(
            f"Prompt file not found: {prompt_path}\n"
            f"Please ensure the prompt file exists in the prompt_gallery directory."
        )

    return prompt_path.read_text(encoding="utf-8").strip()


def find_most_recent_transcript(transcripts_dir: Path, exclude_notes: bool = True) -> Path | None:
    """Find the most recent transcript file in the transcripts directory.

    Args:
        transcripts_dir: Path to the Transcripts directory.
        exclude_notes: If True, exclude files with '_notes' in the name.
            Defaults to True.

    Returns:
        Optional[Path]: Path to the most recent transcript file, or None if
            no files found.
    """
    if not transcripts_dir.exists():
        return None

    # Supported transcript file extensions
    transcript_extensions = [".txt", ".md", ".transcript", ".srt", ".vtt"]

    # Find all transcript files
    transcript_files = []
    for ext in transcript_extensions:
        transcript_files.extend(transcripts_dir.glob(f"*{ext}"))
        transcript_files.extend(transcripts_dir.glob(f"*{ext.upper()}"))

    # Filter out notes files if requested
    if exclude_notes:
        transcript_files = [f for f in transcript_files if "_notes" not in f.name]

    if not transcript_files:
        return None

    # #region agent log
    write_instrumentation_log(
        location="helper.py:248",
        message="Before sorting files",
        data={"file_count": len(transcript_files)},
        hypothesis_id="B",
    )
    # #endregion
    # Sort by modification time, most recent first
    try:
        transcript_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    except (FileNotFoundError, OSError) as e:
        # #region agent log
        write_instrumentation_log(
            location="helper.py:248",
            message="File stat error",
            data={"error_type": type(e).__name__, "error_msg": str(e)},
            hypothesis_id="B",
        )
        # #endregion
        raise
    # #region agent log
    write_instrumentation_log(
        location="helper.py:248",
        message="After sorting",
        data={
            "selected_file": sanitize_path_for_logging(transcript_files[0])
            if transcript_files
            else None
        },
        hypothesis_id="B",
    )
    # #endregion

    return transcript_files[0]


def read_transcript(transcript_path: Path) -> str:
    """Read the transcript file content.

    Args:
        transcript_path: Path to the transcript file.

    Returns:
        str: The transcript content.

    Raises:
        IOError: If the file cannot be read.
    """
    try:
        return transcript_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # Try with different encoding if UTF-8 fails
        try:
            return transcript_path.read_text(encoding="latin-1")
        except Exception as e:
            raise OSError(f"Error reading transcript file: {e}") from e
    except Exception as e:
        raise OSError(f"Error reading transcript file: {e}") from e


def save_meeting_notes(
    notes: str,
    transcript_path: Path,
    output_dir: Path | None = None,
    format: str | None = None,
    model_info: dict[str, Any] | None = None,
) -> Path:
    """Save the generated meeting notes to a file.

    This function uses DocuHelper.save_document() internally to support
    multiple formats (txt, docx, rtf, md) with APA citations.

    Args:
        notes: The meeting notes text (markdown formatted).
        transcript_path: Path to the original transcript (for naming).
        output_dir: Directory to save notes (defaults to same as transcript).
        format: Document format ('txt', 'docx', 'rtf', 'md'). Defaults to 'md' for
            backward compatibility.
        model_info: Dictionary with model information for APA citation.
            Keys: 'model_name', 'version', 'url'. Defaults to Ollama values.

    Returns:
        Path: Path to the saved notes file.
    """
    from meeting_agent.docu_helper import DocumentFormat, save_document

    if output_dir is None:
        output_dir = transcript_path.parent

    # #region agent log
    write_instrumentation_log(
        location="helper.py:297",
        message="Before save",
        data={
            "output_dir": sanitize_path_for_logging(output_dir),
            "output_dir_exists": output_dir.exists(),
        },
        hypothesis_id="C",
    )
    # #endregion

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create notes filename based on transcript name
    transcript_stem = transcript_path.stem

    # Determine format
    doc_format = None
    if format:
        format_lower = format.lower().lstrip(".")
        if format_lower == "txt":
            doc_format = DocumentFormat.TXT
        elif format_lower == "docx":
            doc_format = DocumentFormat.DOCX
        elif format_lower == "rtf":
            doc_format = DocumentFormat.RTF
        elif format_lower == "md":
            doc_format = DocumentFormat.MD
        else:
            # Default to md if invalid format specified
            doc_format = DocumentFormat.MD
    else:
        # Default to md for backward compatibility
        doc_format = DocumentFormat.MD

    # Create base path (extension will be added by save_document)
    notes_path = output_dir / f"{transcript_stem}_notes"

    # Use DocuHelper to save document
    try:
        saved_path = save_document(
            content=notes,
            output_path=notes_path,
            format=doc_format,
            interactive=False,  # Non-interactive for backward compatibility
            model_info=model_info,
        )
    except Exception as e:
        # #region agent log
        write_instrumentation_log(
            location="helper.py:297",
            message="Write error",
            data={"error_type": type(e).__name__, "error_msg": str(e)},
            hypothesis_id="C",
        )
        # #endregion
        raise
    # #region agent log
    write_instrumentation_log(
        location="helper.py:297",
        message="After save",
        data={"notes_path": sanitize_path_for_logging(saved_path)},
        hypothesis_id="C",
    )
    # #endregion

    return saved_path


# Response parsing utilities


def parse_ollama_response(response: Any) -> dict[str, Any]:
    """Parse an Ollama API response to a dictionary.

    Handles both Pydantic models and plain dictionaries.

    Args:
        response: Response from Ollama API (can be Pydantic model or dict).

    Returns:
        Dict[str, Any]: Parsed response as a dictionary.
    """
    if response is None:
        return {}

    if hasattr(response, "model_dump"):
        return response.model_dump()
    if hasattr(response, "dict"):
        return response.dict()
    if isinstance(response, dict):
        return response
    # Try to convert to dict, but handle non-dict-able types
    try:
        return dict(response)
    except (TypeError, ValueError):
        # If conversion fails, wrap in a dict with a 'response' key
        return {"response": response}


def extract_response_text(response_dict: dict[str, Any]) -> str:
    """Extract text content from an Ollama response dictionary.

    Args:
        response_dict: Response dictionary from parse_ollama_response().

    Returns:
        str: The response text content, or empty string if not found.
    """
    if not response_dict:
        return ""

    # Handle dict response
    if isinstance(response_dict, dict):
        if "message" in response_dict:
            message = response_dict["message"]
            if isinstance(message, dict):
                return message.get("content", "")
            if hasattr(message, "content"):
                return message.content
        elif "response" in response_dict:
            return response_dict.get("response", "")

    # Handle Pydantic model response
    if hasattr(response_dict, "message"):
        msg = response_dict.message
        if hasattr(msg, "content"):
            return msg.content
    elif hasattr(response_dict, "response"):
        return response_dict.response

    return ""


# Path utilities


def get_project_root() -> Path:
    """Get the project root directory.

    Returns:
        Path: Path to the project root directory (parent of meeting_agent package).
    """
    # Return parent of package directory (project root)
    return Path(__file__).parent.parent


def get_transcripts_dir() -> Path:
    """Get the transcripts directory path.

    Returns:
        Path: Path to the Soundrecording_raw/Transcripts directory.
    """
    project_root = get_project_root()
    return project_root / "Soundrecording_raw" / "Transcripts"


def is_instrumentation_enabled() -> bool:
    """Check if debug instrumentation is enabled.

    Instrumentation is enabled if the MEETING_AGENT_DEBUG environment variable
    is set to '1', 'true', 'yes', or 'on' (case-insensitive).

    Returns:
        bool: True if instrumentation should be enabled, False otherwise.
    """
    debug_flag = os.getenv("MEETING_AGENT_DEBUG", "").lower()
    return debug_flag in ("1", "true", "yes", "on")


def write_instrumentation_log(
    location: str,
    message: str,
    data: dict[str, Any],
    hypothesis_id: str = "unknown",
    session_id: str = "debug-session",
    run_id: str = "run1",
) -> None:
    """Write an instrumentation log entry if instrumentation is enabled.

    Args:
        location: Code location (e.g., "helper.py:248").
        message: Log message.
        data: Dictionary of data to log.
        hypothesis_id: Hypothesis identifier (default: "unknown").
        session_id: Session identifier (default: "debug-session").
        run_id: Run identifier (default: "run1").
    """
    if not is_instrumentation_enabled():
        return

    try:
        import json
        import time

        log_path = get_project_root() / "debug.log"
        log_entry = {
            "sessionId": session_id,
            "runId": run_id,
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception:
        # Silently fail if logging fails (don't break production code)
        pass


def sanitize_path_for_logging(path: Path) -> str:
    """Sanitize a file path for logging to remove user-specific information.

    Converts absolute paths to relative paths from project root, or uses
    a placeholder for paths outside the project.

    Args:
        path: Path object to sanitize.

    Returns:
        str: Sanitized path string safe for logging.
    """
    try:
        project_root = get_project_root()
        path_obj = Path(path) if not isinstance(path, Path) else path

        # Try to make path relative to project root
        try:
            relative_path = path_obj.relative_to(project_root)
            return str(relative_path)
        except ValueError:
            # Path is outside project root, use filename only
            return f"[external]/{path_obj.name}"
    except Exception:
        # Fallback: just return filename
        try:
            return Path(path).name if path else "[unknown]"
        except Exception:
            return "[unknown]"


# Token counting utilities

# Cache for encoding objects to avoid repeated initialization
_encoding_cache: dict[str, tiktoken.Encoding] = {}


def get_encoding(encoding_name: str = "cl100k_base") -> tiktoken.Encoding:
    """Get a tiktoken encoding object, with caching.

    Args:
        encoding_name: Name of the encoding to use. Defaults to "cl100k_base".

    Returns:
        tiktoken.Encoding: The encoding object.
    """
    if encoding_name not in _encoding_cache:
        _encoding_cache[encoding_name] = tiktoken.get_encoding(encoding_name)
    return _encoding_cache[encoding_name]


def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """Count tokens in a text string.

    Args:
        text: Text to count tokens for.
        encoding_name: Name of the encoding to use. Defaults to "cl100k_base".

    Returns:
        int: Number of tokens.

    Raises:
        TypeError: If text is None or not a string.
    """
    # #region agent log
    write_instrumentation_log(
        location="helper.py:408",
        message="Before count_tokens",
        data={
            "text_type": type(text).__name__,
            "text_is_none": text is None,
            "text_length": len(text) if text else 0,
        },
        hypothesis_id="D",
    )
    # #endregion

    if text is None:
        raise TypeError("text cannot be None")
    if not isinstance(text, str):
        raise TypeError(f"text must be a string, got {type(text).__name__}")

    encoding = get_encoding(encoding_name)
    try:
        result = len(encoding.encode(text))
    except (TypeError, AttributeError) as e:
        # #region agent log
        write_instrumentation_log(
            location="helper.py:408",
            message="Encode error",
            data={"error_type": type(e).__name__, "error_msg": str(e)},
            hypothesis_id="D",
        )
        # #endregion
        raise
    # #region agent log
    write_instrumentation_log(
        location="helper.py:408",
        message="After count_tokens",
        data={"token_count": result},
        hypothesis_id="D",
    )
    # #endregion
    return result


# Message creation utilities


def create_llm_messages(system_prompt: str, user_prompt: str) -> list[dict[str, str]]:
    """Create a list of message dictionaries for LLM API calls.

    Args:
        system_prompt: The system/role prompt.
        user_prompt: The user prompt.

    Returns:
        List[Dict[str, str]]: List of message dictionaries with 'role' and 'content'.
    """
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]


def format_token_usage(input_tokens: int, output_tokens: int) -> dict[str, int]:
    """Format token usage information into a dictionary.

    Args:
        input_tokens: Number of input tokens.
        output_tokens: Number of output tokens.

    Returns:
        Dict[str, int]: Dictionary with input_tokens, output_tokens, and total_tokens.
    """
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }


# Error handling utilities


def handle_prompt_load_error(
    component_name: str, original_error: FileNotFoundError
) -> FileNotFoundError:
    """Create a standardized error message for prompt loading failures.

    Args:
        component_name: Name of the component trying to load prompts (e.g., 'MeetingAgent').
        original_error: The original FileNotFoundError.

    Returns:
        FileNotFoundError: A new error with a standardized message.
    """
    msg = (
        f"Failed to load prompts for {component_name}: {original_error}\n"
        f"Please ensure prompt files exist in the prompt_gallery directory."
    )
    new_error = FileNotFoundError(msg)
    new_error.__cause__ = original_error
    return new_error


def print_traceback() -> None:
    """Print a traceback for the current exception.

    This is a convenience wrapper around traceback.print_exc().
    """
    import traceback

    traceback.print_exc()


def run_main_with_error_handling(main_func, exit_on_keyboard_interrupt: bool = True) -> None:
    """Run a main function with standard error handling.

    Handles KeyboardInterrupt and general exceptions with appropriate exit codes.

    Args:
        main_func: The main function to run (should return an exit code).
        exit_on_keyboard_interrupt: If True, exit on KeyboardInterrupt. Defaults to True.
    """
    import sys

    try:
        exit_code = main_func()
        if exit_code is not None:
            sys.exit(exit_code)
    except KeyboardInterrupt:
        if exit_on_keyboard_interrupt:
            print("\n\nInterrupted by user")
            sys.exit(0)
        else:
            raise
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        print_traceback()
        sys.exit(1)


# Audio transcription utilities


def check_ffmpeg() -> bool:
    """Check if ffmpeg is installed and available.

    Returns:
        bool: True if ffmpeg is available, False otherwise.
    """
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        try:
            result = subprocess.run(
                ["ffmpeg", "-version"], capture_output=True, text=True, timeout=5
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    return False


def install_ffmpeg_instructions() -> str:
    """Get instructions for installing ffmpeg via Homebrew.

    Returns:
        str: Installation instructions.
    """
    return (
        "\nTo install ffmpeg, run:\n"
        "  brew install ffmpeg\n"
        "\nIf Homebrew is not installed, install it first:\n"
        '  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
    )


def transcribe_audio(
    audio_path: Path, model_size: str = "base", output_dir: Path | None = None
) -> Path:
    """Transcribe an audio file using Whisper.

    Args:
        audio_path: Path to the audio file.
        model_size: Whisper model size ('tiny', 'base', 'small', 'medium', 'large').
            Defaults to 'base'.
        output_dir: Directory to save transcript. Defaults to transcripts directory.

    Returns:
        Path: Path to the saved transcript file.

    Raises:
        FileNotFoundError: If audio file doesn't exist.
        RuntimeError: If ffmpeg is not installed or Whisper is not available.
        Exception: For transcription errors.
    """
    if not WHISPER_AVAILABLE:
        raise RuntimeError(
            "Whisper is not installed. Please install it with:\n  pip install openai-whisper"
        )

    if not check_ffmpeg():
        raise RuntimeError(
            f"ffmpeg is not installed or not available in PATH.{install_ffmpeg_instructions()}"
        )

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Set output directory
    if output_dir is None:
        output_dir = get_transcripts_dir()

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine device (use Metal GPU if available, otherwise CPU)
    device = "cpu"
    if torch and torch.backends.mps.is_available():
        device = "mps"
        print(f"{color_info('Using Metal GPU acceleration')} (MPS)")
    else:
        print(f"{color_info('Using CPU')} (Metal GPU not available)")

    # Load Whisper model
    print(f"{color_info('Loading Whisper model')} ({model_size})...")
    try:
        model = whisper.load_model(model_size, device=device)
        print(f"{color_success('SUCCESS:')} Model loaded on {device.upper()}")
    except Exception as e:
        raise RuntimeError(f"Failed to load Whisper model: {e}") from e

    # Transcribe audio
    print(f"\n{color_info('Transcribing audio')} (this may take a while)...")
    print(f"  File: {audio_path.name}")
    print(f"  Model: {model_size}")
    try:
        result = model.transcribe(str(audio_path))
        transcript_text = result["text"]
        print(f"{color_success('SUCCESS:')} Transcription complete")
    except Exception as e:
        raise RuntimeError(f"Transcription failed: {e}") from e

    # Save transcript
    transcript_stem = audio_path.stem
    transcript_path = output_dir / f"{transcript_stem}_transcript.txt"

    print(f"\n{color_info('Saving transcript')}...")
    try:
        transcript_path.write_text(transcript_text, encoding="utf-8")
        print(f"{color_success('SUCCESS:')} Transcript saved to: {transcript_path}")
    except Exception as e:
        raise OSError(f"Failed to save transcript: {e}") from e

    return transcript_path


# Ollama cloud integration utilities


def _get_api_key() -> str:
    """Read API key from api_key.txt file or environment variable.

    Reads the API key from api_key.txt file first, then falls back to
    OLLAMA_API_KEY environment variable.

    Returns:
        str: The API key for Ollama cloud authentication.

    Raises:
        ValueError: If API key is not found in file or environment variable.
    """
    # Try to read from api_key.txt file first
    api_key_file = get_project_root() / "api_key.txt"
    if api_key_file.exists():
        api_key = api_key_file.read_text().strip()
        # Handle "apikey=..." format
        if api_key.startswith("apikey="):
            api_key = api_key[7:].strip()  # Remove "apikey=" prefix
        if api_key:
            return api_key

    # Fallback to environment variable
    api_key = os.getenv("OLLAMA_API_KEY")
    if api_key:
        return api_key

    raise ValueError(
        "API key not found. Please ensure api_key.txt exists with your Ollama API key, "
        "or set the OLLAMA_API_KEY environment variable."
    )


# Initialize the Ollama client with cloud endpoint (lazy initialization)
_client: Client | None = None


def _get_client() -> Client:
    """Get or initialize the Ollama client.

    Returns:
        Client: Initialized Ollama client.

    Raises:
        RuntimeError: If Ollama is not available.
    """
    global _client
    if _client is None:
        if not OLLAMA_AVAILABLE:
            raise RuntimeError(
                "Ollama is not installed. Please install it with: pip install ollama"
            )
        api_key = _get_api_key()
        _client = Client(host="https://ollama.com", headers={"Authorization": f"Bearer {api_key}"})
    return _client


def call_ollama_cloud(
    model: str,
    messages: list | None = None,
    prompt: str | None = None,
    stream: bool = False,
    **kwargs,
):
    """Call an Ollama cloud model.

    Args:
        model: Model name (must include '-cloud' suffix, e.g., 'phi3-cloud',
            'gpt-oss:120b-cloud').
        messages: List of message dictionaries with 'role' and 'content' keys
            (for chat models). Defaults to None.
        prompt: Single prompt string (for completion models). Defaults to None.
        stream: Whether to stream the response. Defaults to False.
        **kwargs: Additional parameters to pass to the API call.

    Returns:
        dict: Response from the cloud model.

    Raises:
        ValueError: If model name doesn't include '-cloud' suffix or if neither
            messages nor prompt is provided.
        Exception: For API errors, authentication errors, or network issues.
    """
    # Ensure model name includes -cloud suffix
    if "-cloud" not in model:
        raise ValueError(
            f"Model '{model}' must include '-cloud' suffix. "
            f"Cloud models are required (e.g., 'phi3-cloud', 'gpt-oss:120b-cloud')."
        ) from None

    try:
        client = _get_client()
        if messages is not None:
            # Use chat API for messages
            response = client.chat(model=model, messages=messages, stream=stream, **kwargs)
        elif prompt is not None:
            # Use generate API for single prompt
            response = client.generate(model=model, prompt=prompt, stream=stream, **kwargs)
        else:
            raise ValueError("Either 'messages' or 'prompt' must be provided")

        return response
    except ValueError:
        raise
    except Exception as e:
        # Provide clearer error messages
        error_msg = str(e)
        if "authentication" in error_msg.lower() or "401" in error_msg or "403" in error_msg:
            raise RuntimeError(
                f"Authentication error: Please verify your API key is valid. Error: {error_msg}"
            ) from e
        if "network" in error_msg.lower() or "connection" in error_msg.lower():
            raise ConnectionError(
                f"Network error: Unable to connect to Ollama cloud API. Error: {error_msg}"
            ) from e
        raise RuntimeError(f"Error calling Ollama cloud model: {error_msg}") from e


# Default cloud model (gpt-oss:120b-cloud)
DEFAULT_CLOUD_MODEL = "gpt-oss:120b-cloud"
