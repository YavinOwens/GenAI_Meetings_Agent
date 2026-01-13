"""
Meeting Agent - Main Script

Main script that can either:
1. Transcribe an audio file and generate meeting notes, or
2. Read an existing transcript and generate meeting notes.

Note: This script should be run after installing the package in editable mode:
    pip install -e .

Or use the entry point after installation:
    meeting-agent
"""

import argparse

from meeting_agent.helper import (
    DEFAULT_CLOUD_MODEL,
    WHISPER_AVAILABLE,
    check_ffmpeg,
    color_error,
    color_header,
    color_header_border,
    color_success,
    find_most_recent_transcript,
    get_transcripts_dir,
    install_ffmpeg_instructions,
    print_traceback,
    read_transcript,
    save_meeting_notes,
    transcribe_audio,
)
from meeting_agent.llm import create_meeting_agent


def main():
    """Main entry point for the meeting agent.

    Returns:
        int: Exit code (0 for success, 1 for error).
    """
    parser = argparse.ArgumentParser(
        description="Meeting Agent - Transcribe audio and/or generate meeting notes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate notes from most recent transcript
  python meeting_agent.py

  # Transcribe audio file and generate notes
  python meeting_agent.py --transcribe "path/to/audio.m4a"

  # Transcribe with specific Whisper model
  python meeting_agent.py --transcribe "path/to/audio.m4a" --whisper-model small

  # Only transcribe (don't generate notes)
  python meeting_agent.py --transcribe "path/to/audio.m4a" --no-notes
        """,
    )
    parser.add_argument(
        "--transcribe",
        type=str,
        metavar="AUDIO_FILE",
        help="Audio file to transcribe before generating notes",
    )
    parser.add_argument(
        "--whisper-model",
        type=str,
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size for transcription (default: base)",
    )
    parser.add_argument(
        "--no-notes",
        action="store_true",
        help="Only transcribe audio, don't generate meeting notes",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Meeting Agent - Transcript to Notes Generator")
    print("=" * 60)
    print(f"Using model: {DEFAULT_CLOUD_MODEL}\n")

    transcript_path: Path | None = None

    # If audio file provided, transcribe it first
    if args.transcribe:
        # Check dependencies for transcription
        if not WHISPER_AVAILABLE:
            print(f"{color_error('ERROR:')} Whisper is not installed.")
            print("Please install it with: pip install openai-whisper")
            return 1

        if not check_ffmpeg():
            print(f"{color_error('ERROR:')} ffmpeg is not installed or not available.")
            print(install_ffmpeg_instructions())
            return 1

        # Validate audio file
        audio_path = Path(args.transcribe)
        if not audio_path.is_absolute():
            audio_path = Path.cwd() / audio_path

        if not audio_path.exists():
            print(f"{color_error('ERROR:')} Audio file not found: {audio_path}")
            return 1

        # Transcribe audio
        print(color_header_border("=" * 60))
        print(color_header("Step 1: Audio Transcription"))
        print(color_header_border("=" * 60))
        print()

        try:
            transcript_path = transcribe_audio(
                audio_path=audio_path, model_size=args.whisper_model
            )
            print(f"\n{color_success('SUCCESS:')} Transcript saved to: {transcript_path}")

            # If --no-notes flag, exit here
            if args.no_notes:
                print("\n" + "=" * 60)
                print("Transcription complete (notes generation skipped)")
                print("=" * 60)
                return 0

        except Exception as e:
            print(f"\n{color_error('ERROR:')} Transcription failed: {e}")
            print_traceback()
            return 1

    # If no audio file provided, find the most recent transcript
    if not transcript_path:
        transcripts_dir = get_transcripts_dir()
        print(f"Looking for transcripts in: {transcripts_dir}")
        transcript_path = find_most_recent_transcript(transcripts_dir)

        if not transcript_path:
            print(f"ERROR: No transcript files found in {transcripts_dir}")
            print("\nSupported formats: .txt, .md, .transcript, .srt, .vtt")
            print("\nTip: Use --transcribe to transcribe an audio file first")
            return 1

        print(f"SUCCESS: Found most recent transcript: {transcript_path.name}")
        print(f"  Modified: {transcript_path.stat().st_mtime}")

    # Read the transcript
    print("\n" + "=" * 60)
    print("Step 2: Reading Transcript")
    print("=" * 60)
    print("\nReading transcript...")
    try:
        transcript = read_transcript(transcript_path)
        transcript_length = len(transcript)
        print(f"SUCCESS: Transcript read successfully ({transcript_length} characters)")
    except Exception as e:
        print(f"ERROR: Error reading transcript: {e}")
        return 1

    # Create meeting agent
    print("\n" + "=" * 60)
    print("Step 3: Generating Meeting Notes")
    print("=" * 60)
    print("\nInitializing Meeting Agent...")
    agent = create_meeting_agent()
    print("SUCCESS: Meeting Agent ready")

    # Generate meeting notes
    print("\nGenerating meeting notes (this may take a moment)...")
    try:
        response = agent.generate_meeting_notes(transcript)
        notes = agent.get_notes_text(response)
        token_usage = response.get("token_usage", {})

        if not notes:
            print("ERROR: No notes generated")
            return 1

        print("SUCCESS: Meeting notes generated successfully")
        print("\nToken usage:")
        print(f"  Input tokens: {token_usage.get('input_tokens', 0):,}")
        print(f"  Output tokens: {token_usage.get('output_tokens', 0):,}")
        print(f"  Total tokens: {token_usage.get('total_tokens', 0):,}")

        # Display notes preview
        print("\n" + "=" * 60)
        print("MEETING NOTES PREVIEW")
        print("=" * 60)
        preview_length = min(500, len(notes))
        print(notes[:preview_length])
        if len(notes) > preview_length:
            print("\n... (truncated)")

        # Save notes to file as Word document
        print("\n" + "=" * 60)
        print("Saving meeting notes as Word document (.docx)...")
        # Prepare model information for APA citation
        model_info = {
            "model_name": DEFAULT_CLOUD_MODEL.replace("-cloud", ""),
            "version": "2024",
            "url": "https://ollama.com",
        }
        notes_path = save_meeting_notes(
            notes, transcript_path, format="docx", model_info=model_info
        )
        print(f"SUCCESS: Meeting notes saved to: {notes_path}")

        print("\n" + "=" * 60)
        print("Meeting Agent processing complete")
        print("=" * 60)

        return 0

    except Exception as e:
        print(f"ERROR: Error generating meeting notes: {e}")
        print_traceback()
        return 1


if __name__ == "__main__":
    from meeting_agent.helper import run_main_with_error_handling

    run_main_with_error_handling(main, exit_on_keyboard_interrupt=True)
