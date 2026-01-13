"""
Meeting Agent CLI - Interactive Question Interface

Interactive command-line interface for asking questions about meeting transcripts.

Note: This script should be run after installing the package in editable mode:
    pip install -e .

Or use the entry point after installation:
    meeting-cli
"""

import sys
from pathlib import Path

from meeting_agent.helper import (
    DEFAULT_CLOUD_MODEL,
    call_ollama_cloud,
    color_answer,
    color_answer_label,
    color_command,
    color_dim,
    color_error,
    color_header,
    color_header_border,
    color_info,
    color_label,
    color_success,
    color_warning,
    create_llm_messages,
    extract_response_text,
    find_most_recent_transcript,
    get_transcripts_dir,
    handle_prompt_load_error,
    load_prompt,
    parse_ollama_response,
    print_traceback,
    read_transcript,
    sanitize_path_for_logging,
    write_instrumentation_log,
)

# Ensures output is not buffered
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, "reconfigure") else None


class MeetingCLI:
    """Interactive CLI for querying meeting transcripts."""

    def __init__(self, model: str | None = None):
        """Initialize the CLI with a model.

        Args:
            model: Model name (defaults to DEFAULT_CLOUD_MODEL).

        Raises:
            FileNotFoundError: If prompt files cannot be loaded.
        """
        self.model = model or DEFAULT_CLOUD_MODEL
        self.encoding_name = "cl100k_base"
        self.current_transcript = None
        self.transcript_text = None

        # Load prompts from prompt_gallery
        try:
            self.system_prompt = load_prompt("cli_qa_system_prompt.txt")
            self.user_prompt_template = load_prompt("cli_qa_user_prompt_template.txt")
        except FileNotFoundError as e:
            raise handle_prompt_load_error("MeetingCLI", e) from e

    def load_transcript(self, transcript_path: Path | None = None) -> bool:
        """Load a transcript file.

        Args:
            transcript_path: Path to transcript file, or None to use most recent.

        Returns:
            bool: True if transcript loaded successfully, False otherwise.
        """
        if transcript_path is None:
            # Find most recent transcript
            transcripts_dir = get_transcripts_dir()
            transcript_path = find_most_recent_transcript(transcripts_dir, exclude_notes=True)

            if not transcript_path:
                print(f"{color_error('ERROR: No transcript files found in')} {transcripts_dir}")
                return False

        if not transcript_path.exists():
            print(f"{color_error('ERROR: Transcript file not found:')} {transcript_path}")
            return False

        try:
            self.transcript_text = read_transcript(transcript_path)
            self.current_transcript = transcript_path
            print(
                f"{color_success('SUCCESS:')} Loaded transcript: {color_header(transcript_path.name)}"
            )
            print(f"  Length: {color_dim(f'{len(self.transcript_text)} characters')}")
            return True
        except Exception as e:
            print(f"{color_error('ERROR: Error reading transcript:')} {e}")
            return False

    def _print_help_commands(self):
        """Print the help commands list."""
        print(f"\n{color_header_border('=' * 60)}")
        print(color_header("Interactive Mode"))
        print(color_header_border("=" * 60))
        print(f"{color_label('Commands:')}")
        print("  - Type your question and press Enter")
        print(f"  - Type {color_command('reload')} to reload the most recent transcript")
        print(f"  - Type {color_command('load <filename>')} to load a specific transcript")
        print(f"  - Type {color_command('quit')} or {color_command('exit')} to exit")
        print(f"  - Type {color_command('help')} for this help message")
        print(f"{color_header_border('=' * 60)}\n")

    def ask_question(self, question: str) -> str:
        """Ask a question about the current transcript.

        Args:
            question: The question to ask.

        Returns:
            str: The answer from the LLM, or an error message if something fails.
        """
        if not self.transcript_text:
            return "Error: No transcript loaded. Please load a transcript first."

        # #region agent log
        write_instrumentation_log(
            location="meeting_cli.py:121",
            message="Before format call",
            data={
                "template_length": len(self.user_prompt_template),
                "question_length": len(question) if question else 0,
                "transcript_length": len(self.transcript_text) if self.transcript_text else 0,
            },
            hypothesis_id="A",
        )
        # #endregion
        # Format user prompt with question and transcript
        try:
            user_prompt = self.user_prompt_template.format(
                question=question, transcript=self.transcript_text
            )
        except (KeyError, ValueError) as e:
            # #region agent log
            write_instrumentation_log(
                location="meeting_cli.py:121",
                message="Format error",
                data={"error_type": type(e).__name__, "error_msg": str(e)},
                hypothesis_id="A",
            )
            # #endregion
            raise
        # #region agent log
        write_instrumentation_log(
            location="meeting_cli.py:121",
            message="After format call",
            data={"user_prompt_length": len(user_prompt)},
            hypothesis_id="A",
        )
        # #endregion

        messages = create_llm_messages(self.system_prompt, user_prompt)

        try:
            response = call_ollama_cloud(model=self.model, messages=messages, stream=False)

            # Extract response text
            response_dict = parse_ollama_response(response)
            response_text = extract_response_text(response_dict)

            if response_text:
                return response_text

            return "Error: Could not parse response"

        except Exception as e:
            return f"Error: {e!s}"

    def interactive_mode(self):
        """Run interactive CLI mode."""
        print(color_header_border("=" * 60))
        print(color_header("Meeting Agent CLI - Interactive Question Interface"))
        print(color_header_border("=" * 60))
        print(f"{color_info('Model:')} {self.model}\n")

        # Load transcript
        if not self.load_transcript():
            print(f"\n{color_error('Exiting. Please ensure transcripts are available.')}")
            return

        self._print_help_commands()

        while True:
            try:
                user_input = input("Question: ").strip()

                if not user_input:
                    continue

                # Handle commands
                if user_input.lower() in ["quit", "exit", "q"]:
                    print(f"\n{color_success('Goodbye!')}")
                    break

                if user_input.lower() == "help":
                    print()
                    self._print_help_commands()
                    continue

                if user_input.lower() == "reload":
                    if self.load_transcript():
                        print(f"{color_success('SUCCESS: Transcript reloaded')}\n")
                    continue

                if user_input.lower().startswith("load "):
                    filename = user_input[5:].strip()
                    # #region agent log
                    write_instrumentation_log(
                        location="meeting_cli.py:186",
                        message="Before path construction",
                        data={
                            "filename": filename,
                            "contains_dotdot": ".." in filename,
                            "contains_slash": "/" in filename or "\\" in filename,
                        },
                        hypothesis_id="E",
                    )
                    # #endregion
                    transcripts_dir = get_transcripts_dir()
                    transcript_path = transcripts_dir / filename

                    # Validate path to prevent directory traversal
                    try:
                        resolved_path = transcript_path.resolve()
                        resolved_dir = transcripts_dir.resolve()
                        if not str(resolved_path).startswith(str(resolved_dir)):
                            print(
                                f"{color_error('ERROR: Invalid path. Cannot access files outside transcripts directory.')}\n"
                            )
                            continue
                    except (OSError, ValueError) as e:
                        print(f"{color_error(f'ERROR: Invalid path: {e}')}\n")
                        continue

                    # #region agent log
                    write_instrumentation_log(
                        location="meeting_cli.py:186",
                        message="After path validation",
                        data={
                            "resolved_path": sanitize_path_for_logging(resolved_path),
                            "is_under_dir": str(resolved_path).startswith(str(resolved_dir)),
                        },
                        hypothesis_id="E",
                    )
                    # #endregion
                    if self.load_transcript(transcript_path):
                        print(f"{color_success('SUCCESS: Transcript loaded')}\n")
                    continue

                # Ask the question
                print(f"\n{color_info('Thinking...')}", flush=True)
                try:
                    answer = self.ask_question(user_input)
                    sys.stdout.flush()
                    if answer and answer.strip():
                        # Display answer in a different color (bright cyan for visibility)
                        print(f"\n{color_answer_label('Answer:')}")
                        print(f"{color_answer(answer)}\n", flush=True)
                    else:
                        print(
                            f"\n{color_error('ERROR: No answer received from LLM')}\n", flush=True
                        )
                    print(f"{color_dim('-' * 60)}\n", flush=True)
                except Exception as e:
                    print(f"\n{color_error(f'ERROR: Error asking question: {e}')}", flush=True)
                    print_traceback()
                    print(f"{color_dim('-' * 60)}\n", flush=True)

            except KeyboardInterrupt:
                print(
                    f"\n\n{color_warning("Interrupted. Type 'quit' to exit or continue asking questions.")}"
                )
            except EOFError:
                print(f"\n\n{color_success('Goodbye!')}")
                break
            except Exception as e:
                print(f"\n{color_error(f'ERROR: Unexpected error: {e}')}\n")
                print_traceback()


def main():
    """Main entry point for the CLI."""
    cli = MeetingCLI()
    cli.interactive_mode()


if __name__ == "__main__":
    from meeting_agent.helper import run_main_with_error_handling

    run_main_with_error_handling(main, exit_on_keyboard_interrupt=True)
