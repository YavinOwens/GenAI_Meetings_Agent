"""
Meeting Agent - LLM Module

A bespoke meeting agent that uses a role/task framework to act as a transcriber
and note taker, providing structured outputs for meetings.
"""

from meeting_agent.helper import (
    DEFAULT_CLOUD_MODEL,
    call_ollama_cloud,
    count_tokens,
    create_llm_messages,
    extract_response_text,
    format_token_usage,
    handle_prompt_load_error,
    load_prompt,
    parse_ollama_response,
    write_instrumentation_log,
)


class MeetingAgent:
    """Meeting Agent that processes transcripts and generates structured meeting notes.

    This class acts as a meeting transcriber and note taker,
    transforming raw transcripts into detailed, structured meeting notes.
    """

    def __init__(self, model: str | None = None, encoding_name: str = "cl100k_base"):
        """Initialize the Meeting Agent.

        Args:
            model: Model name (defaults to DEFAULT_CLOUD_MODEL).
            encoding_name: Tiktoken encoding to use for token counting.

        Raises:
            FileNotFoundError: If prompt files cannot be loaded.
        """
        self.model = model or DEFAULT_CLOUD_MODEL
        self.encoding_name = encoding_name

        # Load prompts from prompt_gallery
        try:
            self.system_prompt = load_prompt("meeting_notes_system_prompt.txt")
            self.user_prompt_template = load_prompt("meeting_notes_user_prompt_template.txt")
        except FileNotFoundError as e:
            raise handle_prompt_load_error("MeetingAgent", e) from e

    def _create_meeting_prompt(self, transcript: str) -> list[dict[str, str]]:
        """Create the prompt for meeting note generation.

        Args:
            transcript: The raw meeting transcript text.

        Returns:
            List[Dict[str, str]]: List of message dictionaries for the LLM.
        """
        # #region agent log
        write_instrumentation_log(
            location="llm.py:58",
            message="Before format call",
            data={
                "template_length": len(self.user_prompt_template),
                "transcript_length": len(transcript) if transcript else 0,
            },
            hypothesis_id="A",
        )
        # #endregion
        # Format user prompt with transcript
        try:
            user_prompt = self.user_prompt_template.format(transcript=transcript)
        except (KeyError, ValueError) as e:
            # #region agent log
            write_instrumentation_log(
                location="llm.py:58",
                message="Format error",
                data={"error_type": type(e).__name__, "error_msg": str(e)},
                hypothesis_id="A",
            )
            # #endregion
            raise
        # #region agent log
        write_instrumentation_log(
            location="llm.py:58",
            message="After format call",
            data={"user_prompt_length": len(user_prompt)},
            hypothesis_id="A",
        )
        # #endregion

        return create_llm_messages(self.system_prompt, user_prompt)

    def generate_meeting_notes(self, transcript: str) -> dict:
        """Generate structured meeting notes from a transcript.

        Args:
            transcript: The raw meeting transcript text.

        Returns:
            Dict: Response containing meeting notes and token usage.

        Raises:
            ValueError: If transcript is empty.
        """
        if not transcript or not transcript.strip():
            raise ValueError("Transcript cannot be empty")

        # Create the prompt
        messages = self._create_meeting_prompt(transcript)

        # Count input tokens
        input_text = " ".join([msg.get("content", "") for msg in messages])
        input_tokens = count_tokens(input_text, self.encoding_name)

        # Call the LLM
        response = call_ollama_cloud(model=self.model, messages=messages, stream=False)

        # Convert response to dict if it's a Pydantic model
        response_dict = parse_ollama_response(response)

        # Count output tokens
        output_text = extract_response_text(response_dict)
        output_tokens = count_tokens(output_text, self.encoding_name) if output_text else 0

        # Add token usage information
        response_dict["token_usage"] = format_token_usage(input_tokens, output_tokens)

        return response_dict

    def get_notes_text(self, response: dict) -> str:
        """Extract the meeting notes text from a response.

        Args:
            response: Response dictionary from generate_meeting_notes().

        Returns:
            str: The meeting notes text.
        """
        return extract_response_text(response)


def create_meeting_agent(model: str | None = None) -> MeetingAgent:
    """Create a MeetingAgent instance.

    Args:
        model: Model name (defaults to DEFAULT_CLOUD_MODEL).

    Returns:
        MeetingAgent: A MeetingAgent instance.
    """
    return MeetingAgent(model=model)
