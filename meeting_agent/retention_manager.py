"""
Retention Manager Module for Meeting Agent.

Handles retention policy matching, application, and data lifecycle management.
"""

import json
from datetime import datetime, timedelta
from typing import Any

from meeting_agent.database import Database
from meeting_agent.helper import (
    call_ollama_cloud,
    create_llm_messages,
    extract_response_text,
    load_prompt,
    parse_ollama_response,
)


class RetentionManager:
    """Manages retention policies and data lifecycle."""

    def __init__(self, database: Database):
        """Initialize retention manager.

        Args:
            database: Database instance.
        """
        self.db = database

    def calculate_retention_date(self, created_at: str | datetime, retention_days: int) -> datetime:
        """Calculate when data should be archived/deleted.

        Args:
            created_at: Creation timestamp.
            retention_days: Retention period in days.

        Returns:
            datetime: Retention date.
        """
        if isinstance(created_at, str):
            # Handle ISO format strings, with or without timezone
            try:
                if "Z" in created_at or "+" in created_at or created_at.count("-") > 2:
                    # Has timezone info
                    created_dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                else:
                    # No timezone, assume local
                    created_dt = datetime.fromisoformat(created_at)
            except ValueError:
                # Fallback: try parsing as simple date string
                created_dt = datetime.strptime(created_at[:10], "%Y-%m-%d")
        else:
            created_dt = created_at

        return created_dt + timedelta(days=retention_days)

    def match_label_to_policy(self, label_text: str, label_type: str) -> list[dict[str, Any]]:
        """Match a label to applicable retention policies.

        Args:
            label_text: Label text.
            label_type: 'auto' or 'custom'.

        Returns:
            list: List of applicable policies (sorted by retention_days, shortest first).
        """
        return self.db.get_policies_for_label(label_text, label_type)

    def get_policy_status(self, transcript_id: int) -> dict[str, Any] | None:
        """Get retention policy status for a transcript.

        Args:
            transcript_id: Transcript ID.

        Returns:
            dict: Policy status information or None.
        """
        transcript = self.db.get_transcript(transcript_id=transcript_id)
        if not transcript:
            return None

        labels = self.db.get_labels(transcript_id)
        if not labels:
            return None

        # Find most restrictive policy (shortest retention period)
        applicable_policies = []
        for label in labels:
            policies = self.match_label_to_policy(label['label_text'], label['label_type'])
            applicable_policies.extend(policies)

        if not applicable_policies:
            return None

        # Sort by retention_days (shortest first = most restrictive)
        applicable_policies.sort(key=lambda p: p['retention_days'])

        most_restrictive = applicable_policies[0]
        retention_date = self.calculate_retention_date(
            transcript['created_at'],
            most_restrictive['retention_days']
        )

        return {
            'policy': most_restrictive,
            'retention_date': retention_date,
            'days_remaining': (retention_date - datetime.now()).days,
            'is_overdue': retention_date < datetime.now(),
            'all_applicable_policies': applicable_policies,
        }

    def check_and_apply_policies(self, dry_run: bool = False) -> dict[str, Any]:
        """Check and apply retention policies to all transcripts.

        Args:
            dry_run: If True, only report what would be done without applying.

        Returns:
            dict: Results with counts and details.
        """
        results = {
            'archived': 0,
            'deleted': 0,
            'errors': 0,
            'details': [],
        }

        transcripts = self.db.get_transcripts_for_retention_check()

        for transcript_row in transcripts:
            transcript_id = transcript_row['id']
            label_text = transcript_row['label_text']
            label_type = transcript_row['label_type']

            # Skip if already archived
            if transcript_row.get('archived', 0):
                continue

            # Get applicable policies
            policies = self.match_label_to_policy(label_text, label_type)
            if not policies:
                continue

            # Use most restrictive policy
            policy = min(policies, key=lambda p: p['retention_days'])

            # Calculate retention date
            retention_date = self.calculate_retention_date(
                transcript_row['created_at'],
                policy['retention_days']
            )

            # Check if retention period has passed
            if retention_date < datetime.now():
                try:
                    if dry_run:
                        results['details'].append({
                            'transcript_id': transcript_id,
                            'action': policy['action'],
                            'policy_id': policy['id'],
                            'label': label_text,
                            'days_overdue': (datetime.now() - retention_date).days,
                        })
                        if policy['action'] == 'archive':
                            results['archived'] += 1
                        else:
                            results['deleted'] += 1
                    else:
                        # Apply action
                        if policy['action'] == 'archive':
                            self.archive_transcript_data(
                                transcript_id,
                                policy['id'],
                                reason=f"Retention policy: {policy['label_text']} ({policy['retention_days']} days)"
                            )
                            results['archived'] += 1
                        else:
                            # Delete action
                            self.delete_transcript_data(transcript_id, policy['id'])
                            results['deleted'] += 1
                except Exception as e:
                    results['errors'] += 1
                    results['details'].append({
                        'transcript_id': transcript_id,
                        'error': str(e),
                    })

        return results

    def archive_transcript_data(
        self,
        transcript_id: int,
        policy_id: int | None = None,
        reason: str | None = None,
    ) -> int:
        """Archive a transcript and preserve all related data.

        Args:
            transcript_id: Transcript ID.
            policy_id: Policy that triggered archive.
            reason: Reason for archiving.

        Returns:
            int: Archive record ID.
        """
        return self.db.archive_transcript(
            transcript_id=transcript_id,
            retention_policy_id=policy_id,
            archived_by="RetentionManager",
            reason=reason,
        )

    def delete_transcript_data(
        self,
        transcript_id: int,
        policy_id: int | None = None,
    ) -> bool:
        """Delete a transcript and all related data permanently.

        Args:
            transcript_id: Transcript ID.
            policy_id: Policy that triggered deletion.

        Returns:
            bool: True if deleted successfully.
        """
        # Delete transcript (cascade will delete related data)
        return self.db.delete_transcript(transcript_id)

    def get_pending_actions(self) -> list[dict[str, Any]]:
        """Get transcripts that are due for retention action.

        Returns:
            list: List of transcripts with pending actions.
        """
        pending = []
        transcripts = self.db.get_transcripts_for_retention_check()

        for transcript_row in transcripts:
            transcript_id = transcript_row['id']
            label_text = transcript_row['label_text']
            label_type = transcript_row['label_type']

            # Skip if already archived
            if transcript_row.get('archived', 0):
                continue

            # Get applicable policies
            policies = self.match_label_to_policy(label_text, label_type)
            if not policies:
                continue

            # Use most restrictive policy
            policy = min(policies, key=lambda p: p['retention_days'])

            # Calculate retention date
            retention_date = self.calculate_retention_date(
                transcript_row['created_at'],
                policy['retention_days']
            )

            # Check if retention period has passed
            if retention_date < datetime.now():
                days_overdue = (datetime.now() - retention_date).days
                pending.append({
                    'transcript_id': transcript_id,
                    'file_path': transcript_row.get('file_path', 'Unknown'),
                    'label': label_text,
                    'policy': policy,
                    'retention_date': retention_date.isoformat(),
                    'days_overdue': days_overdue,
                    'action': policy['action'],
                })

        return pending

    def generate_policies_via_llm(
        self,
        transcript_id: int | None = None,
        analyze_all: bool = False,
        model: str | None = None,
    ) -> dict[str, Any]:
        """Generate retention policy recommendations using LLM.

        Args:
            transcript_id: Specific transcript ID to analyze (if None and analyze_all=False, uses current transcript).
            analyze_all: If True, analyze all transcripts in database.
            model: LLM model to use (defaults to DEFAULT_CLOUD_MODEL).

        Returns:
            dict: LLM analysis with policy recommendations.
        """
        from meeting_agent.helper import DEFAULT_CLOUD_MODEL

        if model is None:
            model = DEFAULT_CLOUD_MODEL

        # Load prompts
        try:
            system_prompt = load_prompt("retention_policy_system_prompt.txt")
            user_prompt_template = load_prompt("retention_policy_user_prompt_template.txt")
        except FileNotFoundError as e:
            return {
                "error": f"Prompt files not found: {e}",
                "recommendations": [],
            }

        # Gather data for analysis
        if analyze_all:
            # Analyze all transcripts
            transcripts = self.db.list_transcripts(limit=1000, include_archived=False)
            labels_summary = {}
            metadata_summary = {}
            sentiment_summary = {}
            transcript_summary = []

            for transcript in transcripts:
                transcript_id_val = transcript['id']
                # Get labels
                labels = self.db.get_labels(transcript_id_val)
                for label in labels:
                    label_text = label['label_text']
                    if label_text not in labels_summary:
                        labels_summary[label_text] = {"count": 0, "type": label['label_type']}
                    labels_summary[label_text]["count"] += 1

                # Get sentiment if available
                sentiment = self.db.get_sentiment_analysis(transcript_id_val)
                if sentiment:
                    overall = sentiment.get('overall_compound', 0.0) if isinstance(sentiment, dict) else 0.0
                    sentiment_summary[transcript_id_val] = overall

                # Metadata
                metadata_summary[transcript_id_val] = {
                    "source": transcript.get('source_type', 'unknown'),
                    "created": transcript.get('created_at', '')[:10] if transcript.get('created_at') else 'unknown',
                    "size": transcript.get('character_count', 0),
                }

                # Transcript summary (first 500 chars)
                transcript_summary.append({
                    "id": transcript_id_val,
                    "preview": transcript.get('content', '')[:500],
                })

            analysis_context = f"Analyzing {len(transcripts)} transcripts in the database to identify patterns and recommend retention policies."
            labels_summary_str = "\n".join([f"- {label}: {info['count']} occurrences ({info['type']} label)" for label, info in labels_summary.items()])
            metadata_summary_str = f"Total transcripts: {len(transcripts)}\nSource types: {set(m.get('source', 'unknown') for m in metadata_summary.values())}"
            sentiment_summary_str = f"Sentiment analyzed for {len(sentiment_summary)} transcripts"
            transcript_summary_str = f"Sample of {min(5, len(transcript_summary))} transcript previews:\n" + "\n---\n".join([f"ID {t['id']}: {t['preview']}..." for t in transcript_summary[:5]])
            additional_context = "Analyzing patterns across all transcripts to recommend comprehensive retention policies."

        else:
            # Analyze specific transcript
            if transcript_id is None:
                return {
                    "error": "transcript_id required when analyze_all is False",
                    "recommendations": [],
                }

            transcript = self.db.get_transcript(transcript_id=transcript_id)
            if not transcript:
                return {
                    "error": f"Transcript {transcript_id} not found",
                    "recommendations": [],
                }

            # Get labels
            labels = self.db.get_labels(transcript_id)
            labels_summary_str = "\n".join([f"- {l['label_text']} ({l['label_type']} label)" for l in labels])

            # Get sentiment
            sentiment = self.db.get_sentiment_analysis(transcript_id)
            sentiment_summary_str = "Not available"
            if sentiment:
                overall = sentiment.get('overall_compound', 0.0) if isinstance(sentiment, dict) else 0.0
                sentiment_summary_str = f"Overall sentiment: {overall:.3f}"

            # Metadata
            metadata_summary_str = f"Source: {transcript.get('source_type', 'unknown')}\nCreated: {transcript.get('created_at', '')[:10] if transcript.get('created_at') else 'unknown'}\nSize: {transcript.get('character_count', 0):,} characters"

            # Transcript summary
            content = transcript.get('content', '')
            transcript_summary_str = content[:2000] + "..." if len(content) > 2000 else content

            analysis_context = f"Analyzing transcript ID {transcript_id} to recommend retention policies."
            additional_context = "Focus on this specific transcript's content, labels, and characteristics."

        # Format user prompt
        user_prompt = user_prompt_template.format(
            analysis_context=analysis_context,
            transcript_summary=transcript_summary_str,
            labels_summary=labels_summary_str if labels_summary_str else "No labels found",
            metadata_summary=metadata_summary_str,
            sentiment_summary=sentiment_summary_str,
            additional_context=additional_context,
        )

        # Create messages
        messages = create_llm_messages(system_prompt, user_prompt)

        # Call LLM
        try:
            response = call_ollama_cloud(model=model, messages=messages, stream=False)
            response_dict = parse_ollama_response(response)
            answer = extract_response_text(response_dict)

            # Parse JSON response
            if answer:
                # Try to extract JSON from response
                answer_clean = answer.strip()
                if answer_clean.startswith("```json"):
                    answer_clean = answer_clean[7:]
                if answer_clean.startswith("```"):
                    answer_clean = answer_clean[3:]
                if answer_clean.endswith("```"):
                    answer_clean = answer_clean[:-3]
                answer_clean = answer_clean.strip()

                try:
                    recommendations = json.loads(answer_clean)
                    return recommendations
                except json.JSONDecodeError:
                    # Try to find JSON object in response
                    import re
                    json_match = re.search(r'\{.*\}', answer_clean, re.DOTALL)
                    if json_match:
                        recommendations = json.loads(json_match.group())
                        return recommendations
                    else:
                        return {
                            "error": "Could not parse LLM response as JSON",
                            "raw_response": answer,
                            "recommendations": [],
                        }
            else:
                return {
                    "error": "No response from LLM",
                    "recommendations": [],
                }
        except Exception as e:
            return {
                "error": f"Error calling LLM: {e}",
                "recommendations": [],
            }
