"""
Database module for Meeting Agent.

Provides SQLite database operations for storing transcripts, sentiment analysis,
meeting notes, labels, and chat history.
"""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from meeting_agent.helper import get_project_root


class Database:
    """SQLite database handler for Meeting Agent data."""

    def __init__(self, db_path: Path | None = None):
        """Initialize database connection.

        Args:
            db_path: Path to database file. Defaults to meeting_agent.db in project root.
        """
        if db_path is None:
            db_path = get_project_root() / "meeting_agent.db"
        self.db_path = db_path
        self._init_database()

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection.

        Returns:
            sqlite3.Connection: Database connection.
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        return conn

    def _init_database(self) -> None:
        """Initialize database schema."""
        conn = self._get_connection()
        cursor = conn.cursor()

        # Create transcripts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS transcripts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT UNIQUE,
                content TEXT NOT NULL,
                source_type TEXT NOT NULL CHECK(source_type IN ('uploaded', 'transcribed')),
                audio_file_path TEXT,
                whisper_model TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                file_size INTEGER,
                character_count INTEGER
            )
        """)

        # Create sentiment_analysis table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sentiment_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                transcript_id INTEGER NOT NULL,
                overall_compound REAL NOT NULL,
                overall_pos REAL NOT NULL,
                overall_neu REAL NOT NULL,
                overall_neg REAL NOT NULL,
                overall_label TEXT NOT NULL,
                analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                model_used TEXT,
                FOREIGN KEY (transcript_id) REFERENCES transcripts(id) ON DELETE CASCADE
            )
        """)

        # Create attendees table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS attendees (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                transcript_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                compound_score REAL NOT NULL,
                pos_score REAL NOT NULL,
                neu_score REAL NOT NULL,
                neg_score REAL NOT NULL,
                label TEXT NOT NULL,
                mentions_count INTEGER NOT NULL,
                FOREIGN KEY (transcript_id) REFERENCES transcripts(id) ON DELETE CASCADE
            )
        """)

        # Create topics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS topics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                transcript_id INTEGER NOT NULL,
                topic_name TEXT NOT NULL,
                compound_score REAL NOT NULL,
                pos_score REAL NOT NULL,
                neu_score REAL NOT NULL,
                neg_score REAL NOT NULL,
                label TEXT NOT NULL,
                segments_count INTEGER NOT NULL,
                is_auto_generated BOOLEAN DEFAULT 1,
                FOREIGN KEY (transcript_id) REFERENCES transcripts(id) ON DELETE CASCADE
            )
        """)

        # Create meeting_notes table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS meeting_notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                transcript_id INTEGER NOT NULL,
                content TEXT NOT NULL,
                format TEXT NOT NULL CHECK(format IN ('txt', 'docx', 'rtf', 'md')),
                file_path TEXT,
                model_used TEXT NOT NULL,
                input_tokens INTEGER,
                output_tokens INTEGER,
                total_tokens INTEGER,
                generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (transcript_id) REFERENCES transcripts(id) ON DELETE CASCADE
            )
        """)

        # Create labels table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS labels (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                transcript_id INTEGER NOT NULL,
                label_text TEXT NOT NULL,
                label_type TEXT NOT NULL CHECK(label_type IN ('custom', 'auto')),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (transcript_id) REFERENCES transcripts(id) ON DELETE CASCADE,
                UNIQUE(transcript_id, label_text)
            )
        """)

        # Create chat_history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                transcript_id INTEGER NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (transcript_id) REFERENCES transcripts(id) ON DELETE CASCADE
            )
        """)

        # Create schema_version table for migrations
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER PRIMARY KEY,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                description TEXT
            )
        """)

        # Migrate database schema
        self._migrate_database(cursor)

        # Create retention_policies table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS retention_policies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                label_text TEXT NOT NULL,
                retention_days INTEGER NOT NULL CHECK(retention_days > 0),
                action TEXT NOT NULL CHECK(action IN ('archive', 'delete')),
                enabled BOOLEAN DEFAULT 1,
                apply_to_auto_labels BOOLEAN DEFAULT 1,
                apply_to_custom_labels BOOLEAN DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_by TEXT,
                description TEXT,
                UNIQUE(label_text, action)
            )
        """)

        # Create archived_transcripts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS archived_transcripts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                original_transcript_id INTEGER NOT NULL,
                archived_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                archived_by TEXT,
                retention_policy_id INTEGER,
                reason TEXT,
                FOREIGN KEY (original_transcript_id) REFERENCES transcripts(id),
                FOREIGN KEY (retention_policy_id) REFERENCES retention_policies(id)
            )
        """)

        # Add new columns to transcripts table (migration handled in _migrate_database)
        # These will be added via ALTER TABLE if they don't exist

        # Create indexes for better query performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_transcripts_file_path ON transcripts(file_path)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_transcripts_created_at ON transcripts(created_at)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_sentiment_transcript_id ON sentiment_analysis(transcript_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_attendees_transcript_id ON attendees(transcript_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_topics_transcript_id ON topics(transcript_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_labels_transcript_id ON labels(transcript_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_labels_text ON labels(label_text)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chat_transcript_id ON chat_history(transcript_id)")
        
        # Indexes for retention policies
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_retention_policies_label ON retention_policies(label_text)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_retention_policies_enabled ON retention_policies(enabled)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_transcripts_archived ON transcripts(archived)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_transcripts_policy_applied ON transcripts(retention_policy_applied)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_archived_transcripts_policy ON archived_transcripts(retention_policy_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_archived_transcripts_date ON archived_transcripts(archived_at)")

        conn.commit()
        conn.close()

    def _migrate_database(self, cursor: sqlite3.Cursor) -> None:
        """Apply database migrations safely.

        Args:
            cursor: Database cursor.
        """
        # Get current schema version
        cursor.execute("SELECT MAX(version) FROM schema_version")
        result = cursor.fetchone()
        current_version = result[0] if result and result[0] else 0

        # Migration 1: Add retention policy columns to transcripts table
        if current_version < 1:
            try:
                # Check if columns exist before adding (SQLite doesn't support IF NOT EXISTS for ALTER TABLE)
                cursor.execute("PRAGMA table_info(transcripts)")
                columns = [row[1] for row in cursor.fetchall()]
                
                if 'archived' not in columns:
                    cursor.execute("ALTER TABLE transcripts ADD COLUMN archived BOOLEAN DEFAULT 0")
                if 'archived_at' not in columns:
                    cursor.execute("ALTER TABLE transcripts ADD COLUMN archived_at TIMESTAMP")
                if 'retention_policy_applied' not in columns:
                    cursor.execute("ALTER TABLE transcripts ADD COLUMN retention_policy_applied INTEGER")
                
                cursor.execute("INSERT INTO schema_version (version, description) VALUES (1, 'Add retention policy columns to transcripts')")
            except sqlite3.OperationalError as e:
                # Column might already exist, continue
                pass

        # Future migrations can be added here
        # if current_version < 2:
        #     ...

    def save_transcript(
        self,
        file_path: str | Path,
        content: str,
        source_type: str,
        audio_file_path: str | Path | None = None,
        whisper_model: str | None = None,
    ) -> int:
        """Save transcript to database.

        Args:
            file_path: Path to transcript file.
            content: Transcript content.
            source_type: 'uploaded' or 'transcribed'.
            audio_file_path: Path to audio file (for transcribed files).
            whisper_model: Whisper model used (for transcribed files).

        Returns:
            int: Transcript ID.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        file_path_str = str(file_path)
        audio_file_path_str = str(audio_file_path) if audio_file_path else None

        cursor.execute("""
            INSERT OR REPLACE INTO transcripts 
            (file_path, content, source_type, audio_file_path, whisper_model, 
             updated_at, file_size, character_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            file_path_str,
            content,
            source_type,
            audio_file_path_str,
            whisper_model,
            datetime.now().isoformat(),
            len(content.encode('utf-8')),
            len(content),
        ))

        # Get the transcript ID
        cursor.execute("SELECT id FROM transcripts WHERE file_path = ?", (file_path_str,))
        result = cursor.fetchone()
        transcript_id = result[0] if result else cursor.lastrowid

        conn.commit()
        conn.close()
        return transcript_id

    def get_transcript(self, transcript_id: int | None = None, file_path: str | Path | None = None) -> dict[str, Any] | None:
        """Get transcript from database.

        Args:
            transcript_id: Transcript ID.
            file_path: Path to transcript file.

        Returns:
            dict: Transcript data or None if not found.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        if transcript_id:
            cursor.execute("SELECT * FROM transcripts WHERE id = ?", (transcript_id,))
        elif file_path:
            cursor.execute("SELECT * FROM transcripts WHERE file_path = ?", (str(file_path),))
        else:
            conn.close()
            return None

        row = cursor.fetchone()
        conn.close()

        if row:
            return dict(row)
        return None

    def list_transcripts(
        self,
        label_filter: str | None = None,
        source_type: str | None = None,
        limit: int | None = None,
        offset: int = 0,
        include_archived: bool = False,
    ) -> list[dict[str, Any]]:
        """List transcripts with optional filtering.

        Args:
            label_filter: Filter by label text.
            source_type: Filter by source type ('uploaded' or 'transcribed').
            limit: Maximum number of results.
            offset: Number of results to skip.
            include_archived: Include archived transcripts (default: False).

        Returns:
            list: List of transcript dictionaries.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        query = "SELECT DISTINCT t.* FROM transcripts t"
        conditions = []
        params = []

        if label_filter:
            query += " JOIN labels l ON t.id = l.transcript_id"
            conditions.append("l.label_text = ?")
            params.append(label_filter)

        if source_type:
            conditions.append("t.source_type = ?")
            params.append(source_type)

        if not include_archived:
            conditions.append("(t.archived = 0 OR t.archived IS NULL)")

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY t.created_at DESC"

        if limit:
            query += " LIMIT ? OFFSET ?"
            params.extend([limit, offset])

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def save_sentiment_analysis(
        self,
        transcript_id: int,
        overall_sentiment: dict[str, Any],
        model_used: str | None = None,
    ) -> int:
        """Save sentiment analysis results.

        Args:
            transcript_id: Transcript ID.
            overall_sentiment: Dictionary with sentiment scores and label.
            model_used: LLM model used for topic extraction.

        Returns:
            int: Sentiment analysis ID.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        # Delete existing sentiment analysis for this transcript
        cursor.execute("DELETE FROM sentiment_analysis WHERE transcript_id = ?", (transcript_id,))

        cursor.execute("""
            INSERT INTO sentiment_analysis 
            (transcript_id, overall_compound, overall_pos, overall_neu, overall_neg, 
             overall_label, analyzed_at, model_used)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            transcript_id,
            overall_sentiment.get("compound", 0.0),
            overall_sentiment.get("pos", 0.0),
            overall_sentiment.get("neu", 0.0),
            overall_sentiment.get("neg", 0.0),
            overall_sentiment.get("label", "Neutral"),
            datetime.now().isoformat(),
            model_used,
        ))

        sentiment_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return sentiment_id

    def save_attendees(self, transcript_id: int, attendees: list[dict[str, Any]]) -> None:
        """Save attendee sentiment data.

        Args:
            transcript_id: Transcript ID.
            attendees: List of attendee sentiment dictionaries.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        # Delete existing attendees for this transcript
        cursor.execute("DELETE FROM attendees WHERE transcript_id = ?", (transcript_id,))

        for attendee in attendees:
            cursor.execute("""
                INSERT INTO attendees 
                (transcript_id, name, compound_score, pos_score, neu_score, 
                 neg_score, label, mentions_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                transcript_id,
                attendee.get("name", ""),
                attendee.get("compound", 0.0),
                attendee.get("pos", 0.0),
                attendee.get("neu", 0.0),
                attendee.get("neg", 0.0),
                attendee.get("label", "Neutral"),
                attendee.get("mentions", 0),
            ))

        conn.commit()
        conn.close()

    def save_topics(self, transcript_id: int, topics: dict[str, dict[str, Any]] | list[str], is_auto_generated: bool = True) -> None:
        """Save topic sentiment data.

        Args:
            transcript_id: Transcript ID.
            topics: Dictionary mapping topic names to sentiment data, or list of topic names.
            is_auto_generated: Whether topics are auto-generated from LLM.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        # Delete existing topics for this transcript (only auto-generated if adding auto)
        if is_auto_generated:
            cursor.execute("DELETE FROM topics WHERE transcript_id = ? AND is_auto_generated = 1", (transcript_id,))
        else:
            cursor.execute("DELETE FROM topics WHERE transcript_id = ? AND is_auto_generated = 0", (transcript_id,))

        # Handle both dict format (with sentiment) and list format (just topic names)
        if isinstance(topics, dict):
            for topic_name, topic_data in topics.items():
                if isinstance(topic_data, dict):
                    cursor.execute("""
                        INSERT INTO topics 
                        (transcript_id, topic_name, compound_score, pos_score, neu_score, 
                         neg_score, label, segments_count, is_auto_generated)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        transcript_id,
                        topic_name,
                        topic_data.get("compound", 0.0),
                        topic_data.get("pos", 0.0),
                        topic_data.get("neu", 0.0),
                        topic_data.get("neg", 0.0),
                        topic_data.get("label", "Neutral"),
                        topic_data.get("segments", 0),
                        1 if is_auto_generated else 0,
                    ))
        elif isinstance(topics, list):
            # Just topic names, save with default neutral sentiment
            for topic_name in topics:
                cursor.execute("""
                    INSERT INTO topics 
                    (transcript_id, topic_name, compound_score, pos_score, neu_score, 
                     neg_score, label, segments_count, is_auto_generated)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    transcript_id,
                    topic_name,
                    0.0, 0.0, 1.0, 0.0, "Neutral", 0,
                    1 if is_auto_generated else 0,
                ))

        conn.commit()
        conn.close()

    def save_meeting_notes(
        self,
        transcript_id: int,
        content: str,
        format: str,
        model_used: str,
        file_path: str | Path | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        total_tokens: int | None = None,
    ) -> int:
        """Save meeting notes.

        Args:
            transcript_id: Transcript ID.
            content: Notes content.
            format: Format ('txt', 'docx', 'rtf', 'md').
            model_used: LLM model used.
            file_path: Path to saved notes file.
            input_tokens: Input token count.
            output_tokens: Output token count.
            total_tokens: Total token count.

        Returns:
            int: Meeting notes ID.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO meeting_notes 
            (transcript_id, content, format, file_path, model_used, 
             input_tokens, output_tokens, total_tokens, generated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            transcript_id,
            content,
            format,
            str(file_path) if file_path else None,
            model_used,
            input_tokens,
            output_tokens,
            total_tokens,
            datetime.now().isoformat(),
        ))

        notes_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return notes_id

    def get_meeting_notes(self, transcript_id: int) -> dict[str, Any] | None:
        """Get meeting notes for a transcript.

        Args:
            transcript_id: Transcript ID.

        Returns:
            dict: Most recent meeting notes or None.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM meeting_notes 
            WHERE transcript_id = ? 
            ORDER BY generated_at DESC 
            LIMIT 1
        """, (transcript_id,))

        row = cursor.fetchone()
        conn.close()

        if row:
            return dict(row)
        return None

    def add_label(self, transcript_id: int, label_text: str, label_type: str = "custom") -> int:
        """Add label to transcript.

        Args:
            transcript_id: Transcript ID.
            label_text: Label text.
            label_type: 'custom' or 'auto'.

        Returns:
            int: Label ID.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute("""
                INSERT INTO labels (transcript_id, label_text, label_type, created_at)
                VALUES (?, ?, ?, ?)
            """, (
                transcript_id,
                label_text,
                label_type,
                datetime.now().isoformat(),
            ))
            label_id = cursor.lastrowid
            conn.commit()
            conn.close()
            return label_id
        except sqlite3.IntegrityError:
            # Label already exists (unique constraint)
            conn.close()
            cursor.execute("SELECT id FROM labels WHERE transcript_id = ? AND label_text = ?", (transcript_id, label_text))
            result = cursor.fetchone()
            return result[0] if result else 0

    def get_labels(self, transcript_id: int) -> list[dict[str, Any]]:
        """Get labels for a transcript.

        Args:
            transcript_id: Transcript ID.

        Returns:
            list: List of label dictionaries.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM labels WHERE transcript_id = ? ORDER BY created_at", (transcript_id,))
        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def remove_label(self, transcript_id: int, label_text: str) -> bool:
        """Remove label from transcript.

        Args:
            transcript_id: Transcript ID.
            label_text: Label text to remove.

        Returns:
            bool: True if label was removed, False if not found.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("DELETE FROM labels WHERE transcript_id = ? AND label_text = ?", (transcript_id, label_text))
        deleted = cursor.rowcount > 0
        conn.commit()
        conn.close()
        return deleted

    def save_chat_message(self, transcript_id: int, role: str, content: str) -> int:
        """Save chat message.

        Args:
            transcript_id: Transcript ID.
            role: 'user' or 'assistant'.
            content: Message content.

        Returns:
            int: Chat message ID.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO chat_history (transcript_id, role, content, created_at)
            VALUES (?, ?, ?, ?)
        """, (
            transcript_id,
            role,
            content,
            datetime.now().isoformat(),
        ))

        message_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return message_id

    def get_chat_history(self, transcript_id: int) -> list[dict[str, Any]]:
        """Get chat history for a transcript.

        Args:
            transcript_id: Transcript ID.

        Returns:
            list: List of chat message dictionaries, ordered by created_at.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM chat_history 
            WHERE transcript_id = ? 
            ORDER BY created_at ASC
        """, (transcript_id,))

        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def get_sentiment_analysis(self, transcript_id: int) -> dict[str, Any] | None:
        """Get sentiment analysis for a transcript.

        Args:
            transcript_id: Transcript ID.

        Returns:
            dict: Sentiment analysis data with attendees and topics, or None.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        # Get overall sentiment
        cursor.execute("SELECT * FROM sentiment_analysis WHERE transcript_id = ?", (transcript_id,))
        sentiment_row = cursor.fetchone()

        if not sentiment_row:
            conn.close()
            return None

        result = dict(sentiment_row)

        # Get attendees
        cursor.execute("SELECT * FROM attendees WHERE transcript_id = ?", (transcript_id,))
        attendee_rows = cursor.fetchall()
        # Convert to dict format with sentiment scores
        attendees_dict = {}
        for row in attendee_rows:
            attendees_dict[row["name"]] = {
                "compound": row["compound_score"],
                "pos": row["pos_score"],
                "neu": row["neu_score"],
                "neg": row["neg_score"],
                "label": row["label"],
                "mentions": row["mentions_count"],
            }
        result["attendees"] = attendees_dict

        # Get topics
        cursor.execute("SELECT * FROM topics WHERE transcript_id = ?", (transcript_id,))
        topic_rows = cursor.fetchall()
        # Convert to dict format with sentiment scores
        topics_dict = {}
        for row in topic_rows:
            topics_dict[row["topic_name"]] = {
                "compound": row["compound_score"],
                "pos": row["pos_score"],
                "neu": row["neu_score"],
                "neg": row["neg_score"],
                "label": row["label"],
                "segments": row["segments_count"],
            }
        result["topics"] = topics_dict

        conn.close()
        return result

    def delete_transcript(self, transcript_id: int) -> bool:
        """Delete transcript and all related data (cascade).

        Args:
            transcript_id: Transcript ID.

        Returns:
            bool: True if deleted, False if not found.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("DELETE FROM transcripts WHERE id = ?", (transcript_id,))
        deleted = cursor.rowcount > 0
        conn.commit()
        conn.close()
        return deleted

    def get_all_labels(self) -> list[str]:
        """Get all unique labels across all transcripts.

        Returns:
            list: List of unique label texts.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT DISTINCT label_text FROM labels ORDER BY label_text")
        rows = cursor.fetchall()
        conn.close()

        return [row[0] for row in rows]

    # Retention Policy Methods

    def create_retention_policy(
        self,
        label_text: str,
        retention_days: int,
        action: str,
        apply_to_auto_labels: bool = True,
        apply_to_custom_labels: bool = False,
        description: str | None = None,
        created_by: str | None = None,
    ) -> int:
        """Create a new retention policy.

        Args:
            label_text: Label to apply policy to.
            retention_days: Number of days to retain data.
            action: 'archive' or 'delete'.
            apply_to_auto_labels: Apply to LLM-generated labels.
            apply_to_custom_labels: Apply to user-created labels.
            description: Optional description.
            created_by: User/system that created the policy.

        Returns:
            int: Policy ID.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute("""
                INSERT INTO retention_policies 
                (label_text, retention_days, action, enabled, apply_to_auto_labels, 
                 apply_to_custom_labels, description, created_by, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                label_text,
                retention_days,
                action,
                True,  # enabled by default
                apply_to_auto_labels,
                apply_to_custom_labels,
                description,
                created_by,
                datetime.now().isoformat(),
                datetime.now().isoformat(),
            ))
            policy_id = cursor.lastrowid
            conn.commit()
            conn.close()
            return policy_id
        except sqlite3.IntegrityError:
            # Policy already exists for this label/action combination
            conn.close()
            raise ValueError(f"Policy already exists for label '{label_text}' with action '{action}'")

    def get_retention_policies(
        self,
        enabled_only: bool = False,
        label_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get retention policies.

        Args:
            enabled_only: Only return enabled policies.
            label_filter: Filter by label text.

        Returns:
            list: List of policy dictionaries.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        query = "SELECT * FROM retention_policies WHERE 1=1"
        params = []

        if enabled_only:
            query += " AND enabled = 1"
        if label_filter:
            query += " AND label_text = ?"
            params.append(label_filter)

        query += " ORDER BY label_text, retention_days"

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def update_retention_policy(
        self,
        policy_id: int,
        retention_days: int | None = None,
        action: str | None = None,
        enabled: bool | None = None,
        apply_to_auto_labels: bool | None = None,
        apply_to_custom_labels: bool | None = None,
        description: str | None = None,
    ) -> bool:
        """Update a retention policy.

        Args:
            policy_id: Policy ID.
            retention_days: New retention days.
            action: New action.
            enabled: Enable/disable policy.
            apply_to_auto_labels: Apply to auto labels.
            apply_to_custom_labels: Apply to custom labels.
            description: New description.

        Returns:
            bool: True if updated, False if not found.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        updates = []
        params = []

        if retention_days is not None:
            updates.append("retention_days = ?")
            params.append(retention_days)
        if action is not None:
            updates.append("action = ?")
            params.append(action)
        if enabled is not None:
            updates.append("enabled = ?")
            params.append(int(enabled))
        if apply_to_auto_labels is not None:
            updates.append("apply_to_auto_labels = ?")
            params.append(int(apply_to_auto_labels))
        if apply_to_custom_labels is not None:
            updates.append("apply_to_custom_labels = ?")
            params.append(int(apply_to_custom_labels))
        if description is not None:
            updates.append("description = ?")
            params.append(description)

        if not updates:
            conn.close()
            return False

        updates.append("updated_at = ?")
        params.append(datetime.now().isoformat())
        params.append(policy_id)

        query = f"UPDATE retention_policies SET {', '.join(updates)} WHERE id = ?"
        cursor.execute(query, params)
        updated = cursor.rowcount > 0
        conn.commit()
        conn.close()
        return updated

    def delete_retention_policy(self, policy_id: int) -> bool:
        """Delete a retention policy.

        Args:
            policy_id: Policy ID.

        Returns:
            bool: True if deleted, False if not found.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("DELETE FROM retention_policies WHERE id = ?", (policy_id,))
        deleted = cursor.rowcount > 0
        conn.commit()
        conn.close()
        return deleted

    def get_policies_for_label(self, label_text: str, label_type: str = "auto") -> list[dict[str, Any]]:
        """Get applicable policies for a label.

        Args:
            label_text: Label text to match.
            label_type: 'auto' or 'custom'.

        Returns:
            list: List of applicable policies.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        if label_type == "auto":
            cursor.execute("""
                SELECT * FROM retention_policies 
                WHERE label_text = ? AND enabled = 1 AND apply_to_auto_labels = 1
                ORDER BY retention_days ASC
            """, (label_text,))
        else:
            cursor.execute("""
                SELECT * FROM retention_policies 
                WHERE label_text = ? AND enabled = 1 AND apply_to_custom_labels = 1
                ORDER BY retention_days ASC
            """, (label_text,))

        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def archive_transcript(
        self,
        transcript_id: int,
        retention_policy_id: int | None = None,
        archived_by: str | None = None,
        reason: str | None = None,
    ) -> int:
        """Archive a transcript.

        Args:
            transcript_id: Transcript ID.
            retention_policy_id: Policy that triggered archive.
            archived_by: User/system that archived.
            reason: Reason for archiving.

        Returns:
            int: Archive record ID.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        # Mark transcript as archived
        cursor.execute("""
            UPDATE transcripts 
            SET archived = 1, archived_at = ?, retention_policy_applied = ?
            WHERE id = ?
        """, (datetime.now().isoformat(), retention_policy_id, transcript_id))

        # Create archive record
        cursor.execute("""
            INSERT INTO archived_transcripts 
            (original_transcript_id, archived_at, archived_by, retention_policy_id, reason)
            VALUES (?, ?, ?, ?, ?)
        """, (
            transcript_id,
            datetime.now().isoformat(),
            archived_by,
            retention_policy_id,
            reason,
        ))

        archive_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return archive_id

    def get_archived_transcripts(
        self,
        limit: int | None = None,
        policy_id: int | None = None,
    ) -> list[dict[str, Any]]:
        """Get archived transcripts.

        Args:
            limit: Maximum number of results.
            policy_id: Filter by policy ID.

        Returns:
            list: List of archived transcript dictionaries.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        query = """
            SELECT t.*, a.archived_at, a.archived_by, a.reason, a.retention_policy_id
            FROM transcripts t
            JOIN archived_transcripts a ON t.id = a.original_transcript_id
            WHERE t.archived = 1
        """
        params = []

        if policy_id:
            query += " AND a.retention_policy_id = ?"
            params.append(policy_id)

        query += " ORDER BY a.archived_at DESC"

        if limit:
            query += " LIMIT ?"
            params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def restore_transcript(self, transcript_id: int) -> bool:
        """Restore an archived transcript.

        Args:
            transcript_id: Transcript ID.

        Returns:
            bool: True if restored, False if not found.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        # Unmark transcript as archived
        cursor.execute("""
            UPDATE transcripts 
            SET archived = 0, archived_at = NULL
            WHERE id = ?
        """, (transcript_id,))

        # Remove archive record
        cursor.execute("DELETE FROM archived_transcripts WHERE original_transcript_id = ?", (transcript_id,))

        restored = cursor.rowcount > 0
        conn.commit()
        conn.close()
        return restored

    def get_transcripts_for_retention_check(self) -> list[dict[str, Any]]:
        """Get transcripts that need retention policy evaluation.

        Returns:
            list: List of transcripts with their labels and policies.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        # Get non-archived transcripts with labels
        cursor.execute("""
            SELECT DISTINCT t.*, l.label_text, l.label_type
            FROM transcripts t
            JOIN labels l ON t.id = l.transcript_id
            WHERE t.archived = 0
            ORDER BY t.created_at
        """)

        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def count_transcripts_by_policy(self, policy_id: int) -> int:
        """Count transcripts that would be affected by a policy.

        Args:
            policy_id: Policy ID.

        Returns:
            int: Number of affected transcripts.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        # Get policy details
        cursor.execute("SELECT * FROM retention_policies WHERE id = ?", (policy_id,))
        policy = cursor.fetchone()
        if not policy:
            conn.close()
            return 0

        policy_dict = dict(policy)

        # Count transcripts with matching label
        query = """
            SELECT COUNT(DISTINCT t.id)
            FROM transcripts t
            JOIN labels l ON t.id = l.transcript_id
            WHERE t.archived = 0
            AND l.label_text = ?
        """
        params = [policy_dict['label_text']]

        if policy_dict['apply_to_auto_labels'] and not policy_dict['apply_to_custom_labels']:
            query += " AND l.label_type = 'auto'"
        elif policy_dict['apply_to_custom_labels'] and not policy_dict['apply_to_auto_labels']:
            query += " AND l.label_type = 'custom'"

        cursor.execute(query, params)
        count = cursor.fetchone()[0]
        conn.close()

        return count


