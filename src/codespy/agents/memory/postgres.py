"""PostgreSQL-backed episode persistence for Hippocampus memory."""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from psycopg import sql
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

if TYPE_CHECKING:
    from codespy.agents.memory.hippocampus.context_memory import (
        ContextMemory,
        Item,
        Mutation,
        Topic,
    )
    from codespy.agents.memory.hippocampus.episode import Episode

logger = logging.getLogger(__name__)


class EpisodeStore:
    """PostgreSQL-backed episode persistence for Hippocampus memory.

    Uses psycopg ConnectionPool for thread-safe background saves.
    Tables are auto-created on first connect if they don't exist.
    """

    def __init__(
        self,
        conninfo: str,
        bank_id: str,
        min_size: int = 1,
        max_size: int = 4,
    ):
        """Create store with a psycopg ConnectionPool, scoped to a bank.

        Args:
            conninfo: PostgreSQL connection string (e.g., postgresql://localhost:5432/dbname)
            bank_id: Identifier for the bank (nickname, username, email, agent name)
            min_size: Minimum connections in pool
            max_size: Maximum connections in pool
        """
        self.bank_id = bank_id
        self._pool = ConnectionPool(
            conninfo=conninfo,
            min_size=min_size,
            max_size=max_size,
        )
        self._pool.open()
        self.ensure_schema()

    def close(self) -> None:
        """Shut down the connection pool."""
        self._pool.close()

    def ensure_schema(self) -> None:
        """Auto-create tables if not present (idempotent)."""
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                # Extensions
                cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")

                # Schema version table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS schema_version (
                        version INT PRIMARY KEY,
                        applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
                    )
                """)

                # Banks table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS banks (
                        id VARCHAR(64) PRIMARY KEY,
                        description TEXT
                    )
                """)

                # Topics table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS topics (
                        bank_id VARCHAR(64) NOT NULL REFERENCES banks(id) ON DELETE CASCADE,
                        id VARCHAR(256) NOT NULL,
                        type VARCHAR(64) NOT NULL,
                        description TEXT NOT NULL,
                        PRIMARY KEY (bank_id, id)
                    )
                """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_topics_id_trgm
                        ON topics USING gin (id gin_trgm_ops)
                """)

                # Episodes table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS episodes (
                        bank_id VARCHAR(64) NOT NULL REFERENCES banks(id) ON DELETE CASCADE,
                        id UUID NOT NULL,
                        run_id VARCHAR(48) NOT NULL,
                        task VARCHAR(64) NOT NULL,
                        module VARCHAR(64) NOT NULL,
                        question TEXT NOT NULL DEFAULT '',
                        timestamp TIMESTAMPTZ NOT NULL,
                        PRIMARY KEY (bank_id, id)
                    )
                """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_episodes_task_time
                        ON episodes (bank_id, task, timestamp DESC)
                """)

                # Episode topics junction
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS episode_topics (
                        bank_id VARCHAR(64) NOT NULL,
                        episode_id UUID NOT NULL,
                        topic_id VARCHAR(256) NOT NULL,
                        PRIMARY KEY (bank_id, episode_id, topic_id),
                        FOREIGN KEY (bank_id, episode_id)
                            REFERENCES episodes(bank_id, id) ON DELETE CASCADE,
                        FOREIGN KEY (bank_id, topic_id)
                            REFERENCES topics(bank_id, id) ON DELETE CASCADE
                    )
                """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_episode_topics_reverse
                        ON episode_topics (bank_id, topic_id)
                """)

                # Observations table with flattened mutation fields
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS observations (
                        bank_id VARCHAR(64) NOT NULL REFERENCES banks(id) ON DELETE CASCADE,
                        id VARCHAR(48) NOT NULL,
                        version INT NOT NULL DEFAULT 1,
                        type VARCHAR(32) NOT NULL,
                        content TEXT,
                        episode_id UUID NOT NULL,
                        step INT NOT NULL DEFAULT 0,
                        op_type VARCHAR(8) NOT NULL,
                        previous_content TEXT,
                        ordinal INT NOT NULL DEFAULT 0,
                        PRIMARY KEY (bank_id, id, version),
                        FOREIGN KEY (bank_id, episode_id)
                            REFERENCES episodes(bank_id, id) ON DELETE CASCADE
                    )
                """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_observations_episode
                        ON observations (bank_id, episode_id)
                """)

                # Observation topics junction
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS observation_topics (
                        bank_id VARCHAR(64) NOT NULL,
                        observation_id VARCHAR(48) NOT NULL,
                        observation_version INT NOT NULL,
                        topic_id VARCHAR(256) NOT NULL,
                        observation_occurrence INT NOT NULL DEFAULT 0,
                        version_occurrence INT NOT NULL DEFAULT 0,
                        PRIMARY KEY (bank_id, observation_id, observation_version, topic_id),
                        FOREIGN KEY (bank_id, observation_id, observation_version)
                            REFERENCES observations(bank_id, id, version) ON DELETE CASCADE,
                        FOREIGN KEY (bank_id, topic_id)
                            REFERENCES topics(bank_id, id) ON DELETE CASCADE
                    )
                """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_observation_topics_reverse
                        ON observation_topics (bank_id, topic_id)
                """)

                # Episode observations junction
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS episode_observations (
                        bank_id VARCHAR(64) NOT NULL,
                        episode_id UUID NOT NULL,
                        observation_id VARCHAR(48) NOT NULL,
                        observation_version INT NOT NULL,
                        PRIMARY KEY (bank_id, episode_id, observation_id),
                        FOREIGN KEY (bank_id, episode_id)
                            REFERENCES episodes(bank_id, id) ON DELETE CASCADE,
                        FOREIGN KEY (bank_id, observation_id, observation_version)
                            REFERENCES observations(bank_id, id, version) ON DELETE CASCADE
                    )
                """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_episode_observations_reverse
                        ON episode_observations (bank_id, observation_id)
                """)

                # Artifacts table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS artifacts (
                        bank_id VARCHAR(64) NOT NULL,
                        episode_id UUID NOT NULL,
                        name TEXT NOT NULL,
                        content TEXT NOT NULL,
                        PRIMARY KEY (bank_id, episode_id, name),
                        FOREIGN KEY (bank_id, episode_id)
                            REFERENCES episodes(bank_id, id) ON DELETE CASCADE
                    )
                """)

                conn.commit()

    def verify_access(self) -> None:
        """Check connection health. Raises on failure."""
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")

    def save_episode(self, episode: Episode) -> None:
        """Persist episode. Idempotent on episode data.

        Single transaction:
        1. Ensure bank row exists (idempotent)
        2. Upsert episode row (no-op if already exists)
        3. Upsert topics, insert items (versioned), insert junctions, insert artifacts
        """
        # Import here to avoid circular imports
        from codespy.agents.memory.hippocampus.context_memory import (
            ContextMemory,
            Item,
            Mutation,
            OpType,
            Topic,
        )

        with self._pool.connection() as conn:
            conn.row_factory = dict_row
            with conn.cursor() as cur:
                # 1. Ensure bank exists
                cur.execute(
                    "INSERT INTO banks (id) VALUES (%s) ON CONFLICT (id) DO NOTHING",
                    (self.bank_id,),
                )

                # 2. Upsert episode
                cur.execute(
                    """
                    INSERT INTO episodes (bank_id, id, run_id, task, module, question, timestamp)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (bank_id, id) DO NOTHING
                    """,
                    (
                        self.bank_id,
                        str(episode.id),
                        episode.run_id,
                        episode.task,
                        episode.module,
                        episode.question,
                        episode.timestamp,
                    ),
                )

                # 3. Upsert topics
                topic_ids: set[str] = set()
                for topic in episode.context_memory.topics:
                    topic_ids.add(topic.id)
                    cur.execute(
                        """
                        INSERT INTO topics (bank_id, id, type, description)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (bank_id, id) DO UPDATE SET type = EXCLUDED.type, description = EXCLUDED.description
                        """,
                        (self.bank_id, topic.id, topic.type, topic.description),
                    )

                # 4. Insert episode_topics junctions
                for topic in episode.context_memory.topics:
                    cur.execute(
                        """
                        INSERT INTO episode_topics (bank_id, episode_id, topic_id)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (bank_id, episode_id, topic_id) DO NOTHING
                        """,
                        (self.bank_id, str(episode.id), topic.id),
                    )

                # 5. Process observations and mutations
                # Build map of item_id -> mutation for quick lookup
                mutation_map: dict[str, Mutation] = {}
                for mut in episode.mutations:
                    mutation_map[mut.item_id] = mut

                # Get all current observations from context_memory
                current_items: dict[str, tuple[str, Item]] = {}  # observation_id -> (type, observation)
                for sec in episode.context_memory.section_names():
                    for item in getattr(episode.context_memory, sec):
                        current_items[item.id] = (sec, item)

                # Track observations we've processed to detect DELETEs
                processed_item_ids: set[str] = set()

                # Process observations in current context memory
                for item_id, (type, item) in current_items.items():
                    processed_item_ids.add(item_id)
                    mutation = mutation_map.get(item_id)

                    # Determine if this is a new item (ADD) or existing (REPLACE/inherited)
                    is_new = False
                    version = 1
                    op_type = "ADD"
                    previous_content: str | None = None

                    if mutation:
                        is_new = mutation.type == OpType.ADD
                        if mutation.type == OpType.REPLACE:
                            op_type = "REPLACE"
                            previous_content = mutation.previous_content
                        elif mutation.type == OpType.ADD:
                            op_type = "ADD"
                    else:
                        # Inherited observation - need to look up existing version
                        cur.execute(
                            """
                            SELECT MAX(version) as max_ver
                            FROM observations
                            WHERE bank_id = %s AND id = %s
                            """,
                            (self.bank_id, item_id),
                        )
                        row = cur.fetchone()
                        if row and row["max_ver"]:
                            version = row["max_ver"]
                            op_type = "REPLACE"  # Already exists

                    # Insert new observation version only if it's ADD or REPLACE
                    if mutation and mutation.type in (OpType.ADD, OpType.REPLACE):
                        # Get next version number
                        cur.execute(
                            """
                            SELECT COALESCE(MAX(version), 0) + 1 as next_ver
                            FROM observations
                            WHERE bank_id = %s AND id = %s
                            """,
                            (self.bank_id, item_id),
                        )
                        row = cur.fetchone()
                        version = row["next_ver"] if row else 1

                        # Find step from mutation
                        step = mutation.step if mutation else 0

                        cur.execute(
                            """
                            INSERT INTO observations
                                (bank_id, id, version, type, content, episode_id, step, op_type, previous_content, ordinal)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            """,
                            (
                                self.bank_id,
                                item_id,
                                version,
                                type,
                                item.content,
                                str(episode.id),
                                step,
                                op_type,
                                previous_content,
                                0,
                            ),
                        )

                        # Insert episode_observations junction
                        cur.execute(
                            """
                            INSERT INTO episode_observations (bank_id, episode_id, observation_id, observation_version)
                            VALUES (%s, %s, %s, %s)
                            ON CONFLICT (bank_id, episode_id, observation_id) DO NOTHING
                            """,
                            (self.bank_id, str(episode.id), item_id, version),
                        )

                        # Insert observation_topics for this version
                        for topic_id in item.topic_ids:
                            # Count occurrences
                            cur.execute(
                                """
                                SELECT observation_occurrence, version_occurrence
                                FROM observation_topics
                                WHERE bank_id = %s AND observation_id = %s AND topic_id = %s
                                ORDER BY observation_version DESC
                                LIMIT 1
                                """,
                                (self.bank_id, item_id, topic_id),
                            )
                            prev_row = cur.fetchone()
                            if prev_row:
                                observation_occ = prev_row["observation_occurrence"] + 1
                                ver_occ = (
                                    prev_row["version_occurrence"] + 1
                                    if is_new
                                    else 1
                                )
                            else:
                                observation_occ = 1
                                ver_occ = 1

                            cur.execute(
                                """
                                INSERT INTO observation_topics
                                    (bank_id, observation_id, observation_version, topic_id, observation_occurrence, version_occurrence)
                                VALUES (%s, %s, %s, %s, %s, %s)
                                ON CONFLICT (bank_id, observation_id, observation_version, topic_id) DO NOTHING
                                """,
                                (
                                    self.bank_id,
                                    item_id,
                                    version,
                                    topic_id,
                                    observation_occ,
                                    ver_occ,
                                ),
                            )
                    else:
                        # Inherited observation - use existing version
                        cur.execute(
                            """
                            SELECT MAX(version) as max_ver
                            FROM observations
                            WHERE bank_id = %s AND id = %s
                            """,
                            (self.bank_id, item_id),
                        )
                        row = cur.fetchone()
                        existing_version = row["max_ver"] if row and row["max_ver"] else 1

                        # Insert episode_observations junction with existing version
                        cur.execute(
                            """
                            INSERT INTO episode_observations (bank_id, episode_id, observation_id, observation_version)
                            VALUES (%s, %s, %s, %s)
                            ON CONFLICT (bank_id, episode_id, observation_id) DO NOTHING
                            """,
                            (
                                self.bank_id,
                                str(episode.id),
                                item_id,
                                existing_version,
                            ),
                        )

                        # Increment observation_occurrence for inherited observations
                        for topic_id in item.topic_ids:
                            cur.execute(
                                """
                                SELECT observation_occurrence
                                FROM observation_topics
                                WHERE bank_id = %s AND observation_id = %s AND topic_id = %s
                                AND observation_version = %s
                                """,
                                (self.bank_id, item_id, topic_id, existing_version),
                            )
                            occ_row = cur.fetchone()
                            if occ_row:
                                new_occ = occ_row["observation_occurrence"] + 1
                                cur.execute(
                                    """
                                    UPDATE observation_topics
                                    SET observation_occurrence = %s
                                    WHERE bank_id = %s AND observation_id = %s AND topic_id = %s
                                    AND observation_version = %s
                                    """,
                                    (
                                        new_occ,
                                        self.bank_id,
                                        item_id,
                                        topic_id,
                                        existing_version,
                                    ),
                                )

                # 6. Process DELETE mutations (observations not in current context)
                for mutation in episode.mutations:
                    if mutation.type == OpType.DELETE and mutation.item_id:
                        if mutation.item_id not in processed_item_ids:
                            # Get next version number
                            cur.execute(
                                """
                                SELECT COALESCE(MAX(version), 0) + 1 as next_ver
                                FROM observations
                                WHERE bank_id = %s AND id = %s
                                """,
                                (self.bank_id, mutation.item_id),
                            )
                            row = cur.fetchone()
                            version = row["next_ver"] if row else 1

                            cur.execute(
                                """
                                INSERT INTO observations
                                    (bank_id, id, version, type, content, episode_id, step, op_type, previous_content, ordinal)
                                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                                """,
                                (
                                    self.bank_id,
                                    mutation.item_id,
                                    version,
                                    mutation.section,
                                    None,  # content is NULL for DELETE
                                    str(episode.id),
                                    mutation.step,
                                    "DELETE",
                                    mutation.previous_content,
                                    0,
                                ),
                            )
                            # Note: DELETE observations are NOT inserted into episode_observations

                # 7. Insert artifacts
                for name, content in (episode.artifacts or {}).items():
                    cur.execute(
                        """
                        INSERT INTO artifacts (bank_id, episode_id, name, content)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (bank_id, episode_id, name) DO UPDATE SET content = EXCLUDED.content
                        """,
                        (self.bank_id, str(episode.id), name, content),
                    )

                conn.commit()

        logger.info(
            "save_episode: persisted episode %s (bank=%s, task=%s, topics=%d, observations=%d)",
            episode.id, self.bank_id, episode.task,
            len(episode.context_memory.topics),
            len(episode.context_memory.all_items()),
        )

    def load_context(
        self,
        task: str,
        topic_ids: list[str] | None = None,
        topic_prefix: str | None = None,
    ) -> ContextMemory | None:
        """Load context for a new episode: latest item versions from the
        most recent episode for this (bank, task, topics).

        Args:
            task: Task name to filter by.
            topic_ids: Exact topic IDs to match (ANY).
            topic_prefix: Prefix match on topic ID (e.g. repo_full_name
                matches 'owner/repo', 'owner/repo/pkg', etc.).
                Mutually exclusive with topic_ids.

        Returns ContextMemory (topics + observations + bindings) or None if
        no prior episode exists. No mutations, artifacts, or episode
        metadata are loaded.
        """
        try:
            # Import here to avoid circular imports
            from codespy.agents.memory.hippocampus.context_memory import (
                ContextMemory,
                Item,
                Topic,
            )

            with self._pool.connection() as conn:
                conn.row_factory = dict_row
                with conn.cursor() as cur:
                    # 1. Find the latest episode for this task + topics
                    topic_filter = ""
                    params: list = [self.bank_id, task]

                    if topic_ids:
                        topic_filter = "AND et.topic_id = ANY(%s)"
                        params.append(topic_ids)
                    elif topic_prefix:
                        topic_filter = "AND et.topic_id LIKE %s"
                        params.append(f"{topic_prefix}%")

                    cur.execute(
                        f"""
                        SELECT e.id FROM episodes e
                        JOIN episode_topics et ON et.bank_id = e.bank_id AND et.episode_id = e.id
                        WHERE e.bank_id = %s AND e.task = %s {topic_filter}
                        ORDER BY e.timestamp DESC
                        LIMIT 1
                        """,
                        params,
                    )
                    row = cur.fetchone()
                    if not row:
                        # Diagnostic: how many episodes exist for this bank+task (ignoring topic filter)?
                        cur.execute(
                            "SELECT COUNT(*) AS cnt FROM episodes WHERE bank_id = %s AND task = %s",
                            (self.bank_id, task),
                        )
                        cnt = cur.fetchone()["cnt"]
                        logger.info(
                            "load_context: no episode matched (bank=%s, task=%s, topics=%s, prefix=%s); "
                            "total episodes for bank+task: %d",
                            self.bank_id, task, topic_ids, topic_prefix, cnt,
                        )
                        return None

                    episode_id = row["id"]
                    logger.debug("load_context: found episode %s (bank=%s, task=%s)", episode_id, self.bank_id, task)

                    # 2. Load topics from that episode
                    cur.execute(
                        """
                        SELECT t.id, t.type, t.description FROM episode_topics et
                        JOIN topics t ON t.bank_id = et.bank_id AND t.id = et.topic_id
                        WHERE et.bank_id = %s AND et.episode_id = %s
                        """,
                        (self.bank_id, episode_id),
                    )
                    topics = [
                        Topic(id=row["id"], type=row["type"], description=row["description"])
                        for row in cur.fetchall()
                    ]

                    # 3. Load observations at their LATEST version (not the pinned version)
                    cur.execute(
                        """
                        SELECT DISTINCT ON (o.id) o.id, o.type, o.content, o.version
                        FROM episode_observations eo
                        JOIN observations o ON o.bank_id = eo.bank_id AND o.id = eo.observation_id
                        WHERE eo.bank_id = %s AND eo.episode_id = %s
                          AND o.op_type != 'DELETE'
                        ORDER BY o.id, o.version DESC
                        """,
                        (self.bank_id, episode_id),
                    )
                    items_by_id: dict[str, dict] = {}
                    for row in cur.fetchall():
                        items_by_id[row["id"]] = {
                            "section": row["type"],
                            "content": row["content"],
                            "version": row["version"],
                        }

                    # 4. Load observation-topic bindings for those latest versions
                    if items_by_id:
                        cur.execute(
                            """
                            SELECT ot.observation_id, ot.topic_id, ot.observation_occurrence, ot.version_occurrence
                            FROM observation_topics ot
                            WHERE ot.bank_id = %s
                              AND (ot.observation_id, ot.observation_version) IN (
                                  SELECT o.id, MAX(o.version)
                                  FROM episode_observations eo
                                  JOIN observations o ON o.bank_id = eo.bank_id AND o.id = eo.observation_id
                                  WHERE eo.bank_id = %s AND eo.episode_id = %s AND o.op_type != 'DELETE'
                                  GROUP BY o.id
                              )
                            """,
                            (self.bank_id, self.bank_id, episode_id),
                        )
                        item_topics: dict[str, list[str]] = {}
                        for row in cur.fetchall():
                            observation_id = row["observation_id"]
                            if observation_id not in item_topics:
                                item_topics[observation_id] = []
                            item_topics[observation_id].append(row["topic_id"])

                        # Build ContextMemory sections
                        sections: dict[str, list[Item]] = {
                            "context_roadmap": [],
                            "context_understanding": [],
                            "domain_constants": [],
                            "actions": [],
                            "parsing_schema": [],
                            "reusable_results": [],
                        }

                        for item_id, item_data in items_by_id.items():
                            section = item_data["section"]
                            if section not in sections:
                                section = "context_understanding"  # fallback

                            item = Item(
                                id=item_id,
                                content=item_data["content"],
                                topic_ids=item_topics.get(item_id, []),
                            )
                            sections[section].append(item)

                        # Build ContextMemory
                        ctx = ContextMemory(topics=topics)
                        for section_name, items in sections.items():
                            setattr(ctx, section_name, items)

                        total_items = sum(len(items) for items in sections.values())
                        logger.debug("load_context: loaded %d topics, %d observations from episode %s", len(topics), total_items, episode_id)

                        return ctx

                    # No observations - return empty ContextMemory with topics
                    logger.debug("load_context: loaded %d topics, 0 observations from episode %s", len(topics), episode_id)
                    return ContextMemory(topics=topics)
        except Exception:
            logger.warning("load_context failed (bank=%s, task=%s)", self.bank_id, task, exc_info=True)
            return None
