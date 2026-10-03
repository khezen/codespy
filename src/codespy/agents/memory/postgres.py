"""PostgreSQL-backed episode persistence for Hippocampus memory."""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING

from psycopg import sql
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

if TYPE_CHECKING:
    from codespy.agents.memory.hippocampus.context_memory import ContextMemory
    from codespy.agents.memory.hippocampus.episode import Episode
    from codespy.agents.memory.recall import RecallRecord

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
        schema: str | None = None,
        min_size: int = 1,
        max_size: int = 4,
    ):
        """Create store with a psycopg ConnectionPool, scoped to a bank.

        Args:
            conninfo: PostgreSQL connection string
            bank_id: Identifier for the bank
            schema: PostgreSQL schema name. When set, CREATE SCHEMA IF NOT
                    EXISTS is run and search_path is set for every pooled
                    connection. Each memory type uses its own schema
                    (e.g. "episodic", "semantic"). None = server default (public).
            min_size: Minimum connections in pool
            max_size: Maximum connections in pool
        """
        self.bank_id = bank_id
        self._schema = schema
        # Inject search_path into the connection string so every pooled
        # connection targets the right schema automatically.
        if schema:
            from urllib.parse import quote_plus
            sep = "&" if "?" in conninfo else "?"
            conninfo = f"{conninfo}{sep}options=-csearch_path%3D{quote_plus(schema)}%2Cpublic"
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
                # Create the target schema if it doesn't exist yet
                if self._schema:
                    cur.execute(
                        sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(
                            sql.Identifier(self._schema)
                        )
                    )

                # Extensions — explicitly in public so they're accessible
                # from any schema's search_path (episodic, semantic, etc.)
                cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm SCHEMA public")

                # Ensure the semantic schema exists (used by Cerebral/MemoryEngine)
                cur.execute("CREATE SCHEMA IF NOT EXISTS semantic")

                # pgvector extension — required by MemoryEngine for semantic search.
                # Wrapped in a nested transaction (savepoint): pgvector may not
                # be installed on all PostgreSQL instances; episodic memory
                # works fine without it.
                try:
                    with conn.transaction():
                        cur.execute("CREATE EXTENSION IF NOT EXISTS vector SCHEMA public")
                except Exception:
                    logger.debug("pgvector extension not available — semantic search requires it")

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

                # Episode observations junction (dropped - no longer needed)
                cur.execute("DROP TABLE IF EXISTS episode_observations")

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

                # Check for legacy recalls table (without 'id' column) and drop it
                cur.execute("""
                    SELECT column_name FROM information_schema.columns
                    WHERE table_name = 'recalls' AND table_schema = COALESCE(current_schema(), 'public')
                """)
                existing_columns = {row[0] for row in cur.fetchall()}
                if existing_columns and 'id' not in existing_columns:
                    logger.warning("Dropping legacy recalls table (missing 'id' column)")
                    cur.execute("DROP TABLE IF EXISTS recalls")

                # Recalls table: one row per Prefrontal recall (monitoring only;
                # never read back into memory, never retained into Cerebral).
                # Redesigned: run-scoped with optional episode_id (NULL for run-level load).
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS recalls (
                        bank_id VARCHAR(64) NOT NULL REFERENCES banks(id) ON DELETE CASCADE,
                        id UUID NOT NULL,
                        run_id VARCHAR(48) NOT NULL,
                        episode_id UUID NULL,
                        task VARCHAR(64) NOT NULL,
                        kind VARCHAR(8) NOT NULL,
                        timestamp TIMESTAMPTZ NOT NULL,
                        query TEXT NOT NULL DEFAULT '',
                        reach VARCHAR(8) NOT NULL,
                        reflects INT NOT NULL DEFAULT 0,
                        status VARCHAR(8) NOT NULL,
                        text TEXT NOT NULL DEFAULT '',
                        model TEXT NOT NULL DEFAULT '',
                        llm_calls INT NOT NULL DEFAULT 0,
                        input_tokens INT NOT NULL DEFAULT 0,
                        output_tokens INT NOT NULL DEFAULT 0,
                        input_cost DOUBLE PRECISION NOT NULL DEFAULT 0,
                        output_cost DOUBLE PRECISION NOT NULL DEFAULT 0,
                        latency_ms INT NOT NULL DEFAULT 0,
                        details JSONB NOT NULL DEFAULT '{}'::jsonb,
                        PRIMARY KEY (bank_id, id),
                        FOREIGN KEY (bank_id, episode_id) REFERENCES episodes(bank_id, id) ON DELETE CASCADE
                    )
                """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_recalls_run ON recalls (bank_id, run_id)
                """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_recalls_time
                        ON recalls (bank_id, timestamp DESC)
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
        3. Upsert topics, insert observations (versioned), insert junctions, insert artifacts
        """
        # Import here to avoid circular imports
        from codespy.agents.memory.hippocampus.context_memory import (
            TOMBSTONE_TYPES,
            Mutation,
            Observation,
            OpType,
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
                # Build map of observation_id -> mutation for quick lookup
                mutation_map: dict[str, Mutation] = {}
                for mut in episode.mutations:
                    mutation_map[mut.observation_id] = mut

                # Get all current observations from context_memory
                current_observations: dict[str, tuple[str, Observation]] = {}  # observation_id -> (type, observation)
                for sec in episode.context_memory.section_names():
                    for obs in getattr(episode.context_memory, sec):
                        current_observations[obs.id] = (sec, obs)

                # Track observations we've processed to detect DELETEs
                processed_observation_ids: set[str] = set()

                # Process observations in current context memory
                for observation_id, (type, obs) in current_observations.items():
                    processed_observation_ids.add(observation_id)
                    mutation = mutation_map.get(observation_id)

                    # Determine if this is a new observation (ADD) or existing (REPLACE/inherited)
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
                            (self.bank_id, observation_id),
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
                            (self.bank_id, observation_id),
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
                                observation_id,
                                version,
                                type,
                                obs.content,
                                str(episode.id),
                                step,
                                op_type,
                                previous_content,
                                0,
                            ),
                        )

                        # Insert observation_topics for this version
                        for topic_id in obs.topic_ids:
                            # Count occurrences
                            cur.execute(
                                """
                                SELECT observation_occurrence, version_occurrence
                                FROM observation_topics
                                WHERE bank_id = %s AND observation_id = %s AND topic_id = %s
                                ORDER BY observation_version DESC
                                LIMIT 1
                                """,
                                (self.bank_id, observation_id, topic_id),
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
                                    observation_id,
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
                            (self.bank_id, observation_id),
                        )
                        row = cur.fetchone()
                        existing_version = row["max_ver"] if row and row["max_ver"] else 1

                        # Increment observation_occurrence for inherited observations
                        for topic_id in obs.topic_ids:
                            cur.execute(
                                """
                                SELECT observation_occurrence
                                FROM observation_topics
                                WHERE bank_id = %s AND observation_id = %s AND topic_id = %s
                                AND observation_version = %s
                                """,
                                (self.bank_id, observation_id, topic_id, existing_version),
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
                                        observation_id,
                                        topic_id,
                                        existing_version,
                                    ),
                                )

                # 6. Write tombstones (DELETE / EVICT) for observations not in
                #    current context. One per id: the id's last mutation decides.
                #    Skip when next_ver == 1: the id was never persisted, so no
                #    orphan tombstone row should be written.
                for mutation in mutation_map.values():
                    if mutation.type in TOMBSTONE_TYPES and mutation.observation_id:
                        if mutation.observation_id not in processed_observation_ids:
                            # Get next version number
                            cur.execute(
                                """
                                SELECT COALESCE(MAX(version), 0) + 1 as next_ver
                                FROM observations
                                WHERE bank_id = %s AND id = %s
                                """,
                                (self.bank_id, mutation.observation_id),
                            )
                            row = cur.fetchone()
                            next_ver = row["next_ver"] if row else 1

                            # Skip if the id was never saved (next_ver == 1)
                            if next_ver == 1:
                                continue

                            cur.execute(
                                """
                                INSERT INTO observations
                                    (bank_id, id, version, type, content, episode_id, step, op_type, previous_content, ordinal)
                                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                                """,
                                (
                                    self.bank_id,
                                    mutation.observation_id,
                                    next_ver,
                                    mutation.section,
                                    None,  # content is NULL for tombstones
                                    str(episode.id),
                                    mutation.step,
                                    mutation.type.value,  # a TOMBSTONE_TYPES member
                                    mutation.previous_content,
                                    0,
                                ),
                            )

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

                # 8. Insert recalls (Prefrontal monitoring) - now with episode_id set
                self._insert_recalls(
                    cur, episode.recalls or [], run_id=episode.run_id, episode_id=episode.id, task=episode.task
                )

                conn.commit()

        logger.info(
            "save_episode: persisted episode %s (bank=%s, task=%s, topics=%d, observations=%d, recalls=%d)",
            episode.id, self.bank_id, episode.task,
            len(episode.context_memory.topics),
            len(episode.context_memory.all_observations()),
            len(episode.recalls or []),
        )

    def load_context(
        self,
        task: str,
        topic_ids: list[str] | None = None,
        topic_prefix: str | None = None,
    ) -> ContextMemory | None:
        """Load context for a new episode: latest observation versions from all episodes.

        Loads every observation from all prior episodes whose latest version
        is not a tombstone (DELETE or EVICT) and bound to any of the specified topics.

        Args:
            task: Task name to filter by.
            topic_ids: Exact topic IDs to match (ANY).
            topic_prefix: Prefix match on topic ID (e.g. repo_full_name
                matches 'owner/repo', 'owner/repo/pkg', etc.).
                Mutually exclusive with topic_ids.

        Returns:
            ContextMemory (topics + observations) or None if no filter provided
            or no observations match. No mutations, artifacts, or episode
            metadata are loaded.
        """
        # Import here to avoid circular imports
        from codespy.agents.memory.hippocampus.context_memory import (
            TOMBSTONE_TYPES,
            ContextMemory,
            Observation,
            Topic,
        )

        if not topic_ids and not topic_prefix:
            logger.debug("load_context: no topic filter provided, returning None")
            return None

        # Single source of truth for which op_types are tombstones.
        tombstone_op_types = sorted(t.value for t in TOMBSTONE_TYPES)

        try:
            with self._pool.connection() as conn:
                conn.row_factory = dict_row
                with conn.cursor() as cur:
                    # 1. Find observations whose latest version is not a tombstone
                    #    and that are bound to matching topics
                    params: list = [self.bank_id, task]

                    if topic_ids:
                        filter_clause = sql.SQL("ot.topic_id = ANY(%s)")
                        params.append(list(topic_ids))
                    else:
                        # Escape LIKE wildcards so 'owner/my_repo' does not
                        # match 'owner/myXrepo/...'. Backslash is the default
                        # LIKE escape character in PostgreSQL.
                        escaped = (
                            topic_prefix.replace("\\", "\\\\")
                            .replace("%", "\\%")
                            .replace("_", "\\_")
                        )
                        filter_clause = sql.SQL("(ot.topic_id = %s OR ot.topic_id LIKE %s)")
                        params.extend([topic_prefix, f"{escaped}/%"])
                    params.append(tombstone_op_types)

                    # Query: get latest non-tombstone observation versions
                    # - Match topics via observation_topics on ANY version
                    # - Require version = MAX(version) over ALL versions of the id
                    # - Require the latest version to be live: op_type not in
                    #   TOMBSTONE_TYPES, and content present. The content check
                    #   also skips tombstones of a type this code does not know,
                    #   instead of failing the whole load on a NULL content.
                    # - Order oldest-first by the timestamp of the episode that
                    #   wrote the latest version. evict() drops the earliest
                    #   positions first on ties, so the oldest facts are the
                    #   ones evicted (and EVICT-tombstoned), not arbitrary ones.
                    query = sql.SQL("""
                        SELECT * FROM (
                            SELECT DISTINCT ON (o.id)
                                o.id,
                                o.version,
                                o.type as section,
                                o.content,
                                o.episode_id,
                                e.timestamp
                            FROM observations o
                            JOIN episodes e ON e.bank_id = o.bank_id AND e.id = o.episode_id
                            WHERE o.bank_id = %s
                              AND e.task = %s
                              AND EXISTS (
                                  SELECT 1 FROM observation_topics ot
                                  WHERE ot.bank_id = o.bank_id AND ot.observation_id = o.id
                                    AND {filter_clause}
                              )
                              AND o.version = (
                                  SELECT MAX(o2.version) FROM observations o2
                                  WHERE o2.bank_id = o.bank_id AND o2.id = o.id
                              )
                              AND o.op_type <> ALL(%s)
                              AND o.content IS NOT NULL
                            ORDER BY o.id, o.version DESC
                        ) latest
                        ORDER BY latest.timestamp ASC, latest.id ASC
                    """).format(filter_clause=filter_clause)

                    cur.execute(query, params)
                    rows = cur.fetchall()

                    if not rows:
                        logger.debug("load_context: no observations found (bank=%s, task=%s)", self.bank_id, task)
                        return None

                    # 2. Load topic bindings of exactly the versions loaded above
                    cur.execute(
                        """
                        SELECT ot.observation_id, ot.topic_id
                        FROM observation_topics ot
                        JOIN unnest(%s::text[], %s::int[]) AS v(id, version)
                          ON v.id = ot.observation_id AND v.version = ot.observation_version
                        WHERE ot.bank_id = %s
                        """,
                        (
                            [row["id"] for row in rows],
                            [row["version"] for row in rows],
                            self.bank_id,
                        ),
                    )

                    obs_topics_map: dict[str, list[str]] = {}
                    for row in cur.fetchall():
                        obs_id = row["observation_id"]
                        if obs_id not in obs_topics_map:
                            obs_topics_map[obs_id] = []
                        obs_topics_map[obs_id].append(row["topic_id"])

                    # 3. Build ContextMemory
                    ctx = ContextMemory(topics=[])

                    # Add observations to sections
                    for row in rows:
                        section = row["section"]
                        if section not in ctx.section_names():
                            section = "context_understanding"  # fallback

                        obs = Observation(
                            id=row["id"],
                            content=row["content"],
                            topic_ids=obs_topics_map.get(row["id"], []),
                        )
                        ctx.section(section).append(obs)

                    # 4. Load Topics referenced by these observations
                    all_topic_ids = set()
                    for obs in ctx.all_observations():
                        all_topic_ids.update(obs.topic_ids)

                    if all_topic_ids:
                        cur.execute(
                            """
                            SELECT t.id, t.type, t.description
                            FROM topics t
                            WHERE t.bank_id = %s AND t.id = ANY(%s)
                            """,
                            (self.bank_id, list(all_topic_ids)),
                        )
                        for row in cur.fetchall():
                            ctx.topics.append(
                                Topic(id=row["id"], type=row["type"], description=row["description"])
                            )

                    total_observations = len(ctx.all_observations())
                    logger.debug(
                        "load_context: loaded %d topics, %d observations (bank=%s, task=%s)",
                        len(ctx.topics), total_observations, self.bank_id, task
                    )

                    return ctx

        except Exception:
            logger.warning(
                "load_context failed (bank=%s, task=%s)",
                self.bank_id, task, exc_info=True
            )
            return None

    def _insert_recalls(
        self,
        cur,
        records: list[RecallRecord],
        *,
        run_id: str,
        episode_id: uuid.UUID | None,
        task: str,
    ) -> None:
        """Insert recall records with run_id, episode_id, and task filled in.

        Args:
            cur: Database cursor
            records: List of RecallRecord to insert
            run_id: Run identifier
            episode_id: Episode UUID (None for run-level recalls)
            task: Task name (consumer task or "review" for run-level)
        """
        import uuid as uuid_module

        for rec in records:
            # Ensure the record has the correct run_id, episode_id, and task
            rec.run_id = run_id
            rec.task = task
            cur.execute(
                """
                INSERT INTO recalls
                    (bank_id, id, run_id, episode_id, task, kind, timestamp, query, reach,
                     reflects, status, text, model, llm_calls, input_tokens,
                     output_tokens, input_cost, output_cost, latency_ms, details)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (bank_id, id) DO NOTHING
                """,
                (
                    self.bank_id,
                    str(rec.id) if rec.id else str(uuid_module.uuid4()),
                    rec.run_id,
                    str(episode_id) if episode_id else None,
                    rec.task,
                    rec.kind,
                    rec.timestamp,
                    rec.query,
                    rec.reach,
                    rec.reflects,
                    rec.status,
                    rec.text,
                    rec.model,
                    rec.llm_calls,
                    rec.input_tokens,
                    rec.output_tokens,
                    rec.input_cost,
                    rec.output_cost,
                    rec.latency_ms,
                    Jsonb(rec.details),
                ),
            )

    def save_recalls(
        self,
        records: list[RecallRecord],
        *,
        run_id: str,
        episode_id: uuid.UUID | None = None,
        task: str = "",
    ) -> None:
        """Persist recall records for a run or episode.

        Args:
            records: List of RecallRecord to persist
            run_id: Run identifier
            episode_id: Optional episode UUID (None for run-level recalls)
            task: Task name (consumer task or "review" for run-level load)
        """
        if not records:
            return

        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                # Ensure bank exists
                cur.execute(
                    "INSERT INTO banks (id) VALUES (%s) ON CONFLICT (id) DO NOTHING",
                    (self.bank_id,),
                )
                self._insert_recalls(cur, records, run_id=run_id, episode_id=episode_id, task=task)
                conn.commit()

        logger.info(
            "save_recalls: persisted %d recalls (bank=%s, run_id=%s, task=%s)",
            len(records), self.bank_id, run_id, task,
        )
