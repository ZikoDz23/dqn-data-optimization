import psycopg2
import numpy as np
import os
import re

class QueryEnv:
    """
    RL Environment for SQL query optimization using JOB benchmark
    """
    def __init__(self, db_config):
        self.db_config = db_config
        self.connection = psycopg2.connect(**db_config)

        self.actions = ["add_index", "rewrite_join", "add_where", "remove_subquery"]
        self.state = None
        self.query = None
        self.current_query_idx = 0

        queries_path = os.path.join("JOB", "queries", "join-order-benchmark")
        self.job_queries = []
        self.query_names = []  # ← à mettre dans __init__
        self.current_query_name = ""  # ← nom actuel du fichier


        for filename in os.listdir(queries_path):
            if filename.endswith(".sql"):
                with open(os.path.join(queries_path, filename), "r", encoding="utf-8") as f:
                    sql = f.read().strip().replace("\n", " ").replace(";", "")
                    if sql.lower().startswith("select"):  # Ne garder que les requêtes SELECT
                        self.job_queries.append(sql)
                        self.query_names.append(filename)

        if not self.job_queries:
            raise ValueError("Aucune requête SELECT JOB trouvée dans le dossier spécifié.")

    def reset(self):
        """
        Reset environment and return initial state
        """
        self.query = self.job_queries[self.current_query_idx]
        self.current_query_idx = (self.current_query_idx + 1) % len(self.job_queries)
        self.current_query_name = self.query_names[self.current_query_idx]


        try:
            with self.connection.cursor() as cursor:
                cursor.execute("EXPLAIN (FORMAT JSON) " + self.query)
                plan = cursor.fetchone()[0]
                plan_dict = plan[0]

                self.state = np.array([
                    len(self.query),
                    len(plan_dict["Plan"].get("Plans", [])),
                    plan_dict["Plan"]["Plan Rows"],
                    self._count_indexes("title"),
                    self._count_indexes("movie_info")
                ], dtype=np.float32)

        except Exception as e:
            print(f"Error in reset: {e}")
            self.state = np.zeros(5, dtype=np.float32)

        return self.state

    def _count_indexes(self, table_name):
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(
                    "SELECT COUNT(*) FROM pg_indexes WHERE tablename = %s",
                    (table_name,)
                )
                return cursor.fetchone()[0]
        except Exception as e:
            print(f"Error counting indexes: {e}")
            return 0

    def _has_alias(self, query, alias):
        """
        Vérifie si un alias (ex: 'mi.') est bien présent dans la requête.
        """
        pattern = rf"\b{alias}\."
        return re.search(pattern, query) is not None

    def step(self, action_idx):
        """
        Execute action and return (new_state, reward, done)
        """
        action = self.actions[action_idx]
        reward = -1000
        done = False

        try:
            with self.connection.cursor() as cursor:
                if action == "add_index":
                    cursor.execute("""
                        CREATE INDEX IF NOT EXISTS idx_movie_info_movie_id 
                        ON movie_info(movie_id)
                    """)
                    self.connection.commit()

                elif action == "rewrite_join":
                    self.query = self.query.replace(
                        "JOIN movie_info mi ON t.id = mi.movie_id",
                        "JOIN movie_info mi ON mi.movie_id = t.id"
                    )

                elif action == "add_where":
                    if self._has_alias(self.query, "mi"):
                        if "WHERE" not in self.query:
                            self.query += " WHERE mi.info_type_id = 3"
                        else:
                            self.query += " AND mi.info_type_id = 3"

                elif action == "remove_subquery":
                    self.query = " ".join(
                        line for line in self.query.split()
                        if "SELECT" not in line.upper() or line.upper() == "SELECT"
                    )

                cursor.execute("EXPLAIN ANALYZE " + self.query)
                result = cursor.fetchall()

                execution_time = 1000
                for line in result:
                    if "actual time=" in line[0]:
                        time_part = line[0].split("actual time=")[1]
                        execution_time = float(time_part.split("..")[0])
                        break

                # Timeout soft
                if execution_time > 5000:
                    print("Query took too long, skipping...")
                    return self.state, -1000, True

                reward = -execution_time
                done = execution_time < 100

        except Exception as e:
            print(f"Error in step: {e}")
            self.connection.rollback()

        new_state = self.reset()
        return new_state, reward, done

    def close(self):
        self.connection.close()
    def get_query_filename(self):
        return self.current_query_name

