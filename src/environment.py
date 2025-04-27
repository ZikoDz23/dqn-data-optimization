import psycopg2
import numpy as np

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
        
        # Sample JOB queries
        self.job_queries = [
            """SELECT COUNT(*) FROM title t 
               JOIN movie_info mi ON t.id = mi.movie_id""",
            """SELECT COUNT(*) FROM cast_info ci
               JOIN movie_info mi ON ci.movie_id = mi.movie_id"""
        ]

    def reset(self):
        """
        Reset environment and return initial state
        """
        self.query = self.job_queries[self.current_query_idx]
        self.current_query_idx = (self.current_query_idx + 1) % len(self.job_queries)
        
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
        """
        Count indexes on specified table
        """
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
                        
                reward = -execution_time
                done = execution_time < 100

        except Exception as e:
            print(f"Error in step: {e}")
            self.connection.rollback()
            
        new_state = self.reset()
        return new_state, reward, done

    def close(self):
        """
        Close database connection
        """
        self.connection.close()
