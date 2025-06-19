import psycopg2
import numpy as np
import os
import re

class QueryEnv:
    """
    Environnement d'apprentissage par renforcement pour l'optimisation des requêtes SQL (benchmark JOB)
    """

    def __init__(self, db_config):
        self.db_config = db_config
        self.connection = psycopg2.connect(**db_config)

        
        self.actions = [
            "no_op",               
            "add_index",           
            "rewrite_join",        
            "add_where",           
            "remove_subquery",     
            "add_order_by",         
            "rewrite_to_explicit_joins",
            "apply_index_on_title",
            "apply_pg_trgm_on_title"
        ]

        self.state = None
        self.query = None
        self.current_query_idx = 0
        self.history = []

        
        self.job_queries = [
            """SELECT title FROM title WHERE production_year IS NOT NULL""",
            """SELECT title FROM title WHERE production_year IS NOT NULL""",
            """SELECT COUNT(*) AS war_movies FROM title WHERE title ILIKE '%war%'""",
            """SELECT COUNT(*) AS war_movies FROM title WHERE title ILIKE '%war%'""",
            """SELECT t.title, cn.name FROM title AS t JOIN movie_companies AS mc ON mc.movie_id = t.id JOIN company_name AS cn ON mc.company_id = cn.id""",
            """SELECT t.title, cn.name FROM title AS t JOIN movie_companies AS mc ON mc.movie_id = t.id JOIN company_name AS cn ON mc.company_id = cn.id""",
            """SELECT MIN(cn.name) AS from_company, MIN(lt.link) AS movie_link_type, MIN(t.title) AS non_polish_sequel_movie FROM company_name AS cn, company_type AS ct, keyword AS k, link_type AS lt, movie_companies AS mc, movie_keyword AS mk, movie_link AS ml, title AS t WHERE cn.country_code !='[pl]' AND (cn.name LIKE '%Film%' OR cn.name LIKE '%Warner%') AND ct.kind ='production companies' AND k.keyword ='sequel' AND lt.link LIKE '%follow%' AND mc.note IS NULL AND t.production_year BETWEEN 1950 AND 2000 AND lt.id = ml.link_type_id AND ml.movie_id = t.id AND t.id = mk.movie_id AND mk.keyword_id = k.id AND t.id = mc.movie_id AND mc.company_type_id = ct.id AND mc.company_id = cn.id AND ml.movie_id = mk.movie_id AND ml.movie_id = mc.movie_id AND mk.movie_id = mc.movie_id""",
            """SELECT MIN(cn.name) AS from_company, MIN(lt.link) AS movie_link_type, MIN(t.title) AS non_polish_sequel_movie FROM company_name AS cn, company_type AS ct, keyword AS k, link_type AS lt, movie_companies AS mc, movie_keyword AS mk, movie_link AS ml, title AS t WHERE cn.country_code !='[pl]' AND (cn.name LIKE '%Film%' OR cn.name LIKE '%Warner%') AND ct.kind ='production companies' AND k.keyword ='sequel' AND lt.link LIKE '%follow%' AND mc.note IS NULL AND t.production_year BETWEEN 1950 AND 2000 AND lt.id = ml.link_type_id AND ml.movie_id = t.id AND t.id = mk.movie_id AND mk.keyword_id = k.id AND t.id = mc.movie_id AND mc.company_type_id = ct.id AND mc.company_id = cn.id AND ml.movie_id = mk.movie_id AND ml.movie_id = mc.movie_id AND mk.movie_id = mc.movie_id"""
        ]

        self.query_names = [
            "req1.sql", "req1.sql", "req2.sql", "req2.sql", "req3.sql", "req3.sql", "req4.sql", "req4.sql"
        ]

    def reset(self):
        self.query = self.job_queries[self.current_query_idx]
        self.current_query_name = self.query_names[self.current_query_idx]
        self.current_query_idx = (self.current_query_idx + 1) % len(self.job_queries)
        self.history = []

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
            print(f"[RESET ERROR] Query {self.current_query_name} failed:\n{e}")
            self.state = np.zeros(5, dtype=np.float32)

        return self.state

    def _count_indexes(self, table_name):
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) FROM pg_indexes WHERE tablename = %s", (table_name,))
                return cursor.fetchone()[0]
        except Exception as e:
            print(f"[INDEX COUNT ERROR] {e}")
            return 0

    def _has_alias(self, query, alias):
        pattern = rf"\b{alias}\."
        return re.search(pattern, query) is not None

    def step(self, action_idx):
        action = self.actions[action_idx]
        self.history.append(action)  # Historique

        reward = -1000
        done = False

        try:
            with self.connection.cursor() as cursor:
                # === Appliquer l’action ===
                if action == "no_op":
                    pass  # ne rien faire

                elif action == "add_index":
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
                        if "WHERE" not in self.query.upper():
                            self.query += " WHERE mi.info_type_id = 3"
                        else:
                            self.query += " AND mi.info_type_id = 3"

                elif action == "remove_subquery":
                    self.query = " ".join(
                        line for line in self.query.split()
                        if "SELECT" not in line.upper() or line.upper() == "SELECT"
                    )

                elif action == "add_order_by":
                    if "ORDER BY" not in self.query.upper():
                        self.query += " ORDER BY 1"
                
                elif action == "rewrite_to_explicit_joins":
                    self.query = re.sub(r"FROM (.+?) WHERE", 
                    lambda m: "FROM " + ", ".join(part.strip() for part in m.group(1).split(",")) + " WHERE",
                    self.query, flags=re.IGNORECASE)

                elif action == "apply_index_on_title":
                    cursor.execute("CREATE INDEX IF NOT EXISTS idx_title_title ON title(title);")
                    self.connection.commit()

                elif action == "apply_pg_trgm_on_title":
                    cursor.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")
                    cursor.execute("CREATE INDEX IF NOT EXISTS trgm_idx_title_title ON title USING gin(title gin_trgm_ops);")
                    self.connection.commit()


                # Mesurer performance
                cursor.execute("EXPLAIN ANALYZE " + self.query)
                result = cursor.fetchall()

                execution_time = 1000
                for line in result:
                    if "Execution Time" in line[0]:
                        time_part = line[0].split(":")[1]
                        execution_time = float(time_part.strip().split()[0])
                        break

                reward = -execution_time
                done = execution_time < 100  # seuil arbitraire

        except Exception as e:
            print(f"[STEP ERROR] Action: {action} → {e}")
            self.connection.rollback()

        new_state = self.reset()
        return new_state, reward, done

    def close(self):
        self.connection.close()

    def get_query_filename(self):
        return self.current_query_name
