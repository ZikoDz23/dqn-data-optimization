import psycopg2
import numpy as np

class QueryEnv:
    """
    Environnement RL pour l'optimisation des requêtes SQL.
    """
    def __init__(self, db_config):
        """
        Initialise l'environnement avec la configuration de la base de données.
        """
        self.db_config = db_config
        self.connection = psycopg2.connect(**db_config)
        self.actions = ["add_index", "rewrite_join", "add_where", "remove_subquery"]
        self.state = None
        self.query = None

    def reset(self):
        """
        Réinitialise l'environnement et retourne l'état initial.
        """
        self.query = "SELECT l_returnflag, l_linestatus, SUM(l_quantity) FROM lineitem WHERE l_shipdate <= '1998-12-01' GROUP BY l_returnflag, l_linestatus"
        with self.connection.cursor() as cursor:
            cursor.execute("EXPLAIN (FORMAT JSON) " + self.query)
            plan = cursor.fetchone()[0]
            plan_dict = plan[0]
            self.state = np.array([
                len(self.query),
                len(plan_dict["Plan"].get("Plans", [])),
                plan_dict["Plan"]["Plan Rows"],
                self._count_indexes()
            ], dtype=np.float32)
        return self.state

    def _count_indexes(self):
        """
        Compte le nombre d'index sur la table lineitem.
        """
        with self.connection.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM pg_indexes WHERE tablename = 'lineitem'")
            return cursor.fetchone()[0]

    def step(self, action_idx):
        action = self.actions[action_idx]
        old_query = self.query

        with self.connection.cursor() as cursor:
            if action == "add_index":
                try:
                    cursor.execute("CREATE INDEX IF NOT EXISTS idx_shipdate ON lineitem(l_shipdate)")
                    self.connection.commit()
                except Exception as e:
                    print(f"Erreur lors de la création de l'index: {e}")
                    self.connection.rollback()
            elif action == "rewrite_join":
                self.query = "SELECT l_returnflag, l_linestatus, SUM(l_quantity) FROM lineitem WHERE l_shipdate <= '1998-12-01' GROUP BY l_returnflag, l_linestatus"
            elif action == "add_where":
                # Ajouter la condition dans la clause WHERE
                if "WHERE" in self.query:
                    # Trouver la position de "GROUP BY"
                    group_by_index = self.query.find("GROUP BY")
                    if group_by_index != -1:
                        # Insérer "AND l_quantity > 10" avant "GROUP BY"
                        self.query = self.query[:group_by_index].strip() + " AND l_quantity > 10 " + self.query[group_by_index:]
                    else:
                        # Si "GROUP BY" n'existe pas, ajouter "AND l_quantity > 10" à la fin
                        self.query += " AND l_quantity > 10"
                else:
                    # Si la clause WHERE n'existe pas, l'ajouter avant "GROUP BY"
                    group_by_index = self.query.find("GROUP BY")
                    if group_by_index != -1:
                        self.query = self.query[:group_by_index].strip() + " WHERE l_quantity > 10 " + self.query[group_by_index:]
                    else:
                        # Si "GROUP BY" n'existe pas, ajouter "WHERE l_quantity > 10" à la fin
                        self.query += " WHERE l_quantity > 10"
            elif action == "remove_subquery":
                self.query = "SELECT l_returnflag, l_linestatus, SUM(l_quantity) FROM lineitem GROUP BY l_returnflag, l_linestatus"

            try:
                cursor.execute("EXPLAIN ANALYZE " + self.query)
                execution_plan = cursor.fetchall()
                if execution_plan and len(execution_plan[-1]) > 0:
                    last_line = execution_plan[-1][0]
                    if "actual time=" in last_line:
                        execution_time = float(last_line.split("actual time=")[1].split("..")[0])
                    else:
                        execution_time = 1000  # Valeur par défaut si le temps d'exécution n'est pas trouvé
                else:
                    execution_time = 1000  # Valeur par défaut si le plan est vide
            except Exception as e:
                print(f"Erreur lors de l'exécution de la requête : {self.query}. Erreur : {e}")
                self.connection.rollback()
                execution_time = 1000

        reward = -execution_time
        self.state = self.reset()
        done = execution_time < 10
        return self.state, reward, done

    def close(self):
        """
        Ferme la connexion à la base de données.
        """
        self.connection.close()