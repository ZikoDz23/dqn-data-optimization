import psycopg2
import json
from ..utils.plan_parser import PlanParser

class CostInterface:
    def __init__(self, db_config):
        self.db_config = db_config
        self.conn = None
        self.parser = PlanParser()

    def connect(self):
        try:
            self.conn = psycopg2.connect(
                dbname=self.db_config['dbname'],
                user=self.db_config['user'],
                password=self.db_config['password'],
                host=self.db_config['host'],
                port=self.db_config['port']
            )
            self.conn.autocommit = True
            # Enforce join order optimization by disabling reordering
            with self.conn.cursor() as cur:
                cur.execute(f"SET join_collapse_limit = {self.db_config.get('join_collapse_limit', 1)};")
                cur.execute("SET enable_nestloop = off;") # Optional: heuristics
        except Exception as e:
            print(f"Failed to connect to DB: {e}")

    def close(self):
        if self.conn:
            self.conn.close()

    def estimate_cost(self, join_order, sql_query_template):
        """
        Estimate cost/runtime for a specific join order.
        args:
            join_order: list of tables in order e.g. ['t1', 't2', 't3']
            sql_query_template: the original query string
        """
        if not self.conn:
            self.connect()

        hint = self._generate_leading_hint(join_order)
        final_query = f"{hint}\n{sql_query_template}"
        
        # We use EXPLAIN (FORMAT JSON) to get cost without executing
        # This is much faster than ANALYZE which actually runs the query.
        explain_cmd = f"EXPLAIN (FORMAT JSON) {final_query}"
        
        try:
            with self.conn.cursor() as cur:
                import time
                start_t = time.time()
                print(f"DEBUG: Estimating cost for join_order={join_order}...")
                cur.execute(explain_cmd)
                print(f"DEBUG: EXPLAIN finished in {time.time() - start_t:.4f}s")
                result = cur.fetchone()[0] # JSON output
                
            parsed = self.parser.parse_explain_json(result)
            # Use estimated total_cost instead of actual execution_time
            return parsed['total_cost'] if parsed['total_cost'] else 100000.0
        except Exception as e:
            print(f"Query execution failed: {e}")
            return 100000.0 # High penalty for failure

    def _generate_leading_hint(self, join_order):
        """
        Generates /*+ Leading(t1 t2 t3) */ hint.
        Assumes join_order is a list of tables or aliases.
        """
        # join_order might be a flat list ['t1', 't2'] or structure
        # User example: Leading(title movie_info cast_info)
        # We assume flat list for now corresponding to the sequence
        joined_str = " ".join(join_order)
        return f"/*+ Leading({joined_str}) */"
