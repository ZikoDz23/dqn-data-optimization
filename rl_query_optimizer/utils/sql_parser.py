import sqlparse
from sqlparse.sql import IdentifierList, Identifier, Where, Comparison
from sqlparse.tokens import Keyword, DML

class SQLParser:
    def __init__(self):
        pass

    def parse(self, sql):
        parsed = sqlparse.parse(sql)[0]
        tables = []
        joins = []
        predicates = []

        # Extract tables from FROM/JOIN clauses
        # This is a simplified extraction logic
        from_seen = False
        for token in parsed.flatten():
            if token.ttype is Keyword and token.value.upper() == 'FROM':
                from_seen = True
            elif token.ttype is Keyword and 'JOIN' in token.value.upper():
                pass # Handle explicit joins if needed
            elif from_seen and token.ttype is Keyword and token.value.upper() in ['WHERE', 'GROUP BY', 'ORDER BY', 'LIMIT']:
                from_seen = False
            
        # A more robust approach using checking token types
        # Re-parsing for structure
        
        # Let's try to extract tables and aliases roughly
        # This is a placeholder for a complex parsing logic. 
        # For the prototype, we assume implicit joins in WHERE mostly or standard ANSI joins.
        
        # NOTE: For a robust implementation, we might want to regex or iterate carefully.
        # But let's build a structure that we can improve.
        
        extracted_tables = self._extract_tables(parsed)
        extracted_predicates = self._extract_predicates(parsed)
        
        # Separate predicates into joins and filters
        for pred in extracted_predicates:
            if self._is_join_predicate(pred):
                joins.append(pred)
            else:
                predicates.append(pred)

        return {
            "tables": extracted_tables,
            "joins": joins,
            "predicates": predicates
        }

    def _extract_tables(self, token):
        tables = []
        from_seen = False
        
        # This is a heuristic parser. 
        # In a real rigorous academic project we might use the Postgres parser via libpg_query
        # but sqlparse is pure python.
        
        for item in token.tokens:
            if item.ttype is Keyword and item.value.upper() == 'FROM':
                from_seen = True
                continue
            
            if from_seen:
                if isinstance(item, IdentifierList):
                    for identifier in item.get_identifiers():
                        tables.append(identifier.get_real_name())
                elif isinstance(item, Identifier):
                    tables.append(item.get_real_name())
                elif item.ttype is Keyword:
                    # Stop if we hit WHERE, etc
                    if item.value.upper() in ['WHERE', 'GROUP BY', 'ORDER BY']:
                         from_seen = False
                         
        return [t for t in tables if t]

    def _extract_predicates(self, token):
        predicates = []
        where_clause = next((t for t in token.tokens if isinstance(t, Where)), None)
        if where_clause:
            # Recursively find comparisons
            for item in where_clause.flatten():
                # This needs to be smarter to group expressions
                # For now let's just capture standard comparisons
                pass
            # Better: iterate top level tokens of where
            for item in where_clause.tokens:
                 if isinstance(item, Comparison):
                     predicates.append(str(item))
        return predicates

    def _is_join_predicate(self, predicate_str):
        # Heuristic: if both sides have a dot '.', it's likely a join
        # e.g. "t1.id = t2.movie_id"
        parts = predicate_str.split('=')
        if len(parts) == 2:
            left, right = parts
            return '.' in left and '.' in right
        return False

# Example usage helper
def parse_sql(sql):
    parser = SQLParser()
    return parser.parse(sql)
