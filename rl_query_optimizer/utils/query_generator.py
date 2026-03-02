import random
import os

class QueryGenerator:
    def __init__(self):
        # Simplified IMDb Schema (Tables and their Foreign Key relationships)
        # Format: "Table": ["NeighborTable.fk_col = Table.id_col"]
        # Or more simply, an adjacency list of valid joins.
        
        self.schema_graph = {
            "title": [
                ("movie_info", "movie_info.movie_id = title.id"),
                # ("movie_info_idx", "movie_info_idx.movie_id = title.id"), # MISSING TABLE
                ("cast_info", "cast_info.movie_id = title.id"),
                ("movie_companies", "movie_companies.movie_id = title.id"),
                ("movie_keyword", "movie_keyword.movie_id = title.id"),
                ("kind_type", "title.kind_id = kind_type.id"),
            ],
            "movie_info": [
                ("title", "movie_info.movie_id = title.id"),
                ("info_type", "movie_info.info_type_id = info_type.id")
            ],
            # "movie_info_idx": [
            #     ("title", "movie_info_idx.movie_id = title.id"),
            #     ("info_type", "movie_info_idx.info_type_id = info_type.id")
            # ],
            "cast_info": [
                ("title", "cast_info.movie_id = title.id"),
                ("name", "cast_info.person_id = name.id"),
                ("char_name", "cast_info.person_role_id = char_name.id"),
                ("role_type", "cast_info.role_id = role_type.id")
            ],
            "movie_companies": [
                ("title", "movie_companies.movie_id = title.id"),
                ("company_name", "movie_companies.company_id = company_name.id"),
                ("company_type", "movie_companies.company_type_id = company_type.id")
            ],
            "movie_keyword": [
                ("title", "movie_keyword.movie_id = title.id"),
                ("keyword", "movie_keyword.keyword_id = keyword.id")
            ],
            "name": [
                ("cast_info", "cast_info.person_id = name.id"),
                ("aka_name", "aka_name.person_id = name.id") # Assuming aka_name links to name
            ],
             "aka_name": [
                 ("name", "aka_name.person_id = name.id")
             ],
            "company_name": [
                ("movie_companies", "movie_companies.company_id = company_name.id")
            ],
            "keyword": [
                ("movie_keyword", "movie_keyword.keyword_id = keyword.id")
            ],
            "kind_type": [
                ("title", "title.kind_id = kind_type.id")
            ],
            "info_type": [
                ("movie_info", "movie_info.info_type_id = info_type.id"),
                # ("movie_info_idx", "movie_info_idx.info_type_id = info_type.id")
            ],
            "char_name": [
                ("cast_info", "cast_info.person_role_id = char_name.id")
            ],
            "role_type": [
                ("cast_info", "cast_info.role_id = role_type.id")
            ],
            "company_type": [
                ("movie_companies", "movie_companies.company_type_id = company_type.id")
            ]
        }
        
        self.tables = list(self.schema_graph.keys())

    def generate_query(self, min_joins=2, max_joins=8):
        """
        Generates a random connected join query.
        """
        # Start with a random table
        current_tables = [random.choice(self.tables)]
        joins = []
        
        num_joins = random.randint(min_joins, max_joins)
        
        # Simple Random Walk to add tables
        # We try to add a neighbor of any currently included table
        
        for _ in range(num_joins):
            # Pick a table currently in the query that has free neighbors
            potential_sources = []
            for t in current_tables:
                neighbors = self.schema_graph.get(t, [])
                # Check for neighbors not yet in query
                valid_neighbors = [n for n in neighbors if n[0] not in current_tables]
                if valid_neighbors:
                    potential_sources.append((t, valid_neighbors))
            
            if not potential_sources:
                break # No more expansion possible
            
            # Pick a source and a neighbor
            src, neighbors = random.choice(potential_sources)
            target, predicate = random.choice(neighbors)
            
            current_tables.append(target)
            joins.append(predicate)
            
        # Construct SQL
        tables_str = ", ".join(current_tables)
        joins_str = " AND ".join(joins)
        
        sql = f"SELECT count(*) FROM {tables_str} WHERE {joins_str};"
        return sql

    def generate_dataset(self, num_queries, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        for i in range(num_queries):
            sql = self.generate_query()
            with open(f"{output_dir}/gen_query_{i:05d}.sql", "w") as f:
                f.write(sql)
                
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=100)
    parser.add_argument("--out", type=str, default="rl_query_optimizer/data/train_queries")
    args = parser.parse_args()
    
    gen = QueryGenerator()
    gen.generate_dataset(args.num, args.out)
    print(f"Generated {args.num} queries in {args.out}")
