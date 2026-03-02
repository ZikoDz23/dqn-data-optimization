import networkx as nx

class QueryGraph:
    def __init__(self):
        self.graph = nx.Graph()

    def build_from_parsed(self, parsed_query):
        """
        Builds the query graph from the output of SQLParser.
        
        Args:
            parsed_query (dict): output from SQLParser.parse()
                                 Expects keys: 'tables', 'joins' (list of Join objects or strings)
        """
        
        # Add nodes for each table
        for table in parsed_query.get('tables', []):
            self.add_relation(table)
            
        # Add edges for joins
        for join_pred in parsed_query.get('joins', []):
            # join_pred is currently a string "t1.col = t2.col"
            # We need to extract table names from it
            # This is a naive extraction
            parts = join_pred.split('=')
            if len(parts) == 2:
                left = parts[0].strip()
                right = parts[1].strip()
                
                # Assume table.column format
                t1 = left.split('.')[0] if '.' in left else None
                t2 = right.split('.')[0] if '.' in right else None
                
                if t1 and t2 and t1 in self.graph.nodes and t2 in self.graph.nodes:
                    self.add_join(t1, t2, predicate=join_pred)

    def add_relation(self, name, cardinality=0, estimated_rows=0, has_index=False):
        """
        Add a relation node with features.
        """
        # Features vector placeholder
        features = {
            "cardinality": cardinality,
            "estimated_rows": estimated_rows,
            "has_index": 1.0 if has_index else 0.0
        }
        self.graph.add_node(name, features=features, type="relation")

    def add_join(self, t1, t2, predicate=""):
        """
        Add a join edge.
        """
        if self.graph.has_edge(t1, t2):
            # If edge exists, maybe update it (multiple join predicates)
            current_pred = self.graph[t1][t2].get('predicate', '')
            new_pred = f"{current_pred} AND {predicate}" if current_pred else predicate
            self.graph[t1][t2]['predicate'] = new_pred
        else:
            self.graph.add_edge(t1, t2, predicate=predicate, type="join")

    def get_features_matrix(self):
        # Return node features as numpy array (for GNN later)
        pass
    
    def get_adjacency_matrix(self):
        return nx.adjacency_matrix(self.graph)
