import json

class PlanParser:
    def __init__(self):
        pass

    def parse_explain_json(self, json_output):
        """
        Parses the JSON output from PostgreSQL EXPLAIN (FORMAT JSON)
        """
        if isinstance(json_output, str):
            plan_data = json.loads(json_output)
        else:
            plan_data = json_output

        # The plan is usually wrapped in a list
        if isinstance(plan_data, list):
            plan_data = plan_data[0]
            
        plan_node = plan_data.get('Plan', {})
        
        return {
            "execution_time": plan_data.get('Execution Time'), # timestamp or float
            "planning_time": plan_data.get('Planning Time'),
            "total_cost": plan_node.get('Total Cost'),
            "startup_cost": plan_node.get('Startup Cost'),
            "plan_tree": self._extract_plan_tree(plan_node)
        }

    def _extract_plan_tree(self, node):
        """
        Recursively extract plan structure
        """
        parsed_node = {
            "node_type": node.get('Node Type'),
            "cost": node.get('Total Cost'),
            "rows": node.get('Plan Rows'),
            "relation": node.get('Relation Name'),
            "alias": node.get('Alias'),
            "children": []
        }
        
        if 'Plans' in node:
            for child in node['Plans']:
                parsed_node['children'].append(self._extract_plan_tree(child))
                
        return parsed_node
