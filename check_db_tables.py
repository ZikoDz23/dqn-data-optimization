import psycopg2
import yaml

def list_tables():
    with open('rl_query_optimizer/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    db_config = config['database']
    try:
        conn = psycopg2.connect(
            dbname=db_config['dbname'],
            user=db_config['user'],
            password=db_config['password'],
            host=db_config['host'],
            port=db_config['port']
        )
        cur = conn.cursor()
        cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';")
        tables = cur.fetchall()
        print("Tables in database:", [t[0] for t in tables])
        
        if 'movie_info_idx' in [t[0] for t in tables]:
            print("movie_info_idx EXISTS.")
        else:
            print("movie_info_idx MISSING.")
            
        conn.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    list_tables()
