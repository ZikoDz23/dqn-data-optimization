import psycopg2
import time
from dotenv import load_dotenv
import os

load_dotenv()

db_config = {
    "host": os.getenv("DB_HOST"),
    "database": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD")
}

# Deux requêtes équivalentes avec ordre de jointure différent
queries = {
    "Join_Order_1": """
        EXPLAIN ANALYZE
        SELECT t.title, cn.name
        FROM title t
        JOIN movie_companies mc ON mc.movie_id = t.id
        JOIN company_name cn ON cn.id = mc.company_id;
    """,

    "Join_Order_2": """
        EXPLAIN ANALYZE
        SELECT t.title, cn.name
        FROM company_name cn
        JOIN movie_companies mc ON cn.id = mc.company_id
        JOIN title t ON mc.movie_id = t.id;
    """
}

# Connexion à PostgreSQL
conn = psycopg2.connect(**db_config)
cur = conn.cursor()

print(" Comparaison des plans d'exécution (ordre des jointures) :\n")

for name, sql in queries.items():
    try:
        cur.execute(sql)
        result = cur.fetchall()

        for row in result:
            line = row[0]
            if "Execution Time" in line:
                time_ms = float(line.split(":")[1].strip().split(" ")[0])
                print(f"{name} → Temps d'exécution : {time_ms} ms")
                break

    except Exception as e:
        print(f"❌ Erreur pour {name}: {e}")

cur.close()
conn.close()
