import psycopg2
import csv
from dotenv import load_dotenv
import os

load_dotenv()
# Configuration PostgreSQL
db_config = {
    "host": os.getenv("DB_HOST"),
    "database": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD")
}

# Liste manuelle des requêtes SQL
job_queries = [
            """SELECT title FROM title WHERE production_year IS NOT NULL""",
            
            """SELECT COUNT(*) AS war_movies FROM title WHERE title ILIKE '%war%'""",
            
            """SELECT t.title, cn.name FROM title AS t JOIN movie_companies AS mc ON mc.movie_id = t.id JOIN company_name AS cn ON mc.company_id = cn.id""",
            
            """SELECT MIN(cn.name) AS from_company, MIN(lt.link) AS movie_link_type, MIN(t.title) AS non_polish_sequel_movie FROM company_name AS cn, company_type AS ct, keyword AS k, link_type AS lt, movie_companies AS mc, movie_keyword AS mk, movie_link AS ml, title AS t WHERE cn.country_code !='[pl]' AND (cn.name LIKE '%Film%' OR cn.name LIKE '%Warner%') AND ct.kind ='production companies' AND k.keyword ='sequel' AND lt.link LIKE '%follow%' AND mc.note IS NULL AND t.production_year BETWEEN 1950 AND 2000 AND lt.id = ml.link_type_id AND ml.movie_id = t.id AND t.id = mk.movie_id AND mk.keyword_id = k.id AND t.id = mc.movie_id AND mc.company_type_id = ct.id AND mc.company_id = cn.id AND ml.movie_id = mk.movie_id AND ml.movie_id = mc.movie_id AND mk.movie_id = mc.movie_id"""
           
        ]

query_names = [
            "req1.sql", "req2.sql", "req3.sql", "req4.sql"
        ]

# Connexion à PostgreSQL
conn = psycopg2.connect(**db_config)
cur = conn.cursor()

# Résultats
results = []

for i in range(len(job_queries)):
    query = job_queries[i]
    query_name = query_names[i]
    try:
        cur.execute("EXPLAIN ANALYZE " + query)
        rows = cur.fetchall()
        exec_time = None
        for row in rows:
            if "Execution Time" in row[0]:
                exec_time = float(row[0].split(":")[1].split("ms")[0].strip())
                break
        results.append((query_name, exec_time))
        print(f"{query_name} → {exec_time:.2f} ms")
    except Exception as e:
        print(f"[ERROR] {query_name} failed: {e}")
        results.append((query_name, None))

# Fermer la connexion
cur.close()
conn.close()

# Écrire dans un fichier CSV
with open("baseline_results.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Query", "ExecutionTime_ms"])
    writer.writerows(results)
