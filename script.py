import psycopg2
import os

db_config = {
    "host": "localhost",
    "database": "jobtest",
    "user": "postgres",
    "password": "123456789"
}

conn = psycopg2.connect(**db_config)
cur = conn.cursor()

query_folder = "JOB/queries/join-order-benchmark"
results = []

for filename in sorted(os.listdir(query_folder)):
    if filename.endswith(".sql"):
        with open(os.path.join(query_folder, filename), "r", encoding="utf-8") as f:
            query = f.read().strip()
            try:
                cur.execute("EXPLAIN ANALYZE " + query)
                rows = cur.fetchall()
                for row in rows:
                    if "Execution Time" in row[0]:
                        time = float(row[0].split(":")[1].split("ms")[0].strip())
                        results.append((filename, time))
                        break
            except Exception as e:
                print(f"Error in {filename}: {e}")
                results.append((filename, None))

cur.close()
conn.close()

# Sauvegarde dans un fichier CSV
import csv
with open("baseline_results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Query", "ExecutionTime_ms"])
    writer.writerows(results)
