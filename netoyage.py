import csv

input_file = "logs/dqn_results.csv"
output_file = "logs/dqn_results_filtered.csv"

best_times = {}

# Lire toutes les lignes et garder la meilleure exécution (temps le plus bas)
with open(input_file, "r", newline='', encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        filename = row["QueryFile"]
        time = float(row["ExecutionTime_ms"])

        if filename not in best_times or time < best_times[filename]:
            best_times[filename] = time

# Écrire le fichier nettoyé
with open(output_file, "w", newline='', encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["QueryFile", "ExecutionTime_ms"])  # En-têtes

    for filename, time in sorted(best_times.items()):
        writer.writerow([filename, round(time, 3)])

print(f"✅ Fichier filtré enregistré dans {output_file}")
