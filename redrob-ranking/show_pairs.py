import csv, difflib

rows = []
with open("submission_phase4_test.csv", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        rows.append(row)

def show_pair(i, j):
    r1, r2 = rows[i-1], rows[j-1]
    s1, s2 = r1["reasoning"], r2["reasoning"]
    sim = difflib.SequenceMatcher(None, s1, s2).ratio()
    print(f"Ranks {i} & {j}: sim={sim:.4f}")
    print(f"  R{i} ({r1['candidate_id']}): {s1}")
    print(f"  R{j} ({r2['candidate_id']}): {s2}")
    print()

for a, b in [(25,51),(33,94),(28,83),(36,77),(83,84),(27,87),(48,58),(23,25),(41,73),(28,67)]:
    show_pair(a, b)
