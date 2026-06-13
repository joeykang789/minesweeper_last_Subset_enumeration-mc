"""Print summary of sweep results"""
import csv
from collections import defaultdict

data = defaultdict(list)
with open('sweep_20x50_results.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        size = int(row['rows'])
        data[size].append((float(row['density']), float(row['win_rate'])*100, float(row['ci95'])*100))

print(f"{'Board':>8} {'Cells':>5} {'Crit_rho':>8} {'rho_50%':>8} {'rho_85%':>8}")
print('-' * 45)
for size in sorted(data.keys()):
    d = sorted(data[size])
    
    # Crit rho (win rate first < 5%)
    rho_crit = 0
    for den, w, _ in d:
        if w < 5:
            rho_crit = den
            break
    
    # rho at 50%
    rho_50 = 0
    for i, (den, w, _) in enumerate(d):
        if i > 0 and d[i-1][1] >= 50 and w < 50:
            rho_50 = (d[i-1][0] + den) / 2
            break
    
    # rho at 85%
    rho_85 = 0
    for i, (den, w, _) in enumerate(d):
        if i > 0 and d[i-1][1] >= 85 and w < 85:
            rho_85 = (d[i-1][0] + den) / 2
            break
    
    total = size * size
    crit_str = f"{rho_crit:.2f}" if rho_crit > 0 else ">0.35"
    r50_str = f"{rho_50:.2f}" if rho_50 > 0 else "N/A"
    r85_str = f"{rho_85:.2f}" if rho_85 > 0 else "N/A"
    print(f"{size:>3}x{size:<3} {total:5d} {crit_str:>8} {r50_str:>8} {r85_str:>8}")

print()
print("All densities:")
density_set = sorted(set(d[0] for d_list in data.values() for d in d_list))
print("  ".join(f"{d:.2f}" for d in density_set))