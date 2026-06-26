"""เทียบสถิติ real vs synthetic — v1.9.3 หลังปรับ."""
import sys, warnings
from pathlib import Path
import pandas as pd
import numpy as np

# Force venv site-packages first
VENV_SP = Path(r"C:\Users\User\Desktop\Hermes_Peet\scratch\thaieda\.venv-baseline\Lib\site-packages")
sys.path.insert(0, str(VENV_SP))
sys.path.insert(0, str(Path(r"C:\Users\User\Desktop\Hermes_Peet\scratch\thaieda\src")))

from thaieda.llm._synthetic import generate_synthetic_data

warnings.simplefilter('ignore')

DATA_DIR = Path(r"C:\Users\User\Desktop\Hermes_Peet\scratch\thaieda\data-example\public-datasets")
datasets = sorted(DATA_DIR.glob("*.csv"))
for sub in sorted(DATA_DIR.iterdir()):
    if sub.is_dir():
        datasets.extend(sorted(sub.glob("*.csv")))

seen = set()
unique = []
for p in datasets:
    key = p.stem.replace("-clean", "")
    if key not in seen:
        seen.add(key)
        unique.append(p)
datasets = unique

results = []
for csv_path in datasets:
    try:
        df = pd.read_csv(csv_path, encoding='utf-8', nrows=2000)
    except Exception:
        try:
            df = pd.read_csv(csv_path, encoding='latin-1', nrows=2000)
        except Exception as e:
            results.append({"dataset": csv_path.name, "error": f"read: {e}"})
            continue
    if len(df) < 10 or len(df.columns) < 2:
        continue
    try:
        synth = generate_synthetic_data(df, random_seed=42)
    except Exception as e:
        results.append({"dataset": csv_path.name, "error": f"gen: {e}"})
        continue

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in df.select_dtypes(include=['object', 'string']).columns if df[c].nunique() <= 50]

    col_stats = []
    for col in numeric_cols:
        real_vals = pd.to_numeric(df[col], errors='coerce').dropna()
        synth_vals = pd.to_numeric(synth[col], errors='coerce').dropna()
        if len(real_vals) < 5 or len(synth_vals) < 5:
            continue
        real_mean = round(float(real_vals.mean()), 4)
        synth_mean = round(float(synth_vals.mean()), 4)
        real_std = round(float(real_vals.std()), 4)
        synth_std = round(float(synth_vals.std()), 4)
        mean_diff_pct = round(abs(real_mean - synth_mean) / (abs(real_mean) + 1e-10) * 100, 2)
        std_diff_pct = round(abs(real_std - synth_std) / (abs(real_std) + 1e-10) * 100, 2)
        try:
            from scipy.stats import ks_2samp
            ks_stat, ks_p = ks_2samp(real_vals, synth_vals)
            ks_stat = round(float(ks_stat), 4)
            ks_p = round(float(ks_p), 4)
        except Exception:
            ks_stat, ks_p = None, None
        col_stats.append({"col": col, "real_mean": real_mean, "synth_mean": synth_mean,
            "mean_diff_pct": mean_diff_pct, "real_std": real_std, "synth_std": synth_std,
            "std_diff_pct": std_diff_pct, "ks_stat": ks_stat, "ks_p": ks_p})

    cat_stats = []
    for col in cat_cols:
        real_vc = df[col].value_counts(normalize=True, dropna=False)
        synth_vc = synth[col].value_counts(normalize=True, dropna=False)
        all_cats = real_vc.index.union(synth_vc.index)
        real_p = real_vc.reindex(all_cats, fill_value=0).values
        synth_p = synth_vc.reindex(all_cats, fill_value=0).values
        tvd = round(float(np.abs(real_p - synth_p).sum() / 2), 4)
        cat_stats.append({"col": col, "tvd": tvd})

    n_num = len(col_stats)
    avg_mean_diff = round(np.mean([c["mean_diff_pct"] for c in col_stats]), 2) if col_stats else 0
    avg_std_diff = round(np.mean([c["std_diff_pct"] for c in col_stats]), 2) if col_stats else 0
    avg_ks_stat = round(np.mean([c["ks_stat"] for c in col_stats if c["ks_stat"] is not None]), 4) if any(c["ks_stat"] is not None for c in col_stats) else None
    ks_pass_count = sum(1 for c in col_stats if c["ks_p"] is not None and c["ks_p"] > 0.05)
    ks_pass_pct = round(ks_pass_count / n_num * 100, 1) if n_num > 0 else 0
    avg_tvd = round(np.mean([c["tvd"] for c in cat_stats]), 4) if cat_stats else 0

    results.append({"dataset": csv_path.name, "shape": f"{df.shape[0]}x{df.shape[1]}",
        "numeric_cols": n_num, "cat_cols": len(cat_stats),
        "avg_mean_diff_pct": avg_mean_diff, "avg_std_diff_pct": avg_std_diff,
        "avg_ks_stat": avg_ks_stat, "ks_pass_pct": ks_pass_pct, "avg_tvd": avg_tvd,
        "col_details": col_stats[:5]})

print("=" * 110)
print(f"{'Dataset':<30} {'Shape':<12} {'Num':<5} {'Cat':<5} {'MeanDiff%':<12} {'StdDiff%':<12} {'KS-stat':<10} {'KS pass%':<10} {'TVD':<8}")
print("-" * 110)
for r in results:
    if "error" in r:
        print(f"{r['dataset']:<30} ERROR: {r['error']}")
        continue
    ks_s = r['avg_ks_stat'] if r['avg_ks_stat'] is not None else "N/A"
    print(f"{r['dataset']:<30} {r['shape']:<12} {r['numeric_cols']:<5} {r['cat_cols']:<5} "
          f"{r['avg_mean_diff_pct']:<12} {r['avg_std_diff_pct']:<12} {str(ks_s):<10} "
          f"{r['ks_pass_pct']:<10} {r['avg_tvd']:<8}")

print("=" * 110)
print("\n=== BEFORE vs AFTER (v1.9.2 -> v1.9.3) ===")
before = {
    "adult.csv": (59.88, 0.0), "online-shoppers.csv": (45.62, 14.3),
    "beijing-pm25.csv": (26.84, 58.3), "synthetic-quality-benchmark.csv": (32.73, 66.7),
    "titanic.csv": (13.89, 28.6), "winequality-red.csv": (1.0, 16.7),
    "california-housing.csv": (2.23, 22.2), "loan-default.csv": (2.9, 25.0),
    "telco-churn.csv": (4.1, 33.3), "bike-sharing-hour.csv": (7.67, 37.5),
    "superstore.csv": (8.66, 33.3), "online-retail.csv": (8.16, 0.0),
    "day.csv": (7.77, 73.3),
}
print(f"{'Dataset':<35} {'Bef dMu%':<10} {'Aft dMu%':<10} {'Bef KS%':<10} {'Aft KS%':<10} {'Better?'}")
print("-" * 85)
for r in results:
    if "error" in r:
        continue
    name = r['dataset']
    b_mean, b_ks = before.get(name, (None, None))
    if b_mean is None:
        continue
    improved = "YES" if r['avg_mean_diff_pct'] < b_mean else "no"
    print(f"{name:<35} {b_mean:<10} {r['avg_mean_diff_pct']:<10} {b_ks:<10} {r['ks_pass_pct']:<10} {improved}")

# Show worst columns for adult and online-shoppers
print("\n=== WORST COLUMNS (adult.csv) ===")
adult_r = next((r for r in results if r['dataset'] == 'adult.csv'), None)
if adult_r:
    for c in sorted(adult_r['col_details'], key=lambda x: -x['mean_diff_pct']):
        ks_info = f"KS={c['ks_stat']}(p={c['ks_p']})" if c['ks_stat'] is not None else "KS=N/A"
        print(f"  {c['col']:<25} dMu={c['mean_diff_pct']}%  {ks_info}")

print("\n=== WORST COLUMNS (online-shoppers.csv) ===")
os_r = next((r for r in results if r['dataset'] == 'online-shoppers.csv'), None)
if os_r:
    for c in sorted(os_r['col_details'], key=lambda x: -x['mean_diff_pct']):
        ks_info = f"KS={c['ks_stat']}(p={c['ks_p']})" if c['ks_stat'] is not None else "KS=N/A"
        print(f"  {c['col']:<25} dMu={c['mean_diff_pct']}%  {ks_info}")