"""Generate a synthetic dataset with known injected issues for quality benchmark.

Injects exactly known defects so we can measure recall:
- 5 outliers in numeric column
- 1 column with 20% missing
- 1 duplicate-heavy column
- 1 constant column
- 1 type-mixed column (numbers as text)
- 1 placeholder column (-, N/A, ไม่มี)
- 1 BE date column (พ.ศ. 2567)
- 1 Thai numerals column (๑๒๓)
- 1 zero-width space column
- 1 mojibake column (TIS-620)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

np.random.seed(42)
N = 2000

OUTPUT_DIR = Path(__file__).parent.parent.parent / "data-example" / "public-datasets"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Base clean data
data = {
    "id": range(1, N + 1),
    "price": np.random.normal(500, 50, N).round(2),
    "quantity": np.random.randint(1, 100, N),
    "category": np.random.choice(["electronics", "clothing", "food", "books"], N),
    "customer_name": [f"Customer_{i}" for i in range(N)],
}

df = pd.DataFrame(data)

# === INJECT KNOWN ISSUES ===

# 1. Outliers in price (5 extreme values)
df.loc[df.index[:5], "price"] = [99999, -500, 50000, -100, 9999]
INJECTED_ISSUES = [{"issue": "outliers", "column": "price", "count": 5}]

# 2. 20% missing in quantity
miss_mask = np.random.random(N) < 0.20
df.loc[miss_mask, "quantity"] = np.nan
INJECTED_ISSUES.append({"issue": "missing_values", "column": "quantity", "count": int(miss_mask.sum())})

# 3. Duplicate-heavy customer_name (many duplicates)
df["customer_name"] = np.random.choice(["John Smith", "Jane Doe", "Bob Wilson", "Alice Brown"], N)
dup_count = df["customer_name"].duplicated().sum()
INJECTED_ISSUES.append({"issue": "duplicates", "column": "customer_name", "count": int(dup_count)})

# 4. Constant column
df["status"] = "active"
INJECTED_ISSUES.append({"issue": "constant_column", "column": "status", "count": 1})

# 5. Type-mixed column (numbers as text with some strings)
df["zip_code"] = np.random.randint(10000, 99999, N).astype(str)
df.loc[df.index[:50], "zip_code"] = "unknown"
INJECTED_ISSUES.append({"issue": "type_inconsistency", "column": "zip_code", "count": 50})

# 6. Placeholder values
df["rating"] = np.random.randint(1, 6, N).astype(str)
df.loc[np.random.choice(df.index, 100, replace=False), "rating"] = "-"
df.loc[np.random.choice(df.index, 50, replace=False), "rating"] = "N/A"
df.loc[np.random.choice(df.index, 30, replace=False), "rating"] = "ไม่มี"
INJECTED_ISSUES.append({"issue": "placeholders", "column": "rating", "count": 180})

# 7. Buddhist Era dates (พ.ศ. 2567 = CE 2024)
df["order_date"] = [f"15/03/{np.random.randint(2560, 2569)}" for _ in range(N)]
INJECTED_ISSUES.append({"issue": "buddhist_era", "column": "order_date", "count": N})

# 8. Thai numerals (๑๒๓ → 123)
df["thai_amount"] = np.random.randint(100, 9999, N)
thai_numerals_map = {"0": "๐", "1": "๑", "2": "๒", "3": "๓", "4": "๔", "5": "๕",
                     "6": "๖", "7": "๗", "8": "๘", "9": "๙"}
df["thai_amount"] = df["thai_amount"].astype(str)
df.loc[df.index[:500], "thai_amount"] = df.loc[df.index[:500], "thai_amount"].apply(
    lambda x: "".join(thai_numerals_map.get(c, c) for c in x)
)
INJECTED_ISSUES.append({"issue": "thai_numerals", "column": "thai_amount", "count": 500})

# 9. Zero-width spaces in category
df["category_text"] = df["category"].copy()
df.loc[df.index[:100], "category_text"] = "elec\u200btro\u200bnics"
INJECTED_ISSUES.append({"issue": "zero_width_space", "column": "category_text", "count": 100})

# 10. Mojibake (TIS-620 encoded text decoded as latin-1)
def make_mojibake(n=100):
    results = []
    thai_words = ["สวัสดี", "ราคา", "สินค้า", "ลูกค้า", "ออร์เดอร์"]
    for i in range(n):
        word = np.random.choice(thai_words)
        try:
            moji = word.encode("tis-620").decode("latin-1")
            results.append(moji)
        except Exception:
            results.append(word)
    return results

df["product_desc"] = ""
df.loc[df.index[:200], "product_desc"] = make_mojibake(200)
INJECTED_ISSUES.append({"issue": "mojibake", "column": "product_desc", "count": 200})

# Save
out_path = OUTPUT_DIR / "synthetic-quality-benchmark.csv"
df.to_csv(out_path, index=False, encoding="utf-8")

# Save ground truth
truth_path = Path(__file__).parent / "quality-ground-truth.json"
truth_path.write_text(json.dumps({
    "dataset": str(out_path),
    "rows": N,
    "cols": len(df.columns),
    "columns": list(df.columns),
    "injected_issues": INJECTED_ISSUES,
    "total_issues": len(INJECTED_ISSUES),
}, indent=2, ensure_ascii=False), encoding="utf-8")

print(f"Dataset: {out_path}")
print(f"Rows: {N} | Cols: {len(df.columns)}")
print(f"Injected issues: {len(INJECTED_ISSUES)}")
for issue in INJECTED_ISSUES:
    print(f"  - {issue['issue']} in {issue['column']} ({issue['count']})")
print(f"\nGround truth: {truth_path}")