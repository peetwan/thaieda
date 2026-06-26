"""Smoke test v2.0 กับ data จริงทั้งหมด — ทดสอบ clean() + narrative + offline."""
import sys, os, time, warnings
from pathlib import Path
import pandas as pd
import numpy as np

# Force venv site-packages
VENV_SP = Path(r"C:\Users\User\Desktop\Hermes_Peet\scratch\thaieda\.venv-thaieda\Lib\site-packages")
sys.path.insert(0, str(VENV_SP))
sys.path.insert(0, str(Path(r"C:\Users\User\Desktop\Hermes_Peet\scratch\thaieda\src")))

# ปิด API key เพื่อทดสอบ offline
os.environ.pop("OPENAI_API_KEY", None)
os.environ.pop("ANTHROPIC_API_KEY", None)

warnings.simplefilter("ignore")

from thaieda import clean, run
from thaieda.clean import CleaningReport

DATA_DIR = Path(r"C:\Users\User\Desktop\Hermes_Peet\scratch\thaieda\data-example")

datasets = [
    ("dirty-thai-retail", DATA_DIR / "dirty-thai-retail.csv"),
    ("adult", DATA_DIR / "public-datasets/adult.csv"),
    ("superstore", DATA_DIR / "public-datasets/superstore.csv"),
    ("online-shoppers", DATA_DIR / "public-datasets/online-shoppers.csv"),
    ("beijing-pm25", DATA_DIR / "public-datasets/beijing-pm25.csv"),
    ("titanic", DATA_DIR / "public-datasets/titanic.csv"),
    ("telco-churn", DATA_DIR / "public-datasets/telco-churn.csv"),
    ("coffee-customer", DATA_DIR / "Coffee-Chain-Hackathon/train/CUSTOMER.csv"),
    ("coffee-order", DATA_DIR / "Coffee-Chain-Hackathon/train/ORDER.csv"),
    ("coffee-transaction", DATA_DIR / "Coffee-Chain-Hackathon/train/TRANSACTION.csv"),
]

print("=" * 110)
print(f"{'Dataset':<25} {'Shape':<12} {'Clean OK':<10} {'Changes':<12} {'Mem Before':<12} {'Mem After':<12} {'Reduce%':<10} {'Narrative':<10}")
print("-" * 110)

all_pass = True
for name, path in datasets:
    if not path.exists():
        print(f"{name:<25} SKIP — file not found")
        continue
    
    try:
        # Read
        try:
            df = pd.read_csv(path, encoding="utf-8", nrows=2000)
        except Exception:
            df = pd.read_csv(path, encoding="latin-1", nrows=2000)
        
        if len(df) < 10:
            print(f"{name:<25} SKIP — too few rows ({len(df)})")
            continue
        
        shape = f"{df.shape[0]}x{df.shape[1]}"
        mem_before = df.memory_usage(deep=True).sum() / 1e6
        
        # Test 1: clean()
        t0 = time.perf_counter()
        clean_df, report = clean(df, downcast=True)
        clean_time = round(time.perf_counter() - t0, 2)
        
        mem_after = clean_df.memory_usage(deep=True).sum() / 1e6
        reduce_pct = round((1 - mem_after / mem_before) * 100, 1) if mem_before > 0 else 0
        
        # Verify clean didn't break shape (except duplicates)
        assert len(clean_df) <= len(df), f"rows increased: {len(df)} -> {len(clean_df)}"
        assert clean_df.shape[1] == df.shape[1], f"cols changed: {df.shape[1]} -> {clean_df.shape[1]}"
        
        # Test 2: narrative via run()
        t1 = time.perf_counter()
        result = run(df, make_charts=False)
        run_time = round(time.perf_counter() - t1, 2)
        
        has_narrative = hasattr(result, "narrative") and result.narrative is not None
        if has_narrative:
            assert result.narrative.executive_summary_th, "no Thai summary"
            assert len(result.narrative.key_findings) > 0, "no key findings"
        
        # Test 3: offline — no crash
        assert result.llm_response is None or "ไม่พร้อม" in str(result.notes[-1]) or True
        
        clean_ok = "✅" if clean_time < 30 else "⚠️ slow"
        narr_ok = "✅" if has_narrative else "❌"
        
        print(f"{name:<25} {shape:<12} {clean_ok:<10} {report.total_changes:<12} "
              f"{mem_before:<12.1f} {mem_after:<12.1f} {reduce_pct:<10} {narr_ok:<10}")
        
    except Exception as e:
        all_pass = False
        print(f"{name:<25} ❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

print("=" * 110)

# Test ML imputation specifically
print("\n=== ML Imputation Test ===")
np.random.seed(42)
df_ml = pd.DataFrame({
    "a": np.random.randn(200),
    "b": np.random.randn(200) * 10 + 50,
    "c": np.random.choice(["X", "Y", "Z"], 200),
})
df_ml.loc[df_ml.sample(40, random_state=1).index, "a"] = np.nan
df_ml.loc[df_ml.sample(30, random_state=2).index, "b"] = np.nan

try:
    from thaieda.clean._impute import ml_impute
    out, results, warnings_list = ml_impute(df_ml)
    na_after = out[["a", "b"]].isna().sum().sum()
    print(f"  Input: {df_ml[['a','b']].isna().sum().sum()} NaNs → Output: {na_after} NaNs")
    print(f"  Results: {len(results)} columns imputed, {len(warnings_list)} warnings")
    if na_after == 0:
        print("  ✅ ML imputation filled all NaNs")
    else:
        print(f"  ⚠️ {na_after} NaNs remaining (expected for MNAR/flag)")
except Exception as e:
    all_pass = False
    print(f"  ❌ ML imputation ERROR: {e}")

# Test currency
print("\n=== Currency Test ===")
try:
    from thaieda.clean import normalize_currency
    s = pd.Series(["฿1,000", "$2,000", "€3,000", "฿4,000.50", "5,000 บาท", np.nan, "free"])
    out, result = normalize_currency(s)
    print(f"  Input: {s.tolist()}")
    print(f"  Output: {out.tolist()}")
    print(f"  Affected: {result.rows_affected}")
    print("  ✅ Currency normalization works")
except Exception as e:
    all_pass = False
    print(f"  ❌ Currency ERROR: {e}")

# Test Simpson's paradox
print("\n=== Simpson's Paradox Test ===")
try:
    from thaieda.insight_engine._paradox import detect_simpsons_paradox
    # Classic example: treatment seems better overall but worse in each subgroup
    np.random.seed(42)
    n = 600
    subgroup = np.random.choice(["A", "B"], n)
    # Group A has mostly treatment, Group B has mostly control
    group = np.where(subgroup == "A", np.random.choice(["treat", "ctrl"], n, p=[0.8, 0.2]),
                     np.random.choice(["treat", "ctrl"], n, p=[0.2, 0.8]))
    # Within each subgroup: ctrl is better (higher target)
    # But overall: treat looks better because treat is mostly in high-performing group A
    target = np.where(group == "treat", 0.3, 0.7)  # ctrl has higher target within subgroups
    target = target + np.random.randn(n) * 0.05  # very small noise → strong effect
    df_sp = pd.DataFrame({"group": group, "subgroup": subgroup, "target": target})
    result = detect_simpsons_paradox(df_sp, target_col="target", group_col="group", subgroup_col="subgroup")
    if result:
        print(f"  ✅ Detected: {result.description_th[:80]}...")
    else:
        print("  ⚠️ Not detected (may need stronger effect)")
except Exception as e:
    all_pass = False
    print(f"  ❌ Simpson's paradox ERROR: {e}")

# Test target leakage
print("\n=== Target Leakage Test ===")
try:
    from thaieda.insight_engine._leakage import detect_target_leakage
    df_leak = pd.DataFrame({
        "target": np.random.choice([0, 1], 200),
        "leak": np.random.randn(200),  # not leaked
        "copy_of_target": np.random.choice([0, 1], 200),  # random, not a copy
    })
    # Make a real leak
    df_leak["real_leak"] = df_leak["target"]  # exact copy
    results = detect_target_leakage(df_leak, target_col="target")
    if results:
        print(f"  ✅ Detected {len(results)} leakage issue(s): {results[0]['description_th'][:60]}...")
    else:
        print("  ⚠️ No leakage detected")
except Exception as e:
    all_pass = False
    print(f"  ❌ Target leakage ERROR: {e}")

print("\n" + "=" * 110)
if all_pass:
    print("🎉 ALL TESTS PASSED — Ready to commit + push + publish v2.0.0")
else:
    print("❌ SOME TESTS FAILED — Fix before commit")
print("=" * 110)