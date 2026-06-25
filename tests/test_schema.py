"""ทดสอบ thaieda.schema — discover_keys, match_relationships, profile_dataset + DatasetReport."""

from __future__ import annotations

import pandas as pd

from thaieda.report._dataset import DatasetReport
from thaieda.schema import (
    DatasetProfile,
    KeyCandidate,
    Relationship,
    TableProfile,
    discover_keys,
    match_relationships,
    profile_dataset,
)


def _profiles(tables: dict[str, pd.DataFrame]) -> dict[str, TableProfile]:
    """สร้าง TableProfile (พร้อม key_candidates) สำหรับชุด DataFrame ในการทดสอบ."""
    out: dict[str, TableProfile] = {}
    for name, df in tables.items():
        out[name] = TableProfile(
            name=name,
            file_path="",
            row_count=len(df),
            column_count=len(df.columns),
            columns=[str(c) for c in df.columns],
            column_types={},
            key_candidates=discover_keys(df, name),
        )
    return out


# ------------------------------------------------------------- discover_keys
def test_discover_keys_simple():
    df = pd.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"], "value": [10, 20, 30]})
    keys = discover_keys(df, "test")
    # มีแค่ "id" (unique + ชื่อบอกใบ้) — "name"/"value" ไม่ซ้ำโดยบังเอิญในตารางจิ๋ว ไม่นับ
    assert len(keys) == 1
    assert keys[0].column == "id"
    assert keys[0].is_unique is True
    assert keys[0].name_hint is True


def test_discover_keys_excludes_boolean_and_constant():
    df = pd.DataFrame(
        {
            "user_id": [1, 2, 3, 4, 5, 6],
            "is_active": [True, False, True, False, True, False],
            "const": [1, 1, 1, 1, 1, 1],
        }
    )
    keys = discover_keys(df, "t")
    cols = {k.column for k in keys}
    assert "user_id" in cols
    assert "is_active" not in cols  # บูลีนไม่ใช่คีย์
    assert "const" not in cols  # ค่าเดียวไม่ใช่คีย์


def test_discover_keys_name_hinted_non_unique_fk():
    # store_id ที่ซ้ำ (FK) ควรถูกเก็บเป็น KeyCandidate ด้วย (is_unique=False)
    df = pd.DataFrame({"store_id": [1, 1, 2, 2, 3, 3], "amount": [10, 20, 30, 40, 50, 60]})
    keys = discover_keys(df, "orders")
    by_col = {k.column: k for k in keys}
    assert "store_id" in by_col
    assert by_col["store_id"].is_unique is False
    assert by_col["store_id"].name_hint is True


# --------------------------------------------------------- match_relationships
def test_match_relationships_basic():
    customers = pd.DataFrame({"customer_id": [1, 2, 3, 4, 5], "name": ["a", "b", "c", "d", "e"]})
    orders = pd.DataFrame({"order_id": [10, 20, 30], "customer_id": [1, 2, 6]})  # 6 = orphan
    tables = {"CUSTOMER": customers, "ORDER": orders}
    rels = match_relationships(tables, _profiles(tables))
    assert len(rels) >= 1
    rel = [r for r in rels if r.from_table == "ORDER" and r.to_table == "CUSTOMER"][0]
    assert rel.from_column == "customer_id"
    assert rel.cardinality == "1:N"
    assert rel.orphan_count == 1  # customer_id=6 ไม่มีใน CUSTOMER
    assert rel.overlap_ratio < 1.0
    assert rel.is_validated is True


def test_date_trap_no_link():
    orders = pd.DataFrame({"date": ["2024-01-01", "2024-01-02"], "amount": [100, 200]})
    inventory = pd.DataFrame({"date": ["2024-01-01", "2024-01-03"], "stock": [5, 10]})
    tables = {"ORDER": orders, "INVENTORY": inventory}
    rels = match_relationships(tables, _profiles(tables))
    # ทั้งสองฝั่ง date ไม่ unique → ไม่ควรเชื่อม
    date_rels = [r for r in rels if r.from_column == "date"]
    assert date_rels == []


def test_date_dimension_links_but_siblings_dont():
    # DATE_DIM.date เป็น PK (unique) → ตารางอื่นที่มี date เชื่อมไปหา DATE_DIM ได้
    date_dim = pd.DataFrame({"date": [f"2024-01-{d:02d}" for d in range(1, 11)]})
    orders = pd.DataFrame({"date": ["2024-01-01"] * 3 + ["2024-01-02"] * 3})
    inventory = pd.DataFrame({"date": ["2024-01-01"] * 4 + ["2024-01-05"] * 4})
    tables = {"DATE_DIM": date_dim, "ORDER": orders, "INVENTORY": inventory}
    rels = match_relationships(tables, _profiles(tables))
    targets = {(r.from_table, r.to_table) for r in rels}
    assert ("ORDER", "DATE_DIM") in targets
    assert ("INVENTORY", "DATE_DIM") in targets
    # ORDER ↔ INVENTORY (date ทั้งคู่ไม่ unique) ต้องไม่เชื่อมกันโดยตรง
    assert ("ORDER", "INVENTORY") not in targets
    assert ("INVENTORY", "ORDER") not in targets


def test_thai_digit_keys():
    # customer_id เป็นเลขไทยในไฟล์หนึ่ง อารบิกในอีกไฟล์ — หลัง normalize ควร overlap ได้
    customers = pd.DataFrame({"customer_id": ["๑๐๐", "๒๐๐", "๓๐๐", "๔๐๐", "๕๐๐"]})
    orders = pd.DataFrame(
        {"customer_id": ["100", "200", "300", "999", "100"], "amount": [1, 2, 3, 4, 5]}
    )
    tables = {"CUSTOMER": customers, "ORDER": orders}
    rels = match_relationships(tables, _profiles(tables))
    rel = [r for r in rels if r.from_table == "ORDER" and r.to_table == "CUSTOMER"][0]
    # distinct FK = {100,200,300,999}; 3 ใน 4 ตรง → overlap 0.75, orphan 1 (999)
    assert rel.orphan_count == 1
    assert rel.overlap_ratio == 0.75


def test_float_id_normalization():
    # ORDER.customer_id ที่เป็น float (มี NaN) "3827.0" ต้อง match กับ int "3827"
    customers = pd.DataFrame({"customer_id": [1, 2, 3827, 4, 5, 6]})
    orders = pd.DataFrame(
        {"customer_id": [1.0, 2.0, 3827.0, float("nan"), 5.0, 1.0], "x": [1, 2, 3, 4, 5, 6]}
    )
    tables = {"CUSTOMER": customers, "ORDER": orders}
    rels = match_relationships(tables, _profiles(tables))
    rel = [r for r in rels if r.from_table == "ORDER" and r.to_table == "CUSTOMER"][0]
    assert rel.orphan_count == 0  # ทุกค่า (ยกเว้น NaN) อยู่ใน CUSTOMER
    assert rel.overlap_ratio == 1.0


def test_one_to_one():
    a = pd.DataFrame({"user_id": [1, 2, 3, 4, 5, 6]})
    b = pd.DataFrame({"user_id": [1, 2, 3, 4, 5, 6], "score": [9, 8, 7, 6, 5, 4]})
    tables = {"A": a, "B": b}
    rels = match_relationships(tables, _profiles(tables))
    assert len(rels) == 1
    assert rels[0].cardinality == "1:1"


def test_confidence_scoring():
    customers = pd.DataFrame({"customer_id": [1, 2, 3, 4, 5, 6]})
    orders = pd.DataFrame({"customer_id": [1, 2, 3, 1, 2, 3], "x": [1, 2, 3, 4, 5, 6]})
    tables = {"CUSTOMER": customers, "ORDER": orders}

    # จับคู่ด้วยชื่ออย่างเดียว → ความมั่นใจต่ำ
    name_only = match_relationships(tables, _profiles(tables), validate_values=False)
    rel_name = [r for r in name_only if r.from_table == "ORDER"][0]
    assert rel_name.match_method == "name"
    assert rel_name.is_validated is False

    # จับคู่ด้วยชื่อ + ค่า (overlap 100%) → ความมั่นใจสูงกว่า
    validated = match_relationships(tables, _profiles(tables), validate_values=True)
    rel_val = [r for r in validated if r.from_table == "ORDER"][0]
    assert rel_val.match_method == "both"
    assert rel_val.confidence > rel_name.confidence
    assert rel_val.confidence >= 0.9


def test_reject_low_overlap():
    # ชื่อคอลัมน์ตรงกัน (code) แต่ค่าไม่เกี่ยวกันเลย → ปฏิเสธ (overlap < 0.5)
    a = pd.DataFrame({"code": [1, 2, 3, 4, 5, 6]})
    b = pd.DataFrame({"code": [100, 200, 300, 400, 500, 600], "v": [1, 2, 3, 4, 5, 6]})
    tables = {"A": a, "B": b}
    rels = match_relationships(tables, _profiles(tables), validate_values=True)
    assert rels == []


def test_no_validate_keeps_name_match():
    # ปิด validate → ไม่ปฏิเสธแม้ค่าไม่ตรง (จับคู่ด้วยชื่ออย่างเดียว)
    a = pd.DataFrame({"code": [1, 2, 3, 4, 5, 6]})
    b = pd.DataFrame({"code": [100, 200, 300, 400, 500, 600], "v": [1, 2, 3, 4, 5, 6]})
    tables = {"A": a, "B": b}
    rels = match_relationships(tables, _profiles(tables), validate_values=False)
    assert len(rels) == 1
    assert rels[0].overlap_ratio == 0.0


# ----------------------------------------------------------- profile_dataset
def _write_coffee(tmp_path):
    """สร้างชุดข้อมูลตัวอย่าง (STORE/CUSTOMER/ORDER) ในไดเรกทอรีชั่วคราว."""
    stores = pd.DataFrame({"store_id": [1, 2, 3, 4, 5], "name": ["a", "b", "c", "d", "e"]})
    customers = pd.DataFrame(
        {
            "customer_id": [1, 2, 3, 4, 5, 6, 7, 8],
            "preferred_store_id": [1, 2, 3, 1, 2, 3, 1, 2],
        }
    )
    orders = pd.DataFrame(
        {
            "order_id": list(range(1, 11)),
            "store_id": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
            "customer_id": [1, 2, 3, 4, 5, 6, 7, 8, 1, 99],  # 99 = orphan
        }
    )
    stores.to_csv(tmp_path / "STORE.csv", index=False, encoding="utf-8")
    customers.to_csv(tmp_path / "CUSTOMER.csv", index=False, encoding="utf-8")
    orders.to_csv(tmp_path / "ORDER.csv", index=False, encoding="utf-8")


def test_profile_dataset_directory(tmp_path):
    _write_coffee(tmp_path)
    ds = profile_dataset(str(tmp_path))
    assert isinstance(ds, DatasetProfile)
    assert len(ds.tables) == 3
    names = {t.name for t in ds.tables}
    assert names == {"STORE", "CUSTOMER", "ORDER"}
    # ORDER → STORE (store_id) และ ORDER → CUSTOMER (customer_id)
    targets = {(r.from_table, r.to_table) for r in ds.relationships}
    assert ("ORDER", "STORE") in targets
    assert ("ORDER", "CUSTOMER") in targets


def test_profile_dataset_file_list(tmp_path):
    _write_coffee(tmp_path)
    files = [str(tmp_path / "STORE.csv"), str(tmp_path / "ORDER.csv")]
    ds = profile_dataset(files)
    assert len(ds.tables) == 2
    targets = {(r.from_table, r.to_table) for r in ds.relationships}
    assert ("ORDER", "STORE") in targets


def test_profile_dataset_orphan_findings(tmp_path):
    _write_coffee(tmp_path)
    ds = profile_dataset(str(tmp_path))
    # customer_id=99 ใน ORDER ไม่มีใน CUSTOMER → ต้องมี orphan finding
    assert any("ORDER" in f and "customer_id" in f for f in ds.orphan_findings)


def test_profile_dataset_skips_large_file(tmp_path):
    _write_coffee(tmp_path)
    ds = profile_dataset(str(tmp_path), max_file_size_mb=0.000001)
    # ทุกไฟล์ใหญ่กว่า 1 ไบต์ → ถูกข้ามทั้งหมด
    assert len(ds.tables) == 0
    assert any("ข้าม" in n for n in ds.notes)


# ----------------------------------------------------------------- to_mermaid
def test_to_mermaid(tmp_path):
    _write_coffee(tmp_path)
    ds = profile_dataset(str(tmp_path))
    mermaid = ds.to_mermaid()
    assert mermaid.startswith("erDiagram")
    assert "STORE {" in mermaid
    assert "ORDER {" in mermaid
    # มีเส้นความสัมพันธ์ (1:N) จาก STORE ไป ORDER
    assert "||--o{" in mermaid
    assert "STORE ||--o{ ORDER" in mermaid
    # คอลัมน์ PK ถูกทำเครื่องหมาย
    assert "store_id PK" in mermaid


def test_to_mermaid_sanitizes_weird_names():
    rel = Relationship(
        from_table="ORDER",
        from_column="store id",
        to_table="STORE-A",
        to_column="store id",
        match_method="both",
        overlap_ratio=1.0,
        orphan_count=0,
        orphan_ratio=0.0,
        cardinality="1:N",
        confidence=1.0,
        description_th="x",
        is_validated=True,
    )
    kc = KeyCandidate("STORE-A", "store id", True, 0.0, 5, False, "object")
    t1 = TableProfile("STORE-A", "", 5, 1, ["store id"], {"store id": "id"}, [kc])
    t2 = TableProfile("ORDER", "", 5, 1, ["store id"], {"store id": "id"}, [])
    ds = DatasetProfile(tables=[t1, t2], relationships=[rel])
    mermaid = ds.to_mermaid()
    # ชื่อที่มีช่องว่าง/ขีด ต้องถูกแปลงเป็น _ (ไม่ทำให้ Mermaid พัง)
    assert "STORE-A" not in mermaid
    assert "STORE_A" in mermaid
    assert "store id" not in mermaid
    assert "store_id" in mermaid


# --------------------------------------------------------------- DatasetReport
def test_dataset_report_html(tmp_path):
    _write_coffee(tmp_path)
    ds = profile_dataset(str(tmp_path))
    report = DatasetReport(ds, lang="th")
    out = tmp_path / "dataset.html"
    html = report.to_html(str(out))
    assert out.is_file()
    assert "<!DOCTYPE html>" in html
    assert 'class="mermaid"' in html
    assert "erDiagram" in html
    assert "mermaid" in html  # CDN script
    # ส่วนต่าง ๆ ปรากฏ
    assert "ความสัมพันธ์" in html
    assert "STORE" in html


def test_dataset_report_json(tmp_path):
    _write_coffee(tmp_path)
    ds = profile_dataset(str(tmp_path))
    report = DatasetReport(ds, lang="th")
    out = tmp_path / "dataset.json"
    report.to_json(str(out))
    import json

    parsed = json.loads(out.read_text(encoding="utf-8"))
    assert parsed["table_count"] == 3
    assert "mermaid" in parsed
    assert "relationships" in parsed


def test_dataset_report_english(tmp_path):
    _write_coffee(tmp_path)
    ds = profile_dataset(str(tmp_path))
    html = DatasetReport(ds, lang="en").to_html()
    assert "Relationships" in html
    assert "ER Diagram" in html


# --------------------------------------------------------------- CLI
def test_cli_dataset(tmp_path, capsys):
    from thaieda.cli import main

    _write_coffee(tmp_path)
    out = tmp_path / "report.html"
    rc = main(["dataset", str(tmp_path), "-o", str(out)])
    assert rc == 0
    assert out.is_file()
    captured = capsys.readouterr().out
    assert "วิเคราะห์ชุดข้อมูล" in captured


def test_cli_dataset_file_list(tmp_path):
    from thaieda.cli import main

    _write_coffee(tmp_path)
    out = tmp_path / "report.html"
    rc = main(
        [
            "dataset",
            str(tmp_path / "STORE.csv"),
            str(tmp_path / "ORDER.csv"),
            "-o",
            str(out),
            "--json",
            str(tmp_path / "out.json"),
        ]
    )
    assert rc == 0
    assert out.is_file()
    assert (tmp_path / "out.json").is_file()


def test_cli_dataset_no_validate(tmp_path):
    from thaieda.cli import main

    _write_coffee(tmp_path)
    out = tmp_path / "report.html"
    rc = main(["dataset", str(tmp_path), "-o", str(out), "--no-validate", "--quiet"])
    assert rc == 0
    assert out.is_file()


def test_cli_profile_autoroutes_directory(tmp_path):
    from thaieda.cli import main

    _write_coffee(tmp_path)
    out = tmp_path / "auto.html"
    # profile กับไดเรกทอรี → auto-route ไป dataset mode
    rc = main(["profile", str(tmp_path), "-o", str(out)])
    assert rc == 0
    assert out.is_file()
    assert "erDiagram" in out.read_text(encoding="utf-8")
