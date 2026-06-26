# ThaiEDA QA Pipeline — Kanban Board

## 📋 Backlog
(none)

## 🔄 In Progress
(none)

## ✅ Done
- [x] **B1**: Claude Code Batch 1 — แก้ 5 defects (C1/C2/D1/R1/U1) — 41 turns, $2.85
- [x] **S1**: แก้ QA scanner false positives (no_data_placeholder) + section matching (aliases + h2 span)
- [x] **D1-D5**: ดาวน์โหลด datasets เดิม 9 อัน (titanic, telco, wine, california, superstore, adult, bank, online-retail, dirty-thai)

## 📊 Dataset Summary (14 total)

### Existing (9)
| # | Dataset | Rows | Cols | Size | Domain | Types |
|---|---------|------|------|------|--------|-------|
| 1 | titanic | 891 | 15 | 60KB | passenger survival | text+num+cat |
| 2 | telco-churn | 7,043 | 21 | 970KB | telecom churn | num+cat |
| 3 | wine-quality | 1,599 | 12 | 84KB | wine chemistry | num only |
| 4 | california-housing | 20,640 | 9 | 1.4MB | housing census | num+text |
| 5 | superstore | 10,800 | 21 | 2.3MB | retail sales | dates+text+num |
| 6 | adult | 32,561 | 15 | 4.0MB | census demographics | text+num+cat |
| 7 | bank-marketing | 41,188 | 21 | 5.8MB | bank marketing | num+cat (sep=;) |
| 8 | online-retail | 541,909 | 8 | 48.6MB | e-commerce | dates+text+num |
| 9 | dirty-thai-retail | 500 | 8 | ~50KB | thai retail (dirty) | dates+thai text |

### New (5)
| # | Dataset | Rows | Cols | Size | Domain | Types |
|---|---------|------|------|------|--------|-------|
| 10 | absenteeism | 740 | 21 | 45KB | workplace | num+cat (sep=;) |
| 11 | online-shoppers | 12,330 | 18 | 1.1MB | e-commerce | num+cat+bool |
| 12 | aps-failure | 16,000 | 171 | 11.9MB | truck diagnostics | num (171 cols!) |
| 13 | beijing-pm25 | 43,824 | 13 | 2.0MB | air quality | dates+num+NA |
| 14 | bike-sharing | 17,379 | 17 | 1.2MB | bike rental | dates+num+cat |

## 🐛 Defects Found & Fixed

### Batch 1 (Fixed by Claude Code)
| ID | Type | Description | File | Fix |
|----|------|-------------|------|-----|
| C1 | Clean | placeholder "-" over-flagging <1% | quality/__init__.py | skip single-dash <1% |
| C2 | Clean | repeated-char spam on ID columns | quality/__init__.py | skip alphanumeric-hyphen codes |
| D1 | Detect | Order ID detected as date | detect/__init__.py | guard alphabetic prefix ≥60% |
| R1 | Report | empty Breakdown label | report/_template.py | {% if breakdown %} guard |
| U1 | EDA | date-parse warnings ×6 | report/insight_engine/timeseries | format="mixed" |

### Scanner Fixes (Fixed manually)
| ID | Type | Description | Fix |
|----|------|-------------|-----|
| S1 | Scanner | false positive "No data" | filter ✓/class="empty"/detected context |
| S2 | Scanner | section matching | aliases + h2 span regex |

### Batch 2 (Pending — Claude Code)
| ID | Severity | Type | Description | Target |
|----|----------|------|-------------|--------|
| P1 | CRITICAL | Insight | generate_insights() ไม่มี cap → 679 insights (aps) | ≤30 insights |
| P2 | HIGH | Report | HTML bloat: bike-sharing 3.7MB/61 charts, aps 2.3MB/208 tables | <2MB |
| P3 | HIGH | Perf | aps 491s (171 cols), online-retail 397s (541K rows) | <120s |
| Q1 | MEDIUM | Clean | handle_missing_values ไม่ handle >40% NA columns | mostly_missing flag |
| Q2 | MEDIUM | Report | Template แสดง all insights ไม่มี collapse | ≤20 + collapse |