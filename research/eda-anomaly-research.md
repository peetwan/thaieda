# ThaiEDA Enhancement Research — 2026-06-24

สรุปเทคนิคใหม่ ๆ จากการ research พร้อมประเมินว่าควรใส่ ThaiEDA หรือไม่

---

## 1. Anomaly Detection Algorithms

### สิ่งที่มีแล้วใน ThaiEDA v0.1
- z-score, modified z-score (MAD), IQR

### เทคนิคใหม่ที่เจอ

**a) Isolation Forest**
- สรุป: ใช้ random decision trees แยก anomalies ออกจาก majority — anomalies ถูก isolate ไวกว่าเพราะ "few and different"
- ข้อดี: เร็วมาก, ทำงานกับ high-dimensional data ได้ดี, ให้ anomaly score (ไม่ใช่แค่ binary)
- ข้อเสีย: ต้องมี scikit-learn (dependency หนัก), ต้อง tuning n_estimators/contamination
- **แนะนำ: ใส่ ✅** เป็น optional (extras `[ml]`), ใช้เมื่อข้อมูล >1000 rows, ให้ score ไม่ใช่แค่ flag
- Source: https://www.digitalocean.com/community/tutorials/anomaly-detection-isolation-forest

**b) Local Outlier Factor (LOF)**
- สรุป: เปรียบเทียบ local density ของจุดกับ neighbors — จุดที่ density ต่ำกว่าเพื่อนบ้าน = outlier
- ข้อดี: จับ local anomalies ได้ (cluster ที่มี outlier แบบ local), ให้ score
- ข้อเสีย: ช้ากับข้อมูลใหญ่ (k-NN computation), ต้องเลือก n_neighbors
- **แนะนำ: ใส่ ✅** เป็น optional, ใช้เมื่อข้อมูลมีหลาย cluster
- Source: https://medium.com/@priyanjalipatel/anomaly-detection-explained-isolation-forest-dbscan-and-local-outlier-factor-0c7af4e2c651

**c) DBSCAN**
- สรุป: density-based clustering — จุดที่ไม่อยู่ใน cluster ใด ๆ = noise/outlier
- ข้อดี: ไม่ต้องรู้จำนวน cluster ล่วงหน้า, จับ anomalies ที่เป็น noise
- ข้อเสีย: ต้อง tune eps/min_samples, ช้ากับข้อมูลใหญ่, ไม่ทำงานกับ high-dim
- **แนะนำ: ไม่ใส่ ❌** เกินจำเป็นสำหรับ EDA tool — IF + LOF ครอบคลุมพอแล้ว, DBSCAN ใช้ทรัพยากรเยอะ
- Source: https://www.itm-conferences.org/articles/itmconf/abs/2025/01/itmconf_dai2024_04012/itmconf_dai2024_04012.html

---

## 2. Modern EDA — Automated Insight Generation

**a) QUIS (Question-guided Insights)**
- สรุป: ระบบ EDA อัตโนมัติที่ generate คำถามเกี่ยวกับข้อมูล แล้ว explore subspaces เพื่อหา insights
- ข้อดี: ค้นพบ insights ที่คนอาจมองข้าม เช่น "ยอดขายสูงสุดในกลุ่มลูกค้า X ช่วงเวลา Y"
- ข้อเสีย: ซับซ้อนมาก, เป็น research system ไม่ใช่ library ที่ใช้งานง่าย
- **แนะนำ: ไม่ใส่ตอนนี้ ❌** แต่เก็บไว้ใน roadmap — ใช้แนวคิดใน v0.3 LLM Q&A แทน
- Source: https://ojs.aaai.org/index.php/AAAI/article/view/35360/37515

**b) Data Drift Detection**
- สรุป: ตรวจจับการเปลี่ยนแปลงของ distribution เมื่อเทียบกับ baseline (มักใช้ใน production ML)
- ข้อดี: มีประโยชน์สำหรับ monitoring ข้อมูล
- ข้อเสีย: ต้องมี baseline dataset — เกินขอบเขต EDA แบบ one-shot
- **แนะนำ: ไม่ใส่ ❌** นอกขอบเขต EDA — เป็น feature ของ MLOps tool ไม่ใช่ EDA tool
- Source: https://www.acceldata.io/blog/data-drift

---

## 3. Data Visualization — Modern Chart Types

### สิ่งที่มีแล้ว
- Histogram, bar chart (top tokens), word cloud

### เทคนิคใหม่ที่ควรใส่

**a) Correlation Heatmap**
- สรุป: แสดง correlation matrix ระหว่าง numeric columns เป็นสี
- ข้อดี: เห็น relationship ระหว่างตัวแปรได้ทันที, เป็นมาตรฐานของ EDA
- **แนะนำ: ใส่ ✅** ต้องมีทุก EDA tool
- Source: https://medium.com/@lomashbhuva/data-visualization-exploratory-data-analysis-eda-1c347fdd7505

**b) Box Plot (distribution comparison)**
- สรุป: เปรียบเทียบ distribution ของหลาย columns ใน chart เดียว
- ข้อดี: เห็น outlier, median, IQR ได้ในมุมเดียว
- **แนะนำ: ใส่ ✅** มีประโยชน์มาก โดยเฉพาะเมื่อมี numeric columns หลายตัว
- Source: https://fivenumbersummary.io/blog/data-visualization-best-practices

**c) Missing Data Matrix (missingno-style)**
- สรุป: แสดงตำแหน่งของ missing values เป็น matrix — แถว = row, คอลัมน์ = column, สีดำ = missing
- ข้อดี: เห็น pattern ของ missing data ได้ทันที (MCAR vs MAR vs MNAR)
- ข้อเสีย: ข้อมูลใหญ่มาก ๆ อาจช้า
- **แนะนำ: ใส่ ✅** ทำเองด้วย matplotlib ไม่ต้อง depend missingno
- Source: https://github.com/residentmario/missingno, https://www.nb-data.com/p/how-to-visualize-missing-data-patterns

**d) Missing Data Heatmap (nullity correlation)**
- สรุป: correlation ระหว่าง missing patterns — "เมื่อคอลัมน์ A missing, B ก็ missing ด้วยไหม"
- ข้อดี: เห็น relationship ของ missing patterns
- **แนะนำ: ใส่ ✅** ใช้คู่กับ missing matrix

**e) Violin Plot**
- สรุป: แสดง distribution shape (ไม่ใช่แค่ summary stats แบบ box plot)
- **แนะนำ: ใส่ ✅** เป็น alternative ของ box plot เมื่อ distribution ไม่ใช่ normal

**f) Auto chart selection**
- สรุป: เลือก chart type อัตโนมัติตาม data type และจำนวน variables
- **แนะนำ: ใส่ ✅** logic: numeric×numeric → scatter, numeric distribution → histogram/violin, categorical → bar, missing → matrix
- Source: https://research.lib.buffalo.edu/dataviz/best-practices

---

## 4. Text Anomaly Detection — Encoding/Mojibake

### สิ่งที่มีแล้ว
- ตรวจจับ mojibake พื้นฐาน (UTF-8 bytes ผิด, replacement char U+FFFD)

### เทคนิคใหม่

**a) ftfy (fixes text for you)**
- สรุป: library ที่ fix mojibake และ Unicode glitches อัตโนมัติ — detect encoding errors แล้ว restore
- ข้อดี: ทำงานได้ดีมาก, มี `fix_and_explain()` บอกว่าแก้อะไรยังไง
- ข้อเสีย: Apache license (ต้อง attribute), บางครั้ง "fix" ข้อมูลที่ไม่ได้เสีย จริง
- **แนะนำ: ใส่ ✅** มีอยู่แล้วใน extras `[fix]` — ปรับให้ใช้ `fix_and_explain()` และแสดงใน report
- Source: https://github.com/rspeer/python-ftfy, https://alexwlchan.net/notes/2025/ftfy-fix-and-explain

**b) Thai-specific encoding detection (tis-620 / cp874)**
- สรุป: tis-620 และ cp874 เป็น encoding ภาษาไทย — mojibake เกิดเมื่อ UTF-8 Thai ถูก decode เป็น Latin-1/cp1252
- รูปแบบที่พบบ่อย: `à¸ªà¸§à¸±à¸ªà¸"à¸µ` = "สวัสดี" ที่ถูก encode UTF-8 แล้ว decode ผิด
- **แนะนำ: ใส่ ✅** เพิ่ม Thai-specific mojibake pattern detection — นี่คือจุดเด่นของ ThaiEDA
- Source: https://forage.ai/blog/character-encoding-bugs-web-scraping-guide

---

## 5. Data Cleaning — Deduplication & Fuzzy Matching

### สิ่งที่มีแล้ว
- fuzzy duplicates ใน categorical (difflib.SequenceMatcher ratio >0.8)

### เทคนิคใหม่

**a) rapidfuzz (แทน difflib)**
- สรุป: fuzzy string matching ที่เร็วกว่า difflib 10-100x, C++ implementation
- ข้อดี: เร็วมาก, มี Levenshtein, Jaro-Winkler, token-based ratios
- ข้อเสีย: dependency เพิ่ม (แต่เบา)
- **แนะนำ: ใส่ ✅** เป็น optional extras `[fuzzy]`, fallback ไป difflib ถ้าไม่ได้ติดตั้ง
- Source: https://medium.com/towards-data-engineering/i-benchmarked-fuzzy-matching-at-1m-rows-heres-where-python-libraries-break-ba270b8b54b1

**b) Python Record Linkage Toolkit**
- สรุป: full record linkage / deduplication framework
- **แนะนำ: ไม่ใส่ ❌** เกินขอบเขต EDA — เป็น tool สำหรับ entity resolution ไม่ใช่ EDA
- Source: https://github.com/J535D165/data-matching-software

---

## สรุป: สิ่งที่ควรใส่ใน v0.1.1 (enhancement)

| Feature | ใส่ไหม | Priority | Dependency |
|---------|--------|----------|------------|
| Isolation Forest anomaly | ✅ | High | scikit-learn (optional `[ml]`) |
| LOF anomaly | ✅ | High | scikit-learn (optional `[ml]`) |
| Correlation heatmap | ✅ | High | matplotlib (มีแล้ว) |
| Box plot | ✅ | High | matplotlib (มีแล้ว) |
| Missing data matrix | ✅ | High | matplotlib (มีแล้ว) |
| Missing data heatmap | ✅ | Medium | matplotlib (มีแล้ว) |
| Violin plot | ✅ | Medium | matplotlib (มีแล้ว) |
| Auto chart selection | ✅ | Medium | logic only |
| ftfy fix_and_explain | ✅ | Medium | ftfy (มีแล้วใน `[fix]`) |
| Thai mojibake patterns | ✅ | High | none |
| rapidfuzz fuzzy matching | ✅ | Low | rapidfuzz (optional `[fuzzy]`) |

### ไม่ใส่
- DBSCAN (เกินจำเป็น)
- QUIS (ซับซ้อนเกินไป, เก็บไว้ใน LLM v0.3)
- Data drift detection (นอกขอบเขต EDA)
- Record Linkage Toolkit (เกินขอบเขต)

---

## 2026-06-25 — Research Update

### Extended Isolation Forest (EIF)

เป็นการพัฒนาของ Isolation Forest โดย Hariri et al. แก้ bias จาก axis-aligned splits ของ IF ดั้งเดิม — ใช้ random slope + intercept (oblique cuts) แทนการเลือก feature + threshold ทำให้ anomaly score แม่นยำขึ้นโดยเฉพาะข้อมูลที่มี structure ซับซ้อน มี `extension_level` ควบคุม degree ของ extension (0 = IF ปกติ, P-1 = full extension) มี implementation ใน H2O และ `sahandha/eif` บน GitHub แต่ยังไม่อยู่ใน scikit-learn อย่างเป็นทางการ
- **เหมาะกับ ThaiEDA:** เป็นทางเลือกขั้นสูงให้ผู้ใช้ที่ต้องการ accuracy สูงกว่า IF ปกติ — แต่ dependency หนัก แนะนำเก็บเป็น roadmap ไม่ใส่ v0.1.1
- Source: https://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/eif.html

### PyCaret Anomaly Detection — Multi-Algorithm API Design

PyCaret มี anomaly module ที่รวมหลาย algorithms ไว้ใน API เดียว (`iforest`, `lof`, `knn`, `svm`, `pca`, `mcd`, `sod`, `abod`, `cluster`, `cof`, `histogram`, `sos`) พร้อม auto-preprocessing pipeline (imputation, encoding, scaling) และ visualization ผ่าน t-SNE/UMAP ใช้ pattern `setup() → create_model() → assign_model() → predict_model()` ที่สะอาดมาก มี `contamination` parameter ควบคุม threshold และ `plot_model()` สร้าง embedding plots
- **เหมาะกับ ThaiEDA:** เอา API design pattern มาอ้างอิง — แนวคิด unified `detect_anomalies(df, method='iforest'|'lof'|'zscore')` ที่รวมหลาย method ในฟังก์ชันเดียว ทำให้ user เปรียบเทียบได้ง่าย
- Source: https://pycaret.readthedocs.io/en/latest/api/anomaly.html

### AutoViz — Smart Chart Selection

AutoViz วิเคราะห์ data structure อัตโนมัติแล้วเลือก chart type ที่เหมาะสม — สร้าง 35+ visualizations ด้วย code บรรทัดเดียว เลือก chart ตาม data type และจำนวน variables: numeric×numeric → scatter, distribution → histogram/violin, categorical → bar, missing → matrix ทำงานกับ CSV/DataFrame ได้โดยตรง
- **เหมาะกับ ThaiEDA:** ยืนยันแนวคิด auto chart selection ที่อยู่ใน research ก่อนหน้า — ควร implement logic แบบนี้ใน `viz/` module โดยไม่ depend AutoViz เอง (เพราะ ThaiEDA ต้องการ Thai font support ที่ AutoViz ไม่มี)
- Source: https://www.autoviz.ai/features

### Sweetviz — Target Analysis & Dataset Comparison

Sweetviz สร้าง HTML report ครอบคลุม distributions, missing values, correlations, data types ด้วย `analyze()` บรรทัดเดียว — จุดเด่นคือ **target variable analysis** (แสดงความสัมพันธ์ของทุก column กับ target) และ **dataset comparison** (เปรียบเทียบ train vs test หรือ 2 dataset ใน report เดียว) เป็น inspiration จาก Pandas-Profiling แต่ focus ที่ comparison
- **เหมาะกับ ThaiEDA:** เอา pattern target analysis มาใช้ใน `report/` module — ถ้า user ระบุ target column ให้ report แสดง correlation/association ของทุก column กับ target และ dataset comparison มีประโยชน์สำหรับเปรียบเทียบ before/after cleaning
- Source: https://www.statology.org/automated-exploratory-data-analysis-with-sweetviz-in-python

### PyThaiNLP `normalize()` — Thai Text Normalization Utility

PyThaiNLP `pythainlp.util.normalize()` เป็นฟังก์ชันรวม normalization ภาษาไทย — เรียก `remove_zw()` (ลบ zero-width spaces), `remove_dup_spaces()` (ลบช่องว่างซ้ำ) ภายในฟังก์ชันเดียว มี `tis620_to_utf8()` สำหรับ convert encoding เก่า และ `eng_to_thai()` แก้ keyboard layout พิมพ์ผิด (Kedmanee) รวมถึง `count_thai_chars()` นับ consonants/vowels/tone marks แยก
- **เหมาะกับ ThaiEDA:** ควร integrate `normalize()` เข้ากับ `clean/` module สำหรับทำ text cleaning ก่อน profiling, และ `tis620_to_utf8()` สำหรับ fix encoding เก่า — เป็น optional dep ที่มีอยู่แล้ว เพิ่ม wrapper ให้ใช้ง่ายใน ThaiEDA context
- Source: https://pythainlp.org/dev-docs/api/util.html

---

## 2026-06-26 — Research Update

### AERCA — Granger Causal Discovery สำหรับ Root Cause ของ Anomaly (ICLR 2025)

AERCA เป็น encoder-decoder architecture ที่เรียนรู้ Granger causality ระหว่าง multivariate time series พร้อม model distribution ของ exogenous variables ทำให้ไม่เพียงแต่บอกว่าเกิด anomaly ที่ตัวแปรใด แต่ระบุ root cause time series และ root cause time steps ได้ด้วย — ผลลัพธ์ state-of-the-art ทั้ง Granger causal discovery และ root cause identification บนหลาย dataset แนวคิดหลักคือสร้าง structural causal model ก่อนแล้ว highlight ช่วงเวลาที่ intervention เกิด
- **เหมาะกับ ThaiEDA:** ยังเกินขอบเขต EDA แบบ one-shot ในขณะนี้ ❌ แต่แนวคิด causal discovery เก็บไว้เป็น roadmap ระยะไกลสำหรับ `timeseries/` module — หากเพิ่ม baseline comparison ในอนาคต การบอก "column ไหนเป็นต้นเหตุ" จะมีค่ามาก
- Source: https://proceedings.iclr.cc/paper_files/paper/2025/file/6fde96479648d71e4fd9724374bf76eb-Paper-Conference.pdf

### Large-Scale Benchmark ของ Data Cleaning Tools (MDPI Data 2025)

งานวิจัยจาก MDPI Data (vol. 10, 2025) เปรียบเทียบ 5 เครื่องมือ data cleaning บน dataset สกปรกขนาดใหญ่ 3 โดเมน (healthcare, finance, industrial telemetry): OpenRefine, Dedupe, Great Expectations, TidyData/PyJanitor และ baseline Pandas pipeline วัด performance + scalability พบว่าแต่ละเครื่องมือมีจุดแข็งต่างกันตาม structure ของ algorithm — Dedupe เก่ง entity resolution, Great Expectations เก่ง validation-driven quarantine ส่วน PyJanitor เป็น lightweight chaining API บน Pandas
- **เหมาะกับ ThaiEDA:** ควรอ้างอิงเป็น justification สำหรับการเลือก lightweight approach (Pandas + vectorized) แทน dependency หนัก ❌ ไม่ adopt เครื่องมือใดตรง ๆ แต่เอาแนวคิด "expectation + quarantine DF" จาก Great Expectations มาใช้ใน `quality/` — แยก bad rows ไว้ review แทนลบทิ้งเงียบ ๆ
- Source: https://www.mdpi.com/2306-5729/10/5/68

### Drift Monitoring ยุค LLM — Embedding Drift + Eval-Score Logging (2026)

แนวทางล่าสุดปี 2026 สำหรับ production ML/LLM คือ instrumentation ทุก trace ด้วย embedding + eval-score logging แล้ว alert เมื่อ input drift และ eval drop เกิดพร้อมกัน (joint condition) แยก data drift (P(X)) จาก model drift (P(Y|X)) ชัดเจน — PSI, KS test, embedding cosine เป็น metrics หลัก สำหรับ LLM จะวัด prompt/embedding drift บน trace และ online faithfulness/groundedness
- **เหมาะกับ ThaiEDA:** ยังนอกขอบเขต EDA one-shot ❌ แต่ถ้าอนาคตเพิ่ม baseline-comparison mode (เปรียบเทียบ 2 dataset) ควรใช้ PSI/KS test แทน correlation แบบเดิม — เป็น upgrade ของ dataset comparison ที่เคยเห็นใน Sweetviz
- Source: https://futureagi.com/blog/model-vs-data-drift-how-to-identify-and-handle-it

### AI-Powered Anomaly-Guided Data Cleaning (2026)

แนวโน้มปี 2026 คือ AI data cleaning ฝังใน data stack ที่ใช้ anomaly detection ชี้นำการทำความสะอาด — ระบบตรวจจับ pattern ผิดปกติ (format, range, semantic) แล้วแนะนำ fix อัตโนมัติพร้อม explain ว่าทำไม ลด manual rule-based cleansing ที่ตามไม่ทันปริมาณข้อมูล จุดสำคัญคือ automation + governance รวมกัน ไม่ใช่แค่ลบข้อมูล
- **เหมาะกับ ThaiEDA:** นี่คือทิศทางที่ ThaiEDA ควรไป — เชื่อม `anomaly/` module กับ `clean/` module ให้ผล anomaly detection ชี้นำว่าควร clean อะไรก่อน ✅ เช่น พบ mojibake → แนะนำ `convert_encoding()`, พบ Buddhist era → แนะนำ `convert_buddhist_era()` แล้วรายงานเป็น actionable note ใน `report/`
- Source: https://www.ovaledge.com/blog/ai-data-cleaning

---

## 2026-06-27 — Research Update

### HBOS (Histogram-based Outlier Score) — Linear-Time Unsupervised Anomaly Detection

HBOS เป็น unsupervised algorithm ที่สร้าง univariate histogram แต่ละ feature แยก แล้วรวม log-probability เป็น anomaly score ในเวลาเชิงเส้น (linear time) โดย assume features independent กัน — เร็วกว่า IF/LOF มากและ scalable กับข้อมูลใหญ่ แตกต่างจาก IF ที่ต้อง build random trees: HBOS ใช้แค่ histogram binning เลยทำงานได้แม้ข้อมูลหลายแสน rows มีใน PyOD (`pyod.models.hbos`) และ dtaianomaly สำหรับ timeseries
- **เหมาะกับ ThaiEDA:** เป็นทางเลือกเร็วสุดสำหรับข้อมูลใหญ่ที่ IF อาจช้า ✅ ใส่เป็น optional method ใน unified `detect_anomalies(df, method='hbos')` — เหมาะเมื่อ profiling ข้อมูลหลายแสน rows ที่ต้องการคร่าว ๆ ก่อนเลือก algorithm ละเอียด ข้อจำกัด: ความแม่นต่ำกว่า IF เพราะไม่จับ feature interaction
- Source: https://pyod.readthedocs.io/en/latest/_modules/pyod/models/hbos.html

### LLM Agents สำหรับ Cleaning Tabular ML Datasets (arXiv 2503.06664, มีนา 2025)

งานวิจัยจาก arXiv (มีนา 2025) ทดลองให้ LLM จับคู่กับ Python runtime ทำความสะอาด training dataset ที่ corrupt แล้ววัดผลจาก downstream model performance พบว่า LLM ใช้ **contextual information จาก feature อื่น ๆ ใน row เดียวกัน** ตรวจจับและแก้ค่าผิด (illogical values, outliers) ได้ พร้อมรับ feedback จาก iteration ก่อน ๆ ปรับปรุงผลลัพธ์ได้เรื่อย ๆ — แต่ LLM ไม่สามารถแก้ feature engineering หรือ training pipeline ได้ จึง focus ที่ row-level correction ล้วน ๆ
- **เหมาะกับ ThaiEDA:** สนับสนุนแนวทาง LLM module ที่มีอยู่แล้ว (v0.9) ✅ ประเพณีใหม่ที่น่าสนใจคือ "contextual row-aware cleaning" — แทนที่จะ clean ทีละ column แบบ isolate ให้ส่ง row เต็ม ๆ ให้ LLM แก้โดยอ้างอิง feature อื่น เช่น พบ "อายุ 200" ก็ดู column "เกิด พ.ศ. 2540" ช่วยยืนยัน ควรเป็น opt-in mode ใน `llm/` module
- Source: https://arxiv.org/html/2503.06664v1

### pyunormalize — Unicode 17.0 (กันยา 2025) Pure-Python NFKC Conformance

`pyunormalize` เป็น pure-Python implementation ของ Unicode normalization algorithm รองรับ Unicode Standard 17.0 (ปล่อยกันยา 2025) ให้ฟังก์ชัน `NFC/NFD/NFKC/NFKD` แยกจาก Python core Unicode database — ทำให้ได้ normalization ที่ conform กับ Unicode เวอร์ชันใหม่ล่าสุดแม้ Python runtime ยังใช้ UCD เวอร์ชันเก่ากว่า มีประโยชน์เมื่อต้องการ consistency ข้าม environment และครอบคลุม character ใหม่ ๆ ที่เพิ่งเข้ามาใน Unicode 17.0
- **เหมาะกับ ThaiEDA:** ThaiEDA มี `normalize_nfkc` อยู่แล้ว (v1.0 feature 2026-06-26) ที่ใช้ `unicodedata.normalize` ของ Python ✅ ไม่ต้อง depend lib ใหม่ แต่ควร document ว่า accuracy ขึ้นกับ Unicode version ของ Python runtime — ถ้าผู้ใช้เจอ edge case ที่ NFKC ไม่ cover ให้ทราบว่าเป็นข้อจำกัดของ UCD เวอร์ชัน pyunormalize เป็น optional escape hatch สำหรับ conformance ล่าสุด
- Source: https://github.com/mlodewijck/pyunormalize

### DataKitchen 2026 Profiling Landscape — Profiling → AI-Generated Data Hygiene Tests

รายงานปี 2026 จาก DataKitchen สรุป open-source data profiling landscape และชี้ช่องว่างสำคัญ: "most open-source profiling tools stop at describing data; almost none automatically convert profiling insights into actionable data hygiene checks" — ทำให้ profiling กลายเป็น passive report ไม่ใช่ active guardrail สำหรับ AI/LLM workload ที่ bad data ไม่ได้แค่พัง dashboard แต่หลอก LLM ได้ DataOps TestGen เป็นตัวอย่างที่ combine deep profiling กับ AI-generated hygiene tests อัตโนมัติ
- **เหมาะกับ ThaiEDA:** ยืนยันทิศทาง "anomaly-guided cleaning" จากงานวิจัยก่อนหน้า และเสริมแนวคิดใหม่ ✅ นอกจากชี้นำ clean แล้ว ควร generate **assertion/hygiene rules** จาก profiling output เช่น พบ column ที่ 95% เป็น Buddhist era → สร้าง rule "คอลัมน์นี้ควรอยู่ในช่วง พ.ศ. 2540–2570" ใส่ใน `quality/` เป็น auto-generated expectation เพื่อ reuse ใน batch ถัดไป
- Source: https://datakitchen.io/blog/the-2026-open-source-data-profiling-software-landscape

### ABOD (Angle-Based Outlier Detection) — High-Dimensional Outlier via Variance of Angles

ABOD วัดความผิดปกติจาก **variance ของมุมระหว่างจุดเป้าหมายกับคู่ neighbors ทั้งหมด** — outlier จะมี variance ของมุมต่ำ (จุดอยู่ "นอกกรอบ" ทุก neighbor มองในทิศทางใกล้เคียงกัน) ประโยชน์หลักคือทำงานได้ดีใน high-dimensional space ที่ distance-based method (LOF, k-NN) เสื่อมประสิทธิภาพเพราะ "curse of dimensionality" มี fast-ABOD ที่ใช้ k-nearest neighbors แทนทุกคู่เพื่อลด cost จาก O(n²) ลง มีใน PyOD (`pyod.models.abod`)
- **เหมาะกับ ThaiEDA:** มีค่าเมื่อ ThaiEDA ใช้กับข้อมูลหลาย numeric columns (เช่น timeseries telemetry) ที่ dimension สูง ✅ ใส่เป็น optional method ใน unified anomaly API แต่ใช้ fast-ABOD เพื่อหลีกเลี่ยง O(n²) ไม่แนะนำเป็น default เพราะช้ากว่า IF/HBOS ในข้อมูลทั่วไป
- Source: https://blog.paperspace.com/outlier-detection-with-abod