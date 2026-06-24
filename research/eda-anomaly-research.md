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