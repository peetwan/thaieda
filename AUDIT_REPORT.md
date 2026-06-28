# Audit Report - ThaiEDA Project

รายงานผลการตรวจสอบโค้ดเพื่อค้นหาข้อบกพร่องและข้อเสนอแนะในการปรับปรุงโครงการ ThaiEDA

---

## Critical Findings

### 1. ปัญหา Out Of Memory (OOM) ในโหมด Batch
- File: [src/thaieda/cli.py:L1242-1253](file:///C:/Users/User/Desktop/Hermes_Peet/scratch/thaieda/src/thaieda/cli.py#L1242-1253)
- คำอธิบาย: ใน `_run_oneliner_batch` มีการโหลดข้อมูลเต็มไฟล์เก็บในหน่วยความจำพร้อมกันทั้งหมดใน list `file_peeks` เพื่อตรวจหา target column ส่งผลให้หน่วยความจำเต็มเมื่อเจอไฟล์ข้อมูลขนาดใหญ่หลายไฟล์
- Proposed Fix: ในขั้นตอน peek ให้โหลดข้อมูลเพียง 10 แถวแรกเพื่อหาคอลัมน์ และไม่ต้องเก็บ DataFrame ตัวเต็มไว้ในลิสต์ ให้เก็บเฉพาะ metadata ที่จำเป็นสำหรับการเลือก target เท่านั้น
```python
df_peek = read_data(fpath, format=args.format, encoding=args.encoding, nrows=10)
```

### 2. ปัญหา ValueError range parameter must be finite (inf) ใน bimodal check
- File: [src/thaieda/insight/__init__.py:L772](file:///C:/Users/User/Desktop/Hermes_Peet/scratch/thaieda/src/thaieda/insight/__init__.py#L772)
- คำอธิบาย: ฟังก์ชัน `_is_bimodal` เรียก `np.histogram` โดยไม่ได้กรองค่า `inf` หรือ `-inf` ออก ส่งผลให้เกิด ValueError เมื่อมีค่า infinity ในชุดข้อมูล
- Proposed Fix: กรองค่าที่ไม่ใช่ finite ออกจาก numpy array ก่อนส่งให้ `np.histogram`
```python
values = values[np.isfinite(values)]
```

### 3. ปัญหา Listwise Deletion ก่อนคำนวณ Correlation Matrix
- File: [src/thaieda/insight_engine/__init__.py:L1094](file:///C:/Users/User/Desktop/Hermes_Peet/scratch/thaieda/src/thaieda/insight_engine/__init__.py#L1094)
- คำอธิบาย: ฟังก์ชัน `_detect_strong_correlations` ใช้ `.dropna()` บน DataFrame ส่งผลให้ลบแถวทั้งหมดหากมีคอลัมน์ใดคอลัมน์หนึ่งมีค่าว่าง (listwise deletion) ทำให้ไม่สามารถคำนวณ correlation ของคอลัมน์อื่นได้
- Proposed Fix: เอาการเรียก `.dropna()` ท้ายประโยคออก แล้วตรวจสอบจำนวนแถวที่มีค่าครบในแต่ละคู่คอลัมน์ตอนวนลูปคำนวณ correlation แทน
```python
numeric = df[measure_cols].apply(pd.to_numeric, errors="coerce")
# ... ตรวจสอบในลูป ...
valid_mask = numeric[col_a].notna() & numeric[col_b].notna()
if valid_mask.sum() < 10:
    continue
```

### 4. KeyError จากการเรียกใช้คอลัมน์ที่เป็นตัวเลข (Non-string Column Name)
- File: [src/thaieda/analysis/__init__.py:L228-237](file:///C:/Users/User/Desktop/Hermes_Peet/scratch/thaieda/src/thaieda/analysis/__init__.py#L228-237)
- คำอธิบาย: ใน `analyze_target` ชื่อคอลัมน์ถูกแปลงเป็น string ล่วงหน้า ทำให้เมื่อเรียก `df[col]` ดึงข้อมูลใน `_associate` จะเกิด KeyError หากตารางเดิมมีคอลัมน์ชื่อเป็นตัวเลข (int)
- Proposed Fix: ส่งทั้งค่าคอลัมน์เดิมสำหรับดึงข้อมูล และชื่อคอลัมน์ที่เป็น string สำหรับแสดงผลแยกกัน
```python
# ส่ง col ดั้งเดิมและ col_name ที่เป็น string แยกกัน
assoc = _associate(df, col, name, target, ...)
```

### 5. KeyError ใน match_relationships บนคอลัมน์ที่เป็นตัวเลข
- File: [src/thaieda/schema/__init__.py:L450](file:///C:/Users/User/Desktop/Hermes_Peet/scratch/thaieda/src/thaieda/schema/__init__.py#L450), [L461](file:///C:/Users/User/Desktop/Hermes_Peet/scratch/thaieda/src/thaieda/schema/__init__.py#L461)
- คำอธิบาย: คล้ายกับกรณีใน analysis ระบบแปลงชื่อคอลัมน์เป็น string ล่วงหน้าใน `name_map` เมื่อดึงข้อมูลด้วย `tables[tname][col]` จะเกิด KeyError
- Proposed Fix: เก็บข้อมูลคีย์ดั้งเดิมไว้ใน `name_map` เพื่อใช้สำหรับดึงข้อมูล
```python
name_map[str(col).strip().lower()].append((tname, col)) # col เดิม
# ...
info = _col_info(tname, str(col), tables[tname][col])
```

### 6. การแครช (Crash) บนคอลัมน์ตัวเลขที่เป็น NaN ทั้งหมด (All-NaN Columns) ใน synthetic generation
- File: [src/thaieda/llm/_synthetic.py:L107-109](file:///C:/Users/User/Desktop/Hermes_Peet/scratch/thaieda/src/thaieda/llm/_synthetic.py#L107-109)
- คำอธิบาย: ใน `_gen_numeric` หากคอลัมน์มีแต่ค่าว่าง ตัวแปร `numeric` จะว่างเปล่า ทำให้การเรียกใช้ `.sample` แครชทันที
- Proposed Fix: เพิ่มเงื่อนไขตรวจสอบลิสต์ `numeric` ว่างเปล่า แล้วส่งค่า NaN คืนกลับไปโดยตรง
```python
if len(numeric) == 0:
    return pd.Series([np.nan] * n)
```

### 7. การแครชเมื่อสร้างข้อมูลจำลองบนคอลัมน์หมวดหมู่ของ DataFrame ที่ไม่มีแถวข้อมูล (Empty DataFrame)
- File: [src/thaieda/llm/_synthetic.py:L372](file:///C:/Users/User/Desktop/Hermes_Peet/scratch/thaieda/src/thaieda/llm/_synthetic.py#L372)
- คำอธิบาย: ใน `_gen_categorical` หาก DataFrame เริ่มต้นไม่มีแถวข้อมูลเลย ตัวแปร `values` และ `probs` จะว่างเปล่า ทำให้การเรียกใช้ `rng.choice` แครชทันที
- Proposed Fix: เพิ่มเงื่อนไขตรวจสอบว่า DataFrame ว่างเปล่า และแก้ไขการสุ่มด้วยการสุ่ม index แทนเพื่อหลีกเลี่ยง mixed type error
```python
if len(vc) == 0:
    return pd.Series([np.nan] * n)
```

---

## Warning Findings

### 8. ปี ค.ศ. 2 หลัก ถูกแปลงเป็นปี ค.ศ. 1977 (Data Corruption)
- File: [src/thaieda/clean/__init__.py:L1135-1151](file:///C:/Users/User/Desktop/Hermes_Peet/scratch/thaieda/src/thaieda/clean/__init__.py#L1135-1151)
- คำอธิบาย: ฟังก์ชัน `_replace_be_year` แปลงปี 2 หลักน้อยกว่าหรือเท่ากับ 75 เป็น พ.ศ. (+2500 และ -543) เสมอ ทำให้วันที่ที่เป็นปี ค.ศ. 2 หลัก เช่น 20/06/20 (ค.ศ. 2020) ถูกแปลงผิดเป็น ค.ศ. 1977
- Proposed Fix: แปลงปี 2 หลักให้เป็น พ.ศ. เฉพาะในกรณีที่ข้อความมีอักษรภาษาไทยปะปนอยู่เท่านั้น เพื่อป้องกันข้อมูลที่เป็น ค.ศ. 2 หลักเสียหาย
```python
has_thai = bool(re.search(r"[\u0e00-\u0e7f]", text))
# ถ้า has_thai เป็น False และเจอปี 2 หลัก ให้คืนค่าเดิมโดยไม่แปลง
```

### 9. บั๊ก Regex จากเครื่องหมายจุดในอักษรย่อเดือนไทย
- File: [src/thaieda/clean/__init__.py:L1093-1096](file:///C:/Users/User/Desktop/Hermes_Peet/scratch/thaieda/src/thaieda/clean/__init__.py#L1093-1096), [src/thaieda/detect/__init__.py:L863-866](file:///C:/Users/User/Desktop/Hermes_Peet/scratch/thaieda/src/thaieda/detect/__init__.py#L863-866)
- คำอธิบาย: การนำอักษรย่อเดือนไทย (เช่น ม.ค., ก.พ.) มารวมใน regex ด้วยเครื่องหมาย `|` โดยไม่ได้ escape จุด ทำให้จุดถูกมองเป็นตัวอักษรใด ๆ ก็ได้ 1 ตัว ส่งผลให้การตรวจจับเดือนแมทช์กับคำแปลกปลอมได้
- Proposed Fix: ใช้ `re.escape()` หรือใส่ backslash หนีจุดใน regex pattern ของชื่อย่อเดือนไทย
```python
# ใน clean:
"|".join(re.escape(k) for k in _THAI_MONTH_MAP.keys())
# ใน detect:
# ใส่ backslash หลีกเลี่ยงตัวย่อ เช่น ม\.ค\.
```

### 10. บั๊กตัวเลขแรกในที่อยู่ไทยถูกตีเป็นบ้านเลขที่
- File: [src/thaieda/detect/_thai_address.py:L32-34](file:///C:/Users/User/Desktop/Hermes_Peet/scratch/thaieda/src/thaieda/detect/_thai_address.py#L32-34), [L190](file:///C:/Users/User/Desktop/Hermes_Peet/scratch/thaieda/src/thaieda/detect/_thai_address.py#L190)
- คำอธิบาย: `_HOUSE_NUMBER_RE` ค้นหาตัวเลขตัวแรกสุดที่พบในข้อความ ทำให้ที่อยู่ที่มีตัวเลขอื่นก่อน เช่น "สาขาที่ 3 เลขที่ 123" ไปจับคู่เลข "3" เป็นบ้านเลขที่แทน
- Proposed Fix: แยกการค้นหาแบบ explicit (มีคำนำหน้า "เลขที่") ก่อน หากไม่พบให้ fallback ไปหาตัวเลขที่อยู่ต้นข้อความสุดเท่านั้น
```python
explicit_re = re.compile(r"เลขที่\s*(?P<num>\d+(?:/\d+)?(?:-\d+)?)")
fallback_re = re.compile(r"^\s*(?P<num>\d+(?:/\d+)?(?:-\d+)?)")
```

### 11. การแมทช์แบบละโมบ (Greedy) กลืนส่วนประกอบที่อยู่อื่น
- File: [src/thaieda/detect/_thai_address.py:L44-58](file:///C:/Users/User/Desktop/Hermes_Peet/scratch/thaieda/src/thaieda/detect/_thai_address.py#L44-58)
- คำอธิบาย: `_SUBDISTRICT_RE` และ `_DISTRICT_RE` ใช้ `[ก-๛A-Za-z0-9.]+` เพื่อสกัดชื่อ ซึ่งหากข้อความป้อนแบบติดกัน เช่น "ต.บางบัวอ.บางบัวจ.กรุงเทพฯ" จะถูกตำบลกลืนไปทั้งหมด
- Proposed Fix: ใส่ lookahead assertion เพื่อจำกัดไม่ให้สแกนข้ามไปเจอกับคำนำหน้าอื่น เช่น อำเภอ, อ., เขต, จังหวัด, จ.
```python
_SUBDISTRICT_RE = re.compile(
    r"(?:ตำบล|ต\.|แขวง)(?:\s*)(?P<name>(?:(?!อำเภอ|อ\.|เขต|จังหวัด|จ\.)[ก-๛A-Za-z0-9.])+)",
)
```

### 12. ข้อมูลส่วนบุคคล (PII) รั่วไหลในรายงานคุณภาพข้อมูล
- File: [src/thaieda/quality/_thai_id.py:L179-182](file:///C:/Users/User/Desktop/Hermes_Peet/scratch/thaieda/src/thaieda/quality/_thai_id.py#L179-182)
- คำอธิบาย: เลขบัตรประชาชนที่ผิด format (เช่น มี 10-12 หลัก) จะถูกดึงมาเก็บใน list examples แบบดิบ ๆ ส่งผลให้เกิด PII รั่วไหลบนรายงาน HTML
- Proposed Fix: ทำการเซ็นเซอร์ตัวเลขตรงกลาง (masking) สำหรับกรณี format_invalid ด้วยเช่นกัน
```python
if len(str_value) >= 7:
    examples.append(str_value[:3] + "..." + str_value[-3:])
else:
    examples.append(str_value)
```

### 13. ข้อมูลสถิติบางส่วนรั่วไหลในโหมด Differential Privacy (dp_noise)
- File: [src/thaieda/llm/_prepare.py:L243-261](file:///C:/Users/User/Desktop/Hermes_Peet/scratch/thaieda/src/thaieda/llm/_prepare.py#L243-261)
- คำอธิบาย: มีการสกัด `std`, `q25`, `q50`, `q75` จากข้อมูลดิบโดยตรงมาใส่ในผลลัพธ์โดยไม่มีการใส่ noise ทำให้อาจรั่วไหลค่าจริง
- Proposed Fix: ลบข้อมูล `std`, `q25`, `q50`, `q75`, `median` ออกไปจากผลลัพธ์ในโหมด dp_noise
```python
for k in ["std", "25%", "50%", "75%", "q25", "q50", "q75", "median"]:
    noisy_numeric[col].pop(k, None)
```

### 14. statsmodels STL Decomposition แครชบนข้อมูลที่สั้นเกินไปในโหมด auto
- File: [src/thaieda/timeseries/__init__.py:L522-530](file:///C:/Users/User/Desktop/Hermes_Peet/scratch/thaieda/src/thaieda/timeseries/__init__.py#L522-530)
- คำอธิบาย: ในโหมด auto หากพบว่า statsmodels ติดตั้งอยู่และข้อมูลยาวพอสมควรตามเงื่อนไข แต่มีจำนวนจุดน้อยเกินกว่า seasonal window จะทำให้ STL แครชได้
- Proposed Fix: ครอบด้วย try-except ในโหมด auto เพื่อ fallback ไปใช้ basic decomposition ได้อย่างปลอดภัย
```python
try:
    parts = _stl_decompose(values, period_for_decompose)
    engine_used = "statsmodels"
except Exception:
    parts = _basic_decompose(values, period_for_decompose)
    engine_used = "basic"
```

### 15. ค่า NaT (Not a Time) ในคอลัมน์ดัชนีเวลาขัดขวางการวิเคราะห์อนุกรมเวลา
- File: [src/thaieda/timeseries/__init__.py:L442-453](file:///C:/Users/User/Desktop/Hermes_Peet/scratch/thaieda/src/thaieda/timeseries/__init__.py#L442-453)
- คำอธิบาย: หากดัชนีเวลามี NaT ปะปนอยู่ จะถูกส่งไปวิเคราะห์ gaps ทำให้คำนวณผิดพลาด
- Proposed Fix: คัดกรองค่า NaT ออกจากดัชนีเวลาและแถวข้อมูลตั้งแต่ขั้นตอนแรกของการวิเคราะห์
```python
if dt_index is not None:
    valid_time = dt_index.notna()
    work = series.loc[valid_time]
    dt_index = dt_index[valid_time]
```

### 16. ลำดับการเช็คประเภทวันที่ในไฟล์ thai_holidays.py ผิดพลาด
- File: [src/thaieda/timeseries/_thai_holidays.py:L139-150](file:///C:/Users/User/Desktop/Hermes_Peet/scratch/thaieda/src/thaieda/timeseries/_thai_holidays.py#L139-150)
- คำอธิบาย: เงื่อนไข `isinstance(dt, date)` อยู่ก่อน `isinstance(dt, datetime)` ทำให้ datetime ซึ่งเป็น subclass ของ date ถูกตรวจจับผิด และไม่ได้เรียก `.date()`
- Proposed Fix: สลับลำดับการตรวจสอบให้เช็ค `datetime` ก่อน `date`
```python
if isinstance(dt, datetime):
    return dt.date()
if isinstance(dt, date):
    return dt
```

---

## Info Findings

### 17. การนับซ้ำ (Double Counting) ในการจัดรูปแบบช่องว่าง
- File: [src/thaieda/clean/_smart.py:L123-131](file:///C:/Users/User/Desktop/Hermes_Peet/scratch/thaieda/src/thaieda/clean/_smart.py#L123-131)
- คำอธิบาย: การนับสะสมจำนวนช่องว่างเกินตรงกลาง และช่องว่างเกินหน้าหลังแยกกันในตัวแปรเดียวกัน หากมีปัญหากระทบทั้งสองพร้อมกันจะถูกนับเป็น 2 จุด
- Proposed Fix: ใช้ logic OR หรือรวม mask ก่อนหาผลรวม
```python
mask = s.str.contains(r"  +", regex=True, na=False) | s.str.contains(r"^\s|\s$", regex=True, na=False)
count += mask.sum()
```

### 18. เกณฑ์การจัดเป็น ID ในการตรวจ target leakage ต่ำเกินไป
- File: [src/thaieda/insight_engine/_leakage.py:L198-201](file:///C:/Users/User/Desktop/Hermes_Peet/scratch/thaieda/src/thaieda/insight_engine/_leakage.py#L198-201)
- คำอธิบาย: ค่าคงที่ `_ID_CARDINALITY` ตั้งไว้สูงเกินไป ทำให้ ID หลุดไปคำนวณ leakage เกิด False Positive
- Proposed Fix: ปรับลดเกณฑ์ลงเหลือ 0.50 - 0.70

### 19. กราฟ Q-Q plot และ Scatter Plot โต้ตอบ (Plotly) ไม่จำกัดจำนวนจุดข้อมูล
- File: [src/thaieda/viz/_extra_charts.py:L223-228](file:///C:/Users/User/Desktop/Hermes_Peet/scratch/thaieda/src/thaieda/viz/_extra_charts.py#L223-228), [src/thaieda/viz/_interactive.py:L262](file:///C:/Users/User/Desktop/Hermes_Peet/scratch/thaieda/src/thaieda/viz/_interactive.py#L262)
- คำอธิบาย: ไม่มีการจำกัดจุดข้อมูลสำหรับ Q-Q Plot และ Scatter Interactive ส่งผลให้รายงานบวมและเว็บบราวเซอร์ค้างเมื่อเจอชุดข้อมูลขนาดใหญ่
- Proposed Fix: ทำการ subsample ให้เหลือไม่เกิน 5,000 จุดก่อนคำนวณ/วาดกราฟ

### 20. กลุ่มเป้าหมายกลายเป็นกลุ่ม อื่นๆ ใน Box plot สหสัมพันธ์
- File: [src/thaieda/viz/__init__.py:L1130-1131](file:///C:/Users/User/Desktop/Hermes_Peet/scratch/thaieda/src/thaieda/viz/__init__.py#L1130-1131)
- คำอธิบาย: กรองกลุ่มแบบ top-9 โดยไม่ได้เก็บกลุ่มเด่น (top_segment) เสมอ ทำให้กลุ่มเด่นอาจถูกยุบเป็น "อื่น ๆ"
- Proposed Fix: คัดลอกกลุ่มเด่นเข้ามาใน top_groups ตั้งแต่วันแรก แล้วเติมกลุ่มอื่นให้ครบ
