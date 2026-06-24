# Contributing to ThaiEDA

ขอบคุณที่สนใจร่วมพัฒนา! 🙏

## การร่วมพัฒนา

1. **Fork** repo และสร้าง branch ใหม่
   ```bash
   git checkout -b feat/your-feature-name
   ```

2. **ติดตั้งสำหรับ development**
   ```bash
   pip install -e ".[dev,thai,viz]"
   pre-commit install
   ```

3. **เขียน tests** สำหรับ code ใหม่ทุกครั้ง
   ```bash
   pytest tests/ -v
   ```

4. **Code style** — ใช้ ruff ตรวจสอบ
   ```bash
   ruff check src/ tests/
   ruff format src/ tests/
   ```

5. **สร้าง Pull Request** พร้อมคำอธิบายชัดเจน

## Code conventions

- Python 3.10+
- Type hints ทุก public function
- Docstrings เป็นภาษาไทยหรืออังกฤษก็ได้ (แต่ต้องชัดเจน)
- Thai text processing: ระบุ engine ที่ใช้เสมอ (transparency)
- ไม่ทำ silent fallback — ถ้า tokenizer ไม่พร้อม ให้ fail พร้อมข้อความชัดเจน

## การรายงาน bug

เปิด issue พร้อม:
- ข้อมูลตัวอย่าง (ถ้าเป็นไปได้)
- โค้ดที่ทำให้เกิดปัญหา
- ผลที่ได้ vs ผลที่คาดหวัง
- เวอร์ชัน Python และ ThaiEDA

## Code of Conduct

โปรดอ่าน [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) — ทุกคนต้องปฏิบัติตาม