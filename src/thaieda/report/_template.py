"""Jinja2 HTML template สำหรับรายงาน ThaiEDA — เก็บเป็นสตริงเพื่อให้แพ็กเกจ self-contained."""
# ruff: noqa: E501, W293 — เทมเพลต HTML มีบรรทัดยาวโดยธรรมชาติ (แอตทริบิวต์/markup)

from __future__ import annotations

# ธีมเข้ม, modern, CSS ฝังในตัว, รูปเป็น base64 — ไม่พึ่งไฟล์ภายนอก
# Macro render_issue/render_anomaly: ใช้ซ้ำในส่วนหลัก + ส่วน collapse (top-50 + ที่เหลือ) — P2
# วางก่อน <!DOCTYPE> โดยไม่มีช่องว่างคั่น เพื่อไม่ให้ HTML ขึ้นต้นด้วย whitespace (กัน quirks mode)
REPORT_TEMPLATE = r"""{% macro render_issue(iss, sev_icons, L) %}
    <div class="issue {{ iss.severity }}">
      <div><span class="sev {{ iss.severity }}">{{ sev_icons[iss.severity] }} {{ L('severity_' ~ iss.severity) }}</span><b>{{ iss.column }}</b> · <span class="ng">{{ iss.check_name }}</span></div>
      <div class="meta">{{ L('rows_affected') }}: {{ "{:,}".format(iss.count) }} ({{ iss.percentage }}% {{ L('of_rows') }})</div>
      <div class="desc-th">{{ iss.description_th }}</div>
      {% if iss.examples %}<div class="examples">{% for ex in iss.examples[:5] %}<span class="ex mono">{{ ex }}</span>{% endfor %}</div>{% endif %}
      <div class="so-what"><span class="lbl">{{ L('so_what') }}</span> {{ iss.suggestion_th }}</div>
      <details><summary class="ng">{{ L('show_details') }}</summary><div class="desc-en">{{ iss.description }} — {{ iss.suggestion }}</div></details>
    </div>
{% endmacro %}{% macro render_anomaly(an, sev_icons, L) %}
    <div class="issue {{ an.severity }}">
      <div><span class="sev {{ an.severity }}">{{ sev_icons[an.severity] }} {{ L('severity_' ~ an.severity) }}</span><b>{{ an.column }}</b> · <span class="ng">{{ an.check_name }}</span> <span class="badge">{{ an.type_label }}</span></div>
      <div class="meta">{{ L('rows_affected') }}: {{ "{:,}".format(an.count) }} ({{ an.percentage }}% {{ L('of_rows') }})</div>
      <div class="desc-th">{{ an.description_th }}</div>
      {% if an.examples %}<div class="examples">{% for ex in an.examples[:5] %}<span class="ex mono">{{ ex }}</span>{% endfor %}</div>{% endif %}
      <div class="so-what"><span class="lbl">{{ L('so_what') }}</span> {{ an.suggestion_th }}</div>
      <details><summary class="ng">{{ L('show_details') }}</summary><div class="desc-en">{{ an.description }} — {{ an.suggestion }}</div></details>
    </div>
{% endmacro %}<!DOCTYPE html>
<html lang="{{ lang }}">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{{ L('report_title') }}</title>
<style>
  :root {
    --bg: #0b0f19;
    --panel: #151f32;
    --panel-hover: #1e2c45;
    --panel2: #1e293b;
    --panel3: #334155;
    --fg: #f1f5f9;
    --muted: #94a3b8;
    --border: #1e293b;
    --border-light: #334155;
    --accent: #3b82f6;
    --accent-glow: rgba(59, 130, 246, 0.15);
    --critical: #ef4444;
    --critical-bg: rgba(239, 68, 68, 0.08);
    --warning: #f59e0b;
    --warning-bg: rgba(245, 158, 11, 0.08);
    --info: #06b6d4;
    --info-bg: rgba(6, 182, 212, 0.08);
    --ok: #10b981;
    --ok-bg: rgba(16, 185, 129, 0.08);
    --shadow: 0 10px 30px -3px rgba(0, 0, 0, 0.3), 0 4px 6px -4px rgba(0, 0, 0, 0.3);
    --font-sans: "Segoe UI", "Sarabun", "Noto Sans Thai", -apple-system, BlinkMacSystemFont, Tahoma, sans-serif;
  }
  [data-theme="light"] {
    --bg: #f8fafc;
    --panel: #ffffff;
    --panel-hover: #f1f5f9;
    --panel2: #f1f5f9;
    --panel3: #e2e8f0;
    --fg: #0f172a;
    --muted: #64748b;
    --border: #e2e8f0;
    --border-light: #cbd5e1;
    --accent: #2563eb;
    --accent-glow: rgba(37, 99, 235, 0.1);
    --critical: #e11d48;
    --critical-bg: rgba(225, 29, 72, 0.04);
    --warning: #d97706;
    --warning-bg: rgba(217, 119, 6, 0.04);
    --info: #0891b2;
    --info-bg: rgba(8, 145, 178, 0.04);
    --ok: #059669;
    --ok-bg: rgba(5, 150, 105, 0.04);
    --shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.05), 0 8px 10px -6px rgba(0, 0, 0, 0.05);
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    background: radial-gradient(circle at top left, var(--accent-glow), transparent 40rem), var(--bg);
    color: var(--fg);
    font-family: var(--font-sans);
    line-height: 1.6;
    font-size: 15px;
    transition: background 0.3s ease, color 0.3s ease;
  }
  .wrap { max-width: 1200px; margin: 0 auto; padding: 40px 24px 100px; }
  header { display: flex; justify-content: space-between; gap: 20px; align-items: flex-start; margin-bottom: 24px; }
  header h1 { font-size: 32px; font-weight: 800; margin: 0 0 8px; letter-spacing: -.5px; }
  header .sub, .muted { color: var(--muted); font-size: 14px; }
  h2 { font-size: 24px; font-weight: 700; margin: 48px 0 20px; padding-bottom: 10px; border-bottom: 1px solid var(--border); scroll-margin-top: 80px; }
  h3 { font-size: 18px; font-weight: 600; margin: 28px 0 12px; }
  a { color: inherit; text-decoration: none; }
  a:hover { color: var(--accent); }
  
  /* Tables */
  table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    margin: 16px 0;
    border: 1px solid var(--border);
    border-radius: 12px;
    overflow: hidden;
  }
  th, td {
    text-align: left;
    padding: 12px 16px;
    border-bottom: 1px solid var(--border);
    font-size: 14px;
    vertical-align: middle;
    transition: background-color 0.2s ease;
  }
  th {
    background: var(--panel2);
    color: var(--muted);
    font-weight: 600;
    text-transform: uppercase;
    font-size: 11px;
    letter-spacing: 0.5px;
  }
  tr:last-child td {
    border-bottom: none;
  }
  tbody tr:nth-child(even) {
    background-color: rgba(0, 0, 0, 0.02);
  }
  [data-theme="light"] tbody tr:nth-child(even) {
    background-color: rgba(0, 0, 0, 0.01);
  }
  tbody tr:hover td {
    background-color: var(--panel-hover);
  }
  
  code, .mono {
    font-family: "Cascadia Code", Consolas, monospace;
    font-size: 12.5px;
    background: var(--panel2);
    padding: 2px 6px;
    border-radius: 6px;
    word-break: break-all;
    border: 1px solid var(--border);
  }
  
  .nav {
    position: sticky;
    top: 16px;
    z-index: 10;
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    background: rgba(21, 23, 28, 0.85);
    backdrop-filter: blur(12px);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 8px 12px;
    margin: 20px 0;
  }
  [data-theme="light"] .nav {
    background: rgba(255, 255, 255, 0.85);
  }
  .nav a {
    color: var(--muted);
    text-decoration: none;
    font-size: 13px;
    font-weight: 600;
    padding: 6px 12px;
    border-radius: 10px;
    white-space: nowrap;
    transition: all 0.2s ease;
  }
  .nav a:hover {
    color: var(--fg);
    background: var(--panel2);
  }

  .hero-grid { display: grid; grid-template-columns: 1.35fr .65fr; gap: 20px; margin: 20px 0 24px; }
  
  /* Data type banner */
  .data-type-banner { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0 24px; }
  .dtype-card { background: var(--panel); border: 1px solid var(--border); border-radius: 16px; padding: 24px; box-shadow: var(--shadow); }
  .dtype-label { color: var(--accent); font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px; }
  .dtype-value { font-size: 28px; font-weight: 800; margin-bottom: 12px; letter-spacing: -.5px; }
  .dtype-summary { font-size: 15px; line-height: 1.6; }
  .dtype-focus { background: var(--panel2); border: 1px solid var(--border); border-radius: 16px; padding: 24px; }
  .dtype-focus-label { color: var(--accent); font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 12px; }
  .dtype-focus ul { margin: 0; padding-left: 20px; }
  .dtype-focus li { margin: 8px 0; font-size: 14px; }
  
  /* Key findings */
  .key-findings-list { display: grid; gap: 16px; margin: 16px 0; }
  .key-finding {
    display: flex;
    gap: 20px;
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 24px;
    border-left: 5px solid var(--accent);
    box-shadow: var(--shadow);
    transition: transform 0.2s ease;
  }
  .key-finding:hover {
    transform: translateY(-2px);
  }
  .key-finding.critical { border-left-color: var(--critical); }
  .key-finding.warning { border-left-color: var(--warning); }
  .key-finding.info { border-left-color: var(--info); }
  .kf-num { font-size: 32px; font-weight: 900; color: var(--muted); opacity: 0.5; min-width: 40px; text-align: center; }
  .kf-body { flex: 1; }
  .kf-business { margin: 10px 0 6px; font-size: 16px; font-weight: 600; line-height: 1.6; color: var(--fg); }
  .kf-tech { font-size: 13.5px; margin-top: 6px; color: var(--muted); }
  .kf-impact { margin-top: 10px; font-size: 14px; }
  .kf-impact .lbl { color: var(--accent); font-weight: 700; }
  
  .hero, .panel, .card, .issue, .col, details.block {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 16px;
    box-shadow: var(--shadow);
  }
  
  .hero { padding: 28px; border-left: 6px solid var(--accent); }
  .hero.critical { border-left-color: var(--critical); }
  .hero.warning { border-left-color: var(--warning); }
  .hero.good { border-left-color: var(--ok); }
  .hero .eyebrow { color: var(--accent); font-weight: 800; font-size: 12px; letter-spacing: 1px; text-transform: uppercase; }
  .hero.critical .eyebrow { color: var(--critical); }
  .hero.warning .eyebrow { color: var(--warning); }
  .hero.good .eyebrow { color: var(--ok); }
  .hero .verdict { font-size: 26px; font-weight: 800; line-height: 1.3; margin: 10px 0 12px; letter-spacing: -.5px; }
  .hero .summary { font-size: 16px; color: var(--fg); opacity: 0.95; }
  .hero ul, .panel ul { margin: 12px 0 0 20px; padding: 0; }
  .hero li { margin: 6px 0; }
  .panel { padding: 24px; }
  .panel h3 { margin-top: 0; }
  
  .health-meter { display: grid; gap: 12px; }
  .health-row { display: flex; justify-content: space-between; gap: 12px; color: var(--muted); border-bottom: 1px solid var(--border); padding-bottom: 10px; font-size: 14px; }
  .health-row:last-child { border-bottom: 0; padding-bottom: 0; }
  .health-row b { color: var(--fg); }
  
  .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 16px; margin: 20px 0; }
  .card { padding: 20px; box-shadow: none; transition: border-color 0.2s, box-shadow 0.2s; }
  .card:hover { border-color: var(--accent); box-shadow: var(--shadow); }
  .card .k { color: var(--muted); font-size: 13px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }
  .card .v { font-size: 28px; font-weight: 800; margin-top: 6px; letter-spacing: -.5px; }
  
  .typedist { display: flex; flex-wrap: wrap; gap: 8px; margin: 16px 0; }
  .chip, .badge {
    background: var(--panel2);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 6px 14px;
    font-size: 12.5px;
    color: var(--accent);
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-weight: 600;
  }
  .chip b { color: var(--fg); }
  .chip.t-numeric, .badge.t-numeric { border-color: var(--info); color: var(--info); }
  .chip.t-thai_text, .badge.t-thai_text { border-color: var(--ok); color: var(--ok); }
  .chip.t-mixed_text, .badge.t-mixed_text { border-color: #8b5cf6; color: #8b5cf6; }
  .chip.t-english_text, .badge.t-english_text { border-color: #3b82f6; color: #3b82f6; }
  .chip.t-categorical, .badge.t-categorical { border-color: #f59e0b; color: #f59e0b; }
  .chip.t-datetime, .badge.t-datetime { border-color: #10b981; color: #10b981; }
  .chip.t-id, .badge.t-id { border-color: #ec4899; color: #ec4899; }
  .chip.t-phone_number, .badge.t-phone_number { border-color: #eab308; color: #eab308; }
  .chip.t-empty, .badge.t-empty { border-color: var(--muted); color: var(--muted); }

  .action-list { display: grid; gap: 14px; margin: 18px 0; }
  .action {
    display: grid;
    grid-template-columns: auto 1fr;
    gap: 16px;
    background: var(--panel);
    border: 1px solid var(--border);
    border-left: 5px solid var(--info);
    border-radius: 16px;
    padding: 20px;
    box-shadow: var(--shadow);
  }
  .action.critical { border-left-color: var(--critical); }
  .action.warning { border-left-color: var(--warning); }
  .action .num { width: 32px; height: 32px; display: inline-grid; place-items: center; border-radius: 50%; background: var(--panel2); color: var(--fg); font-weight: 800; font-size: 14px; border: 1px solid var(--border); }
  .action b { font-size: 17px; }
  
  .so-what {
    margin-top: 10px;
    background: var(--accent-glow);
    border: 1px solid rgba(59, 130, 246, 0.25);
    border-radius: 12px;
    padding: 12px 16px;
    font-size: 14px;
    line-height: 1.6;
  }
  .so-what .lbl, .rec .lbl, .suggest .lbl { color: var(--accent); font-weight: 700; margin-right: 6px; }

  .issue { border-left: 5px solid var(--border); padding: 18px 22px; margin: 16px 0; box-shadow: none; }
  .issue.critical { border-left-color: var(--critical); background: var(--critical-bg); }
  .issue.warning { border-left-color: var(--warning); background: var(--warning-bg); }
  .issue.info { border-left-color: var(--info); background: var(--info-bg); }
  .issue .meta, .ng, .note { color: var(--muted); font-size: 13px; }
  .issue .desc-th { margin: 8px 0 4px; font-weight: 500; }
  .issue .desc-en { color: var(--muted); font-size: 13.5px; line-height: 1.5; }
  
  .biz-impact { margin: 10px 0; background: var(--ok-bg); border: 1px solid rgba(16, 185, 129, 0.25); border-radius: 12px; padding: 12px 16px; font-size: 14px; }
  .biz-impact .lbl { color: var(--ok); font-weight: 700; margin-right: 6px; }
  .technical { margin-top: 10px; color: var(--muted); font-size: 13px; }

  .sev { display: inline-flex; align-items: center; gap: 4px; padding: 2px 8px; border-radius: 6px; font-size: 12px; font-weight: 700; margin-right: 8px; }
  .sev.critical { background: var(--critical-bg); color: var(--critical); }
  .sev.warning { background: var(--warning-bg); color: var(--warning); }
  .sev.info { background: var(--info-bg); color: var(--info); }
  
  .sevcount { display: inline-flex; gap: 8px; flex-wrap: wrap; margin-left: 8px; }
  .cat { float: right; }
  .examples { margin: 10px 0 0; }
  .examples .ex { display: block; margin: 5px 0; font-family: "Cascadia Code", Consolas, monospace; }

  .insight-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; }
  .watch-list { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 16px; margin: 18px 0; }
  .watch { padding: 18px; background: var(--panel); border: 1px solid var(--border); border-left: 5px solid var(--warning); border-radius: 16px; box-shadow: var(--shadow); }
  .watch.critical { border-left-color: var(--critical); }
  .watch.info { border-left-color: var(--info); }
  .watch ul { margin: 10px 0 0 20px; padding: 0; }
  .watch li { margin: 6px 0; font-size: 13.5px; }

  .imgrow { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 18px; }
  .imgrow.full { grid-template-columns: 1fr; }
  .imgrow img { width: 100%; border-radius: 12px; border: 1px solid var(--border); background: var(--panel2); box-shadow: var(--shadow); transition: transform 0.2s; }
  .imgrow img:hover { transform: scale(1.005); }
  .imgcap { color: var(--muted); font-size: 13px; font-weight: 600; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.5px; }
  .chart-note { margin-top: 10px; color: var(--muted); font-size: 13.5px; border-left: 4px solid var(--accent); padding-left: 12px; line-height: 1.5; }

  details.block { margin: 18px 0; overflow: hidden; box-shadow: none; }
  details.block summary { cursor: pointer; padding: 16px 20px; list-style: none; display: flex; align-items: center; gap: 10px; flex-wrap: wrap; font-weight: 600; border-bottom: 1px solid transparent; }
  details.block[open] summary { border-bottom-color: var(--border); background: var(--panel2); }
  details.block summary::-webkit-details-marker { display: none; }
  details.block summary:after { content: "ดูเพิ่มเติม ▾"; margin-left: auto; color: var(--muted); font-size: 12.5px; background: var(--panel2); padding: 4px 10px; border-radius: 8px; border: 1px solid var(--border); }
  details.block[open] summary:after { content: "ซ่อน ▴"; background: var(--accent); color: #fff; border-color: var(--accent); }
  details.block .body { padding: 20px; }
  
  .col { padding: 0; margin: 16px 0; box-shadow: none; overflow: hidden; }
  .col .head { display: flex; align-items: baseline; gap: 12px; flex-wrap: wrap; padding: 16px 20px; background: var(--panel2); border-bottom: 1px solid var(--border); }
  .col .head .nm { font-size: 18px; font-weight: 700; letter-spacing: -.2px; }
  
  .diff .b { color: var(--critical); text-decoration: line-through; text-decoration-color: rgba(239, 68, 68, 0.4); padding: 1px 4px; background: rgba(239, 68, 68, 0.05); border-radius: 4px; }
  .diff .arrow { color: var(--muted); margin: 0 8px; font-weight: bold; }
  .diff .a { color: var(--ok); padding: 1px 4px; background: rgba(16, 185, 129, 0.05); border-radius: 4px; }
  .diff .row { display: block; margin: 5px 0; font-family: "Cascadia Code", Consolas, monospace; }
  .empty { color: var(--muted); font-style: italic; padding: 16px; text-align: center; background: var(--panel2); border-radius: 12px; border: 1px dashed var(--border); }
  
  footer { margin-top: 60px; padding: 24px 0; border-top: 1px solid var(--border); color: var(--muted); font-size: 13px; text-align: center; }

  @media (max-width: 820px) {
    .wrap { padding: 24px 16px 80px; }
    header { flex-direction: column; align-items: stretch; gap: 16px; }
    header div[style*="display:flex"] { flex-wrap: wrap; }
    .hero-grid, .data-type-banner, .imgrow { grid-template-columns: 1fr; display: block; }
    .panel { margin-top: 16px; }
    .cards { grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); }
    table { display: block; overflow-x: auto; -webkit-overflow-scrolling: touch; }
    .nav { position: static; }
    .cat { float: none; display: block; margin-top: 6px; }
  }
  @media print {
    :root { --bg: #fff; --panel: #fff; --panel2: #f8fafc; --panel3: #e2e8f0; --fg: #0f172a; --muted: #64748b; --border: #cbd5e1; }
    body { font-size: 12px; background: #fff; }
    .nav { display: none; }
    .wrap { max-width: none; padding: 0; }
    .issue, .col, .card, .hero, .panel, .action { break-inside: avoid; box-shadow: none; border-radius: 0; }
    img { max-width: 100%; }
  }
  
  /* Tabs */
  .tab-bar { display: flex; gap: 8px; flex-wrap: wrap; margin: 0 0 24px; border-bottom: 1px solid var(--border); padding-bottom: 8px; }
  .tab-btn {
    padding: 8px 16px;
    font-size: 14px;
    font-weight: 600;
    color: var(--muted);
    background: none;
    border: 1px solid transparent;
    cursor: pointer;
    border-radius: 8px;
    transition: all 0.2s ease;
  }
  .tab-btn:hover { color: var(--fg); background: var(--panel-hover); }
  .tab-btn.active { color: var(--fg); background: var(--accent); box-shadow: 0 4px 12px var(--accent-glow); }
  .tab-panel { display: none; }
  .tab-panel.active { display: block; }
  
  .col-filter { margin: 0 0 20px; }
  .col-filter select { padding: 10px 16px; font-size: 14px; background: var(--panel); color: var(--fg); border: 1px solid var(--border); border-radius: 10px; cursor: pointer; outline: none; transition: border-color 0.2s; }
  .col-filter select:focus { border-color: var(--accent); }
  
  .theme-toggle {
    padding: 8px 16px;
    font-size: 13.5px;
    font-weight: 600;
    background: var(--panel2);
    color: var(--fg);
    border: 1px solid var(--border);
    border-radius: 20px;
    cursor: pointer;
    transition: all 0.2s ease;
    display: inline-flex;
    align-items: center;
    gap: 6px;
  }
  .theme-toggle:hover { background: var(--panel3); border-color: var(--accent); }
  
  #back-to-top {
    position: fixed;
    bottom: 24px;
    right: 24px;
    width: 48px;
    height: 48px;
    border-radius: 50%;
    background: var(--accent);
    color: #fff;
    border: none;
    font-size: 20px;
    cursor: pointer;
    opacity: 0;
    visibility: hidden;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    z-index: 999;
    box-shadow: 0 4px 14px var(--accent-glow);
    display: flex;
    align-items: center;
    justify-content: center;
  }
  #back-to-top.visible { opacity: 1; visibility: visible; }
  #back-to-top:hover { transform: translateY(-4px); box-shadow: 0 6px 20px var(--accent-glow); }
  
  /* Modeling Blueprint */
  .blueprint-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; margin: 20px 0; }
  .blueprint-card { background: var(--panel); border: 1px solid var(--border); border-radius: 16px; padding: 22px; box-shadow: var(--shadow); border-top: 4px solid var(--accent); }
  .blueprint-card.critical { border-top-color: var(--critical); }
  .blueprint-card h3 { margin: 0 0 12px; font-size: 16px; color: var(--accent); text-transform: uppercase; letter-spacing: 0.5px; }
  .blueprint-card ul { margin: 8px 0 0 18px; padding: 0; }
  .blueprint-card li { margin: 6px 0; font-size: 14px; }
  .blueprint-checklist { list-style: none; margin: 0; padding: 0; }
  .blueprint-checklist li { padding: 10px 14px; margin: 8px 0; background: var(--panel2); border-radius: 10px; border-left: 4px solid var(--ok); }
  .leakage-item { background: var(--critical-bg); border: 1px solid rgba(239,68,68,0.3); border-radius: 12px; padding: 14px 16px; margin: 10px 0; }

  .plotly-chart { margin-top: 18px; border: 1px solid var(--border); border-radius: 12px; overflow: hidden; background: var(--panel); box-shadow: var(--shadow); }
</style>
<script>
  function switchTab(tabId) {
    document.querySelectorAll('.tab-btn').forEach(function(b) { b.classList.remove('active'); });
    document.querySelectorAll('.tab-panel').forEach(function(p) { p.classList.remove('active'); });
    var btn = document.querySelector('[data-tab="' + tabId + '"]');
    var panel = document.getElementById('tab-' + tabId);
    if (btn) btn.classList.add('active');
    if (panel) panel.classList.add('active');
  }
  function filterColumns(sel) {
    var val = sel.value;
    document.querySelectorAll('[data-col-name]').forEach(function(card) {
      if (val === 'all' || card.getAttribute('data-col-name') === val) {
        card.style.display = '';
      } else {
        card.style.display = 'none';
      }
    });
  }
  function toggleTheme() {
    var body = document.body;
    var cur = body.getAttribute('data-theme');
    var next = cur === 'light' ? 'dark' : 'light';
    if (next === 'dark') { body.removeAttribute('data-theme'); }
    else { body.setAttribute('data-theme', 'light'); }
    try { localStorage.setItem('thaieda-theme', next); } catch(e) {}
  }
  window.addEventListener('scroll', function() {
    var btn = document.getElementById('back-to-top');
    if (btn) { if (window.scrollY > 200) { btn.classList.add('visible'); } else { btn.classList.remove('visible'); } }
  });
  (function() {
    try { var saved = localStorage.getItem('thaieda-theme'); if (saved === 'light') { document.body.setAttribute('data-theme', 'light'); } } catch(e) {}
  })();
</script>
</head>
<body>
<div class="wrap">
  <header>
    <div>
      <h1>{{ L('report_title') }}</h1>
      <div class="sub">{{ L('generated_by') }} ThaiEDA v{{ version }} · {{ overview.rows }} {{ L('rows') }} × {{ overview.columns }} {{ L('columns') }}</div>
    </div>
    <div style="display:flex;gap:12px;align-items:flex-start;">
      <div class="badge">{{ L('how_to_read') }}: {{ L('how_to_read_desc') }}</div>
      <button class="theme-toggle" onclick="toggleTheme()" title="Toggle light/dark mode">🌙 / ☀️</button>
    </div>
  </header>

  <div class="tab-bar">
    <button class="tab-btn active" data-tab="overview" onclick="switchTab('overview')">{{ L('tab_overview') }}</button>
    <button class="tab-btn" data-tab="insights" onclick="switchTab('insights')">{{ L('tab_insights') }}</button>
    <button class="tab-btn" data-tab="quality" onclick="switchTab('quality')">{{ L('tab_quality') }}</button>
    <button class="tab-btn" data-tab="anomalies" onclick="switchTab('anomalies')">{{ L('tab_anomalies') }}</button>
    <button class="tab-btn" data-tab="columns" onclick="switchTab('columns')">{{ L('tab_columns') }}</button>
  </div>

  <!-- TAB: OVERVIEW -->
  <div id="tab-overview" class="tab-panel active">

  <!-- ============ WHAT IS THIS DATA ============ -->
  {% if data_type %}
  <section id="what-is-this" class="data-type-banner">
    <div class="dtype-card">
      <div class="dtype-label">{{ L('detected_data_type') }}</div>
      <div class="dtype-value">{{ data_type.display_label or data_type.label_th }}</div>
      <div class="dtype-summary muted">{{ data_type.summary }}</div>
      {% if data_type.language %}
      <div class="so-what"><span class="lbl">{{ L('detected_language') }}</span> {{ L('language_' ~ data_type.language.language) }}</div>
      {% if data_type.language_impact %}<div class="muted">{{ L('language_impact') }}: {{ data_type.language_impact }}</div>{% endif %}
      {% endif %}
    </div>
    <div class="dtype-focus">
      <div class="dtype-focus-label">{{ L('eda_focus') }}</div>
      <ul>
        {% for f in data_type.focus %}<li>{{ f }}</li>{% endfor %}
      </ul>
      {% if data_type.show_thai_specific and data_type.thai_recommendations %}
      <h3>{{ L('thai_specific_recommendations') }}</h3>
      <ul>
        {% for f in data_type.thai_recommendations %}<li>{{ f }}</li>{% endfor %}
      </ul>
      {% endif %}
    </div>
  </section>
  {% endif %}

  <!-- ============ KEY FINDINGS (top 5) ============ -->
  {% if key_findings %}
  <h2 id="key-findings">{{ L('most_important') }}</h2>
  <div class="key-findings-list">
    {% for kf in key_findings %}
    <div class="key-finding {{ kf.severity }}">
      <div class="kf-num">{{ loop.index }}</div>
      <div class="kf-body">
        <span class="sev {{ kf.severity }}">{{ sev_icons[kf.severity] }} {{ L('severity_' ~ kf.severity) }}</span>
        <b>{{ kf.title if kf.title else kf.column }}</b>
        {% if kf.business %}<div class="kf-business">{{ kf.business }}</div>{% endif %}
        <div class="kf-tech muted">{{ kf.description or kf.technical }}</div>
        {% if kf.percentage %}<div class="kf-impact"><span class="lbl">{{ L('impact') }}</span> {{ kf.percentage }}%</div>{% endif %}
      </div>
    </div>
    {% endfor %}
  </div>
  {% endif %}

  <!-- ============ EXECUTIVE SUMMARY ============ -->
  <section id="summary" class="hero-grid">
    <div class="hero {{ report_summary.status }}">
      <div class="eyebrow">{{ L('executive_summary') }}</div>
      <div class="verdict">{{ report_summary.verdict }}</div>
      <div class="summary">
        {% if insight_section %}{{ insight_section.executive_summary }}{% else %}{{ L('no_insights_yet') }}{% endif %}
      </div>
      <ul>
        {% for h in report_summary.highlights %}<li>{{ h }}</li>{% endfor %}
      </ul>
    </div>
    <aside class="panel">
      <h3>{{ L('data_health') }}</h3>
      <div class="health-meter">
        <div class="health-row"><span>{{ L('severity_critical') }}</span><b>{{ issue_summary.critical }}</b></div>
        <div class="health-row"><span>{{ L('severity_warning') }}</span><b>{{ issue_summary.warning }}</b></div>
        <div class="health-row"><span>{{ L('severity_info') }}</span><b>{{ issue_summary.info }}</b></div>
        {% if overview.raw_missing_pct is defined %}
        <div class="health-row"><span>{{ L('raw_missing') }}</span><b>{{ overview.raw_missing_pct }}%</b></div>
        <div class="health-row"><span>{{ L('post_clean_missing') }}</span><b>{{ report_summary.missing_pct }}%</b></div>
        {% else %}
        <div class="health-row"><span>{{ L('missing_cells') }}</span><b>{{ report_summary.missing_pct }}%</b></div>
        {% endif %}
        <div class="health-row"><span>{{ L('duplicate_rows') }}</span><b>{{ report_summary.duplicate_pct }}%</b></div>
      </div>
    </aside>
  </section>

  <!-- ============ PRIORITY ACTIONS ============ -->
  <h2 id="actions">{{ L('priority_actions') }}</h2>
  {% if priority_actions %}
  <div class="action-list">
    {% for a in priority_actions %}
    <div class="action {{ a.severity }}">
      <div class="num">{{ loop.index }}</div>
      <div>
        <span class="sev {{ a.severity }}">{{ sev_icons[a.severity] }} {{ L('severity_' ~ a.severity) }}</span>
        <b>{{ a.title }}</b>
        <div class="muted">{{ a.why }}</div>
        <div class="so-what"><span class="lbl">{{ L('so_what') }}</span> {{ a.action }}</div>
      </div>
    </div>
    {% endfor %}
  </div>
  {% else %}
    <p class="empty">✓ {{ L('no_priority_actions') }}</p>
  {% endif %}


  {% if modeling_blueprint %}
  <h2 id="modeling-blueprint">{{ L('modeling_blueprint') }}</h2>
  <div class="blueprint-grid">
    {% if modeling_blueprint.target_baseline %}
    <div class="blueprint-card">
      <h3>{{ L('target_baseline') }}</h3>
      <div><b>{{ modeling_blueprint.target_baseline.target_column }}</b></div>
      {% if modeling_blueprint.target_baseline.is_binary %}
      <div class="meta">{{ L('positive_rate') }}: <b>{{ modeling_blueprint.target_baseline.positive_rate_pct }}%</b></div>
      <div class="meta">{{ L('class_balance') }}: {{ modeling_blueprint.target_baseline.balance_label }}</div>
      {% endif %}
      <ul>{% for cls, cnt in modeling_blueprint.target_baseline.class_counts %}<li><code>{{ cls }}</code>: {{ "{:,}".format(cnt) }}</li>{% endfor %}</ul>
    </div>
    {% endif %}
    <div class="blueprint-card {% if modeling_blueprint.leakage %}critical{% endif %}">
      <h3>{{ L('leakage_suspects') }}</h3>
      {% if modeling_blueprint.leakage %}
        {% for lf in modeling_blueprint.leakage %}
        <div class="leakage-item"><b>{{ lf.feature }}</b> <span class="badge">{{ lf.kind }}</span><div class="muted">{{ lf.description }}</div></div>
        {% endfor %}
      {% else %}<p class="empty">✓ {{ L('no_leakage_suspects') }}</p>{% endif %}
    </div>
    <div class="blueprint-card">
      <h3>{{ L('strong_features') }}</h3>
      {% if modeling_blueprint.strong_features %}
      <ul>{% for sf in modeling_blueprint.strong_features %}<li><b>{{ sf.column }}</b> <span class="badge">{{ sf.association_type }}</span> {{ L('association_score') }}={{ sf.score }}</li>{% endfor %}</ul>
      {% else %}<p class="muted">—</p>{% endif %}
    </div>
    <div class="blueprint-card">
      <h3>{{ L('columns_to_drop') }}</h3>
      {% if modeling_blueprint.columns_to_drop %}
      <ul>{% for c in modeling_blueprint.columns_to_drop %}<li><code>{{ c }}</code></li>{% endfor %}</ul>
      {% else %}<p class="muted">—</p>{% endif %}
    </div>
  </div>
  <h3>{{ L('modeling_next_steps') }}</h3>
  <ul class="blueprint-checklist">{% for step in modeling_blueprint.next_steps %}<li>{{ step }}</li>{% endfor %}</ul>
  {% endif %}

  <!-- ============ OVERVIEW ============ -->
  <h2 id="overview">{{ L('overview') }}</h2>
  <div class="cards">
    <div class="card"><div class="k">{{ L('rows') }}</div><div class="v">{{ "{:,}".format(overview.rows) }}</div></div>
    <div class="card"><div class="k">{{ L('columns') }}</div><div class="v">{{ overview.columns }}</div></div>
    <div class="card"><div class="k">{{ L('total_cells') }}</div><div class="v">{{ "{:,}".format(overview.total_cells) }}</div></div>
    <div class="card"><div class="k">{{ L('missing_cells') }}</div><div class="v">{{ "{:,}".format(overview.missing_cells) }} <span class="muted">({{ overview.missing_pct }}%)</span></div></div>
    <div class="card"><div class="k">{{ L('duplicate_rows') }}</div><div class="v">{{ "{:,}".format(overview.duplicate_rows) }}</div></div>
  </div>
  <h3>{{ L('column_types') }}</h3>
  <div class="typedist">
    {% for tk, tlabel, cnt in type_distribution %}<span class="chip t-{{ tk }}">{{ tlabel }} <b>{{ cnt }}</b></span>{% endfor %}
  </div>
  {% if notes %}{% for n in notes %}<div class="note">⚠ {{ n }}</div>{% endfor %}{% endif %}

  {% if top_columns_to_watch %}
  <h3>{{ L('top_columns_to_watch') }}</h3>
  <div class="watch-list">
    {% for w in top_columns_to_watch %}
    <div class="watch {{ w.severity }}">
      <span class="sev {{ w.severity }}">{{ sev_icons[w.severity] }} {{ L('severity_' ~ w.severity) }}</span>
      <b>{{ w.column }}</b>
      <ul>{% for r in w.reasons %}<li>{{ r }}</li>{% endfor %}</ul>
    </div>
    {% endfor %}
  </div>
  {% endif %}

  <!-- ============ CHARTS ============ -->
  <h2 id="charts">{{ L('distributions_correlations') }}</h2>
  {% if has_dist_charts %}
    {% if dist_charts.correlation_heatmap_plotly %}
   <div class="plotly-chart">{{ dist_charts.correlation_heatmap_plotly | safe }}</div>
   {% elif dist_charts.correlation_heatmap %}<div class="imgrow full"><div><div class="imgcap">{{ L('correlation_heatmap') }}</div><img src="data:image/png;base64,{{ dist_charts.correlation_heatmap }}" alt="correlation heatmap"><div class="chart-note"><b>{{ L('chart_insight') }}:</b> ดูว่าคอลัมน์ตัวเลขใดเคลื่อนไปด้วยกันสูงมาก อาจใช้แทนกันได้หรือส่งผลต่อโมเดล</div></div></div>{% endif %}
    {% if dist_charts.scatter_matrix %}<div class="imgrow full"><div><div class="imgcap">{{ L('scatter_matrix') }}</div><img src="data:image/png;base64,{{ dist_charts.scatter_matrix }}" alt="scatter matrix"><div class="chart-note"><b>{{ L('chart_insight') }}:</b> ใช้ดู pattern ระหว่างตัวเลข เช่น เส้นตรง กลุ่มย่อย หรือ outlier</div></div></div>{% endif %}
    <div class="imgrow">{% if dist_charts.boxplot %}<div><div class="imgcap">{{ L('boxplot') }}</div><img src="data:image/png;base64,{{ dist_charts.boxplot }}" alt="box plot"><div class="chart-note"><b>{{ L('chart_insight') }}:</b> จุดที่หลุดจากกล่องคือค่าที่ควรตรวจว่าเป็น outlier จริงหรือไม่</div></div>{% endif %}{% if dist_charts.violinplot %}<div><div class="imgcap">{{ L('violinplot') }}</div><img src="data:image/png;base64,{{ dist_charts.violinplot }}" alt="violin plot"><div class="chart-note"><b>{{ L('chart_insight') }}:</b> ดูรูปทรงการกระจายว่าเบ้ มีหลายกลุ่ม หรือกระจุกตัวตรงไหน</div></div>{% endif %}</div>
  {% else %}<p class="empty">{{ L('no_dist_charts') }}</p>{% endif %}

  <h2>{{ L('missing_data') }}</h2>
  {% if has_missing_charts %}
  <div class="imgrow">{% if missing_charts.missing_matrix %}<div><div class="imgcap">{{ L('missing_matrix') }}</div><img src="data:image/png;base64,{{ missing_charts.missing_matrix }}" alt="missing value matrix"><div class="chart-note"><b>{{ L('chart_insight') }}:</b> แถบว่างช่วยบอกว่าค่าว่างกระจุกตัวช่วงใดหรือคอลัมน์ใด</div></div>{% endif %}{% if missing_charts.missing_heatmap %}<div><div class="imgcap">{{ L('missing_heatmap') }}</div><img src="data:image/png;base64,{{ missing_charts.missing_heatmap }}" alt="missing nullity correlation heatmap"><div class="chart-note"><b>{{ L('chart_insight') }}:</b> ถ้าค่าว่างเกิดพร้อมกันหลายคอลัมน์ อาจเกิดจากขั้นตอนเก็บข้อมูลเดียวกัน</div></div>{% endif %}</div>
  {% else %}<p class="empty">✓ {{ L('no_missing') }}</p>{% endif %}

  </div><!-- /tab-overview -->

  <!-- TAB: INSIGHTS -->
  <div id="tab-insights" class="tab-panel">
  <!-- ============ KEY INSIGHTS ============ -->
  {% if insight_section %}
  <h2 id="insights">{{ L('auto_insights') }}
    <span class="sevcount">
      {% if insight_section.critical_count %}<span class="sev critical">{{ insight_section.critical_count }} {{ L('severity_critical') }}</span>{% endif %}
      {% if insight_section.warning_count %}<span class="sev warning">{{ insight_section.warning_count }} {{ L('severity_warning') }}</span>{% endif %}
      {% if insight_section.info_count %}<span class="sev info">{{ insight_section.info_count }} {{ L('severity_info') }}</span>{% endif %}
    </span>
  </h2>
  {% if insight_section.total_generated and insight_section.total_generated > insight_section.insights|length %}
  <div class="note">{{ L('showing_top_of') }} {{ insight_section.insights|length }} {{ L('of_total') }} {{ insight_section.total_generated }} {{ L('auto_insights') }}</div>
  {% endif %}
  <div class="insight-grid">
    {% for ins in insight_section.insights[:20] %}
    <div class="issue insight {{ ins.severity }}">
      <div>
        <span class="sev {{ ins.severity }}">{{ sev_icons[ins.severity] }} {{ L('severity_' ~ ins.severity) }}</span>
        <span class="badge cat">{{ ins.category_label }}</span>
        <b>{{ ins.title_th }}</b>
      </div>
      <div class="desc-th">{{ ins.description_th }}</div>
      <div class="so-what"><span class="lbl">{{ L('so_what') }}</span> {{ ins.recommendation_th }}</div>
    </div>
    {% endfor %}
  </div>
  {% if insight_section.insights|length > 20 %}
  <details class="block">
    <summary class="ng">{{ L('show_more_insights') }} <b>({{ insight_section.insights|length - 20 }})</b></summary>
    <div class="body">
      <div class="insight-grid">
        {% for ins in insight_section.insights[20:] %}
        <div class="issue insight {{ ins.severity }}">
          <div>
            <span class="sev {{ ins.severity }}">{{ sev_icons[ins.severity] }} {{ L('severity_' ~ ins.severity) }}</span>
            <span class="badge cat">{{ ins.category_label }}</span>
            <b>{{ ins.title_th }}</b>
          </div>
          <div class="desc-th">{{ ins.description_th }}</div>
          <div class="so-what"><span class="lbl">{{ L('so_what') }}</span> {{ ins.recommendation_th }}</div>
        </div>
        {% endfor %}
      </div>
    </div>
  </details>
  {% endif %}
  {% endif %}

  <!-- ============ CROSS-COLUMN INSIGHTS ============ -->
  {% if business_section %}
  <h2 id="business-insights">{{ L('business_insights') }} <span class="ng">({{ business_section.total }})</span></h2>
  {% for c in business_section.cards %}
  <div class="issue insight {{ c.severity if c.severity else 'info' }}">
    <div>
      <span class="badge t-numeric">{{ c.pattern_label }}</span>
      <b>{{ c.title_th }}</b>
      <span class="cat ng">{% if c.perspective.breakdown %}{{ L('breakdown') }}: <code>{{ c.perspective.breakdown }}</code>{% endif %}{% if c.perspective.measure %} · {{ L('measure') }}: <code>{{ c.perspective.measure }}</code>{% endif %} · {{ c.perspective.agg }}</span>
    </div>
    <div class="desc-th">{{ c.description_th }}</div>
    {% if c.chart %}
    <div class="imgrow full"><div><div class="imgcap">{{ L('chart_insight') }}</div><img src="data:image/png;base64,{{ c.chart }}" alt="insight chart"></div></div>
    {% endif %}
    {% if c.evidence.top_segments and c.pattern != 'trend' %}
    <table><tr><th>{{ L('segment') }}</th><th>{{ L('value') }}</th></tr>{% for seg, val in c.evidence.top_segments %}<tr><td>{{ seg }}</td><td class="ng">{{ "{:,}".format(val) }}</td></tr>{% endfor %}</table>
    {% endif %}
    {% if c.pattern == 'comparison' %}<div class="meta">{{ L('lift') }}: <b>{{ c.evidence.lift_pct }}%</b> · {{ L('mean') }}: {{ "{:,}".format(c.evidence.top_mean) }} vs {{ "{:,}".format(c.evidence.rest_mean) }}{% if c.evidence.p_value is not none %} · {{ L('p_value') }}={{ c.evidence.p_value }}{% endif %} · n={{ "{:,}".format(c.evidence.n_top) }}/{{ "{:,}".format(c.evidence.n_rest) }}</div>{% endif %}
    {% if c.pattern == 'attribution' %}<div class="meta">{{ L('share') }}: <b>{{ c.evidence.share }}%</b></div>{% endif %}
    {% if c.pattern == 'trend' %}<div class="meta">τ={{ c.evidence.tau }}{% if c.evidence.p_value is not none %} · {{ L('p_value') }}={{ c.evidence.p_value }}{% endif %} · {{ c.evidence.first_bucket }} → {{ c.evidence.last_bucket }} ({{ "{:,}".format(c.evidence.first_value) }} → {{ "{:,}".format(c.evidence.last_value) }})</div>{% endif %}
    <div class="so-what"><span class="lbl">{{ L('so_what') }}</span> {{ c.recommendation_th }}</div>
  </div>
  {% endfor %}
  {% endif %}

  </div><!-- /tab-insights -->

  <!-- TAB: QUALITY -->
  <div id="tab-quality" class="tab-panel">
  {% if quality_comparison %}
  <h2 id="quality-comparison">{{ L('quality_before_after') }}</h2>
  <div class="panel">
    <table>
      <tr><th></th><th>{{ L('critical') }}</th><th>{{ L('warning') }}</th><th>{{ L('info') }}</th><th>{{ L('quality_score_before') }}/{{ L('quality_score_after') }}</th></tr>
      <tr><td>{{ L('before') }}</td><td>{{ quality_comparison.before.critical_count }}</td><td>{{ quality_comparison.before.warning_count }}</td><td>{{ quality_comparison.before.info_count }}</td><td><b>{{ quality_comparison.score_before }}</b> ({{ quality_comparison.grade_before }})</td></tr>
      <tr><td>{{ L('after') }}</td><td>{{ quality_comparison.after.critical_count }}</td><td>{{ quality_comparison.after.warning_count }}</td><td>{{ quality_comparison.after.info_count }}</td><td><b>{{ quality_comparison.score_after }}</b> ({{ quality_comparison.grade_after }})</td></tr>
    </table>
    {% if quality_comparison.fixed_checks %}
    <div class="note">{{ L('quality_fixed_checks') }}: <span class="mono">{{ quality_comparison.fixed_checks|join(', ') }}</span></div>
    {% endif %}
  </div>
  {% endif %}
  {% if cleaning_plan and cleaning_plan.actions %}
  <h2 id="cleaning-plan">{{ L('cleaning_plan') }}</h2>
  <div class="panel">
    <div><b>{{ L('cleaning_plan_actions') }}:</b> {{ cleaning_plan.actions|join(', ') }}</div>
    {% if cleaning_plan.skipped %}<div class="ng">{{ L('cleaning_plan_skipped') }}: {{ cleaning_plan.skipped|join(', ') }}</div>{% endif %}
  </div>
  {% endif %}
  <!-- ============ QUALITY ISSUES ============ -->
  <h2 id="quality">{{ L('quality_issues') }} <span class="ng">({{ quality_issues|length }})</span></h2>
  {% if quality_issues %}
    {% if quality_issues|length > 50 %}<div class="note">{{ L('showing_top_of') }} 50 {{ L('of_total') }} {{ quality_issues|length }}</div>{% endif %}
    {% for iss in quality_issues[:50] %}{{ render_issue(iss, sev_icons, L) }}{% endfor %}
    {% if quality_issues|length > 50 %}
    <details class="block"><summary class="ng">{{ L('show_details') }} <b>({{ quality_issues|length - 50 }})</b></summary>
      <div class="body">{% for iss in quality_issues[50:] %}{{ render_issue(iss, sev_icons, L) }}{% endfor %}</div>
    </details>
    {% endif %}
  {% else %}
    <p class="empty">✓ {{ L('no_issues') }}</p>
  {% endif %}

  </div><!-- /tab-quality -->

  <!-- TAB: ANOMALIES -->
  <div id="tab-anomalies" class="tab-panel">
  <!-- ============ ANOMALIES ============ -->
  <h2 id="anomalies">{{ L('anomalies') }} <span class="ng">({{ anomalies|length }})</span></h2>
  {% if anomalies %}
    {% if anomalies|length > 50 %}<div class="note">{{ L('showing_top_of') }} 50 {{ L('of_total') }} {{ anomalies|length }}</div>{% endif %}
    {% for an in anomalies[:50] %}{{ render_anomaly(an, sev_icons, L) }}{% endfor %}
    {% if anomalies|length > 50 %}
    <details class="block"><summary class="ng">{{ L('show_details') }} <b>({{ anomalies|length - 50 }})</b></summary>
      <div class="body">{% for an in anomalies[50:] %}{{ render_anomaly(an, sev_icons, L) }}{% endfor %}</div>
    </details>
    {% endif %}
  {% else %}
    <p class="empty">✓ {{ L('no_anomalies') }}</p>
  {% endif %}

  <!-- ============ TARGET ANALYSIS ============ -->
  {% if target_section %}
  <h2 id="target">{{ L('target_analysis') }} <span class="ng">({{ L('target_column') }}: {{ target_section.target_column }})</span></h2>
  {% if target_section.associations %}
    <table><tr><th>{{ L('column') }}</th><th>{{ L('association') }}</th><th>{{ L('score') }}</th><th>{{ L('p_value') }}</th><th>{{ L('so_what') }}</th></tr>
      {% for a in target_section.associations %}<tr><td><b>{{ a.column }}</b></td><td><span class="badge">{{ a.type_label }}</span></td><td>{{ a.score if a.score is not none else '—' }}</td><td>{{ a.p_value if a.p_value is not none else '—' }}</td><td class="ng">{{ a.description_th }}</td></tr>{% endfor %}
    </table>
  {% else %}<p class="empty">{{ L('no_target') }}</p>{% endif %}
  {% endif %}

  <!-- ============ NAMED ENTITIES ============ -->
  {% if ner_sections %}
  <h2>{{ L('named_entities') }}</h2>
  {% for sec in ner_sections %}
    <details class="block"><summary><b>{{ sec.column }}</b><span class="badge">{{ L('total_entities') }}: {{ "{:,}".format(sec.result.total_entities) }}</span><span class="badge">{{ L('ner_engine') }}: {{ sec.result.engine_used }}</span></summary>
      <div class="body"><table><tr><th>{{ L('entity_type') }}</th><th>{{ L('count') }}</th><th>{{ L('top_entities') }}</th></tr>
        {% for etype, cnt in sec.result.entity_counts.items() %}<tr><td><b>{{ etype }}</b></td><td>{{ "{:,}".format(cnt) }}</td><td>{% for ent, c in sec.result.top_entities.get(etype, [])[:10] %}<span class="mono">{{ ent }} ({{ c }})</span> {% endfor %}</td></tr>{% endfor %}
      </table></div>
    </details>
  {% endfor %}
  {% endif %}

  <!-- ============ TIMESERIES ANALYSIS ============ -->
  {% if timeseries_section %}
  <h2 id="timeseries">{{ L('timeseries') }} <span class="ng">({{ L('type_datetime') }}: {{ timeseries_section.time_column }})</span></h2>
  {% if timeseries_section.trend_count or timeseries_section.seasonal_count %}<div class="panel">📈 พบสัญญาณตามเวลา: {% if timeseries_section.trend_count %}<b>{{ timeseries_section.trend_count }}</b> คอลัมน์มีแนวโน้ม{% endif %}{% if timeseries_section.trend_count and timeseries_section.seasonal_count %} · {% endif %}{% if timeseries_section.seasonal_count %}<b>{{ timeseries_section.seasonal_count }}</b> คอลัมน์มีฤดูกาล{% endif %}</div>{% endif %}
  {% for ts in timeseries_section.columns %}
  <details class="block" open><summary><b>{{ ts.column }}</b><span class="badge">{{ L('ts_frequency') }}: {{ ts.result.frequency_th }}</span>{% if ts.result.has_trend %}<span class="badge">{{ L('ts_trend') }}: {{ ts.result.trend_direction_th }}</span>{% endif %}{% if ts.result.has_seasonality %}<span class="badge">{{ L('ts_seasonality') }}: {{ ts.result.seasonal_period_th }}</span>{% endif %}</summary>
    <div class="body">
      <table><tr><th>{{ L('ts_trend') }}</th><td>{{ ts.result.trend_direction_th if ts.result.has_trend else L('ts_none') }}</td><th>{{ L('ts_seasonality') }}</th><td>{{ ts.result.seasonal_period_th if ts.result.has_seasonality else L('ts_none') }}</td></tr><tr><th>{{ L('ts_gaps') }}</th><td>{{ ts.result.gap_count }}</td><th>{{ L('ts_anomalies') }}</th><td>{{ ts.result.anomaly_count }}</td></tr><tr><th>{{ L('ts_autocorr') }}</th><td>{{ ts.result.stats.autocorr_lag1 }}</td><th>{{ L('mean') }}</th><td>{{ ts.result.stats.mean }}</td></tr></table>
      {% if ts.result.insights %}<div class="so-what"><span class="lbl">{{ L('so_what') }}</span>{% for ins in ts.result.insights %}<div>• {{ ins }}</div>{% endfor %}</div>{% endif %}
      {% if ts.charts.line %}<div class="imgrow full"><div><div class="imgcap">{{ L('ts_timeseries_plot') }}</div><img src="data:image/png;base64,{{ ts.charts.line }}" alt="timeseries line"></div></div>{% endif %}
      {% if ts.charts.decomposition %}<div class="imgrow full"><div><div class="imgcap">{{ L('ts_decomposition') }}</div><img src="data:image/png;base64,{{ ts.charts.decomposition }}" alt="STL decomposition"></div></div>{% endif %}
      {% if ts.charts.acf %}<div class="imgrow full"><div><div class="imgcap">{{ L('ts_acf_plot') }}</div><img src="data:image/png;base64,{{ ts.charts.acf }}" alt="ACF"></div></div>{% endif %}
    </div>
  </details>
  {% endfor %}
  {% endif %}

  <!-- ============ CLEANING APPLIED ============ -->
  {% if cleaning_diff %}
  <h2 id="cleaning">{{ L('cleaning_diff') }} <span class="ng">({{ cleaning_diff|length }})</span></h2>
  {% if cleaning_diff_summary %}<div class="panel">{{ L('values_changed') }}: <b>{{ "{:,}".format(cleaning_diff_summary.values_changed) }}</b>{% if cleaning_diff_summary.rows_removed %} · {{ L('rows_removed') }}: <b{% if cleaning_diff_summary.high_row_loss %} class="sev warning"{% endif %}>{{ "{:,}".format(cleaning_diff_summary.rows_removed) }}</b> ({{ cleaning_diff_summary.rows_removed_pct }}% {{ L('of_rows') }}){% if cleaning_diff_summary.high_row_loss %} ⚠ <span class="ng">{{ L('high_row_loss_warn') }}</span>{% endif %}{% endif %} · {{ L('most_impactful') }}: <b>{{ cleaning_diff_summary.most_impactful_op }}</b> <span class="ng">({{ cleaning_diff_summary.most_impactful_th }} — {{ "{:,}".format(cleaning_diff_summary.most_impactful_rows) }})</span></div>{% endif %}
  <table><tr><th>{{ L('column') }}</th><th>{{ L('operation') }}</th><th>{{ L('rows_affected') }}</th><th>{{ L('before') }} → {{ L('after') }}</th></tr>{% for c in cleaning_diff %}<tr><td><b>{{ c.column }}</b></td><td>{{ c.operation }}<div class="ng">{{ c.description_th }}</div>{% if c.explanation %}<div class="ng">{{ L('explanation') }}: <span class="mono">{{ c.explanation }}</span></div>{% endif %}</td><td>{{ "{:,}".format(c.rows_affected) }}</td><td class="diff">{% for ex in c.before_examples %}<span class="row"><span class="b mono">{{ ex }}</span><span class="arrow">→</span><span class="a mono">{{ c.after_examples[loop.index0] }}</span></span>{% endfor %}</td></tr>{% endfor %}</table>
  {% endif %}

  <!-- ============ CLEANING SUGGESTIONS ============ -->
  <h2 id="cleaning-suggestions">{{ L('cleaning_suggestions') }} <span class="ng">({{ cleaning_suggestions|length }})</span></h2>
  {% if cleaning_suggestions %}
    <table><tr><th>{{ L('column') }}</th><th>{{ L('operation') }}</th><th>{{ L('rows_affected') }}</th><th>{{ L('before') }} → {{ L('after') }}</th></tr>{% for c in cleaning_suggestions %}<tr><td><b>{{ c.column }}</b></td><td>{{ c.operation }}<div class="ng">{{ c.description_th }}</div>{% if c.explanation %}<div class="ng">{{ L('explanation') }}: <span class="mono">{{ c.explanation }}</span></div>{% endif %}</td><td>{{ "{:,}".format(c.rows_affected) }}</td><td class="diff">{% for ex in c.before_examples %}<span class="row"><span class="b mono">{{ ex }}</span><span class="arrow">→</span><span class="a mono">{{ c.after_examples[loop.index0] }}</span></span>{% endfor %}</td></tr>{% endfor %}</table>
  {% else %}<p class="empty">✓ {{ L('no_cleaning') }}</p>{% endif %}

  </div><!-- /tab-anomalies -->

  <!-- TAB: COLUMNS -->
  <div id="tab-columns" class="tab-panel">
  <!-- ============ COLUMN DETAILS ============ -->
  <h2 id="columns">{{ L('column_details') }}</h2>
  {% if columns|length > 60 %}
  <p class="muted">คอลัมน์จำนวนมาก ({{ columns|length }}) — สรุปเป็นตารางแทนการ์ดรายคอลัมน์เพื่อลดขนาดไฟล์ HTML</p>
  <table>
    <tr><th>{{ L('column') }}</th><th>{{ L('column_types') }}</th>{% for k, v in columns[0].basic_stats[:3] %}<th>{{ k }}</th>{% endfor %}</tr>
    {% for col in columns %}
    <tr><td><b>{{ col.name }}</b></td><td><span class="badge t-{{ col.type_key }}">{{ col.type_label }}</span></td>{% for k, v in col.basic_stats[:3] %}<td>{{ v }}</td>{% endfor %}</tr>
    {% endfor %}
  </table>
  {% else %}
  <div class="col-filter">
    <select onchange="filterColumns(this)">
      <option value="all">All Columns ({{ columns|length }})</option>
      {% for col in columns %}<option value="{{ col.name }}">{{ col.name }}</option>{% endfor %}
    </select>
  </div>
  <p class="muted">ส่วนนี้ซ่อนไว้เป็นรายคอลัมน์เพื่อลดความรก เปิดเฉพาะคอลัมน์ที่ต้องการตรวจละเอียด</p>
  {% for col in columns %}
  <details class="block col" data-col-name="{{ col.name }}">
    <summary><span class="nm">{{ col.name }}</span><span class="badge t-{{ col.type_key }}">{{ col.type_label }}</span>{% if col.dist_chart or col.valuecounts_chart or col.charts.top_tokens %}<span class="badge">มีกราฟ</span>{% endif %}</summary>
    <div class="body">
      {% if col.is_text and col.metrics %}
        <table><tr><th>{{ L('engine_used') }}</th><td>{{ col.metrics.engine_used }}</td><th>{{ L('avg_char_length') }}</th><td>{{ col.metrics.avg_char_length }}</td></tr><tr><th>{{ L('avg_token_length') }}</th><td>{{ col.metrics.avg_token_length }}</td><th>{{ L('median_char_length') }}</th><td>{{ col.metrics.median_char_length }}</td></tr><tr><th>{{ L('total_tokens') }}</th><td>{{ "{:,}".format(col.metrics.total_tokens) }}</td><th>{{ L('unique_tokens') }}</th><td>{{ "{:,}".format(col.metrics.unique_tokens) }}</td></tr><tr><th>{{ L('min_char_length') }}</th><td>{{ col.metrics.min_char_length }}</td><th>{{ L('max_char_length') }}</th><td>{{ col.metrics.max_char_length }}</td></tr></table>
        {% if col.metrics.sampled_cells < col.metrics.non_null_cells %}<div class="note">{{ L('sampled_note') }}: {{ "{:,}".format(col.metrics.sampled_cells) }} {{ L('of') }} {{ "{:,}".format(col.metrics.non_null_cells) }}</div>{% endif %}
        {% if col.metrics.top_bigrams %}<p class="ng">{{ L('top_bigrams') }}: {% for ng, c in col.metrics.top_bigrams[:8] %}<span class="mono">{{ ng }} ({{ c }})</span> {% endfor %}</p>{% endif %}
        <div class="imgrow">{% if col.charts.top_tokens %}<div><div class="imgcap">{{ L('top_tokens') }}</div><img src="data:image/png;base64,{{ col.charts.top_tokens }}" alt="top tokens"></div>{% endif %}{% if col.charts.length_hist %}<div><div class="imgcap">{{ L('length_distribution') }}</div><img src="data:image/png;base64,{{ col.charts.length_hist }}" alt="length histogram"></div>{% endif %}</div>
        {% if col.charts.wordcloud %}<div class="imgrow full"><div><div class="imgcap">{{ L('wordcloud') }}</div><img src="data:image/png;base64,{{ col.charts.wordcloud }}" alt="word cloud"></div></div>{% endif %}
      {% else %}
        <table>{% for k, v in col.basic_stats %}<tr><th>{{ k }}</th><td>{{ v }}</td></tr>{% endfor %}</table>
        {% if col.dist_chart %}<div class="imgrow full"><div><div class="imgcap">{{ L('distribution') }}</div><img src="data:image/png;base64,{{ col.dist_chart }}" alt="value distribution"></div></div>{% endif %}
        {% if col.valuecounts_chart %}<div class="imgrow full"><div><div class="imgcap">{{ L('value_counts') }}</div><img src="data:image/png;base64,{{ col.valuecounts_chart }}" alt="value counts"></div></div>{% endif %}
        {% if col.top_values %}<h3>{{ L('top_values') }}</h3><table>{% for val, cnt in col.top_values %}<tr><td>{{ val }}</td><td class="ng">{{ "{:,}".format(cnt) }}</td></tr>{% endfor %}</table>{% endif %}
      {% endif %}
    </div>
  </details>
  {% endfor %}
  {% endif %}

  </div><!-- /tab-columns -->

  <footer>{{ L('generated_by') }} <b>ThaiEDA</b> v{{ version }} — AutoEDA สำหรับข้อมูลภาษาไทย</footer>
</div>
<button id="back-to-top" onclick="window.scrollTo({top:0,behavior:'smooth'})" title="Back to top">↑</button>
</body>
</html>"""
