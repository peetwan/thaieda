"""Jinja2 HTML template สำหรับรายงานชุดข้อมูลหลายตาราง (DatasetReport) — v0.5."""
# ruff: noqa: E501, W293 — เทมเพลต HTML มีบรรทัดยาวโดยธรรมชาติ (แอตทริบิวต์/markup)

from __future__ import annotations

# ธีมเข้มเดียวกับ ProfileReport + Mermaid.js (CDN) สำหรับแผนผัง ER
DATASET_TEMPLATE = r"""<!DOCTYPE html>
<html lang="{{ lang }}">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{{ L('dataset_report_title') }}</title>
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
    --accent: #3b82f6;
    --accent-glow: rgba(59, 130, 246, 0.15);
    --critical: #ef4444;
    --warning: #f59e0b;
    --info: #06b6d4;
    --ok: #10b981;
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
    --accent: #2563eb;
    --accent-glow: rgba(37, 99, 235, 0.1);
    --shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.05), 0 8px 10px -6px rgba(0, 0, 0, 0.05);
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--bg); color: var(--fg);
    font-family: var(--font-sans);
    line-height: 1.6; font-size: 15px;
    transition: background 0.3s ease, color 0.3s ease;
  }
  .wrap { max-width: 1200px; margin: 0 auto; padding: 40px 24px 100px; }
  header h1 { font-size: 32px; font-weight: 800; margin: 0 0 8px; letter-spacing: -.5px; }
  header .sub { color: var(--muted); font-size: 14px; }
  h2 { font-size: 24px; font-weight: 700; margin: 48px 0 20px; padding-bottom: 10px;
       border-bottom: 1px solid var(--border); scroll-margin-top: 80px; }
  h3 { font-size: 18px; font-weight: 600; margin: 28px 0 12px; }
  
  .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
           gap: 16px; margin: 20px 0; }
  .card { background: var(--panel); border: 1px solid var(--border);
          border-radius: 16px; padding: 20px; box-shadow: var(--shadow); transition: border-color 0.2s, box-shadow 0.2s; }
  .card:hover { border-color: var(--accent); }
  .card .k { color: var(--muted); font-size: 13px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }
  .card .v { font-size: 28px; font-weight: 800; margin-top: 6px; letter-spacing: -.5px; }
  
  .typedist { display: flex; flex-wrap: wrap; gap: 8px; margin: 16px 0; }
  .chip { background: var(--panel2); border: 1px solid var(--border);
          border-radius: 20px; padding: 6px 14px; font-size: 12.5px; font-weight: 600; color: var(--fg); }
  .chip b { color: var(--accent); }
  
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
  tbody tr:hover td {
    background-color: var(--panel-hover);
  }
  
  code, .mono {
    font-family: "Cascadia Code", Consolas, monospace;
    font-size: 12.5px;
    background: var(--panel2);
    padding: 2px 6px;
    border-radius: 6px;
    border: 1px solid var(--border);
  }
  
  .col {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 24px;
    margin: 20px 0;
    box-shadow: var(--shadow);
  }
  .col .head { display: flex; align-items: baseline; gap: 12px; flex-wrap: wrap; margin-bottom: 12px; }
  .col .head .nm { font-size: 20px; font-weight: 700; letter-spacing: -.3px; }
  .col .head .meta { color: var(--muted); font-size: 13.5px; }
  
  .badge {
    font-size: 12px;
    padding: 4px 10px;
    border-radius: 12px;
    background: var(--panel2);
    border: 1px solid var(--border);
    color: var(--accent);
    font-weight: 600;
  }
  .badge.pk { border-color: #ec4899; color: #ec4899; }
  .badge.fk { border-color: var(--info); color: var(--info); }
  
  .issue {
    background: var(--panel);
    border: 1px solid var(--border);
    border-left: 5px solid var(--border);
    border-radius: 16px;
    padding: 16px 20px;
    margin: 14px 0;
    box-shadow: var(--shadow);
  }
  .issue.critical { border-left-color: var(--critical); background: rgba(239, 68, 68, 0.08); }
  .issue.warning { border-left-color: var(--warning); background: rgba(245, 158, 11, 0.08); }
  .issue.info { border-left-color: var(--info); background: rgba(6, 182, 212, 0.08); }
  .empty { color: var(--muted); font-style: italic; text-align: center; padding: 16px; }
  .ng { color: var(--muted); font-size: 13.5px; }
  .note { color: var(--muted); font-size: 13.5px; font-style: italic; margin-top: 10px; }
  
  footer {
    margin-top: 60px;
    padding: 24px 0;
    border-top: 1px solid var(--border);
    color: var(--muted);
    font-size: 13px;
    text-align: center;
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
  
  .mermaid-box {
    background: #ffffff;
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 24px;
    margin: 20px 0;
    overflow-x: auto;
    box-shadow: var(--shadow);
  }
  
  .conf-bar {
    display: inline-block;
    width: 90px;
    height: 8px;
    border-radius: 4px;
    background: var(--panel2);
    overflow: hidden;
    vertical-align: middle;
    margin-right: 8px;
  }
  .conf-bar > span { display: block; height: 100%; }
  .conf-bar > span.info { background: var(--ok); }
  .conf-bar > span.warning { background: var(--warning); }
  .conf-bar > span.critical { background: var(--critical); }
  .card-num { color: #ec4899; font-weight: 700; }
  
  .hero {
    background: var(--panel);
    border: 1px solid var(--border);
    border-left: 6px solid var(--ok);
    border-radius: 16px;
    padding: 24px;
    margin: 20px 0;
    box-shadow: var(--shadow);
  }
  .hero.warning { border-left-color: var(--warning); }
  .hero.critical { border-left-color: var(--critical); }
  .hero .lbl { color: var(--accent); font-weight: 800; font-size: 12px; text-transform: uppercase; letter-spacing: 1px; }
  .hero .verdict { font-size: 24px; font-weight: 800; margin: 8px 0 10px; letter-spacing: -.5px; }
  .hero ul { margin: 10px 0 0 20px; padding: 0; }
  .hero li { margin: 6px 0; }
  
  .actions { display: grid; gap: 10px; margin: 16px 0; }
  .action {
    background: var(--panel2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 12px 16px;
    font-size: 14.5px;
    transition: transform 0.2s ease;
  }
  .action:hover {
    transform: translateX(4px);
  }
  
  @media (max-width: 820px) {
    .wrap { padding: 24px 16px 80px; }
    .cards { grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); }
    .nav { position: static; }
    table { display: block; overflow-x: auto; -webkit-overflow-scrolling: touch; }
  }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>{{ L('dataset_report_title') }}</h1>
    <div class="sub">{{ L('generated_by') }} ThaiEDA v{{ version }} · {{ overview.table_count }} {{ L('tables') }} · {{ overview.relationship_count }} {{ L('relationships') }}</div>
  </header>

  <!-- ============ NAV ============ -->
  <nav class="nav">
    <a href="#overview">{{ L('schema_overview') }}</a>
    <a href="#er">{{ L('er_diagram') }}</a>
    <a href="#tables">{{ L('tables') }}</a>
    <a href="#relationships">{{ L('relationships') }}</a>
    {% if orphan_findings %}<a href="#orphans">{{ L('orphan_findings') }}</a>{% endif %}
  </nav>

  <!-- ============ OVERVIEW ============ -->
  <h2 id="overview">{{ L('schema_overview') }}</h2>
  <div class="hero {{ summary.status }}">
    <div class="lbl">{{ L('executive_summary') }}</div>
    <div class="verdict">{{ summary.verdict }}</div>
    <ul>{% for h in summary.highlights %}<li>{{ h }}</li>{% endfor %}</ul>
    <h3>{{ L('recommended_actions') }}</h3>
    <div class="actions">{% for a in summary.actions %}<div class="action">{{ loop.index }}. {{ a }}</div>{% endfor %}</div>
  </div>
  <div class="cards">
    <div class="card"><div class="k">{{ L('table_count') }}</div><div class="v">{{ overview.table_count }}</div></div>
    <div class="card"><div class="k">{{ L('relationship_count') }}</div><div class="v">{{ overview.relationship_count }}</div></div>
    <div class="card"><div class="k">{{ L('row_count') }}</div><div class="v">{{ "{:,}".format(overview.total_rows) }}</div></div>
    <div class="card"><div class="k">{{ L('orphan_count_total') }}</div><div class="v">{{ overview.orphan_count }}</div></div>
  </div>
  {% if notes %}
    {% for n in notes %}<div class="note">⚠ {{ n }}</div>{% endfor %}
  {% endif %}

  <!-- ============ ER DIAGRAM (Mermaid) ============ -->
  <h2 id="er">{{ L('er_diagram') }}</h2>
  <div class="mermaid-box">
    <pre class="mermaid">{{ mermaid | safe }}</pre>
  </div>

  <!-- ============ TABLES ============ -->
  <h2 id="tables">{{ L('tables') }} <span class="ng">({{ tables|length }})</span></h2>
  {% for t in tables %}
  <div class="col">
    <div class="head">
      <span class="nm">{{ t.name }}</span>
      <span class="meta">{{ "{:,}".format(t.row_count) }} {{ L('row_count') }} × {{ t.column_count }} {{ L('columns') }}</span>
    </div>
    <div class="typedist">
      {% for tk, tlabel, cnt in t.type_dist %}
        <span class="chip t-{{ tk }}">{{ tlabel }} <b>{{ cnt }}</b></span>
      {% endfor %}
    </div>
    {% if t.key_candidates %}
    <h3>{{ L('key_candidates') }}</h3>
    <table>
      <thead><tr><th>{{ L('column') }}</th><th></th><th>{{ L('unique') }}</th><th>{{ L('cardinality') }}</th><th>dtype</th></tr></thead>
      <tbody>
      {% for k in t.key_candidates %}
        <tr>
          <td class="mono">{{ k.column }}</td>
          <td><span class="badge {{ 'pk' if k.is_unique else 'fk' }}">{{ k.role }}</span></td>
          <td>{{ '✓' if k.is_unique else '—' }}</td>
          <td>{{ "{:,}".format(k.cardinality) }}</td>
          <td class="ng">{{ k.dtype }}</td>
        </tr>
      {% endfor %}
      </tbody>
    </table>
    {% endif %}
  </div>
  {% endfor %}

  <!-- ============ RELATIONSHIPS ============ -->
  <h2 id="relationships">{{ L('relationships') }} <span class="ng">({{ relationships|length }})</span></h2>
  {% if relationships %}
  <table>
    <thead>
      <tr>
        <th>{{ L('from_table') }}</th>
        <th>{{ L('to_table') }}</th>
        <th>{{ L('column') }}</th>
        <th>{{ L('cardinality') }}</th>
        <th>{{ L('overlap') }}</th>
        <th>{{ L('orphan_count') }}</th>
        <th>{{ L('confidence') }}</th>
        <th>{{ L('match_method') }}</th>
      </tr>
    </thead>
    <tbody>
    {% for r in relationships %}
      <tr>
        <td><b>{{ r.from_table }}</b></td>
        <td><b>{{ r.to_table }}</b></td>
        <td class="mono">{{ r.from_column }}</td>
        <td>{{ r.cardinality }}</td>
        <td>{% if r.is_validated %}{{ r.overlap_pct }}%{% else %}<span class="ng">—</span>{% endif %}</td>
        <td>{% if r.orphan_count %}<span class="card-num">{{ "{:,}".format(r.orphan_count) }}</span>{% else %}0{% endif %}</td>
        <td>
          <span class="conf-bar"><span class="{{ r.conf_class }}" style="width: {{ r.conf_pct }}%"></span></span>
          {{ r.conf_pct }}%
        </td>
        <td><span class="badge">{{ r.method_label }}</span></td>
      </tr>
    {% endfor %}
    </tbody>
  </table>
  {% else %}
    <p class="empty">{{ L('no_relationships') }}</p>
  {% endif %}

  <!-- ============ ORPHAN FINDINGS ============ -->
  {% if orphan_findings %}
  <h2 id="orphans">{{ L('orphan_findings') }} <span class="ng">({{ orphan_findings|length }})</span></h2>
  {% for o in orphan_findings %}
  <div class="issue warning">{{ o }}</div>
  {% endfor %}
  {% endif %}

  <footer>
    {{ L('generated_by') }} ThaiEDA v{{ version }}
  </footer>
</div>

<script type="module">
  import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs';
  mermaid.initialize({ startOnLoad: true, theme: 'neutral', securityLevel: 'loose' });
</script>
</body>
</html>
"""
