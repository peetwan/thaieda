"""Jinja2 HTML template สำหรับรายงานชุดข้อมูลหลายตาราง (DatasetReport) — v0.5."""
# ruff: noqa: E501 — เทมเพลต HTML มีบรรทัดยาวโดยธรรมชาติ (แอตทริบิวต์/markup)

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
    --bg: #15171c; --panel: #1d2027; --panel2: #23272f; --fg: #e6e6e6;
    --muted: #9aa0aa; --border: #2e333c; --accent: #4dabf7;
    --critical: #ff6b6b; --warning: #ffd43b; --info: #4dabf7; --ok: #51cf66;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--bg); color: var(--fg);
    font-family: "Segoe UI", "Sarabun", "Noto Sans Thai", Tahoma, sans-serif;
    line-height: 1.6; font-size: 15px;
  }
  .wrap { max-width: 1100px; margin: 0 auto; padding: 32px 24px 80px; }
  header h1 { font-size: 28px; margin: 0 0 4px; }
  header .sub { color: var(--muted); font-size: 14px; }
  h2 { font-size: 21px; margin: 40px 0 16px; padding-bottom: 8px;
       border-bottom: 1px solid var(--border); scroll-margin-top: 60px; }
  h3 { font-size: 17px; margin: 18px 0 10px; }
  .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
           gap: 14px; margin: 18px 0; }
  .card { background: var(--panel); border: 1px solid var(--border);
          border-radius: 10px; padding: 16px; }
  .card .k { color: var(--muted); font-size: 13px; }
  .card .v { font-size: 24px; font-weight: 600; margin-top: 4px; }
  .typedist { display: flex; flex-wrap: wrap; gap: 8px; margin: 12px 0; }
  .chip { background: var(--panel2); border: 1px solid var(--border);
          border-radius: 20px; padding: 5px 14px; font-size: 13px; }
  .chip b { color: var(--accent); }
  table { width: 100%; border-collapse: collapse; margin: 10px 0; }
  th, td { text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--border);
           font-size: 14px; vertical-align: top; }
  th { color: var(--muted); font-weight: 600; }
  code, .mono { font-family: "Cascadia Code", Consolas, monospace; font-size: 12.5px;
                background: var(--panel2); padding: 1px 6px; border-radius: 4px; }
  .col { background: var(--panel); border: 1px solid var(--border);
         border-radius: 10px; padding: 18px 20px; margin: 16px 0; }
  .col .head { display: flex; align-items: baseline; gap: 12px; flex-wrap: wrap; }
  .col .head .nm { font-size: 18px; font-weight: 600; }
  .col .head .meta { color: var(--muted); font-size: 13px; }
  .badge { font-size: 12px; padding: 2px 10px; border-radius: 14px;
           background: var(--panel2); border: 1px solid var(--border); color: var(--accent); }
  .badge.pk { border-color: #f783ac; color: #f783ac; }
  .badge.fk { border-color: var(--info); color: var(--info); }
  .issue { background: var(--panel); border: 1px solid var(--border);
           border-left: 4px solid var(--border); border-radius: 8px;
           padding: 12px 18px; margin: 10px 0; }
  .issue.critical { border-left-color: var(--critical); }
  .issue.warning { border-left-color: var(--warning); }
  .issue.info { border-left-color: var(--info); }
  .empty { color: var(--muted); font-style: italic; }
  .ng { color: var(--muted); font-size: 13px; }
  .note { color: var(--muted); font-size: 13px; font-style: italic; margin-top: 8px; }
  footer { margin-top: 48px; padding-top: 16px; border-top: 1px solid var(--border);
           color: var(--muted); font-size: 13px; }
  .nav { position: sticky; top: 0; z-index: 10; display: flex; flex-wrap: wrap; gap: 6px;
         background: rgba(21,23,28,.92); backdrop-filter: blur(6px);
         border: 1px solid var(--border); border-radius: 10px;
         padding: 8px 10px; margin: 18px 0 8px; }
  .nav a { color: var(--muted); text-decoration: none; font-size: 13px; font-weight: 600;
           padding: 4px 10px; border-radius: 16px; white-space: nowrap; }
  .nav a:hover { color: var(--fg); background: var(--panel2); }
  .mermaid-box { background: #fff; border: 1px solid var(--border); border-radius: 10px;
                 padding: 16px; margin: 14px 0; overflow-x: auto; }
  .conf-bar { display: inline-block; width: 90px; height: 8px; border-radius: 4px;
              background: var(--panel2); overflow: hidden; vertical-align: middle; margin-right: 6px; }
  .conf-bar > span { display: block; height: 100%; }
  .conf-bar > span.info { background: var(--ok); }
  .conf-bar > span.warning { background: var(--warning); }
  .conf-bar > span.critical { background: var(--critical); }
  .card-num { color: #f783ac; }
  @media (max-width: 720px) {
    .wrap { padding: 20px 14px 60px; }
    .cards { grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); }
    .nav { position: static; }
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
