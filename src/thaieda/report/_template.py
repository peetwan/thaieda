"""Jinja2 HTML template สำหรับรายงาน ThaiEDA — เก็บเป็นสตริงเพื่อให้แพ็กเกจ self-contained."""
# ruff: noqa: E501 — เทมเพลต HTML มีบรรทัดยาวโดยธรรมชาติ (แอตทริบิวต์/markup)

from __future__ import annotations

# ธีมเข้ม, modern, CSS ฝังในตัว, รูปเป็น base64 — ไม่พึ่งไฟล์ภายนอก
REPORT_TEMPLATE = r"""<!DOCTYPE html>
<html lang="{{ lang }}">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{{ L('report_title') }}</title>
<style>
  :root {
    --bg: #15171c; --panel: #1d2027; --panel2: #23272f; --panel3: #292e38;
    --fg: #e6e6e6; --muted: #9aa0aa; --border: #2e333c; --accent: #4dabf7;
    --critical: #ff6b6b; --warning: #ffd43b; --info: #4dabf7; --ok: #51cf66;
    --shadow: 0 18px 50px rgba(0,0,0,.22);
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: radial-gradient(circle at top left, rgba(77,171,247,.12), transparent 32rem), var(--bg);
    color: var(--fg); font-family: "Segoe UI", "Sarabun", "Noto Sans Thai", Tahoma, sans-serif;
    line-height: 1.65; font-size: 15px;
  }
  .wrap { max-width: 1160px; margin: 0 auto; padding: 32px 24px 80px; }
  header { display: flex; justify-content: space-between; gap: 20px; align-items: flex-start; margin-bottom: 16px; }
  header h1 { font-size: 30px; margin: 0 0 6px; letter-spacing: -.2px; }
  header .sub, .muted { color: var(--muted); font-size: 14px; }
  h2 { font-size: 22px; margin: 42px 0 16px; padding-bottom: 8px; border-bottom: 1px solid var(--border); scroll-margin-top: 70px; }
  h3 { font-size: 17px; margin: 22px 0 10px; }
  a { color: inherit; }
  table { width: 100%; border-collapse: collapse; margin: 10px 0; }
  th, td { text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--border); font-size: 14px; vertical-align: top; }
  th { color: var(--muted); font-weight: 600; }
  code, .mono { font-family: "Cascadia Code", Consolas, monospace; font-size: 12.5px; background: var(--panel2); padding: 1px 6px; border-radius: 4px; word-break: break-all; }
  .nav { position: sticky; top: 0; z-index: 10; display: flex; flex-wrap: wrap; gap: 6px; background: rgba(21,23,28,.92); backdrop-filter: blur(8px); border: 1px solid var(--border); border-radius: 12px; padding: 8px 10px; margin: 18px 0 18px; }
  .nav a { color: var(--muted); text-decoration: none; font-size: 13px; font-weight: 650; padding: 5px 11px; border-radius: 16px; white-space: nowrap; }
  .nav a:hover { color: var(--fg); background: var(--panel2); }

  .hero-grid { display: grid; grid-template-columns: 1.35fr .65fr; gap: 16px; margin: 18px 0 20px; }
  /* Data type banner */
  .data-type-banner { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin: 18px 0 20px; }
  .dtype-card { background: var(--panel); border: 1px solid var(--border); border-radius: 14px; padding: 20px 24px; }
  .dtype-label { color: var(--accent); font-size: 13px; font-weight: 700; text-transform: uppercase; letter-spacing: .5px; margin-bottom: 6px; }
  .dtype-value { font-size: 26px; font-weight: 800; margin-bottom: 8px; }
  .dtype-summary { font-size: 15px; line-height: 1.6; }
  .dtype-focus { background: var(--panel2); border: 1px solid var(--border); border-radius: 14px; padding: 20px 24px; }
  .dtype-focus-label { color: var(--accent); font-size: 13px; font-weight: 700; text-transform: uppercase; letter-spacing: .5px; margin-bottom: 10px; }
  .dtype-focus ul { margin: 0; padding-left: 20px; }
  .dtype-focus li { margin: 6px 0; font-size: 14px; }
  /* Key findings */
  .key-findings-list { display: grid; gap: 14px; margin: 14px 0; }
  .key-finding { display: flex; gap: 16px; background: var(--panel); border: 1px solid var(--border); border-radius: 14px; padding: 18px 22px; border-left: 4px solid var(--accent); }
  .key-finding.critical { border-left-color: var(--critical); }
  .key-finding.warning { border-left-color: var(--warning); }
  .key-finding.info { border-left-color: var(--info); }
  .kf-num { font-size: 28px; font-weight: 800; color: var(--muted); min-width: 36px; text-align: center; }
  .kf-body { flex: 1; }
  .kf-business { margin: 8px 0 4px; font-size: 16px; line-height: 1.6; color: var(--fg); }
  .kf-tech { font-size: 13px; margin-top: 4px; }
  .kf-impact { margin-top: 8px; font-size: 14px; }
  .kf-impact .lbl { color: var(--accent); font-weight: 700; }
  .hero, .panel, .card, .issue, .col, details.block { background: rgba(29,32,39,.96); border: 1px solid var(--border); border-radius: 14px; box-shadow: var(--shadow); }
  .hero { padding: 24px; border-left: 5px solid var(--accent); }
  .hero.critical { border-left-color: var(--critical); }
  .hero.warning { border-left-color: var(--warning); }
  .hero.good { border-left-color: var(--ok); }
  .hero .eyebrow { color: var(--accent); font-weight: 800; font-size: 13px; letter-spacing: .5px; text-transform: uppercase; }
  .hero.critical .eyebrow { color: var(--critical); }
  .hero.warning .eyebrow { color: var(--warning); }
  .hero.good .eyebrow { color: var(--ok); }
  .hero .verdict { font-size: 24px; font-weight: 800; line-height: 1.35; margin: 8px 0 10px; }
  .hero .summary { font-size: 16px; color: #f2f4f8; }
  .hero ul, .panel ul { margin: 10px 0 0 20px; padding: 0; }
  .panel { padding: 18px 20px; }
  .panel h3 { margin-top: 0; }
  .health-meter { display: grid; gap: 10px; }
  .health-row { display: flex; justify-content: space-between; gap: 12px; color: var(--muted); border-bottom: 1px solid var(--border); padding-bottom: 8px; }
  .health-row:last-child { border-bottom: 0; padding-bottom: 0; }
  .health-row b { color: var(--fg); }

  .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 14px; margin: 18px 0; }
  .card { padding: 16px; box-shadow: none; }
  .card .k { color: var(--muted); font-size: 13px; }
  .card .v { font-size: 25px; font-weight: 750; margin-top: 3px; }
  .typedist { display: flex; flex-wrap: wrap; gap: 8px; margin: 12px 0; }
  .chip, .badge { background: var(--panel2); border: 1px solid var(--border); border-radius: 999px; padding: 4px 12px; font-size: 12.5px; color: var(--accent); display: inline-flex; align-items: center; gap: 4px; }
  .chip b { color: var(--fg); }
  .chip.t-numeric, .badge.t-numeric { border-color: var(--info); color: var(--info); }
  .chip.t-thai_text, .badge.t-thai_text { border-color: #51cf66; color: #51cf66; }
  .chip.t-mixed_text, .badge.t-mixed_text { border-color: #9775fa; color: #9775fa; }
  .chip.t-english_text, .badge.t-english_text { border-color: #4dabf7; color: #4dabf7; }
  .chip.t-categorical, .badge.t-categorical { border-color: #ffa94d; color: #ffa94d; }
  .chip.t-datetime, .badge.t-datetime { border-color: #38d9a9; color: #38d9a9; }
  .chip.t-id, .badge.t-id { border-color: #f783ac; color: #f783ac; }
  .chip.t-phone_number, .badge.t-phone_number { border-color: #ffd43b; color: #ffd43b; }
  .chip.t-empty, .badge.t-empty { border-color: var(--muted); color: var(--muted); }

  .action-list { display: grid; gap: 12px; margin: 14px 0; }
  .action { display: grid; grid-template-columns: auto 1fr; gap: 12px; background: var(--panel); border: 1px solid var(--border); border-left: 4px solid var(--info); border-radius: 12px; padding: 14px 16px; }
  .action.critical { border-left-color: var(--critical); }
  .action.warning { border-left-color: var(--warning); }
  .action .num { width: 28px; height: 28px; display: inline-grid; place-items: center; border-radius: 50%; background: var(--panel2); color: var(--fg); font-weight: 800; }
  .action b { font-size: 16px; }
  .so-what { margin-top: 8px; background: rgba(77,171,247,.08); border: 1px solid rgba(77,171,247,.22); border-radius: 10px; padding: 10px 12px; }
  .so-what .lbl, .rec .lbl, .suggest .lbl { color: var(--accent); font-weight: 750; }

  .issue { border-left: 4px solid var(--border); padding: 14px 18px; margin: 12px 0; box-shadow: none; }
  .issue.critical { border-left-color: var(--critical); }
  .issue.warning { border-left-color: var(--warning); }
  .issue.info { border-left-color: var(--info); }
  .issue .meta, .ng, .note { color: var(--muted); font-size: 13px; }
  .issue .desc-th { margin: 7px 0 2px; }
  .issue .desc-en { color: var(--muted); font-size: 13px; }
  .biz-impact { margin: 9px 0; background: rgba(81,207,102,.08); border: 1px solid rgba(81,207,102,.22); border-radius: 10px; padding: 10px 12px; }
  .biz-impact .lbl { color: var(--ok); font-weight: 800; }
  .technical { margin-top: 8px; color: var(--muted); font-size: 13px; }

  .sev.critical { background: rgba(255,107,107,.18); color: var(--critical); }
  .sev.warning { background: rgba(255,212,59,.18); color: var(--warning); }
  .sev.info { background: rgba(77,171,247,.18); color: var(--info); }
  .sevcount { display: inline-flex; gap: 8px; flex-wrap: wrap; margin-left: 8px; }
  .cat { float: right; }
  .examples { margin: 8px 0 0; }
  .examples .ex { display: block; margin: 3px 0; }

  .insight-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 12px; }
  .watch-list { display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 12px; margin: 14px 0; }
  .watch { padding: 14px 16px; background: var(--panel); border: 1px solid var(--border); border-left: 4px solid var(--warning); border-radius: 12px; }
  .watch.critical { border-left-color: var(--critical); }
  .watch.info { border-left-color: var(--info); }
  .watch ul { margin: 8px 0 0 18px; padding: 0; }

  .imgrow { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 14px; }
  .imgrow.full { grid-template-columns: 1fr; }
  .imgrow img { width: 100%; border-radius: 10px; border: 1px solid var(--border); background: var(--panel2); }
  .imgcap { color: var(--muted); font-size: 13px; margin-bottom: 6px; }
  .chart-note { margin-top: 8px; color: var(--muted); font-size: 13px; border-left: 3px solid var(--accent); padding-left: 10px; }

  details.block { margin: 14px 0; overflow: hidden; box-shadow: none; }
  details.block summary { cursor: pointer; padding: 14px 18px; list-style: none; display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
  details.block summary::-webkit-details-marker { display: none; }
  details.block summary:after { content: "ดูเพิ่มเติม"; margin-left: auto; color: var(--muted); font-size: 13px; }
  details.block[open] summary:after { content: "ซ่อน"; }
  details.block .body { padding: 0 18px 18px; }
  .col { padding: 0; margin: 12px 0; box-shadow: none; }
  .col .head { display: flex; align-items: baseline; gap: 12px; flex-wrap: wrap; }
  .col .head .nm { font-size: 17px; font-weight: 750; }
  .diff .b { color: var(--critical); text-decoration: line-through; text-decoration-color: rgba(255,107,107,.6); }
  .diff .arrow { color: var(--muted); margin: 0 6px; }
  .diff .a { color: #51cf66; }
  .diff .row { display: block; margin: 3px 0; }
  .empty { color: var(--muted); font-style: italic; }
  footer { margin-top: 48px; padding-top: 16px; border-top: 1px solid var(--border); color: var(--muted); font-size: 13px; }

  @media (max-width: 820px) {
    .wrap { padding: 20px 14px 60px; }
    header, .hero-grid, .imgrow { grid-template-columns: 1fr; display: block; }
    .panel { margin-top: 14px; }
    .cards { grid-template-columns: repeat(auto-fit, minmax(125px, 1fr)); }
    table, thead, tbody, th, td, tr { display: block; }
    thead tr { position: absolute; left: -9999px; }
    td { border: none; border-bottom: 1px solid var(--border); padding: 6px 8px; }
    .nav { position: static; }
    .cat { float: none; }
  }
  @media print {
    :root { --bg: #fff; --panel: #fff; --panel2: #f4f4f4; --panel3: #f4f4f4; --fg: #111; --muted: #555; --border: #ccc; }
    body { font-size: 12px; background: #fff; }
    .nav { display: none; }
    .wrap { max-width: none; padding: 0; }
    .issue, .col, .card, .hero, .panel, .action { break-inside: avoid; box-shadow: none; }
    img { max-width: 100%; }
  }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div>
      <h1>{{ L('report_title') }}</h1>
      <div class="sub">{{ L('generated_by') }} ThaiEDA v{{ version }} · {{ overview.rows }} {{ L('rows') }} × {{ overview.columns }} {{ L('columns') }}</div>
    </div>
    <div class="badge">{{ L('how_to_read') }}: {{ L('how_to_read_desc') }}</div>
  </header>

  <nav class="nav">
    {% if data_type %}<a href="#what-is-this">{{ L('what_is_this_data') }}</a>{% endif %}
    {% if key_findings %}<a href="#key-findings">{{ L('most_important') }}</a>{% endif %}
    <a href="#summary">{{ L('executive_summary') }}</a>
    <a href="#actions">{{ L('priority_actions') }}</a>
    <a href="#overview">{{ L('overview') }}</a>
    {% if insight_section %}<a href="#insights">{{ L('auto_insights') }}</a>{% endif %}
    {% if business_section %}<a href="#business-insights">{{ L('business_insights') }}</a>{% endif %}
    <a href="#quality">{{ L('quality_issues') }}</a>
    <a href="#anomalies">{{ L('anomalies') }}</a>
    {% if target_section %}<a href="#target">{{ L('target_analysis') }}</a>{% endif %}
    {% if timeseries_section %}<a href="#timeseries">{{ L('timeseries') }}</a>{% endif %}
    {% if cleaning_diff %}<a href="#cleaning">{{ L('cleaning_diff') }}</a>{% endif %}
    <a href="#charts">{{ L('distributions_correlations') }}</a>
    <a href="#columns">{{ L('column_details') }}</a>
  </nav>

  <!-- ============ WHAT IS THIS DATA ============ -->
  {% if data_type %}
  <section id="what-is-this" class="data-type-banner">
    <div class="dtype-card">
      <div class="dtype-label">{{ L('detected_data_type') }}</div>
      <div class="dtype-value">{{ data_type.label_th }}</div>
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
        <div class="kf-tech muted">{{ kf.description }}</div>
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
        {% if insight_section %}{{ insight_section.executive_summary_th }}{% else %}ชุดข้อมูลนี้ยังไม่พบข้อค้นพบสำคัญจากระบบอัตโนมัติ{% endif %}
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
        <div class="health-row"><span>{{ L('missing_cells') }}</span><b>{{ report_summary.missing_pct }}%</b></div>
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

  <!-- ============ KEY INSIGHTS ============ -->
  {% if insight_section %}
  <h2 id="insights">{{ L('auto_insights') }}
    <span class="sevcount">
      {% if insight_section.critical_count %}<span class="sev critical">{{ insight_section.critical_count }} {{ L('severity_critical') }}</span>{% endif %}
      {% if insight_section.warning_count %}<span class="sev warning">{{ insight_section.warning_count }} {{ L('severity_warning') }}</span>{% endif %}
      {% if insight_section.info_count %}<span class="sev info">{{ insight_section.info_count }} {{ L('severity_info') }}</span>{% endif %}
    </span>
  </h2>
  <div class="insight-grid">
    {% for ins in insight_section.insights %}
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
  {% endif %}

  <!-- ============ CROSS-COLUMN INSIGHTS ============ -->
  {% if business_section %}
  <h2 id="business-insights">{{ L('business_insights') }} <span class="ng">({{ business_section.total }})</span></h2>
  {% for c in business_section.cards %}
  <div class="issue insight {{ c.severity if c.severity else 'info' }}">
    <div>
      <span class="badge t-numeric">{{ c.pattern_label }}</span>
      <b>{{ c.title_th }}</b>
      <span class="cat ng">{{ L('breakdown') }}: <code>{{ c.perspective.breakdown }}</code>{% if c.perspective.measure %} · {{ L('measure') }}: <code>{{ c.perspective.measure }}</code>{% endif %} · {{ c.perspective.agg }}</span>
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

  <!-- ============ QUALITY ISSUES ============ -->
  <h2 id="quality">{{ L('quality_issues') }} <span class="ng">({{ quality_issues|length }})</span></h2>
  {% if quality_issues %}
    {% for iss in quality_issues %}
    <div class="issue {{ iss.severity }}">
      <div><span class="sev {{ iss.severity }}">{{ sev_icons[iss.severity] }} {{ L('severity_' ~ iss.severity) }}</span><b>{{ iss.column }}</b> · <span class="ng">{{ iss.check_name }}</span></div>
      <div class="meta">{{ L('count') }}: {{ "{:,}".format(iss.count) }} ({{ iss.percentage }}%)</div>
      <div class="desc-th">{{ iss.description_th }}</div>
      {% if iss.examples %}<div class="examples">{% for ex in iss.examples[:5] %}<span class="ex mono">{{ ex }}</span>{% endfor %}</div>{% endif %}
      <div class="so-what"><span class="lbl">{{ L('so_what') }}</span> {{ iss.suggestion_th }}</div>
      <details><summary class="ng">{{ L('show_details') }}</summary><div class="desc-en">{{ iss.description }} — {{ iss.suggestion }}</div></details>
    </div>
    {% endfor %}
  {% else %}
    <p class="empty">✓ {{ L('no_issues') }}</p>
  {% endif %}

  <!-- ============ ANOMALIES ============ -->
  <h2 id="anomalies">{{ L('anomalies') }} <span class="ng">({{ anomalies|length }})</span></h2>
  {% if anomalies %}
    {% for an in anomalies %}
    <div class="issue {{ an.severity }}">
      <div><span class="sev {{ an.severity }}">{{ sev_icons[an.severity] }} {{ L('severity_' ~ an.severity) }}</span><b>{{ an.column }}</b> · <span class="ng">{{ an.check_name }}</span> <span class="badge">{{ an.type_label }}</span></div>
      <div class="meta">{{ L('count') }}: {{ "{:,}".format(an.count) }} ({{ an.percentage }}%)</div>
      <div class="desc-th">{{ an.description_th }}</div>
      {% if an.examples %}<div class="examples">{% for ex in an.examples[:5] %}<span class="ex mono">{{ ex }}</span>{% endfor %}</div>{% endif %}
      <div class="so-what"><span class="lbl">{{ L('so_what') }}</span> {{ an.suggestion_th }}</div>
      <details><summary class="ng">{{ L('show_details') }}</summary><div class="desc-en">{{ an.description }} — {{ an.suggestion }}</div></details>
    </div>
    {% endfor %}
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
  {% if cleaning_diff_summary %}<div class="panel">{{ L('total_cells_changed') }}: <b>{{ "{:,}".format(cleaning_diff_summary.total_cells_changed) }}</b> · {{ L('most_impactful') }}: <b>{{ cleaning_diff_summary.most_impactful_op }}</b> <span class="ng">({{ cleaning_diff_summary.most_impactful_th }} — {{ "{:,}".format(cleaning_diff_summary.most_impactful_rows) }})</span></div>{% endif %}
  <table><tr><th>{{ L('column') }}</th><th>{{ L('operation') }}</th><th>{{ L('rows_affected') }}</th><th>{{ L('before') }} → {{ L('after') }}</th></tr>{% for c in cleaning_diff %}<tr><td><b>{{ c.column }}</b></td><td>{{ c.operation }}<div class="ng">{{ c.description_th }}</div>{% if c.explanation %}<div class="ng">{{ L('explanation') }}: <span class="mono">{{ c.explanation }}</span></div>{% endif %}</td><td>{{ "{:,}".format(c.rows_affected) }}</td><td class="diff">{% for ex in c.before_examples %}<span class="row"><span class="b mono">{{ ex }}</span><span class="arrow">→</span><span class="a mono">{{ c.after_examples[loop.index0] }}</span></span>{% endfor %}</td></tr>{% endfor %}</table>
  {% endif %}

  <!-- ============ CLEANING SUGGESTIONS ============ -->
  <h2 id="cleaning-suggestions">{{ L('cleaning_suggestions') }} <span class="ng">({{ cleaning_suggestions|length }})</span></h2>
  {% if cleaning_suggestions %}
    <table><tr><th>{{ L('column') }}</th><th>{{ L('operation') }}</th><th>{{ L('rows_affected') }}</th><th>{{ L('before') }} → {{ L('after') }}</th></tr>{% for c in cleaning_suggestions %}<tr><td><b>{{ c.column }}</b></td><td>{{ c.operation }}<div class="ng">{{ c.description_th }}</div>{% if c.explanation %}<div class="ng">{{ L('explanation') }}: <span class="mono">{{ c.explanation }}</span></div>{% endif %}</td><td>{{ "{:,}".format(c.rows_affected) }}</td><td class="diff">{% for ex in c.before_examples %}<span class="row"><span class="b mono">{{ ex }}</span><span class="arrow">→</span><span class="a mono">{{ c.after_examples[loop.index0] }}</span></span>{% endfor %}</td></tr>{% endfor %}</table>
  {% else %}<p class="empty">✓ {{ L('no_cleaning') }}</p>{% endif %}

  <!-- ============ CHARTS ============ -->
  <h2 id="charts">{{ L('distributions_correlations') }}</h2>
  {% if has_dist_charts %}
    {% if dist_charts.correlation_heatmap %}<div class="imgrow full"><div><div class="imgcap">{{ L('correlation_heatmap') }}</div><img src="data:image/png;base64,{{ dist_charts.correlation_heatmap }}" alt="correlation heatmap"><div class="chart-note"><b>{{ L('chart_insight') }}:</b> ดูว่าคอลัมน์ตัวเลขใดเคลื่อนไปด้วยกันสูงมาก อาจใช้แทนกันได้หรือส่งผลต่อโมเดล</div></div></div>{% endif %}
    {% if dist_charts.scatter_matrix %}<div class="imgrow full"><div><div class="imgcap">{{ L('scatter_matrix') }}</div><img src="data:image/png;base64,{{ dist_charts.scatter_matrix }}" alt="scatter matrix"><div class="chart-note"><b>{{ L('chart_insight') }}:</b> ใช้ดู pattern ระหว่างตัวเลข เช่น เส้นตรง กลุ่มย่อย หรือ outlier</div></div></div>{% endif %}
    <div class="imgrow">{% if dist_charts.boxplot %}<div><div class="imgcap">{{ L('boxplot') }}</div><img src="data:image/png;base64,{{ dist_charts.boxplot }}" alt="box plot"><div class="chart-note"><b>{{ L('chart_insight') }}:</b> จุดที่หลุดจากกล่องคือค่าที่ควรตรวจว่าเป็น outlier จริงหรือไม่</div></div>{% endif %}{% if dist_charts.violinplot %}<div><div class="imgcap">{{ L('violinplot') }}</div><img src="data:image/png;base64,{{ dist_charts.violinplot }}" alt="violin plot"><div class="chart-note"><b>{{ L('chart_insight') }}:</b> ดูรูปทรงการกระจายว่าเบ้ มีหลายกลุ่ม หรือกระจุกตัวตรงไหน</div></div>{% endif %}</div>
  {% else %}<p class="empty">ไม่มีกราฟการกระจายที่เหมาะสมสำหรับข้อมูลชุดนี้</p>{% endif %}

  <h2>{{ L('missing_data') }}</h2>
  {% if has_missing_charts %}
  <div class="imgrow">{% if missing_charts.missing_matrix %}<div><div class="imgcap">{{ L('missing_matrix') }}</div><img src="data:image/png;base64,{{ missing_charts.missing_matrix }}" alt="missing value matrix"><div class="chart-note"><b>{{ L('chart_insight') }}:</b> แถบว่างช่วยบอกว่าค่าว่างกระจุกตัวช่วงใดหรือคอลัมน์ใด</div></div>{% endif %}{% if missing_charts.missing_heatmap %}<div><div class="imgcap">{{ L('missing_heatmap') }}</div><img src="data:image/png;base64,{{ missing_charts.missing_heatmap }}" alt="missing nullity correlation heatmap"><div class="chart-note"><b>{{ L('chart_insight') }}:</b> ถ้าค่าว่างเกิดพร้อมกันหลายคอลัมน์ อาจเกิดจากขั้นตอนเก็บข้อมูลเดียวกัน</div></div>{% endif %}</div>
  {% else %}<p class="empty">✓ {{ L('no_missing') }}</p>{% endif %}

  <!-- ============ COLUMN DETAILS ============ -->
  <h2 id="columns">{{ L('column_details') }}</h2>
  <p class="muted">ส่วนนี้ซ่อนไว้เป็นรายคอลัมน์เพื่อลดความรก เปิดเฉพาะคอลัมน์ที่ต้องการตรวจละเอียด</p>
  {% for col in columns %}
  <details class="block col">
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
  <div class="note">{{ L('more_columns_note') }}</div>

  <h2 id="recommended-actions">{{ L('recommended_actions') }}</h2>
  {% if priority_actions %}
  <ol>
    {% for a in priority_actions %}<li><b>{{ a.title }}</b> — {{ a.action }}</li>{% endfor %}
  </ol>
  {% else %}<p class="empty">✓ {{ L('no_priority_actions') }}</p>{% endif %}

  <footer>{{ L('generated_by') }} <b>ThaiEDA</b> v{{ version }} — AutoEDA สำหรับข้อมูลภาษาไทย</footer>
</div>
</body>
</html>"""
