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
    --bg: #15171c; --panel: #1d2027; --panel2: #23272f; --fg: #e6e6e6;
    --muted: #9aa0aa; --border: #2e333c; --accent: #4dabf7;
    --critical: #ff6b6b; --warning: #ffd43b; --info: #4dabf7;
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
       border-bottom: 1px solid var(--border); }
  h3 { font-size: 17px; margin: 24px 0 10px; }
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
  .issue { background: var(--panel); border: 1px solid var(--border);
           border-left: 4px solid var(--border); border-radius: 8px;
           padding: 14px 18px; margin: 12px 0; }
  .issue.critical { border-left-color: var(--critical); }
  .issue.warning { border-left-color: var(--warning); }
  .issue.info { border-left-color: var(--info); }
  .sev { display: inline-block; font-size: 11px; font-weight: 700; text-transform: uppercase;
         letter-spacing: .5px; padding: 2px 8px; border-radius: 4px; margin-right: 8px; }
  .sev.critical { background: rgba(255,107,107,.18); color: var(--critical); }
  .sev.warning { background: rgba(255,212,59,.18); color: var(--warning); }
  .sev.info { background: rgba(77,171,247,.18); color: var(--info); }
  .issue .meta { color: var(--muted); font-size: 13px; margin: 4px 0; }
  .issue .desc-th { margin: 6px 0 2px; }
  .issue .desc-en { color: var(--muted); font-size: 13px; }
  .issue .suggest { margin-top: 8px; font-size: 13px; }
  .issue .suggest .lbl { color: var(--accent); font-weight: 600; }
  code, .mono { font-family: "Cascadia Code", Consolas, monospace; font-size: 12.5px;
                background: var(--panel2); padding: 1px 6px; border-radius: 4px;
                word-break: break-all; }
  .examples { margin: 8px 0 0; }
  .examples .ex { display: block; margin: 3px 0; }
  .col { background: var(--panel); border: 1px solid var(--border);
         border-radius: 10px; padding: 18px 20px; margin: 16px 0; }
  .col .head { display: flex; align-items: baseline; gap: 12px; flex-wrap: wrap; }
  .col .head .nm { font-size: 18px; font-weight: 600; }
  .badge { font-size: 12px; padding: 2px 10px; border-radius: 14px;
           background: var(--panel2); border: 1px solid var(--border); color: var(--accent); }
  .imgrow { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 14px; }
  .imgrow.full { grid-template-columns: 1fr; }
  .imgrow img { width: 100%; border-radius: 8px; border: 1px solid var(--border); }
  .imgcap { color: var(--muted); font-size: 13px; margin-bottom: 6px; }
  .ng { color: var(--muted); font-size: 13px; }
  .empty { color: var(--muted); font-style: italic; }
  footer { margin-top: 48px; padding-top: 16px; border-top: 1px solid var(--border);
           color: var(--muted); font-size: 13px; }
  .note { color: var(--muted); font-size: 13px; font-style: italic; margin-top: 8px; }
  .exec { background: var(--panel2); border: 1px solid var(--border);
          border-left: 4px solid var(--accent); border-radius: 10px;
          padding: 16px 20px; margin: 14px 0 18px; font-size: 15px; }
  .exec .lbl { color: var(--accent); font-weight: 700; font-size: 13px;
               text-transform: uppercase; letter-spacing: .5px; display: block; margin-bottom: 6px; }
  .insight .rec { margin-top: 8px; font-size: 13px; }
  .insight .rec .lbl { color: var(--accent); font-weight: 600; }
  .insight .cat { float: right; }
  /* --- v0.6: cross-column insight evidence mini-table --- */
  .evtable { width: auto; margin: 8px 0 4px; }
  .evtable th, .evtable td { padding: 4px 14px 4px 0; border-bottom: none; font-size: 13px; }
  .evtable th { color: var(--muted); }
  .sevcount { display: inline-flex; gap: 8px; margin-left: 8px; }
  /* --- v0.4.1: sticky section navigation --- */
  .nav { position: sticky; top: 0; z-index: 10; display: flex; flex-wrap: wrap; gap: 6px;
         background: rgba(21,23,28,.92); backdrop-filter: blur(6px);
         border: 1px solid var(--border); border-radius: 10px;
         padding: 8px 10px; margin: 18px 0 8px; }
  .nav a { color: var(--muted); text-decoration: none; font-size: 13px; font-weight: 600;
           padding: 4px 10px; border-radius: 16px; white-space: nowrap; }
  .nav a:hover { color: var(--fg); background: var(--panel2); }
  h2 { scroll-margin-top: 60px; }
  /* --- v0.4.1: bigger executive summary --- */
  .exec.hero { font-size: 16px; line-height: 1.7; border-left-width: 5px;
               background: linear-gradient(180deg, var(--panel2), var(--panel)); }
  /* --- v0.4.1: severity emoji icon on cards --- */
  .ico { margin-right: 6px; font-size: 14px; }
  /* --- v0.4.1: type-colored badges/chips --- */
  .chip.t-numeric, .badge.t-numeric { border-color: var(--info); color: var(--info); }
  .chip.t-thai_text, .badge.t-thai_text { border-color: #51cf66; color: #51cf66; }
  .chip.t-mixed_text, .badge.t-mixed_text { border-color: #9775fa; color: #9775fa; }
  .chip.t-english_text, .badge.t-english_text { border-color: #4dabf7; color: #4dabf7; }
  .chip.t-categorical, .badge.t-categorical { border-color: #ffa94d; color: #ffa94d; }
  .chip.t-datetime, .badge.t-datetime { border-color: #38d9a9; color: #38d9a9; }
  .chip.t-id, .badge.t-id { border-color: #f783ac; color: #f783ac; }
  .chip.t-phone_number, .badge.t-phone_number { border-color: #ffd43b; color: #ffd43b; }
  .chip.t-empty, .badge.t-empty { border-color: var(--muted); color: var(--muted); }
  /* --- v0.4.1: timeseries highlight banner --- */
  .banner { background: rgba(56,217,169,.10); border: 1px solid #2b7a64;
            border-left: 4px solid #38d9a9; border-radius: 10px;
            padding: 12px 16px; margin: 14px 0; font-size: 14px; }
  .banner b { color: #38d9a9; }
  /* --- v0.4.1: cleaning diff (strikethrough red -> green) --- */
  .diff .b { color: var(--critical); text-decoration: line-through;
             text-decoration-color: rgba(255,107,107,.6); }
  .diff .arrow { color: var(--muted); margin: 0 6px; }
  .diff .a { color: #51cf66; }
  .diff .row { display: block; margin: 3px 0; }
  /* --- v0.4.1: responsive (stack on mobile) --- */
  @media (max-width: 720px) {
    .wrap { padding: 20px 14px 60px; }
    .imgrow { grid-template-columns: 1fr; }
    .cards { grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); }
    table, thead, tbody, th, td, tr { display: block; }
    thead tr { position: absolute; left: -9999px; }
    td { border: none; border-bottom: 1px solid var(--border); padding: 6px 8px; }
    .nav { position: static; }
    .insight .cat { float: none; display: inline-block; }
  }
  /* --- v0.4.1: print-friendly --- */
  @media print {
    :root { --bg: #fff; --panel: #fff; --panel2: #f4f4f4; --fg: #111;
            --muted: #555; --border: #ccc; }
    body { font-size: 12px; }
    .nav { display: none; }
    .wrap { max-width: none; padding: 0; }
    .issue, .col, .card, .exec, .banner { break-inside: avoid; }
    img { max-width: 100%; }
    a { color: inherit; text-decoration: none; }
  }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>{{ L('report_title') }}</h1>
    <div class="sub">{{ L('generated_by') }} ThaiEDA v{{ version }} · {{ overview.rows }} {{ L('rows') }} × {{ overview.columns }} {{ L('columns') }}</div>
  </header>

  <!-- ============ SECTION NAV (sticky) ============ -->
  <nav class="nav">
    <a href="#overview">{{ L('overview') }}</a>
    {% if insight_section %}<a href="#insights">{{ L('auto_insights') }}</a>{% endif %}
    {% if business_section %}<a href="#business-insights">{{ L('business_insights') }}</a>{% endif %}
    <a href="#quality">{{ L('quality_issues') }}</a>
    <a href="#anomalies">{{ L('anomalies') }}</a>
    {% if target_section %}<a href="#target">{{ L('target_analysis') }}</a>{% endif %}
    {% if timeseries_section %}<a href="#timeseries">{{ L('timeseries') }}</a>{% endif %}
    {% if cleaning_diff %}<a href="#cleaning">{{ L('cleaning_diff') }}</a>{% endif %}
    <a href="#columns">{{ L('column_details') }}</a>
  </nav>

  <!-- ============ OVERVIEW ============ -->
  <h2 id="overview">{{ L('overview') }}</h2>
  <div class="cards">
    <div class="card"><div class="k">{{ L('rows') }}</div><div class="v">{{ "{:,}".format(overview.rows) }}</div></div>
    <div class="card"><div class="k">{{ L('columns') }}</div><div class="v">{{ overview.columns }}</div></div>
    <div class="card"><div class="k">{{ L('total_cells') }}</div><div class="v">{{ "{:,}".format(overview.total_cells) }}</div></div>
    <div class="card"><div class="k">{{ L('missing_cells') }}</div><div class="v">{{ "{:,}".format(overview.missing_cells) }} <span style="font-size:14px;color:var(--muted)">({{ overview.missing_pct }}%)</span></div></div>
    <div class="card"><div class="k">{{ L('duplicate_rows') }}</div><div class="v">{{ "{:,}".format(overview.duplicate_rows) }}</div></div>
  </div>

  <h3>{{ L('column_types') }}</h3>
  <div class="typedist">
    {% for tk, tlabel, cnt in type_distribution %}
      <span class="chip t-{{ tk }}">{{ tlabel }} <b>{{ cnt }}</b></span>
    {% endfor %}
  </div>
  {% if notes %}
    {% for n in notes %}<div class="note">⚠ {{ n }}</div>{% endfor %}
  {% endif %}

  <!-- ============ KEY INSIGHTS (AUTO) ============ -->
  {% if insight_section %}
  <h2 id="insights">{{ L('auto_insights') }}
    <span class="sevcount">
      {% if insight_section.critical_count %}<span class="sev critical">{{ insight_section.critical_count }} {{ L('severity_critical') }}</span>{% endif %}
      {% if insight_section.warning_count %}<span class="sev warning">{{ insight_section.warning_count }} {{ L('severity_warning') }}</span>{% endif %}
      {% if insight_section.info_count %}<span class="sev info">{{ insight_section.info_count }} {{ L('severity_info') }}</span>{% endif %}
    </span>
  </h2>
  <div class="exec hero">
    <span class="lbl">{{ L('executive_summary') }}</span>
    {{ insight_section.executive_summary_th }}
  </div>
  {% for ins in insight_section.insights %}
  <div class="issue insight {{ ins.severity }}">
    <div>
      <span class="sev {{ ins.severity }}"><span class="ico">{{ sev_icons[ins.severity] }}</span>{{ L('severity_' ~ ins.severity) }}</span>
      <b>{{ ins.title_th }}</b>
      <span class="badge cat">{{ ins.category_label }}</span>
    </div>
    <div class="desc-th">{{ ins.description_th }}</div>
    <div class="rec"><span class="lbl">{{ L('recommendation') }}:</span> {{ ins.recommendation_th }}</div>
  </div>
  {% endfor %}
  {% endif %}

  <!-- ============ CROSS-COLUMN INSIGHTS (ENGINE, v0.6) ============ -->
  {% if business_section %}
  <h2 id="business-insights">{{ L('business_insights') }} <span class="ng">({{ business_section.total }})</span></h2>
  {% for c in business_section.cards %}
  <div class="issue insight info">
    <div>
      <span class="badge t-numeric">{{ c.pattern_label }}</span>
      <b>{{ c.title_th }}</b>
      <span class="cat ng">{{ L('breakdown') }}: <code>{{ c.perspective.breakdown }}</code>{% if c.perspective.measure %} · {{ L('measure') }}: <code>{{ c.perspective.measure }}</code>{% endif %} · {{ c.perspective.agg }}</span>
    </div>
    <div class="desc-th">{{ c.description_th }}</div>
    {% if c.chart %}
    <div class="imgrow full" style="margin-top: 10px;">
      <img src="data:image/png;base64,{{ c.chart }}" alt="insight chart">
    </div>
    {% endif %}
    {% if c.evidence.top_segments and c.pattern != 'trend' %}
    <table class="evtable">
      <tr><th>{{ L('segment') }}</th><th>{{ L('value') }}</th></tr>
      {% for seg, val in c.evidence.top_segments %}
      <tr><td>{{ seg }}</td><td class="ng">{{ "{:,}".format(val) }}</td></tr>
      {% endfor %}
    </table>
    {% endif %}
    {% if c.pattern == 'comparison' %}
    <div class="meta">{{ L('lift') }}: <b>{{ c.evidence.lift_pct }}%</b> · {{ L('mean') }}: {{ "{:,}".format(c.evidence.top_mean) }} vs {{ "{:,}".format(c.evidence.rest_mean) }}{% if c.evidence.p_value is not none %} · {{ L('p_value') }}={{ c.evidence.p_value }}{% endif %} · n={{ "{:,}".format(c.evidence.n_top) }}/{{ "{:,}".format(c.evidence.n_rest) }}</div>
    {% endif %}
    {% if c.pattern == 'attribution' %}
    <div class="meta">{{ L('share') }}: <b>{{ c.evidence.share }}%</b></div>
    {% endif %}
    {% if c.pattern == 'trend' %}
    <div class="meta">τ={{ c.evidence.tau }}{% if c.evidence.p_value is not none %} · {{ L('p_value') }}={{ c.evidence.p_value }}{% endif %} · {{ c.evidence.first_bucket }} → {{ c.evidence.last_bucket }} ({{ "{:,}".format(c.evidence.first_value) }} → {{ "{:,}".format(c.evidence.last_value) }})</div>
    {% endif %}
    <div class="rec"><span class="lbl">{{ L('recommendation') }}:</span> {{ c.recommendation_th }}</div>
  </div>
  {% endfor %}
  {% endif %}

  <!-- ============ QUALITY ISSUES ============ -->
  <h2 id="quality">{{ L('quality_issues') }} <span class="ng">({{ quality_issues|length }})</span></h2>
  {% if quality_issues %}
    {% for iss in quality_issues %}
    <div class="issue {{ iss.severity }}">
      <div>
        <span class="sev {{ iss.severity }}"><span class="ico">{{ sev_icons[iss.severity] }}</span>{{ L('severity_' ~ iss.severity) }}</span>
        <b>{{ iss.column }}</b> · <span class="ng">{{ iss.check_name }}</span>
      </div>
      <div class="meta">{{ L('count') }}: {{ "{:,}".format(iss.count) }} ({{ iss.percentage }}%)</div>
      <div class="desc-th">{{ iss.description_th }}</div>
      <div class="desc-en">{{ iss.description }}</div>
      {% if iss.examples %}
      <div class="examples">
        {% for ex in iss.examples %}<span class="ex mono">{{ ex }}</span>{% endfor %}
      </div>
      {% endif %}
      <div class="suggest"><span class="lbl">{{ L('suggestion') }}:</span> {{ iss.suggestion_th }} <span class="ng">— {{ iss.suggestion }}</span></div>
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
      <div>
        <span class="sev {{ an.severity }}"><span class="ico">{{ sev_icons[an.severity] }}</span>{{ L('severity_' ~ an.severity) }}</span>
        <b>{{ an.column }}</b> · <span class="ng">{{ an.check_name }}</span>
        <span class="badge">{{ an.type_label }}</span>
      </div>
      <div class="meta">{{ L('count') }}: {{ "{:,}".format(an.count) }} ({{ an.percentage }}%)</div>
      <div class="desc-th">{{ an.description_th }}</div>
      <div class="desc-en">{{ an.description }}</div>
      {% if an.examples %}
      <div class="examples">
        {% for ex in an.examples %}<span class="ex mono">{{ ex }}</span>{% endfor %}
      </div>
      {% endif %}
      <div class="suggest"><span class="lbl">{{ L('suggestion') }}:</span> {{ an.suggestion_th }} <span class="ng">— {{ an.suggestion }}</span></div>
    </div>
    {% endfor %}
  {% else %}
    <p class="empty">✓ {{ L('no_anomalies') }}</p>
  {% endif %}

  <!-- ============ TARGET ANALYSIS ============ -->
  {% if target_section %}
  <h2 id="target">{{ L('target_analysis') }} <span class="ng">({{ L('target_column') }}: {{ target_section.target_column }})</span></h2>
  {% if target_section.associations %}
    <table>
      <tr>
        <th>{{ L('column') }}</th>
        <th>{{ L('association') }}</th>
        <th>{{ L('score') }}</th>
        <th>{{ L('p_value') }}</th>
        <th>{{ L('suggestion') }}</th>
      </tr>
      {% for a in target_section.associations %}
      <tr>
        <td><b>{{ a.column }}</b></td>
        <td><span class="badge">{{ a.type_label }}</span></td>
        <td>{{ a.score if a.score is not none else '—' }}</td>
        <td>{{ a.p_value if a.p_value is not none else '—' }}</td>
        <td class="ng">{{ a.description_th }}</td>
      </tr>
      {% endfor %}
    </table>
  {% else %}
    <p class="empty">{{ L('no_target') }}</p>
  {% endif %}
  {% endif %}

  <!-- ============ NAMED ENTITIES (NER) ============ -->
  {% if ner_sections %}
  <h2>{{ L('named_entities') }}</h2>
  {% for sec in ner_sections %}
    <div class="col">
      <div class="head">
        <span class="nm">{{ sec.column }}</span>
        <span class="badge">{{ L('total_entities') }}: {{ "{:,}".format(sec.result.total_entities) }}</span>
        <span class="badge">{{ L('ner_engine') }}: {{ sec.result.engine_used }}</span>
      </div>
      <table>
        <tr><th>{{ L('entity_type') }}</th><th>{{ L('count') }}</th><th>{{ L('top_entities') }}</th></tr>
        {% for etype, cnt in sec.result.entity_counts.items() %}
        <tr>
          <td><b>{{ etype }}</b></td>
          <td>{{ "{:,}".format(cnt) }}</td>
          <td>{% for ent, c in sec.result.top_entities.get(etype, [])[:10] %}<span class="mono">{{ ent }} ({{ c }})</span> {% endfor %}</td>
        </tr>
        {% endfor %}
      </table>
    </div>
  {% endfor %}
  {% endif %}

  <!-- ============ TIMESERIES ANALYSIS ============ -->
  {% if timeseries_section %}
  <h2 id="timeseries">{{ L('timeseries') }} <span class="ng">({{ L('type_datetime') }}: {{ timeseries_section.time_column }})</span></h2>
  {% if timeseries_section.trend_count or timeseries_section.seasonal_count %}
  <div class="banner">
    📈 {{ L('ts_trend') }}/{{ L('ts_seasonality') }}:
    {% if timeseries_section.trend_count %}<b>{{ timeseries_section.trend_count }}</b> {{ L('column') }} {{ L('ts_trend') }}{% endif %}
    {% if timeseries_section.trend_count and timeseries_section.seasonal_count %} · {% endif %}
    {% if timeseries_section.seasonal_count %}<b>{{ timeseries_section.seasonal_count }}</b> {{ L('column') }} {{ L('ts_seasonality') }}{% endif %}
  </div>
  {% endif %}
  {% for ts in timeseries_section.columns %}
  <div class="col">
    <div class="head">
      <span class="nm">{{ ts.column }}</span>
      <span class="badge">{{ L('ts_frequency') }}: {{ ts.result.frequency_th }}</span>
      {% if ts.result.has_trend %}<span class="badge">{{ L('ts_trend') }}: {{ ts.result.trend_direction_th }}</span>{% endif %}
      {% if ts.result.has_seasonality %}<span class="badge">{{ L('ts_seasonality') }}: {{ ts.result.seasonal_period_th }}</span>{% endif %}
      <span class="badge">{{ L('ts_engine') }}: {{ ts.result.engine_used }}</span>
    </div>
    <table>
      <tr>
        <th>{{ L('ts_trend') }}</th><td>{{ ts.result.trend_direction_th if ts.result.has_trend else L('ts_none') }}</td>
        <th>{{ L('ts_seasonality') }}</th><td>{{ ts.result.seasonal_period_th if ts.result.has_seasonality else L('ts_none') }}</td>
      </tr>
      <tr>
        <th>{{ L('ts_gaps') }}</th><td>{{ ts.result.gap_count }}</td>
        <th>{{ L('ts_anomalies') }}</th><td>{{ ts.result.anomaly_count }}</td>
      </tr>
      <tr>
        <th>{{ L('ts_autocorr') }}</th><td>{{ ts.result.stats.autocorr_lag1 }}</td>
        <th>{{ L('mean') }}</th><td>{{ ts.result.stats.mean }}</td>
      </tr>
    </table>
    {% if ts.result.insights %}
    <div class="examples">
      {% for ins in ts.result.insights %}<span class="ex">• {{ ins }}</span>{% endfor %}
    </div>
    {% endif %}
    {% if ts.charts.line %}
    <div class="imgrow full">
      <div><div class="imgcap">{{ L('ts_timeseries_plot') }}</div><img src="data:image/png;base64,{{ ts.charts.line }}" alt="timeseries line"></div>
    </div>
    {% endif %}
    {% if ts.charts.decomposition %}
    <div class="imgrow full">
      <div><div class="imgcap">{{ L('ts_decomposition') }}</div><img src="data:image/png;base64,{{ ts.charts.decomposition }}" alt="STL decomposition"></div>
    </div>
    {% endif %}
    {% if ts.charts.acf %}
    <div class="imgrow full">
      <div><div class="imgcap">{{ L('ts_acf_plot') }}</div><img src="data:image/png;base64,{{ ts.charts.acf }}" alt="ACF"></div>
    </div>
    {% endif %}
  </div>
  {% endfor %}
  {% endif %}

  <!-- ============ CLEANING APPLIED (DIFF) ============ -->
  {% if cleaning_diff %}
  <h2 id="cleaning">{{ L('cleaning_diff') }} <span class="ng">({{ cleaning_diff|length }})</span></h2>
  {% if cleaning_diff_summary %}
  <div class="exec">
    {{ L('total_cells_changed') }}: <b>{{ "{:,}".format(cleaning_diff_summary.total_cells_changed) }}</b>
    · {{ L('most_impactful') }}: <b>{{ cleaning_diff_summary.most_impactful_op }}</b>
    <span class="ng">({{ cleaning_diff_summary.most_impactful_th }} — {{ "{:,}".format(cleaning_diff_summary.most_impactful_rows) }})</span>
  </div>
  {% endif %}
  <table>
    <tr>
      <th>{{ L('column') }}</th>
      <th>{{ L('operation') }}</th>
      <th>{{ L('rows_affected') }}</th>
      <th>{{ L('before') }} → {{ L('after') }}</th>
    </tr>
    {% for c in cleaning_diff %}
    <tr>
      <td><b>{{ c.column }}</b></td>
      <td>{{ c.operation }}<div class="ng">{{ c.description_th }}</div>{% if c.explanation %}<div class="ng">{{ L('explanation') }}: <span class="mono">{{ c.explanation }}</span></div>{% endif %}</td>
      <td>{{ "{:,}".format(c.rows_affected) }}</td>
      <td class="diff">{% for ex in c.before_examples %}<span class="row"><span class="b mono">{{ ex }}</span><span class="arrow">→</span><span class="a mono">{{ c.after_examples[loop.index0] }}</span></span>{% endfor %}</td>
    </tr>
    {% endfor %}
  </table>
  {% endif %}

  <!-- ============ CLEANING SUGGESTIONS ============ -->
  <h2 id="cleaning-suggestions">{{ L('cleaning_suggestions') }} <span class="ng">({{ cleaning_suggestions|length }})</span></h2>
  {% if cleaning_suggestions %}
    <table>
      <tr>
        <th>{{ L('column') }}</th>
        <th>{{ L('operation') }}</th>
        <th>{{ L('rows_affected') }}</th>
        <th>{{ L('before') }} → {{ L('after') }}</th>
      </tr>
      {% for c in cleaning_suggestions %}
      <tr>
        <td><b>{{ c.column }}</b></td>
        <td>{{ c.operation }}<div class="ng">{{ c.description_th }}</div>{% if c.explanation %}<div class="ng">{{ L('explanation') }}: <span class="mono">{{ c.explanation }}</span></div>{% endif %}</td>
        <td>{{ "{:,}".format(c.rows_affected) }}</td>
        <td class="diff">{% for ex in c.before_examples %}<span class="row"><span class="b mono">{{ ex }}</span><span class="arrow">→</span><span class="a mono">{{ c.after_examples[loop.index0] }}</span></span>{% endfor %}</td>
      </tr>
      {% endfor %}
    </table>
  {% else %}
    <p class="empty">✓ {{ L('no_cleaning') }}</p>
  {% endif %}

  <!-- ============ DISTRIBUTIONS & CORRELATIONS ============ -->
  {% if has_dist_charts %}
  <h2>{{ L('distributions_correlations') }}</h2>
  {% if dist_charts.correlation_heatmap %}
  <div class="imgrow full">
    <div><div class="imgcap">{{ L('correlation_heatmap') }}</div><img src="data:image/png;base64,{{ dist_charts.correlation_heatmap }}" alt="correlation heatmap"></div>
  </div>
  {% endif %}
  {% if dist_charts.scatter_matrix %}
  <div class="imgrow full">
    <div><div class="imgcap">{{ L('scatter_matrix') }}</div><img src="data:image/png;base64,{{ dist_charts.scatter_matrix }}" alt="scatter matrix"></div>
  </div>
  {% endif %}
  <div class="imgrow">
    {% if dist_charts.boxplot %}
    <div><div class="imgcap">{{ L('boxplot') }}</div><img src="data:image/png;base64,{{ dist_charts.boxplot }}" alt="box plot"></div>
    {% endif %}
    {% if dist_charts.violinplot %}
    <div><div class="imgcap">{{ L('violinplot') }}</div><img src="data:image/png;base64,{{ dist_charts.violinplot }}" alt="violin plot"></div>
    {% endif %}
  </div>
  {% endif %}

  <!-- ============ MISSING DATA ============ -->
  <h2>{{ L('missing_data') }}</h2>
  {% if has_missing_charts %}
  <div class="imgrow">
    {% if missing_charts.missing_matrix %}
    <div><div class="imgcap">{{ L('missing_matrix') }}</div><img src="data:image/png;base64,{{ missing_charts.missing_matrix }}" alt="missing value matrix"></div>
    {% endif %}
    {% if missing_charts.missing_heatmap %}
    <div><div class="imgcap">{{ L('missing_heatmap') }}</div><img src="data:image/png;base64,{{ missing_charts.missing_heatmap }}" alt="missing nullity correlation heatmap"></div>
    {% endif %}
  </div>
  {% else %}
  <p class="empty">✓ {{ L('no_missing') }}</p>
  {% endif %}

  <!-- ============ COLUMN DETAILS ============ -->
  <h2 id="columns">{{ L('column_details') }}</h2>
  {% for col in columns %}
  <div class="col">
    <div class="head">
      <span class="nm">{{ col.name }}</span>
      <span class="badge t-{{ col.type_key }}">{{ col.type_label }}</span>
    </div>

    {% if col.is_text and col.metrics %}
      <table>
        <tr><th>{{ L('engine_used') }}</th><td>{{ col.metrics.engine_used }}</td>
            <th>{{ L('avg_char_length') }}</th><td>{{ col.metrics.avg_char_length }}</td></tr>
        <tr><th>{{ L('avg_token_length') }}</th><td>{{ col.metrics.avg_token_length }}</td>
            <th>{{ L('median_char_length') }}</th><td>{{ col.metrics.median_char_length }}</td></tr>
        <tr><th>{{ L('total_tokens') }}</th><td>{{ "{:,}".format(col.metrics.total_tokens) }}</td>
            <th>{{ L('unique_tokens') }}</th><td>{{ "{:,}".format(col.metrics.unique_tokens) }}</td></tr>
        <tr><th>{{ L('min_char_length') }}</th><td>{{ col.metrics.min_char_length }}</td>
            <th>{{ L('max_char_length') }}</th><td>{{ col.metrics.max_char_length }}</td></tr>
      </table>
      {% if col.metrics.sampled_cells < col.metrics.non_null_cells %}
      <div class="note">{{ L('sampled_note') }}: {{ "{:,}".format(col.metrics.sampled_cells) }} {{ L('of') }} {{ "{:,}".format(col.metrics.non_null_cells) }}</div>
      {% endif %}

      {% if col.metrics.top_bigrams %}
      <p class="ng" style="margin-top:10px">{{ L('top_bigrams') }}:
        {% for ng, c in col.metrics.top_bigrams[:8] %}<span class="mono">{{ ng }} ({{ c }})</span> {% endfor %}
      </p>
      {% endif %}

      <div class="imgrow">
        {% if col.charts.top_tokens %}
        <div><div class="imgcap">{{ L('top_tokens') }}</div><img src="data:image/png;base64,{{ col.charts.top_tokens }}" alt="top tokens"></div>
        {% endif %}
        {% if col.charts.length_hist %}
        <div><div class="imgcap">{{ L('length_distribution') }}</div><img src="data:image/png;base64,{{ col.charts.length_hist }}" alt="length histogram"></div>
        {% endif %}
      </div>
      {% if col.charts.wordcloud %}
      <div class="imgrow full">
        <div><div class="imgcap">{{ L('wordcloud') }}</div><img src="data:image/png;base64,{{ col.charts.wordcloud }}" alt="word cloud"></div>
      </div>
      {% endif %}

    {% else %}
      <!-- numeric / categorical / datetime / id -->
      <table>
        {% for k, v in col.basic_stats %}
        <tr><th>{{ k }}</th><td>{{ v }}</td></tr>
        {% endfor %}
      </table>
      {% if col.dist_chart %}
      <div class="imgrow full">
        <div><div class="imgcap">{{ L('distribution') }}</div><img src="data:image/png;base64,{{ col.dist_chart }}" alt="value distribution"></div>
      </div>
      {% endif %}
      {% if col.valuecounts_chart %}
      <div class="imgrow full">
        <div><div class="imgcap">{{ L('value_counts') }}</div><img src="data:image/png;base64,{{ col.valuecounts_chart }}" alt="value counts"></div>
      </div>
      {% endif %}
      {% if col.top_values %}
      <h3>{{ L('top_values') }}</h3>
      <table>
        {% for val, cnt in col.top_values %}
        <tr><td>{{ val }}</td><td class="ng">{{ "{:,}".format(cnt) }}</td></tr>
        {% endfor %}
      </table>
      {% endif %}
    {% endif %}
  </div>
  {% endfor %}
  <div class="note">{{ L('more_columns_note') }}</div>

  <footer>
    {{ L('generated_by') }} <b>ThaiEDA</b> v{{ version }} — AutoEDA สำหรับข้อมูลภาษาไทย
  </footer>
</div>
</body>
</html>"""
