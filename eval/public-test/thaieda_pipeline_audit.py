import sys, json, time, warnings, traceback
from pathlib import Path

sys.path.insert(0, 'src')
import pandas as pd
import thaieda
from pandas.testing import assert_frame_equal

DATASETS = {
    'titanic': 'eval/public-test/titanic.csv',
    'iris': 'eval/public-test/iris.csv',
    'housing': 'eval/public-test/housing.csv',
}

OUTDIR = Path('eval/public-test/audit_outputs')
OUTDIR.mkdir(parents=True, exist_ok=True)

def safe(obj):
    if hasattr(obj, 'to_dict'):
        try:
            return obj.to_dict()
        except Exception:
            pass
    if isinstance(obj, dict):
        return {str(k): safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [safe(x) for x in obj]
    return obj

def df_diff_summary(before, after):
    summary = {
        'before_shape': list(before.shape),
        'after_shape': list(after.shape),
        'same_shape': before.shape == after.shape,
        'dtypes_before': {c: str(t) for c, t in before.dtypes.items()},
        'dtypes_after': {c: str(t) for c, t in after.dtypes.items()},
        'missing_before': {c: int(before[c].isna().sum()) for c in before.columns},
        'missing_after': {c: int(after[c].isna().sum()) for c in after.columns if c in after.columns},
        'duplicates_before': int(before.duplicated().sum()),
        'duplicates_after': int(after.duplicated().sum()),
        'columns_changed': [],
        'cell_changes_sample': {},
        'equal': False,
    }
    try:
        assert_frame_equal(before, after, check_dtype=True, check_like=False)
        summary['equal'] = True
    except Exception as e:
        summary['equal'] = False
        summary['assert_error'] = str(e)[:500]
    common = [c for c in before.columns if c in after.columns]
    if before.shape[0] == after.shape[0]:
        for c in common:
            b = before[c]
            a = after[c]
            neq = ~((b == a) | (b.isna() & a.isna()))
            cnt = int(neq.sum())
            if cnt:
                summary['columns_changed'].append({'column': c, 'changed_cells': cnt})
                idxs = list(neq[neq].index[:5])
                summary['cell_changes_sample'][c] = [
                    {'index': int(i) if isinstance(i, int) else str(i), 'before': repr(b.loc[i]), 'after': repr(a.loc[i])}
                    for i in idxs
                ]
    return summary

def html_probe(html):
    lower = html.lower()
    return {
        'html_len': len(html),
        'contains_traceback': 'traceback' in lower,
        'contains_error_word': 'error' in lower or 'exception' in lower,
        'empty_section_markers': html.count('>—<') + html.count('ไม่มีข้อมูล'),
        'thai_specific_terms': {
            'buddhist_era': ('พ.ศ' in html or 'Buddhist' in html),
            'thai_numerals': ('เลขไทย' in html or 'Thai numerals' in html or '๐' in html),
            'thai_specific_phrase': ('Thai-specific' in html or 'ข้าม Thai-specific checks' in html),
        },
        'sections': {s: (s in html) for s in [
            'Executive', 'สรุป', 'Key Findings', 'ข้อค้นพบ', 'Business', 'Report', 'Quality', 'Anomal', 'Cleaning', 'Charts'
        ]}
    }

all_summary = {}
for name, path in DATASETS.items():
    print(f'=== {name} ===', flush=True)
    df = pd.read_csv(path)
    print('input shape', df.shape, 'dtypes', {c: str(t) for c, t in df.dtypes.items()}, flush=True)
    t0 = time.time()
    warn_list = []
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            result = thaieda.run(df, make_charts=True)
            warn_list = [str(x.message) for x in w]
        elapsed = time.time() - t0
        html_path = OUTDIR / f'{name}-audit-report.html'
        html = result.to_html(str(html_path))
        dct = result.to_dict()
        cleaned = result.cleaned_df
        info = {
            'ok': True,
            'elapsed_sec': round(elapsed, 3),
            'input_shape': list(df.shape),
            'clean_diff': df_diff_summary(df, cleaned),
            'overview': safe(result.overview),
            'data_type': safe(dct.get('data_type')),
            'column_types': {k: str(v) for k, v in result.report.column_types.items()},
            'quality_issues': safe(result.quality_issues),
            'anomalies': safe(result.anomalies),
            'insights': safe(result.insights),
            'insight_engine': safe(dct.get('insight_engine')),
            'cleaning_suggestions': safe(dct.get('cleaning_suggestions')),
            'columns': safe(dct.get('columns')),
            'notes': safe(result.notes),
            'warnings': warn_list,
            'html_path': str(html_path),
            'html_probe': html_probe(html),
        }
        info['input_missing_total'] = int(df.isna().sum().sum())
        info['input_missing_by_col'] = {c: int(df[c].isna().sum()) for c in df.columns}
        info['input_duplicates'] = int(df.duplicated().sum())
        info['input_unique_by_col'] = {c: int(df[c].nunique(dropna=True)) for c in df.columns}
        print('elapsed', elapsed, 'clean_equal', info['clean_diff']['equal'], 'quality', len(info['quality_issues'] or []), 'anom', len(info['anomalies'] or []), flush=True)
        print('language', info['data_type'].get('language', {}).get('language') if info['data_type'] else None, 'data_type', info['data_type'].get('key') if info['data_type'] else None, flush=True)
        print('column_types', info['column_types'], flush=True)
        print('quality', json.dumps(info['quality_issues'], ensure_ascii=False, default=str)[:1200], flush=True)
        print('anomalies', json.dumps(info['anomalies'], ensure_ascii=False, default=str)[:1200], flush=True)
    except Exception as e:
        info = {'ok': False, 'error': repr(e), 'traceback': traceback.format_exc()}
        print('ERROR', repr(e), flush=True)
        print(traceback.format_exc(), flush=True)
    all_summary[name] = info

out_json = OUTDIR / 'pipeline_audit_summary.json'
out_json.write_text(json.dumps(all_summary, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
print('WROTE', out_json)
