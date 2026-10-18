"""UI simulations, explicitly not a real Streamlit browser session."""
from pathlib import Path
import sys
import pandas as pd
from test_v19_core import prices,cfg,panel
from test_v19_ui import ui
from v191_audit import parse_markdown


def old_report():
    parsed=parse_markdown(Path(__file__).with_name('reference_validation_v190.md').read_text())
    records=[]
    for row in parsed['records']:
        for code,key in [('baseline','baseline'),('adaptive','adaptive'),('benchmark_0050','benchmark')]:
            val=row[key]
            records.append(dict(date=row['date'],horizon=row['horizon'],strategy=code,mean_net=val,
                                complete=val is not None))
    return {'version':'19.0.0','results':records,'settings':parsed['settings'],'status':'RESEARCH_PARTIAL',
            'cutoffs':parsed['cutoffs'],'data_kind':'PROVIDER_HISTORY'}


def test_historical_report_page_shows_matched_cases(ui,panel,cfg):
    module,fake=ui
    module.page_validation(panel,old_report(),cfg)
    tables=[args[0] for name,args,_ in fake.calls if name=='dataframe']
    assert any(len(x)==3 for x in tables)
    assert any(name=='warning' for name,_,_ in fake.calls)
    assert not any(name=='error' for name,_,_ in fake.calls)


def test_new_guard_message_present_without_evidence(ui,panel,cfg):
    module,fake=ui
    module.page_decisions(panel,{},cfg)
    assert any(name=='warning' and '0050' in str(args[0]) for name,args,_ in fake.calls)


def test_diagnostic_download_is_first_class(ui,panel,cfg,tmp_path,monkeypatch):
    from v19_data import atomic_json
    module,fake=ui
    run=module.DATA/'runs'/'test';run.mkdir(parents=True)
    (run/'v191_diagnostic_bundle.zip').write_bytes(b'zipfixture')
    atomic_json(module.DATA/'last_success.json',{'directory':str(run.resolve())})
    module.page_validation(panel,old_report(),cfg)
    assert any(name=='download_button' and kwargs.get('file_name')=='v191_diagnostic_bundle.zip'
               for name,_,kwargs in fake.calls)
