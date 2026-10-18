"""Alpha Radar V19.8.2 -- isolated, recoverable Streamlit Cloud entry point.

Nothing related to stock data is imported before the user deliberately opens the
analysis interface.  Both the home page and a subprocess import probe are
independent of the potentially memory-heavy analysis code.
"""
from __future__ import annotations

import importlib.metadata
import os
from pathlib import Path
import platform
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parent
LOG_TAG = '[ALPHARADAR_CLOUD]'


def record(stage: str, error: BaseException | None = None) -> None:
    print(f'{LOG_TAG} stage={stage}', file=sys.stderr, flush=True)
    if error is not None:
        traceback.print_exception(type(error), error, error.__traceback__, file=sys.stderr)


def version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return 'NOT_INSTALLED'


def run_import_probe(timeout: int = 18) -> dict:
    """Run model imports in a DIFFERENT process so a crash can't kill the UI.

    A successful import is useful, but does not certify Cloud WebSocket/health
    checks, production data, memory safety during feature calculation, or profit.
    """
    script = (
        "from v19_config import Settings; "
        "import v19_features, v19_service, v19_ui; "
        "print('MODEL_IMPORT_OK', Settings().factor_design)"
    )
    env = os.environ.copy()
    env.update(OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    try:
        proc = subprocess.run(
            [sys.executable, '-c', script], cwd=str(ROOT), env=env,
            capture_output=True, text=True, timeout=timeout, check=False,
        )
        # Print only brief diagnostic excerpts -- never dump user's environment.
        return {
            'status': 'OK' if proc.returncode == 0 else 'IMPORT_ERROR',
            'exit_code': proc.returncode,
            'stdout': proc.stdout[-600:],
            'stderr_tail': proc.stderr[-1800:],
        }
    except subprocess.TimeoutExpired:
        return {'status': 'TIMEOUT', 'seconds': timeout}
    except OSError as error:
        return {'status': 'LAUNCH_ERROR', 'error_type': type(error).__name__}


def lightweight_home(st) -> None:
    """Renders without importing pandas, NumPy, Plotly, yfinance or model code."""
    st.title('Alpha Radar · V19.8.2')
    st.success('安全啟動首頁已載入。這一步尚未下載行情或建立歷史因子資料。')
    st.write('為避免雲端記憶體不足，完整選股系統及大型歷史資料只會在你主動開啟時載入。')

    if st.button('開啟完整選股系統', type='primary', key='ar_cloud_open'):
        st.session_state['ar1981_mode'] = 'full'
        record('open_full_clicked')
        st.rerun()

    if st.button('檢查程式模組（獨立程序，不下載行情）', key='ar_cloud_probe'):
        with st.spinner('檢查研究模組能否匯入……'):
            result = run_import_probe()
        st.session_state['ar_cloud_probe_result'] = result
        record('probe_' + result['status'].lower())

    result = st.session_state.get('ar_cloud_probe_result')
    if result:
        if result['status'] == 'OK':
            st.success('核心模組匯入成功。下一步仍須檢查正式資料與Cloud資源。')
        else:
            st.error('核心模組自檢失敗：' + result['status'])
        with st.expander('匯入檢查詳細資料'):
            st.json(result)

    with st.expander('部署資訊與資料狀態'):
        history = ROOT / 'data' / 'v19' / 'history_prices.csv.gz'
        st.write({
            'Python': platform.python_version(),
            'Streamlit': version('streamlit'),
            'Pandas': version('pandas'),
            'Model source': 'OK' if (ROOT/'v19_ui.py').is_file() else 'MISSING',
            'History cache': 'FOUND' if history.is_file() else 'NO LOCAL CACHE',
            'History cache compressed MiB': round(history.stat().st_size/(1024*1024), 2)
                if history.is_file() else 0,
        })
        st.caption('Cloud每次重新部署可能不保留本地產生的快取；請先備份資料。')
    st.caption('若連這個簡單首頁也出現平台「Oh no」，請優先檢查Cloud健康檢查、Python環境或WebSocket，而不是選股因子。')
    record('home_render_ok')


def full_dashboard(st) -> None:
    try:
        record('full_import_start')
        from v19_ui import main
        record('full_import_ok')
        main(configure_page=False)
        record('full_render_ok')
    except Exception as error:
        record('FULL_DASHBOARD_ERROR', error)
        st.error('完整選股頁面載入失敗，安全首頁仍可使用。')
        st.warning('例外種類：' + type(error).__name__)
        st.caption('請至 Manage app → Logs 搜尋 [ALPHARADAR_CLOUD] stage=FULL_DASHBOARD_ERROR，並提供Traceback。')
        if st.button('返回安全首頁', key='ar_cloud_back_error'):
            st.session_state['ar1981_mode'] = 'safe'
            st.session_state.pop('ar_cloud_allow_panel', None)
            st.rerun()


def run() -> None:
    record('script_entry')
    try:
        import streamlit as st
    except Exception as error:
        record('STREAMLIT_IMPORT_ERROR', error)
        raise
    # Set page configuration EXACTLY ONCE, before emitting other elements.
    st.set_page_config(page_title='Alpha Radar V19.8.2', page_icon='📈',
                       layout='wide', initial_sidebar_state='collapsed')
    mode = st.session_state.get('ar1981_mode', 'safe')
    if mode == 'full':
        with st.sidebar:
            if st.button('返回安全首頁', key='ar_cloud_back'):
                st.session_state['ar1981_mode'] = 'safe'
                st.rerun()
        full_dashboard(st)
    else:
        lightweight_home(st)


if __name__ == '__main__':
    run()
