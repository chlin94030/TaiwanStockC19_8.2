# Alpha Radar V19.8.2 -- Streamlit Cloud stability repair

IMPORTANT: the V19.8.2 release fixes deployment and memory hazards. It does
NOT change factor weights, backtest trade costs, selected returns or claim
improved investment performance. The numeric research engine remains V19.8.

## Replace files in GitHub

Use the FULL zip (100 root files) to replace your repository sources, or the
9-file PATCH only if the repository is exactly the previously delivered
V19.8.1 100-file release. Extract the ZIP before uploading; GitHub must
contain app.py and requirements.txt at its ROOT, not inside a ZIP or folder.
Keep your private data/, database, Streamlit Secrets and FINMIND_TOKEN.

The entrypoint remains app.py. The PATCH contains changed source/tests, this
README and SHA256_MANIFEST.json for verification.

requirements.txt is intentionally PINNED, including Streamlit 1.59.2 and
pyarrow 24.0.0. Streamlit 1.58.0 has a known bug with @st.fragment(run_every),
fixed starting in 1.59.0. Avoid unrestricted auto-upgrade to 1.66.0 until
successful Cloud web-session verification.

## Launch sequence

1. Push files to GitHub, then wait for Cloud to reinstall dependencies.
2. Open the app from Safari (rather than an embedded app browser).
3. The first page MUST say 'Alpha Radar V19.8.2' and 'safe startup'. It loads
   no stock modules and makes no provider API requests.
4. Click 'Check program modules' (isolated subprocess) if needed.
5. Click 'Open complete dashboard'. This imports numerical modules but does NOT
   automatically compute the full historical panel.
6. Use the download/cache buttons for research tasks. To view stock cards and
   charts, click 'Load latest stock analysis'. Only recent per-symbol bars are
   read for the DISPLAY. The original full history is still used by backtests.
7. Database ZIP bytes are only loaded after 'Prepare database download' is
   clicked. An archive >150 MiB requires an offline/persistent-storage route.

## If the WHOLE Cloud site still says 'Oh no'

Open Manage app > Logs. If there is no [ALPHARADAR_CLOUD] stage=script_entry,
the Python script never ran. Investigate deployment health/WebSocket/runtime,
not the model. If this persists, save secrets, repo/branch, custom URL and
consider redeploying with Python 3.13 via Cloud Advanced Settings. A runtime.txt
file alone does NOT change a deployed Community Cloud Python version.

If stage=FULL_DASHBOARD_ERROR exists, copy the full traceback from Logs. If
the safe page runs but stock data fails, use the isolated module probe, then
load the dashboard and its data separately to identify the failing phase.

## Real testing limits

The source was compiled and unit/integration tested with Python 3.13.5 and
fake-Streamlit page execution. A real 63,843-row provider-price archive was
loaded with the new bounded UI path: 30,471 rows, 92 codes including 0050,
159 panel columns, approximately 62 MiB final DataFrame. This is not a live
Cloud 3.14.8/browser/session test. No web/API credential or live fill was tested.
The Cloud page cannot prevent a system OOM kill, proxy issue or segfault that
happens before app.py runs. Never claim 'fixed on Cloud' without deployment
confirmation.
