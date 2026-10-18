"""Market-snapshot integrity audit: distinguish source, synchronized EOD and coverage.

No network calls.  Source time must never be synthesized from cache dates.
A common-day rollback can be safe for historical research without being current;
missing exchange provenance is unverified, not silently declared synchronized.
"""
from __future__ import annotations
from collections import Counter
from datetime import date


def _nonnegative(value):
    try:
        return max(0, int(value))
    except (TypeError, ValueError, OverflowError):
        return 0


def _day(value):
    if value is None:
        return ""
    s = str(value).strip()[:10].replace("/", "-")
    try:
        return date.fromisoformat(s).isoformat()
    except ValueError:
        return ""


def _percent(part, whole):
    return round(100 * part / whole, 1) if whole else None


def assess_snapshot(snapshot: dict | None) -> dict:
    if not isinstance(snapshot, dict) or not snapshot:
        return {
            "status": "NO_DATA", "label": "尚未建立完整日線模型", "issues": ["按上方『背景更新日線模型』建立基準。"],
            "blockers": [], "metrics": {}, "dates": {}, "rates": {},
            "safe_to_claim_synced": False, "safe_for_research": False,
            "sync_state": "NO_DATA", "sync_label": "尚無資料", "source_verified": False,
            "generated_at": "", "source_errors": [],
        }
    c = snapshot.get("coverage") or {}
    o = c.get("official_eod") or {}
    s = snapshot.get("session") or {}
    provisional_cache = bool(o.get('provisional_cache') or s.get('provisional_cache'))
    # raw_* fields refer to the ACTUAL official response, not cache-target dates.
    tw = _day(o.get("raw_twse_date") if "raw_twse_date" in o else o.get("twse_date"))
    ot = _day(o.get("raw_tpex_date") if "raw_tpex_date" in o else o.get("tpex_date"))
    idx = _day(o.get("raw_index_date") if "raw_index_date" in o else o.get("index_date"))
    explicit_common = _day(s.get("common_complete_date") or o.get("model_sync_date"))
    price = _day(snapshot.get("price_date"))
    common = explicit_common or price
    mother, valid = _nonnegative(c.get("requested")), _nonnegative(c.get("feature_valid"))
    model = _nonnegative(snapshot.get("full_universe_model_n") or snapshot.get("candidate_n"))
    tw_count, ot_count = _nonnegative(c.get("twse_universe")), _nonnegative(c.get("tpex_universe"))
    research = _nonnegative(snapshot.get("research_breadth_n"))
    issues, blockers, notes = [], [], []

    def problem(message, blocking=False):
        if message not in issues:
            issues.append(message)
        if blocking and message not in blockers:
            blockers.append(message)

    if not common or not price:
        problem("完整模型日或快照價格日缺漏，不可核發同步狀態。", True)
    if not explicit_common:
        problem("缺少共同完整交易日欄位；只能看到快照日期，尚無法確認同步來源。")
    if price and explicit_common and price != explicit_common:
        problem(f"快照價格日 {price} 與共同模型日 {explicit_common} 不同。", True)
    if not tw or not ot:
        problem("上市或上櫃的官方原始來源日期未取得；不可用快取日期代替官方驗證。")
    if tw and ot and tw != ot:
        if provisional_cache and common and common > min(tw, ot):
            problem(f"備援日線模型日 {common} 較官方最後可見日期新，僅供研究，未視為官方完成核價。")
        elif common and common <= min(tw, ot):
            problem(f"上市 {tw}／上櫃 {ot} 不同日；已採共同模型日 {common}，較新交易日須等來源補齊。")
        else:
            problem(f"上市 {tw}／上櫃 {ot} 不同日且未安全統一模型日。", True)
    if common and any(d and common > d for d in (tw, ot, idx)):
        if provisional_cache and _day(o.get('cache_observed_date')) == common:
            problem("模型採兩市行情快取的新日期，尚未完成官方原始來源交叉核對；只能供研究，不可產生可買判斷。")
        else:
            problem("模型日比已確認的交易所來源更新，存在跨日混用風險。", True)
    if common and tw and ot and common < min(tw, ot):
        problem(f"官方來源已可取得 {min(tw, ot)}；模型仍停留在 {common}，應檢查快取覆蓋及重新更新。")
    if not mother or not model:
        problem("股票母體或完整模型家數為零；不能代表全市場結果。", True)
    if not tw_count or not ot_count:
        problem("上市或上櫃股票母體家數缺漏，無法確認兩市場均有納入。", True)
    if mother and tw_count and ot_count and mother != tw_count + ot_count:
        problem("普通股總母體與上市＋上櫃分項家數不符，須核對去重口徑。")
    if valid and model > valid:
        problem("可投資模型家數超過有效歷史家數，無法成立。", True)
    if mother and valid > mother:
        problem("歷史有效家數超過普通股母體家數，無法成立。", True)
    if mother and model > mother:
        problem("可投資模型家數超過普通股母體家數，無法成立。", True)

    stocks = snapshot.get("stocks")
    stock_date_counts = {}
    if isinstance(stocks, list) and stocks:
        raw_counts = Counter(_day(row.get("price_date")) for row in stocks if isinstance(row, dict))
        stock_date_counts = {k or "未標示": n for k, n in raw_counts.items()}
        mixed = {k: v for k, v in raw_counts.items() if not k or k != common}
        if mixed:
            description = "、".join(f"{k or '未標示'}：{v}檔" for k, v in sorted(mixed.items()))
            problem(f"模型內股票的價格日並未全數同步於 {common}（{description}）。", True)
        if model != len(stocks):
            problem(f"模型家數 {model:,} 與快照股票紀錄 {len(stocks):,} 不一致。", True)

    if s.get("postclose_pending"):
        problem("盤後官方收盤資料尚未全數齊備，先使用上一完整交易日；不是當日最新行情。")
    errors = [str(x) for x in (c.get("errors") or []) if str(x).strip()]
    if errors:
        if o.get("reconciled_retry") and tw and ot and tw == ot == common:
            notes.append(f"來源過程曾有 {len(errors)} 則警示，但交易所重試已同步；明細仍供追查。")
        else:
            notes.append(f"來源擷取另有 {len(errors)} 則警示；必要時展開紀錄查核。")
    stale = _nonnegative(c.get("stale_excluded"))
    if stale:
        notes.append(f"另有 {stale:,} 檔因價格日落後而被排除，不代表模型內部混用日期。")

    source_verified = bool(tw and ot)
    if blockers:
        status, sync_state, sync_label = "BLOCK", "INVALID", "資料異常｜停止視為最新推薦"
    elif provisional_cache:
        status, sync_state, sync_label = "CHECK", "PROVISIONAL", "當日日線已更新｜官方來源待核；僅供研究"
    elif not source_verified:
        status, sync_state, sync_label = "CHECK", "UNVERIFIED", "來源待核｜尚不能確認兩市場日期"
    elif tw == ot == common and not s.get("postclose_pending"):
        status, sync_state, sync_label = "OK", "ALIGNED", "同步完成｜上市上櫃同一模型日"
    else:
        status, sync_state, sync_label = "CHECK", "ROLLBACK", "保守回退｜模型同日但非最新兩市場"
    # A good synchronized model may have recoverable network warnings; warnings
    # are visible but must not masquerade as a date-synchronization failure.
    if status == "OK" and issues:
        status = "CHECK"
        sync_label = "已同步日期｜仍有覆蓋或來源待核事項"
    if status == "OK" and errors and not o.get("reconciled_retry"):
        status = "CHECK"
        sync_label = "已同步日期｜另有來源擷取警示"

    return {
        "status": status,
        "label": {"BLOCK": "資料內部檢核未通過", "CHECK": "模型可查看，部分資料仍待核對", "OK": "完整日線快照內部檢核通過"}[status],
        "issues": issues, "blockers": blockers, "notes": notes,
        "safe_to_claim_synced": bool(sync_state == "ALIGNED" and tw == ot == common and not blockers),
        # Only time/source-verified official snapshots may be archived as PIT
        # evidence or enter the executable risk engine. Provisional Yahoo
        # prices are display-only research, a deliberately separate flag.
        "safe_for_research": bool(not blockers and tw and ot and common
                                   and common <= tw and common <= ot
                                   and not provisional_cache and model > 0
                                   and not s.get('postclose_pending')),
        "safe_for_display_research": bool(not blockers and common and model > 0),
        "provisional_cache": provisional_cache,
        "sync_state": sync_state, "sync_label": sync_label, "source_verified": source_verified,
        "metrics": {
            "上市母體": tw_count, "上櫃母體": ot_count, "普通股母體": mother,
            "歷史有效": valid, "可投資模型": model,
            "公司／法人可比資料": research, "排除過期資料": stale,
        },
        "rates": {
            "history_pct": _percent(valid, mother), "investable_pct": _percent(model, mother),
            "research_pct": _percent(research, model),
        },
        "dates": {"上市來源": tw or "未驗證", "上櫃來源": ot or "未驗證", "統一模型日": common or "未建立"},
        "market_source": {"twse": str(o.get("twse_source") or "未標示"), "tpex": str(o.get("tpex_source") or "未標示")},
        "stock_date_counts": stock_date_counts,
        "generated_at": str(snapshot.get("generated_at") or ""),
        "source_errors": errors,
        "calendar_expected_date": str(c.get("calendar_expected_date") or ""),
    }
