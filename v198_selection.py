"""V19.8 preregistered horizon-specific hypotheses (development, not optimal)."""
from __future__ import annotations
from v19_config import Settings, TECH, prior


def profiles_for_v198(h,cfg=Settings()):
    core={'legacy_control':prior(h,TECH)}
    H='p197_annual_high';D='p197_downside';C='p197_compression';L='p197_liquidity'
    if h<=7:
        core.update({
            'high_52':{H:1.0},
            'lower_tail':{D:1.0},
            'high_defense_65_35':{H:.65,D:.35},
            'high_defense_break_45_35_20':{H:.45,D:.35,C:.20},
        })
    elif h<=30:
        core.update({
            'high_52':{H:1.0},
            'squeeze_breakout':{C:.6,H:.4},
            'defensive_breakout_35_40_25':{H:.35,D:.40,C:.25},
            'high_defense_60_40':{H:.60,D:.40},
        })
    else:
        core.update({
            'liquid_capacity':{L:1.0},
            'liquid_high_70_30':{L:.70,H:.30},
            'liquid_defense_70_30':{L:.70,D:.30},
            'liquid_breakout_65_35':{L:.65,C:.35},
        })
    if cfg.optional_data:
        # Observed fundamentals/chip data are accepted ONLY when available at T.
        # Strict disclosure policy is enforced by the base feature/data layers.
        if h<=30:
            core['trust_high_research']={'p197_trust':.45,H:.55}
        else:
            core['quality_liquid_research']={'p197_quality':.45,L:.35,H:.20}
    return core


def fit_before(panel,cutoff,h,cfg=Settings(),seed=1980,context=None):
    from v197_selection import fit_before as fit_prequential
    w,info,trace=fit_prequential(panel,cutoff,h,cfg,seed,context)
    info.update(factor_design='balanced_v198',
        hypothesis_status='PRE_REGISTERED_AFTER_EXPOSED_197_HISTORY_NOT_BLIND',
        non_target_profiles_can_be_zero=True,
        universe_expansion_does_not_prove_old_date_performance=True)
    return w,info,trace
