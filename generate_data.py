"""
generate_data.py
-----------------
Generates a synthetic-but-realistic Earned Value Management (EVM)
dataset for infrastructure projects (roads, bridges, water treatment
plants, transmission lines, etc.), tracked across milestones.

For each project we simulate a hidden risk profile (contractor
quality, site/geotechnical risk, regulatory/permitting risk, supply
chain exposure, design complexity) that drives a compounding drift in
actual cost and schedule performance relative to the baseline plan.
From the raw PV/EV/AC series we engineer the classic EVM indicators:

    CPI  = EV / AC                         (cost performance index)
    SPI  = EV / PV                         (schedule performance index)
    CV   = EV - AC                          (cost variance)
    SV   = EV - PV                          (schedule variance)
    EAC  = BAC / CPI                        (estimate at completion)
    TCPI = (BAC - EV) / (BAC - AC)          (to-complete performance index)

plus short-horizon trend/volatility features (3-period rolling slope
and std of CPI/SPI) that give a classifier something to learn from
beyond the raw indicator level.

LABEL: project-level binary outcome, computed from the FINAL milestone
    overrun = 1  if final CPI < 0.95  OR final SPI < 0.90
    overrun = 0  otherwise

The training table only keeps milestone rows from early/mid-project
(at least 2-3 milestones before project completion), so a model
trained on this table is, by construction, predicting the eventual
overrun outcome 2-3 milestones ahead of it actually happening --
exactly the "early warning" framing of the project.
"""

import numpy as np
import pandas as pd
import os

np.random.seed(7)

OUT = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(OUT, exist_ok=True)

N_PROJECTS = 260
MILESTONES = 14           # milestones per project (~7% planned progress each)
LOOKAHEAD = 3              # hold out last 3 milestones from the training feature window

PROJECT_TYPES = ["Highway", "Bridge", "Water Treatment Plant", "Transmission Line",
                  "Airport Terminal", "Rail Corridor", "Dam/Reservoir", "Hospital (Infra)"]
REGIONS = ["Northeast", "Midwest", "South", "West", "Pacific Northwest"]


def simulate_project(pid):
    ptype = np.random.choice(PROJECT_TYPES)
    region = np.random.choice(REGIONS)
    bac = np.random.uniform(15, 400) * 1e6          # budget at completion, $15M-$400M
    duration_planned = MILESTONES                     # in "milestone units"

    # Hidden risk factors (not directly given to the model as-is, but they
    # drive AC/EV drift -- the model must infer risk from EVM trend behavior)
    contractor_quality = np.random.beta(5, 2)          # higher = better (0-1)
    site_risk = np.random.beta(2, 5)                    # higher = worse (0-1)
    regulatory_risk = np.random.beta(2, 6)
    supply_chain_exposure = np.random.beta(2, 5)
    design_complexity = np.random.beta(3, 4)

    # Composite drift rate: how much AC/schedule tends to slip per milestone
    # (dampened so it compounds to a realistic spread of outcomes rather than
    # runaway overruns across nearly every project)
    cost_drift_rate = (0.040 * site_risk + 0.030 * supply_chain_exposure +
                        0.020 * design_complexity - 0.045 * contractor_quality
                        + np.random.normal(0, 0.014))
    schedule_drift_rate = (0.035 * regulatory_risk + 0.028 * site_risk +
                            0.018 * design_complexity - 0.040 * contractor_quality
                            + np.random.normal(0, 0.014))

    # Some projects experience a discrete shock (e.g. permitting delay, storm,
    # supplier default) partway through
    shock_milestone = np.random.choice(range(3, MILESTONES - 2)) if np.random.rand() < 0.35 else None
    shock_magnitude_cost = np.random.uniform(0.05, 0.20) if shock_milestone else 0
    shock_magnitude_sched = np.random.uniform(0.05, 0.18) if shock_milestone else 0

    rows = []
    cum_ac_factor = 1.0     # multiplicative drift applied to planned cost -> actual cost
    cum_sched_factor = 1.0  # multiplicative drift applied to planned progress -> actual progress

    for m in range(1, MILESTONES + 1):
        planned_pct = m / MILESTONES
        pv = bac * planned_pct

        cum_ac_factor *= (1 + cost_drift_rate + np.random.normal(0, 0.02))
        cum_sched_factor *= (1 - schedule_drift_rate + np.random.normal(0, 0.02))
        if shock_milestone and m == shock_milestone:
            cum_ac_factor *= (1 + shock_magnitude_cost)
            cum_sched_factor *= (1 - shock_magnitude_sched)

        actual_pct_complete = np.clip(planned_pct * cum_sched_factor, 0.01, 1.15)
        ev = bac * min(actual_pct_complete, 1.0)
        ac = pv * cum_ac_factor
        ac = max(ac, ev * 0.5)  # sanity floor

        # Reporting/measurement noise: real EVM data is never perfectly clean
        # (rounding in progress %, invoice timing lags, etc.)
        ev = ev * np.random.normal(1.0, 0.026)
        ac = ac * np.random.normal(1.0, 0.030)

        cpi = ev / ac if ac > 0 else np.nan
        spi = ev / pv if pv > 0 else np.nan
        cv = ev - ac
        sv = ev - pv
        eac = bac / cpi if cpi and cpi > 0 else np.nan
        denom = (bac - ac)
        tcpi = (bac - ev) / denom if abs(denom) > 1e-6 else np.nan

        rows.append({
            "project_id": pid, "project_type": ptype, "region": region, "bac_usd": bac,
            "milestone": m, "planned_pct_complete": planned_pct, "actual_pct_complete": actual_pct_complete,
            "pv": pv, "ev": ev, "ac": ac, "cpi": cpi, "spi": spi, "cv": cv, "sv": sv, "eac": eac, "tcpi": tcpi,
            "contractor_quality": contractor_quality, "site_risk": site_risk,
            "regulatory_risk": regulatory_risk, "supply_chain_exposure": supply_chain_exposure,
            "design_complexity": design_complexity, "had_shock": int(shock_milestone == m) if shock_milestone else 0,
        })

    df = pd.DataFrame(rows)

    final = df.iloc[-1]
    overrun = int((final["cpi"] < 0.92) or (final["spi"] < 0.86))
    df["final_overrun"] = overrun
    df["final_cpi"] = final["cpi"]
    df["final_spi"] = final["spi"]
    return df


def add_trend_features(df):
    """3-period rolling trend/volatility features per project, computed causally
    (only using data up to and including the current milestone)."""
    df = df.sort_values(["project_id", "milestone"]).copy()
    out = []
    for pid, g in df.groupby("project_id"):
        g = g.sort_values("milestone").reset_index(drop=True)
        g["cpi_roll_mean3"] = g["cpi"].rolling(3, min_periods=1).mean()
        g["spi_roll_mean3"] = g["spi"].rolling(3, min_periods=1).mean()
        g["cpi_roll_std3"] = g["cpi"].rolling(3, min_periods=1).std().fillna(0)
        g["spi_roll_std3"] = g["spi"].rolling(3, min_periods=1).std().fillna(0)
        g["cpi_trend3"] = g["cpi"].diff(2) / 2
        g["spi_trend3"] = g["spi"].diff(2) / 2
        g["cpi_trend3"] = g["cpi_trend3"].fillna(0)
        g["spi_trend3"] = g["spi_trend3"].fillna(0)
        g["cv_pct_bac"] = g["cv"] / g["bac_usd"]
        g["sv_pct_bac"] = g["sv"] / g["bac_usd"]
        out.append(g)
    return pd.concat(out, ignore_index=True)


def build_dataset():
    all_projects = [simulate_project(pid) for pid in range(1, N_PROJECTS + 1)]
    full = pd.concat(all_projects, ignore_index=True)
    full = add_trend_features(full)

    # Training table: only keep milestones up to (MILESTONES - LOOKAHEAD),
    # i.e. features are observed at least `LOOKAHEAD` milestones before the
    # project's actual completion -> genuine early-warning framing.
    cutoff = MILESTONES - LOOKAHEAD
    train_view = full[full.milestone <= cutoff].copy()

    full.to_csv(os.path.join(OUT, "evm_full_timeseries.csv"), index=False)
    train_view.to_csv(os.path.join(OUT, "evm_milestone_features.csv"), index=False)

    print(f"Projects: {N_PROJECTS} | Milestones/project: {MILESTONES} | Lookahead: {LOOKAHEAD}")
    print(f"Full timeseries rows: {len(full)} | Training-view rows (<=milestone {cutoff}): {len(train_view)}")
    print(f"Overrun rate (project-level): {full.groupby('project_id')['final_overrun'].first().mean():.1%}")
    return full, train_view


if __name__ == "__main__":
    build_dataset()
