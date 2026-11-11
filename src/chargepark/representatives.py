"""Reduce the eligible charging curves to k representative real curves.

Curves are first capped at the site charger power (at a 150 kW site many
curves become identical), then clustered with weighted k-medoids (PAM) on
  features = capped kW at SOC 0, step, ..., 100  +  battery_kwh x battery_weight
using Euclidean distance. Each representative (medoid) is a real curve; its
weight is the sum of its members' weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .curves import N_SOC, SOC_COLS, CurveSet, norm_name


# =============================================================================
# Weighted PAM
# =============================================================================
def pam_cost(D: np.ndarray, w: np.ndarray, medoids) -> float:
    return float(w @ D[:, list(medoids)].min(axis=1))


def _build(D, w, k):
    med = [int(np.argmin(w @ D))]
    near = D[:, med[0]].copy()
    for _ in range(1, k):
        gain = w @ np.maximum(near[:, None] - D, 0.0)
        gain[med] = -np.inf
        j = int(np.argmax(gain))
        med.append(j)
        near = np.minimum(near, D[:, j])
    return med


def _swap(D, w, med, max_iter=500):
    med = list(med)
    n = D.shape[0]
    cost = pam_cost(D, w, med)
    for _ in range(max_iter):
        best = (cost, None, None)
        for a in range(len(med)):
            others = med[:a] + med[a + 1:]
            d_other = D[:, others].min(axis=1) if others else np.full(n, np.inf)
            # cost of replacing med[a] by every candidate h at once
            c = w @ np.minimum(d_other[:, None], D)
            c[med] = np.inf
            h = int(np.argmin(c))
            if c[h] < best[0] - 1e-9 * max(1.0, abs(best[0])):
                best = (float(c[h]), a, h)
        if best[1] is None:
            break
        cost, a, h = best
        med[a] = h
    return med, cost


def pam(D: np.ndarray, w: np.ndarray, k: int, seed: int = 0, n_init: int = 1):
    """Weighted k-medoids. Start 0 is the deterministic BUILD; starts 1..n_init-1
    are random (seeded). Returns (medoid indices sorted ascending, cost)."""
    n = D.shape[0]
    if k >= n:
        return np.arange(n), 0.0
    rng = np.random.default_rng(seed)
    starts = [_build(D, w, k)] + [list(rng.choice(n, k, replace=False)) for _ in range(n_init - 1)]
    best_med, best_cost = None, np.inf
    for s in starts:
        med, cost = _swap(D, w, s)
        if best_med is None or cost < best_cost - 1e-9 * max(1.0, abs(best_cost)):
            best_med, best_cost = med, cost
    return np.array(sorted(best_med)), best_cost


# =============================================================================
# Features, weights, error metrics
# =============================================================================
def capped_curves(cs: CurveSet, charger_kw: float) -> np.ndarray:
    return np.minimum(cs.power_kw, charger_kw)


def features(capped: np.ndarray, battery_kwh: np.ndarray, soc_step: int, battery_weight: float) -> np.ndarray:
    return np.column_stack([capped[:, ::soc_step], np.asarray(battery_kwh, float) * battery_weight])


def distance_matrix(X: np.ndarray) -> np.ndarray:
    sq = (X ** 2).sum(axis=1)
    return np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * X @ X.T, 0.0))


def charge_minutes(capped: np.ndarray, battery_kwh, soc_from=10, soc_to=80, vehicle_eff=1.0,
                   min_kw=1.0) -> np.ndarray:
    """Minutes to charge from soc_from to soc_to % with the (capped) DC curve:
    t = battery / eff x integral of dSOC / P(SOC), trapezoid on 1 % steps."""
    p = np.maximum(np.atleast_2d(capped)[:, soc_from:soc_to + 1], min_kw)
    inv = 0.5 * (1 / p[:, :-1] + 1 / p[:, 1:])
    return np.asarray(battery_kwh, float) * 0.01 * inv.sum(axis=1) / vehicle_eff * 60


def member_weights(cs: CurveSet, cfg: dict) -> tuple[np.ndarray, list[str]]:
    """Weight per curve. Default: default_weight per curve. With weights_csv
    (brand, model, weight): a model's weight is split equally over its
    versions; a curve gets the sum over the versions it represents."""
    r = cfg["representatives"]
    if not r["weights_csv"]:
        return np.full(len(cs), float(r["default_weight"])), [f"member weight {r['default_weight']} per curve"]
    ext = pd.read_csv(r["weights_csv"])
    for c in ("brand", "model", "weight"):
        if c not in ext.columns:
            raise ValueError(f"{r['weights_csv']}: missing column {c!r}")
    ext_key = norm_name(ext["brand"]) + "|" + norm_name(ext["model"])
    wmap = ext.groupby(ext_key)["weight"].sum()
    ver = cs.versions.assign(key=norm_name(cs.versions["brand"]) + "|" + norm_name(cs.versions["model"]))
    per_model = ver.groupby("key").size()
    model_w = ver["key"].map(wmap)
    matched = model_w.notna()
    model_w = model_w.fillna(float(r["default_weight"]))
    ver["w"] = model_w / ver["key"].map(per_model)
    w = cs.meta["curve_id"].map(ver.groupby("curve_id")["w"].sum()).fillna(0.0).to_numpy(float)
    n_models = ver["key"].nunique()
    log = [f"external weights {r['weights_csv']}: {ver.loc[matched, 'key'].nunique()} of {n_models} "
           f"models matched; unmatched get default_weight {r['default_weight']}"]
    return w, log


# =============================================================================
# Representatives
# =============================================================================
@dataclass
class Representatives:
    k_label: str                 # "8" or "all"
    curves: CurveSet             # all eligible curves
    capped: np.ndarray
    weights: np.ndarray          # member weight per curve
    medoids: np.ndarray          # curve index per cluster, clusters ordered by weight desc
    labels: np.ndarray           # cluster index per curve
    table: pd.DataFrame          # one row per cluster
    members: pd.DataFrame        # one row per curve
    cost: float
    log: list[str]

    @property
    def k(self) -> int:
        return len(self.medoids)

    def catalog(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
        """(power_kw [k, 101] uncapped medoid curves, battery_kwh [k], share [k], labels)."""
        m = self.curves.meta.iloc[self.medoids]
        names = (m["brand"] + " " + m["model"] + " " + m["version"].astype(str)).str.strip().tolist()
        return (self.curves.power_kw[self.medoids].copy(), m["battery_kwh"].to_numpy(float),
                self.table["weight"].to_numpy(float), names)


def build_representatives(cs: CurveSet, cfg: dict, k=None) -> Representatives:
    r = cfg["representatives"]
    k = r["k"] if k is None else k
    n = len(cs)
    if n == 0:
        raise ValueError("no eligible curves")
    capped = capped_curves(cs, cfg["site"]["charger_kw"])
    bat = cs.meta["battery_kwh"].to_numpy(float)
    w, log = member_weights(cs, cfg)
    if w.sum() <= 0:
        raise ValueError("member weights sum to zero")

    if k == "all" or int(k) >= n:
        med, cost = np.arange(n), 0.0
        k_label = "all" if k == "all" else str(k)
    else:
        D = distance_matrix(features(capped, bat, r["feature_soc_step"], r["battery_weight"]))
        med, cost = pam(D, w, int(k), seed=r["seed"], n_init=int(r["n_init"]))
        k_label = str(k)

    # assignment (medoids always label themselves)
    if len(med) == n:
        lab = np.arange(n)
    else:
        lab = np.argmin(D[:, med], axis=1)
        lab[med] = np.arange(len(med))
    # order clusters by weight desc, then by curve_id for determinism
    cw = np.bincount(lab, weights=w, minlength=len(med))
    ids = cs.meta["curve_id"].to_numpy()
    order = sorted(range(len(med)), key=lambda c: (-cw[c], ids[med[c]]))
    remap = np.empty(len(med), int)
    remap[order] = np.arange(len(med))
    med, lab = med[order], remap[lab]

    # errors of every member against its medoid
    veff = cfg["vehicles"].get("vehicle_efficiency", 1.0)
    t_own = charge_minutes(capped, bat, vehicle_eff=veff)
    t_med = t_own[med][lab]
    mae = np.abs(capped - capped[med][lab]).mean(axis=1)
    members = cs.meta[["curve_id", "source", "brand", "model", "version", "battery_kwh", "peak_kw"]].assign(
        cluster=lab + 1, medoid_id=ids[med][lab], weight=w, mae_kw=mae,
        t10_80_min=t_own, t10_80_dev_min=t_med - t_own)

    rows = []
    m = cs.meta
    for c, i in enumerate(med):
        sel = lab == c
        ws = w[sel]
        avg = lambda x: float(np.average(x[sel], weights=ws)) if ws.sum() > 0 else float(np.mean(x[sel]))
        rows.append({
            "cluster": c + 1, "medoid_id": ids[i], "source": m.at[i, "source"],
            "brand": m.at[i, "brand"], "model": m.at[i, "model"], "version": m.at[i, "version"],
            "battery_kwh": m.at[i, "battery_kwh"], "peak_kw": m.at[i, "peak_kw"],
            "peak_kw_capped": float(capped[i].max()), "charger_class": m.at[i, "charger_class"],
            "weight": cw[order[c]] / w.sum(), "weight_raw": cw[order[c]],
            "cluster_size": int(sel.sum()), "n_versions": int(m.loc[sel, "n_versions"].sum()),
            "mae_kw": avg(mae), "t10_80_medoid_min": float(t_own[i]),
            "t10_80_mad_min": avg(np.abs(t_med - t_own)), "t10_80_bias_min": avg(t_med - t_own),
        })
    table = pd.DataFrame(rows)
    log = list(cs.log) + log + [f"k = {k_label}: {len(med)} representatives from {n} curves "
                                f"(capped at {cfg['site']['charger_kw']:.0f} kW)"]
    return Representatives(k_label, cs, capped, w, med, lab, table, members, cost, log)


def overall_errors(rep: Representatives) -> dict[str, float]:
    m, w = rep.members, rep.weights
    return {"k": rep.k, "cost": rep.cost,
            "mae_kw": float(np.average(m["mae_kw"], weights=w)),
            "t10_80_mad_min": float(np.average(m["t10_80_dev_min"].abs(), weights=w)),
            "t10_80_bias_min": float(np.average(m["t10_80_dev_min"], weights=w))}


def elbow(cs: CurveSet, cfg: dict, ks=None) -> pd.DataFrame:
    ks = ks or range(2, int(cfg["representatives"]["elbow_k_max"]) + 1)
    return pd.DataFrame([overall_errors(build_representatives(cs, cfg, k)) for k in ks if k < len(cs)])


# =============================================================================
# Output
# =============================================================================
def write_outputs(rep: Representatives, cfg: dict, plot: bool = True) -> list[Path]:
    out = Path(cfg["representatives"]["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    base = out / f"representatives_k{rep.k_label}"
    paths = [base.with_suffix(".csv"), Path(f"{base}_members.csv"), Path(f"{base}_curves.csv")]
    rep.table.to_csv(paths[0], index=False)
    rep.members.to_csv(paths[1], index=False)
    curves = pd.DataFrame(rep.curves.power_kw[rep.medoids], columns=SOC_COLS)
    curves.insert(0, "medoid_id", rep.table["medoid_id"].to_numpy())
    curves.to_csv(paths[2], index=False)
    if plot and rep.k <= 40:
        paths.append(Path(f"{base}_clusters.png"))
        plot_clusters(rep, cfg, paths[-1])
    return paths


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": "#E6E6E6", "grid.linewidth": 0.6})
    return plt


def plot_clusters(rep: Representatives, cfg: dict, path) -> None:
    plt = _style()
    c = cfg["output"]["colors"]
    ncol = min(4, rep.k)
    nrow = int(np.ceil(rep.k / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 2.7 * nrow), squeeze=False, sharey=True)
    soc = np.arange(N_SOC)
    charger = cfg["site"]["charger_kw"]
    for ci, ax in enumerate(axes.flat):
        if ci >= rep.k:
            ax.axis("off")
            continue
        row = rep.table.iloc[ci]
        sel = np.flatnonzero(rep.labels == ci)
        for j, i in enumerate(sel):
            ax.plot(soc, rep.capped[i], color=c["purple"], lw=0.8, alpha=0.35,
                    label="Members (capped)" if j == 0 else None)
        ax.plot(soc, rep.capped[rep.medoids[ci]], color=c["green"], lw=2.2, label="Medoid (capped)")
        ax.axhline(charger, color=c["pink"], lw=1, ls="--", label=f"Charger {charger:.0f} kW")
        ax.set_title(f"C{ci + 1}: {row['brand']} {row['model']}\n"
                     f"{row['battery_kwh']:.0f} kWh | n={row['cluster_size']} | w={row['weight']:.1%} | "
                     f"MAE {row['mae_kw']:.1f} kW | dt {row['t10_80_mad_min']:.1f} min",
                     loc="left", fontsize=7.5)
        ax.set_xlim(0, 100)
        if ci % ncol == 0:
            ax.set_ylabel("DC kW")
        if ci >= rep.k - ncol:
            ax.set_xlabel("SOC (%)")
        ax.legend(frameon=False, fontsize=6, loc="lower left")
    fig.suptitle(f"Representative charging curves, k = {rep.k_label} "
                 f"(curves capped at {charger:.0f} kW; source: Fastned databank)", x=0.01, ha="left",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_elbow(df: pd.DataFrame, cfg: dict, path) -> None:
    plt = _style()
    c = cfg["output"]["colors"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].plot(df["k"], df["mae_kw"], color=c["green"], marker="o", ms=4, lw=2)
    ax[0].set_title("Mean absolute power deviation from medoid", loc="left", fontweight="bold")
    ax[0].set_ylabel("kW (weighted mean over curves)")
    ax[1].plot(df["k"], df["t10_80_mad_min"], color=c["blue"], marker="o", ms=4, lw=2,
               label="Mean absolute deviation")
    ax[1].plot(df["k"], df["t10_80_bias_min"], color=c["purple"], marker="s", ms=4, lw=1.5, ls="--",
               label="Mean deviation (bias)")
    ax[1].axhline(0, color=c["navy"], lw=0.8)
    ax[1].set_title("10-80 % charging time: medoid vs. member", loc="left", fontweight="bold")
    ax[1].set_ylabel("minutes")
    ax[1].legend(frameon=False)
    for a in ax:
        a.set_xlabel("k (number of representatives)")
        a.set_xticks(df["k"])
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
