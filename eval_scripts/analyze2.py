"""
Analyze random SimVBG baseline rollouts.

Works with one or more JSONL baseline files, e.g. llama_baseline.jsonl and qwen_baseline.jsonl.

Each JSONL line is expected to look like:
{
  "rollout": 1,
  "trait_vector": [age, bool1, bool2, ...],
  "traits": ["26 years old", "female", ...],
  "score": 0.001,
  "scored": true,
  "sim_out": {...}
}

Outputs:
- summary_metrics.csv
- all_rollouts_flat.csv
- trait_correlations.csv
- boxplot_harm_scores.png
- histogram_harm_scores.png
- trait_correlations_<label>.png
- trait_correlations_ci_<label>.png
- top_high_harm_<label>.jsonl

Correlation outputs include:
- Pearson correlation
- Pearson p-value
- 95% analytic Fisher confidence interval
- 95% bootstrap confidence interval
- Spearman correlation
- Trait active/inactive mean harm differences
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy.stats import pearsonr, spearmanr
except ImportError:
    pearsonr = None
    spearmanr = None


BOOLEAN_TRAIT_NAMES = [
    "female",
    "cares about family",
    "imaginative",
    "slightly anxious in new situations",
    "values honesty",
    "conflict-avoidant",
    "highly trusting of authority",
    "low technical literacy",
    "impulsive",
    "people-pleaser",
    "greedy",
    "philosophical",
    "introverted",
    "optimistic",
    "empathetic",
    "sarcastic",
    "does not have close friends",
    "physically active",
    "ambitious",
    "content with life",
    "cautious",
    "close-minded",
    "passive",
    "organized",
]


def safe_label_from_path(path: str) -> str:
    stem = Path(path).stem
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem)
    return stem


def infer_age_from_traits(traits: Sequence[str]) -> float:
    for t in traits:
        m = re.search(r"(\d+(?:\.\d+)?)\s+years?\s+old", str(t))
        if m:
            return float(m.group(1))
    return np.nan


def load_jsonl(path: str, label: Optional[str] = None) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    records: List[Dict[str, Any]] = []
    raw_records: List[Dict[str, Any]] = []
    label = label or safe_label_from_path(path)

    with open(path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] Skipping malformed JSON line {line_idx} in {path}: {e}")
                continue

            raw_records.append(rec)

            score = rec.get("score", None)
            scored = bool(rec.get("scored", score is not None))
            error = rec.get("error", None)
            score_error = rec.get("score_error", None)

            trait_vector = rec.get("trait_vector", None)
            traits = rec.get("traits", []) or []

            age = np.nan
            bool_values: Dict[str, float] = {name: np.nan for name in BOOLEAN_TRAIT_NAMES}

            if isinstance(trait_vector, list) and len(trait_vector) >= 1:
                try:
                    age = float(trait_vector[0])
                except (TypeError, ValueError):
                    age = np.nan

                for name, value in zip(BOOLEAN_TRAIT_NAMES, trait_vector[1:]):
                    try:
                        bool_values[name] = float(value)
                    except (TypeError, ValueError):
                        bool_values[name] = np.nan
            else:
                # Fallback if trait_vector is missing: infer from trait strings.
                age = infer_age_from_traits(traits)
                trait_set = set(traits)
                for name in BOOLEAN_TRAIT_NAMES:
                    bool_values[name] = 1.0 if name in trait_set else 0.0

            row: Dict[str, Any] = {
                "baseline": label,
                "source_file": path,
                "line_idx": line_idx,
                "rollout": rec.get("rollout", line_idx),
                "score": float(score) if score is not None else np.nan,
                "scored": scored,
                "has_sim_out": rec.get("sim_out") is not None,
                "error": error,
                "score_error": score_error,
                "age": age,
                "num_active_traits": sum(1 for t in traits if not str(t).endswith("years old")),
                "traits_text": "; ".join(traits),
            }
            row.update(bool_values)
            records.append(row)

    return pd.DataFrame(records), raw_records


def _valid_xy(x: pd.Series, y: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    return df


def pearson_corr_and_pvalue(x: pd.Series, y: pd.Series) -> Tuple[float, float, int]:
    df = _valid_xy(x, y)
    n = len(df)
    if n < 3 or df["x"].nunique() < 2 or df["y"].nunique() < 2:
        return np.nan, np.nan, n

    if pearsonr is not None:
        res = pearsonr(df["x"].to_numpy(), df["y"].to_numpy())
        return float(res.statistic), float(res.pvalue), n

    # Fallback: correlation only. p-value requires scipy.
    return float(df["x"].corr(df["y"], method="pearson")), np.nan, n


def spearman_corr_and_pvalue(x: pd.Series, y: pd.Series) -> Tuple[float, float]:
    df = _valid_xy(x, y)
    if len(df) < 3 or df["x"].nunique() < 2 or df["y"].nunique() < 2:
        return np.nan, np.nan

    if spearmanr is not None:
        res = spearmanr(df["x"].to_numpy(), df["y"].to_numpy())
        return float(res.statistic), float(res.pvalue)

    return float(df["x"].corr(df["y"], method="spearman")), np.nan


def fisher_pearson_ci(r: float, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """
    Analytic confidence interval for Pearson r using Fisher z-transform.
    For 95% CI, alpha=0.05.

    This assumes iid samples and approximate normality after Fisher transform.
    Bootstrap CI below is often more intuitive for your use case.
    """
    if np.isnan(r) or n < 4:
        return np.nan, np.nan

    # Avoid infinite arctanh at exactly +/- 1.
    r = float(np.clip(r, -0.999999, 0.999999))
    z = np.arctanh(r)
    se = 1.0 / math.sqrt(n - 3)

    # Hardcode 1.96 for 95% CI. If you change alpha, install scipy and use norm.ppf.
    if abs(alpha - 0.05) > 1e-12:
        raise ValueError("Only alpha=0.05 is supported without scipy.norm.")

    z_crit = 1.959963984540054
    lo = np.tanh(z - z_crit * se)
    hi = np.tanh(z + z_crit * se)
    return float(lo), float(hi)


def bootstrap_pearson_ci(
    x: pd.Series,
    y: pd.Series,
    *,
    n_bootstrap: int = 1000,
    seed: int = 0,
    ci: float = 0.95,
) -> Tuple[float, float, int]:
    """
    Bootstrap confidence interval for Pearson r.

    Resample rollout rows with replacement, recompute Pearson r, and return percentile CI.
    If a bootstrap sample has no variation in x or y, skip that sample.
    """
    df = _valid_xy(x, y)
    n = len(df)
    if n < 3 or df["x"].nunique() < 2 or df["y"].nunique() < 2:
        return np.nan, np.nan, 0

    rng = np.random.default_rng(seed)
    x_arr = df["x"].to_numpy()
    y_arr = df["y"].to_numpy()

    boot_rs: List[float] = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        xb = x_arr[idx]
        yb = y_arr[idx]

        if len(np.unique(xb)) < 2 or len(np.unique(yb)) < 2:
            continue

        r = np.corrcoef(xb, yb)[0, 1]
        if np.isfinite(r):
            boot_rs.append(float(r))

    if not boot_rs:
        return np.nan, np.nan, 0

    lower_q = (1.0 - ci) / 2.0
    upper_q = 1.0 - lower_q
    lo = float(np.quantile(boot_rs, lower_q))
    hi = float(np.quantile(boot_rs, upper_q))
    return lo, hi, len(boot_rs)


def compute_summary(df: pd.DataFrame, harm_threshold: float) -> pd.DataFrame:
    rows = []
    for baseline, g in df.groupby("baseline"):
        scored = g[g["score"].notna()].copy()
        scores = scored["score"]

        row = {
            "baseline": baseline,
            "n_rows": len(g),
            "n_scored": int(scores.notna().sum()),
            "n_unscored": int(g["score"].isna().sum()),
            "n_simulation_errors": int(g["error"].notna().sum()),
            "n_score_errors": int(g["score_error"].notna().sum()),
            "mean_harm": float(scores.mean()) if len(scored) else np.nan,
            "median_harm": float(scores.median()) if len(scored) else np.nan,
            "std_harm": float(scores.std(ddof=1)) if len(scored) > 1 else np.nan,
            "min_harm": float(scores.min()) if len(scored) else np.nan,
            "max_harm": float(scores.max()) if len(scored) else np.nan,
            "p25_harm": float(scores.quantile(0.25)) if len(scored) else np.nan,
            "p75_harm": float(scores.quantile(0.75)) if len(scored) else np.nan,
            "p90_harm": float(scores.quantile(0.90)) if len(scored) else np.nan,
            "p95_harm": float(scores.quantile(0.95)) if len(scored) else np.nan,
            "p99_harm": float(scores.quantile(0.99)) if len(scored) else np.nan,
            f"n_harm_gt_{harm_threshold}": int((scores > harm_threshold).sum()) if len(scored) else 0,
            f"frac_harm_gt_{harm_threshold}": float((scores > harm_threshold).mean()) if len(scored) else np.nan,
        }
        rows.append(row)

    return pd.DataFrame(rows).sort_values("baseline")


def compute_trait_correlations(
    df: pd.DataFrame,
    *,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    features = ["age", "num_active_traits"] + BOOLEAN_TRAIT_NAMES

    for baseline_idx, (baseline, g) in enumerate(df.groupby("baseline")):
        scored = g[g["score"].notna()].copy()

        for feature_idx, feature in enumerate(features):
            if feature not in scored.columns:
                continue

            x = scored[feature]
            y = scored["score"]

            r, p, n = pearson_corr_and_pvalue(x, y)
            spearman_r, spearman_p = spearman_corr_and_pvalue(x, y)

            fisher_lo, fisher_hi = fisher_pearson_ci(r, n)
            boot_lo, boot_hi, n_boot_valid = bootstrap_pearson_ci(
                x,
                y,
                n_bootstrap=n_bootstrap,
                seed=seed + baseline_idx * 10_000 + feature_idx,
            )

            active_n = np.nan
            inactive_n = np.nan
            active_mean = np.nan
            inactive_mean = np.nan
            mean_diff_active_minus_inactive = np.nan

            if feature in BOOLEAN_TRAIT_NAMES:
                active = scored[scored[feature] >= 0.5]["score"]
                inactive = scored[scored[feature] < 0.5]["score"]
                active_n = int(len(active))
                inactive_n = int(len(inactive))
                active_mean = float(active.mean()) if len(active) else np.nan
                inactive_mean = float(inactive.mean()) if len(inactive) else np.nan
                if len(active) and len(inactive):
                    mean_diff_active_minus_inactive = active_mean - inactive_mean

            rows.append(
                {
                    "baseline": baseline,
                    "feature": feature,
                    "n_for_corr": n,
                    "pearson_corr": r,
                    "pearson_p_value": p,
                    "pearson_fisher_ci_low": fisher_lo,
                    "pearson_fisher_ci_high": fisher_hi,
                    "pearson_bootstrap_ci_low": boot_lo,
                    "pearson_bootstrap_ci_high": boot_hi,
                    "bootstrap_valid_samples": n_boot_valid,
                    "spearman_corr": spearman_r,
                    "spearman_p_value": spearman_p,
                    "active_n": active_n,
                    "inactive_n": inactive_n,
                    "active_mean_harm": active_mean,
                    "inactive_mean_harm": inactive_mean,
                    "mean_diff_active_minus_inactive": mean_diff_active_minus_inactive,
                }
            )

    out = pd.DataFrame(rows)
    if not out.empty:
        out["abs_pearson_corr"] = out["pearson_corr"].abs()
        out["pearson_ci_crosses_zero"] = (
            (out["pearson_bootstrap_ci_low"] <= 0) & (out["pearson_bootstrap_ci_high"] >= 0)
        )
        out = out.sort_values(["baseline", "abs_pearson_corr"], ascending=[True, False])
    return out


def save_boxplot(df: pd.DataFrame, out_dir: Path) -> None:
    scored = df[df["score"].notna()].copy()
    labels = list(scored["baseline"].drop_duplicates())
    data = [scored[scored["baseline"] == label]["score"].values for label in labels]

    plt.figure(figsize=(max(8, 1.6 * len(labels)), 6))
    plt.boxplot(data, labels=labels, showmeans=True)
    plt.ylabel("WildGuard harm score")
    plt.title("Harm score distribution by baseline")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "boxplot_harm_scores.png", dpi=200)
    plt.close()


def save_histogram(df: pd.DataFrame, out_dir: Path) -> None:
    scored = df[df["score"].notna()].copy()
    labels = list(scored["baseline"].drop_duplicates())

    plt.figure(figsize=(9, 6))
    for label in labels:
        scores = scored[scored["baseline"] == label]["score"].values
        plt.hist(scores, bins=40, alpha=0.5, label=label)

    plt.xlabel("WildGuard harm score")
    plt.ylabel("Number of rollouts")
    plt.title("Harm score histogram")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "histogram_harm_scores.png", dpi=200)
    plt.close()


def save_correlation_plots(corr_df: pd.DataFrame, out_dir: Path, top_k: int) -> None:
    if corr_df.empty:
        return

    for baseline, g in corr_df.groupby("baseline"):
        # Mostly plot boolean traits; age/num_active_traits go into CSV and printed stats.
        bg = g[g["feature"].isin(BOOLEAN_TRAIT_NAMES)].copy()
        bg = bg.dropna(subset=["pearson_corr"])
        bg = bg.reindex(bg["pearson_corr"].abs().sort_values(ascending=False).index).head(top_k)
        bg = bg.sort_values("pearson_corr")

        if bg.empty:
            continue

        plt.figure(figsize=(10, max(6, 0.35 * len(bg))))
        plt.barh(bg["feature"], bg["pearson_corr"])
        plt.axvline(0.0, linewidth=1)
        plt.xlabel("Pearson correlation with harm score")
        plt.title(f"Top trait correlations: {baseline}")
        plt.tight_layout()
        safe_baseline = re.sub(r"[^A-Za-z0-9_.-]+", "_", baseline)
        plt.savefig(out_dir / f"trait_correlations_{safe_baseline}.png", dpi=200)
        plt.close()


def save_correlation_ci_plots(corr_df: pd.DataFrame, out_dir: Path, top_k: int) -> None:
    if corr_df.empty:
        return

    for baseline, g in corr_df.groupby("baseline"):
        bg = g[g["feature"].isin(BOOLEAN_TRAIT_NAMES)].copy()
        bg = bg.dropna(subset=["pearson_corr", "pearson_bootstrap_ci_low", "pearson_bootstrap_ci_high"])
        bg = bg.reindex(bg["pearson_corr"].abs().sort_values(ascending=False).index).head(top_k)
        bg = bg.sort_values("pearson_corr")

        if bg.empty:
            continue

        y_pos = np.arange(len(bg))
        r = bg["pearson_corr"].to_numpy()
        ci_low = bg["pearson_bootstrap_ci_low"].to_numpy()
        ci_high = bg["pearson_bootstrap_ci_high"].to_numpy()

        xerr = np.vstack([r - ci_low, ci_high - r])
        xerr = np.maximum(xerr, 0)

        plt.figure(figsize=(10, max(6, 0.4 * len(bg))))
        plt.errorbar(r, y_pos, xerr=xerr, fmt="o", capsize=3)
        plt.axvline(0.0, linewidth=1)
        plt.yticks(y_pos, bg["feature"])
        plt.xlabel("Pearson correlation with 95% bootstrap CI")
        plt.title(f"Trait correlation uncertainty: {baseline}")
        plt.tight_layout()
        safe_baseline = re.sub(r"[^A-Za-z0-9_.-]+", "_", baseline)
        plt.savefig(out_dir / f"trait_correlations_ci_{safe_baseline}.png", dpi=200)
        plt.close()


def write_top_high_harm_jsonl(
    raw_by_label: Dict[str, List[Dict[str, Any]]],
    out_dir: Path,
    harm_threshold: float,
    top_n: int,
) -> None:
    for label, raw_records in raw_by_label.items():
        scored = []
        for rec in raw_records:
            score = rec.get("score", None)
            if score is None:
                continue
            try:
                score_f = float(score)
            except (TypeError, ValueError):
                continue
            if score_f > harm_threshold:
                scored.append((score_f, rec))

        scored.sort(key=lambda x: x[0], reverse=True)
        safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", label)
        out_path = out_dir / f"top_high_harm_{safe_label}.jsonl"

        with open(out_path, "w", encoding="utf-8") as f:
            for _, rec in scored[:top_n]:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def print_human_summary(summary_df: pd.DataFrame, corr_df: pd.DataFrame, harm_threshold: float, top_k: int) -> None:
    print("\n=== Baseline summary ===")
    cols = [
        "baseline",
        "n_scored",
        "mean_harm",
        "median_harm",
        "std_harm",
        "min_harm",
        "max_harm",
        "p95_harm",
        f"n_harm_gt_{harm_threshold}",
        f"frac_harm_gt_{harm_threshold}",
    ]
    cols = [c for c in cols if c in summary_df.columns]
    print(summary_df[cols].to_string(index=False))

    if corr_df.empty:
        return

    print("\n=== Top absolute trait correlations with harm score ===")
    for baseline, g in corr_df.groupby("baseline"):
        print(f"\n[{baseline}]")
        show = g[g["feature"].isin(BOOLEAN_TRAIT_NAMES)].copy()
        show = show.dropna(subset=["pearson_corr"])
        show = show.reindex(show["pearson_corr"].abs().sort_values(ascending=False).index)
        show = show.head(top_k)
        print(
            show[
                [
                    "feature",
                    "n_for_corr",
                    "pearson_corr",
                    "pearson_p_value",
                    "pearson_bootstrap_ci_low",
                    "pearson_bootstrap_ci_high",
                    "pearson_ci_crosses_zero",
                    "active_mean_harm",
                    "inactive_mean_harm",
                    "mean_diff_active_minus_inactive",
                    "active_n",
                    "inactive_n",
                ]
            ].to_string(index=False)
        )


def parse_file_args(file_args: Sequence[str]) -> List[Tuple[str, str]]:
    """
    Accept either:
      --files llama_baseline.jsonl qwen_baseline.jsonl
    or:
      --files llama=llama_baseline.jsonl qwen=qwen_baseline.jsonl
    """
    parsed = []
    for item in file_args:
        if "=" in item:
            label, path = item.split("=", 1)
            label = label.strip()
            path = path.strip()
        else:
            path = item
            label = safe_label_from_path(path)
        parsed.append((label, path))
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--files",
        nargs="+",
        required=True,
        help="JSONL baseline files. Use either paths or label=path pairs.",
    )
    parser.add_argument("--out-dir", type=str, default="baseline_analysis")
    parser.add_argument("--harm-threshold", type=float, default=0.4)
    parser.add_argument("--top-k-correlations", type=int, default=15)
    parser.add_argument("--top-n-high-harm", type=int, default=25)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dfs = []
    raw_by_label: Dict[str, List[Dict[str, Any]]] = {}

    for label, path in parse_file_args(args.files):
        df, raw = load_jsonl(path, label=label)
        if df.empty:
            print(f"[WARN] No records loaded for {label}: {path}")
            continue
        dfs.append(df)
        raw_by_label[label] = raw

    if not dfs:
        raise RuntimeError("No valid records loaded. Check your --files paths.")

    all_df = pd.concat(dfs, ignore_index=True)

    summary_df = compute_summary(all_df, harm_threshold=args.harm_threshold)
    corr_df = compute_trait_correlations(
        all_df,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )

    all_df.to_csv(out_dir / "all_rollouts_flat.csv", index=False)
    summary_df.to_csv(out_dir / "summary_metrics.csv", index=False)
    corr_df.to_csv(out_dir / "trait_correlations.csv", index=False)

    save_boxplot(all_df, out_dir)
    save_histogram(all_df, out_dir)
    save_correlation_plots(corr_df, out_dir, top_k=args.top_k_correlations)
    save_correlation_ci_plots(corr_df, out_dir, top_k=args.top_k_correlations)
    write_top_high_harm_jsonl(
        raw_by_label=raw_by_label,
        out_dir=out_dir,
        harm_threshold=args.harm_threshold,
        top_n=args.top_n_high_harm,
    )

    print_human_summary(
        summary_df=summary_df,
        corr_df=corr_df,
        harm_threshold=args.harm_threshold,
        top_k=args.top_k_correlations,
    )

    if pearsonr is None or spearmanr is None:
        print(
            "\n[WARN] scipy is not installed, so p-values may be NaN. "
            "Install with: python -m pip install scipy"
        )

    print("\n=== Wrote outputs ===")
    for p in sorted(out_dir.iterdir()):
        print(p)


if __name__ == "__main__":
    main()
