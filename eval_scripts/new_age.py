"""
Plot and quantify the relationship between age and harm score.

Works with one or more JSONL baseline files, e.g.:
  - llama_baseline.jsonl
  - qwen_baseline.jsonl

Outputs:
  - age_harm_flat.csv
  - age_harm_summary.csv
  - age_harm_correlations.csv
  - age_bin_indicator_correlations.csv
  - age_vs_harm_scatter.png
  - age_bin_boxplot_<baseline>.png
  - age_bin_mean_harm.png
  - age_bin_correlations_ci_<baseline>.png

The age-bin correlation plot treats each 5-year age bin as a binary indicator:
  age in bin = 1
  age not in bin = 0

Then it computes Pearson correlation between that indicator and harm score, with
a 95% bootstrap confidence interval. This mirrors the boolean-trait correlation
plot, but uses age-bin membership instead of trait presence.
"""

from __future__ import annotations

import argparse
import json
import math
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


def safe_label_from_path(path: str) -> str:
    stem = Path(path).stem
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", stem)


def infer_age_from_traits(traits: Sequence[str]) -> float:
    for t in traits:
        m = re.search(r"(\d+(?:\.\d+)?)\s+years?\s+old", str(t))
        if m:
            return float(m.group(1))
    return np.nan


def parse_file_args(file_args: Sequence[str]) -> List[Tuple[str, str]]:
    parsed = []
    for item in file_args:
        if "=" in item:
            label, path = item.split("=", 1)
            parsed.append((label.strip(), path.strip()))
        else:
            parsed.append((safe_label_from_path(item), item))
    return parsed


def load_jsonl_age_score(path: str, label: Optional[str] = None) -> pd.DataFrame:
    label = label or safe_label_from_path(path)
    rows: List[Dict[str, Any]] = []

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

            score = rec.get("score", None)
            if score is None:
                continue

            try:
                score_f = float(score)
            except (TypeError, ValueError):
                continue

            age = np.nan
            trait_vector = rec.get("trait_vector", None)
            traits = rec.get("traits", []) or []

            if isinstance(trait_vector, list) and len(trait_vector) >= 1:
                try:
                    age = float(trait_vector[0])
                except (TypeError, ValueError):
                    age = np.nan

            if np.isnan(age):
                age = infer_age_from_traits(traits)

            if np.isnan(age):
                continue

            rows.append(
                {
                    "baseline": label,
                    "source_file": path,
                    "line_idx": line_idx,
                    "rollout": rec.get("rollout", line_idx),
                    "age": age,
                    "age_rounded": int(round(age)),
                    "score": score_f,
                    "traits_text": "; ".join(traits),
                }
            )

    return pd.DataFrame(rows)


def _valid_xy(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def pearson_corr_and_pvalue(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, int]:
    x, y = _valid_xy(x, y)
    n = len(x)
    if n < 3 or len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return np.nan, np.nan, n

    if pearsonr is not None:
        res = pearsonr(x, y)
        return float(res.statistic), float(res.pvalue), n

    return float(np.corrcoef(x, y)[0, 1]), np.nan, n


def spearman_corr_and_pvalue(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    x, y = _valid_xy(x, y)
    if len(x) < 3 or len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return np.nan, np.nan

    if spearmanr is not None:
        res = spearmanr(x, y)
        return float(res.statistic), float(res.pvalue)

    return float(pd.Series(x).corr(pd.Series(y), method="spearman")), np.nan


def bootstrap_corr_ci(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_bootstrap: int = 1000,
    seed: int = 0,
    ci: float = 0.95,
) -> Tuple[float, float, int]:
    x, y = _valid_xy(x, y)
    rng = np.random.default_rng(seed)
    n = len(x)

    if n < 3 or len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return np.nan, np.nan, 0

    rs: List[float] = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        xb = x[idx]
        yb = y[idx]

        if len(np.unique(xb)) < 2 or len(np.unique(yb)) < 2:
            continue

        r = np.corrcoef(xb, yb)[0, 1]
        if np.isfinite(r):
            rs.append(float(r))

    if not rs:
        return np.nan, np.nan, 0

    lower_q = (1.0 - ci) / 2.0
    upper_q = 1.0 - lower_q
    return float(np.quantile(rs, lower_q)), float(np.quantile(rs, upper_q)), len(rs)


def compute_correlations(df: pd.DataFrame, n_bootstrap: int, seed: int) -> pd.DataFrame:
    """
    Continuous age-vs-harm correlation.
    """
    rows = []

    for baseline_idx, (baseline, g) in enumerate(df.groupby("baseline")):
        g = g.dropna(subset=["age", "score"])
        x = g["age"].to_numpy()
        y = g["score"].to_numpy()

        pearson_r, pearson_p, n = pearson_corr_and_pvalue(x, y)
        spearman_r, spearman_p = spearman_corr_and_pvalue(x, y)

        boot_low, boot_high, boot_n = bootstrap_corr_ci(
            x,
            y,
            n_bootstrap=n_bootstrap,
            seed=seed + baseline_idx * 10000,
        )

        rows.append(
            {
                "baseline": baseline,
                "n": int(n),
                "mean_age": float(g["age"].mean()),
                "min_age": float(g["age"].min()),
                "max_age": float(g["age"].max()),
                "pearson_corr_age_harm": pearson_r,
                "pearson_p_value": pearson_p,
                "pearson_bootstrap_ci_low": boot_low,
                "pearson_bootstrap_ci_high": boot_high,
                "bootstrap_valid_samples": boot_n,
                "spearman_corr_age_harm": spearman_r,
                "spearman_p_value": spearman_p,
            }
        )

    return pd.DataFrame(rows)


def add_age_bins(df: pd.DataFrame, bin_width: int) -> pd.DataFrame:
    out = df.copy()
    min_age = math.floor(out["age"].min() / bin_width) * bin_width
    max_age = math.ceil(out["age"].max() / bin_width) * bin_width
    bins = list(range(int(min_age), int(max_age + bin_width), bin_width))
    if len(bins) < 2:
        bins = [int(min_age), int(min_age + bin_width)]

    labels = [f"{bins[i]}-{bins[i + 1] - 1}" for i in range(len(bins) - 1)]
    out["age_bin"] = pd.cut(
        out["age"],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=False,
    )
    return out


def compute_age_bin_summary(df: pd.DataFrame, bin_width: int) -> pd.DataFrame:
    binned = add_age_bins(df, bin_width)
    rows = []

    for baseline, g in binned.groupby("baseline"):
        for age_bin, bg in g.groupby("age_bin", observed=True):
            if len(bg) == 0:
                continue
            rows.append(
                {
                    "baseline": baseline,
                    "age_bin": str(age_bin),
                    "n": int(len(bg)),
                    "mean_harm": float(bg["score"].mean()),
                    "median_harm": float(bg["score"].median()),
                    "std_harm": float(bg["score"].std(ddof=1)) if len(bg) > 1 else np.nan,
                    "min_harm": float(bg["score"].min()),
                    "max_harm": float(bg["score"].max()),
                }
            )

    return pd.DataFrame(rows)


def compute_age_bin_indicator_correlations(
    df: pd.DataFrame,
    *,
    bin_width: int,
    n_bootstrap: int,
    seed: int,
    min_bin_n: int,
) -> pd.DataFrame:
    """
    For each age bin, compute Pearson correlation between:
      x = 1 if rollout age is in this bin else 0
      y = harm score

    This is analogous to boolean trait correlation, where each age bin is treated
    as a binary feature.
    """
    binned = add_age_bins(df, bin_width)
    rows = []

    for baseline_idx, (baseline, g) in enumerate(binned.groupby("baseline")):
        g = g.dropna(subset=["age_bin", "score"]).copy()
        y = g["score"].to_numpy()

        age_bins = list(g["age_bin"].dropna().cat.categories)
        for bin_idx, age_bin in enumerate(age_bins):
            age_bin_str = str(age_bin)
            in_bin = (g["age_bin"].astype(str) == age_bin_str).astype(float).to_numpy()

            active_scores = g.loc[in_bin >= 0.5, "score"]
            inactive_scores = g.loc[in_bin < 0.5, "score"]

            active_n = int(len(active_scores))
            inactive_n = int(len(inactive_scores))

            if active_n < min_bin_n or inactive_n < min_bin_n:
                r = p = boot_low = boot_high = np.nan
                boot_n = 0
                n = active_n + inactive_n
            else:
                r, p, n = pearson_corr_and_pvalue(in_bin, y)
                boot_low, boot_high, boot_n = bootstrap_corr_ci(
                    in_bin,
                    y,
                    n_bootstrap=n_bootstrap,
                    seed=seed + baseline_idx * 10000 + bin_idx,
                )

            active_mean = float(active_scores.mean()) if active_n else np.nan
            inactive_mean = float(inactive_scores.mean()) if inactive_n else np.nan
            mean_diff = active_mean - inactive_mean if active_n and inactive_n else np.nan

            rows.append(
                {
                    "baseline": baseline,
                    "age_bin": age_bin_str,
                    "n_for_corr": int(active_n + inactive_n),
                    "bin_n": active_n,
                    "not_bin_n": inactive_n,
                    "pearson_corr": r,
                    "pearson_p_value": p,
                    "pearson_bootstrap_ci_low": boot_low,
                    "pearson_bootstrap_ci_high": boot_high,
                    "bootstrap_valid_samples": boot_n,
                    "bin_mean_harm": active_mean,
                    "not_bin_mean_harm": inactive_mean,
                    "mean_diff_bin_minus_not_bin": mean_diff,
                    "ci_crosses_zero": (
                        bool(boot_low <= 0 <= boot_high)
                        if np.isfinite(boot_low) and np.isfinite(boot_high)
                        else np.nan
                    ),
                }
            )

    return pd.DataFrame(rows)


def save_scatter(df: pd.DataFrame, out_dir: Path, add_trend: bool) -> None:
    labels = list(df["baseline"].drop_duplicates())

    plt.figure(figsize=(10, 6))
    for label in labels:
        g = df[df["baseline"] == label]
        plt.scatter(g["age"], g["score"], alpha=0.45, label=label)

        if add_trend and len(g) >= 2 and g["age"].nunique() >= 2:
            x = g["age"].to_numpy()
            y = g["score"].to_numpy()
            coef = np.polyfit(x, y, deg=1)
            x_line = np.linspace(x.min(), x.max(), 100)
            y_line = coef[0] * x_line + coef[1]
            plt.plot(x_line, y_line, linewidth=2)

    plt.xlabel("Age")
    plt.ylabel("WildGuard harm score")
    plt.title("Age vs harm score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "age_vs_harm_scatter.png", dpi=200)
    plt.close()


def save_age_bin_boxplot(df: pd.DataFrame, out_dir: Path, bin_width: int) -> None:
    binned = add_age_bins(df, bin_width)

    for baseline, g in binned.groupby("baseline"):
        bins = [str(x) for x in g["age_bin"].dropna().unique()]
        data = [g[g["age_bin"].astype(str) == b]["score"].values for b in bins]
        if not data:
            continue

        plt.figure(figsize=(max(9, 0.7 * len(bins)), 6))
        plt.boxplot(data, labels=bins, showmeans=True)
        plt.xlabel("Age bin")
        plt.ylabel("WildGuard harm score")
        plt.title(f"Harm score by age bin: {baseline}")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", baseline)
        plt.savefig(out_dir / f"age_bin_boxplot_{safe}.png", dpi=200)
        plt.close()


def save_age_bin_mean_plot(bin_summary: pd.DataFrame, out_dir: Path) -> None:
    if bin_summary.empty:
        return

    plt.figure(figsize=(10, 6))
    for baseline, g in bin_summary.groupby("baseline"):
        g = g.copy()
        g["age_bin"] = g["age_bin"].astype(str)
        plt.plot(g["age_bin"], g["mean_harm"], marker="o", label=baseline)

    plt.xlabel("Age bin")
    plt.ylabel("Mean WildGuard harm score")
    plt.title("Mean harm score by age bin")
    plt.xticks(rotation=30, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "age_bin_mean_harm.png", dpi=200)
    plt.close()


def save_age_bin_corr_ci_plots(age_bin_corr: pd.DataFrame, out_dir: Path) -> None:
    if age_bin_corr.empty:
        return

    for baseline, g in age_bin_corr.groupby("baseline"):
        g = g.copy()
        g = g.dropna(subset=["pearson_corr", "pearson_bootstrap_ci_low", "pearson_bootstrap_ci_high"])
        if g.empty:
            continue

        # Preserve numeric bin order.
        def bin_start(bin_label: str) -> int:
            m = re.match(r"(-?\d+)-", str(bin_label))
            return int(m.group(1)) if m else 0

        g["bin_start"] = g["age_bin"].map(bin_start)
        g = g.sort_values("bin_start")

        y_pos = np.arange(len(g))
        r = g["pearson_corr"].to_numpy()
        ci_low = g["pearson_bootstrap_ci_low"].to_numpy()
        ci_high = g["pearson_bootstrap_ci_high"].to_numpy()
        xerr = np.vstack([r - ci_low, ci_high - r])
        xerr = np.maximum(xerr, 0)

        plt.figure(figsize=(10, max(6, 0.38 * len(g))))
        plt.errorbar(r, y_pos, xerr=xerr, fmt="o", capsize=3)
        plt.axvline(0.0, linewidth=1)
        plt.yticks(y_pos, g["age_bin"])
        plt.xlabel("Pearson correlation with 95% bootstrap CI")
        plt.ylabel("Age bin")
        plt.title(f"Age-bin correlation uncertainty: {baseline}")
        plt.tight_layout()

        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", baseline)
        plt.savefig(out_dir / f"age_bin_correlations_ci_{safe}.png", dpi=200)
        plt.close()


def print_summary(
    corr_df: pd.DataFrame,
    bin_summary: pd.DataFrame,
    age_bin_corr: pd.DataFrame,
) -> None:
    print("\n=== Continuous age-harm correlations ===")
    print(corr_df.to_string(index=False))

    print("\n=== Mean harm by age bin ===")
    if bin_summary.empty:
        print("[No age-bin summary available]")
    else:
        print(bin_summary.to_string(index=False))

    print("\n=== Age-bin indicator correlations ===")
    if age_bin_corr.empty:
        print("[No age-bin correlation summary available]")
    else:
        cols = [
            "baseline",
            "age_bin",
            "bin_n",
            "pearson_corr",
            "pearson_p_value",
            "pearson_bootstrap_ci_low",
            "pearson_bootstrap_ci_high",
            "ci_crosses_zero",
            "bin_mean_harm",
            "not_bin_mean_harm",
            "mean_diff_bin_minus_not_bin",
        ]
        print(age_bin_corr[cols].to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--files",
        nargs="+",
        required=True,
        help="JSONL baseline files. Use paths or label=path pairs.",
    )
    parser.add_argument("--out-dir", type=str, default="age_harm_analysis")
    parser.add_argument("--age-bin-width", type=int, default=5)
    parser.add_argument("--min-bin-n", type=int, default=5)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-trend", action="store_true")

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dfs = []
    for label, path in parse_file_args(args.files):
        df = load_jsonl_age_score(path, label=label)
        if df.empty:
            print(f"[WARN] No scored age records loaded for {label}: {path}")
            continue
        dfs.append(df)

    if not dfs:
        raise RuntimeError("No valid scored age records found.")

    all_df = pd.concat(dfs, ignore_index=True)
    bin_summary = compute_age_bin_summary(all_df, bin_width=args.age_bin_width)
    corr_df = compute_correlations(all_df, n_bootstrap=args.n_bootstrap, seed=args.seed)
    age_bin_corr = compute_age_bin_indicator_correlations(
        all_df,
        bin_width=args.age_bin_width,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        min_bin_n=args.min_bin_n,
    )

    all_df.to_csv(out_dir / "age_harm_flat.csv", index=False)
    bin_summary.to_csv(out_dir / "age_harm_summary.csv", index=False)
    corr_df.to_csv(out_dir / "age_harm_correlations.csv", index=False)
    age_bin_corr.to_csv(out_dir / "age_bin_indicator_correlations.csv", index=False)

    save_scatter(all_df, out_dir, add_trend=not args.no_trend)
    save_age_bin_boxplot(all_df, out_dir, bin_width=args.age_bin_width)
    save_age_bin_mean_plot(bin_summary, out_dir)
    save_age_bin_corr_ci_plots(age_bin_corr, out_dir)

    print_summary(corr_df, bin_summary, age_bin_corr)

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
