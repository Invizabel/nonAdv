#!/usr/bin/env python3
"""
Analyze a CEM experiment from:
1) rollout JSON/JSONL file with one record per rollout
2) trait distribution JSON/JSONL file with one record per CEM round

Outputs:
- harm_by_round.png
- trait_distribution_means_every_10_rounds.png
- age_distribution_every_10_rounds.png
- pearson_correlations_with_bootstrap_ci.png
- harm_threshold_percent_by_round.png
- round_summary.csv
- pearson_correlations.csv

Example:
python analyze_cem_experiment.py \
  --rollouts rollouts.json \
  --distributions trait_distribs.json \
  --outdir cem_analysis \
  --elite-k 20
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_records(path: str):
    """
    Supports either:
    - JSON list: [{...}, {...}]
    - JSONL: one JSON object per line
    """
    path = Path(path)
    text = path.read_text()

    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            # Handles cases like {"records": [...]} if you ever save that way.
            for key in ["records", "data", "rollouts", "distributions", "trait_distributions"]:
                if key in data and isinstance(data[key], list):
                    return data[key]
            return [data]
    except json.JSONDecodeError:
        pass

    records = []
    for i, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise ValueError(f"Could not parse line {i} in {path}: {e}") from e

    return records


def infer_score_key(records):
    for key in ["score", "harm_score", "harmScore", "max_score"]:
        if key in records[0]:
            return key
    raise KeyError(
        "Could not find score key. Expected one of: "
        "'score', 'harm_score', 'harmScore', 'max_score'."
    )


def build_distribution_df(distribution_records):
    rows = []

    for rec in distribution_records:
        round_id = int(rec["round"])

        age = rec.get("age", {})
        if isinstance(age, dict):
            rows.append({
                "round": round_id,
                "trait": "age",
                "mean": age.get("mean", np.nan),
                "std": age.get("std", np.nan),
                "alpha": np.nan,
                "beta": np.nan,
                "variance": np.nan,
            })

        for bt in rec.get("beta_traits", []):
            rows.append({
                "round": round_id,
                "trait": bt["trait"],
                "mean": bt.get("mean", np.nan),
                "std": np.nan,
                "alpha": bt.get("alpha", np.nan),
                "beta": bt.get("beta", np.nan),
                "variance": bt.get("variance", np.nan),
            })

    return pd.DataFrame(rows)


def build_rollout_df(rollout_records, trait_names, score_key):
    rows = []

    expected_len = 1 + len(trait_names)

    for rec in rollout_records:
        trait_vector = rec.get("trait_vector", None)
        if trait_vector is None:
            raise KeyError("Each rollout record must contain 'trait_vector'.")

        if len(trait_vector) != expected_len:
            raise ValueError(
                f"Trait vector length mismatch. Got {len(trait_vector)}, "
                f"expected {expected_len}: age + {len(trait_names)} traits."
            )

        row = {
            "round": int(rec["round"]),
            "rollout": int(rec.get("rollout", len(rows) + 1)),
            "score": float(rec[score_key]),
            "age": float(trait_vector[0]),
        }

        for name, val in zip(trait_names, trait_vector[1:]):
            row[name] = float(val)

        # Use explicit elite flag if your file has one. Otherwise we infer top-k later.
        if "elite" in rec:
            row["elite"] = bool(rec["elite"])
        elif "is_elite" in rec:
            row["elite"] = bool(rec["is_elite"])

        rows.append(row)

    return pd.DataFrame(rows)


def add_inferred_elites(df, elite_k):
    """
    If no elite flag exists, mark top-k score rollouts in each round as elites.
    """
    df = df.copy()

    if "elite" in df.columns:
        df["elite"] = df["elite"].astype(bool)
        return df

    df["elite"] = False

    for round_id, group in df.groupby("round"):
        idx = group.sort_values("score", ascending=False).head(elite_k).index
        df.loc[idx, "elite"] = True

    return df


def pearson_r(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 3:
        return np.nan

    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan

    return float(np.corrcoef(x, y)[0, 1])


def bootstrap_pearson_ci(x, y, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    n = len(x)
    r_hat = pearson_r(x, y)

    if n < 3 or not np.isfinite(r_hat):
        return r_hat, np.nan, np.nan

    boot_rs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        r = pearson_r(x[idx], y[idx])
        if np.isfinite(r):
            boot_rs.append(r)

    if len(boot_rs) == 0:
        return r_hat, np.nan, np.nan

    lo, hi = np.percentile(boot_rs, [2.5, 97.5])
    return r_hat, float(lo), float(hi)


def plot_harm_by_round(df, outpath):
    summary = (
        df.groupby("round")
        .agg(
            mean_score=("score", "mean"),
            median_score=("score", "median"),
            max_score=("score", "max"),
            n=("score", "size"),
        )
        .reset_index()
    )

    elite_summary = (
        df[df["elite"]]
        .groupby("round")
        .agg(mean_elite_score=("score", "mean"))
        .reset_index()
    )

    summary = summary.merge(elite_summary, on="round", how="left")

    plt.figure(figsize=(10, 6))
    plt.plot(summary["round"], summary["mean_score"], marker="o", label="Mean harm score")
    plt.plot(summary["round"], summary["mean_elite_score"], marker="o", label="Mean elite harm score")
    plt.xlabel("CEM round")
    plt.ylabel("Harm score")
    plt.title("Mean harm score and mean elite harm score by CEM round")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

    return summary


def select_rounds(rounds, step=10):
    rounds = sorted(int(r) for r in rounds)
    selected = [r for r in rounds if r == 1 or r % step == 0]

    # Always include the final round.
    if rounds[-1] not in selected:
        selected.append(rounds[-1])

    return sorted(set(selected))


def plot_trait_distribution_means(dist_df, outpath, round_step=10):
    beta_df = dist_df[dist_df["trait"] != "age"].copy()
    selected = select_rounds(beta_df["round"].unique(), step=round_step)

    pivot = (
        beta_df[beta_df["round"].isin(selected)]
        .pivot(index="trait", columns="round", values="mean")
    )

    # Keep traits in a stable readable order.
    pivot = pivot.loc[pivot.index.tolist()]

    plt.figure(figsize=(max(9, 0.55 * len(selected) + 5), max(8, 0.35 * len(pivot))))
    im = plt.imshow(pivot.values, aspect="auto", vmin=0, vmax=1)
    plt.colorbar(im, label="Beta distribution mean / probability trait is sampled")
    plt.xticks(np.arange(len(pivot.columns)), pivot.columns)
    plt.yticks(np.arange(len(pivot.index)), pivot.index)
    plt.xlabel("CEM round")
    plt.ylabel("Trait")
    plt.title(f"Trait distribution means every {round_step} rounds")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_age_distribution(dist_df, outpath, round_step=10):
    age_df = dist_df[dist_df["trait"] == "age"].copy()
    selected = select_rounds(age_df["round"].unique(), step=round_step)
    age_df = age_df[age_df["round"].isin(selected)].sort_values("round")

    plt.figure(figsize=(10, 6))
    plt.errorbar(
        age_df["round"],
        age_df["mean"],
        yerr=age_df["std"],
        marker="o",
        capsize=4,
        label="Age mean ± std",
    )
    plt.xlabel("CEM round")
    plt.ylabel("Age")
    plt.title(f"Age distribution every {round_step} rounds")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_pearson_correlations(df, trait_columns, outpath, csv_outpath, n_boot=2000, seed=0):
    rows = []

    for i, trait in enumerate(trait_columns):
        r, lo, hi = bootstrap_pearson_ci(
            df[trait].values,
            df["score"].values,
            n_boot=n_boot,
            seed=seed + i,
        )
        rows.append({
            "trait": trait,
            "pearson_r": r,
            "ci_low": lo,
            "ci_high": hi,
            "n": int(df[[trait, "score"]].dropna().shape[0]),
        })

    corr_df = pd.DataFrame(rows)
    corr_df = corr_df.sort_values("pearson_r", ascending=True, na_position="first")
    corr_df.to_csv(csv_outpath, index=False)

    plot_df = corr_df.dropna(subset=["pearson_r"]).copy()

    y = np.arange(len(plot_df))
    x = plot_df["pearson_r"].values
    xerr_low = x - plot_df["ci_low"].values
    xerr_high = plot_df["ci_high"].values - x

    plt.figure(figsize=(10, max(7, 0.35 * len(plot_df))))
    plt.errorbar(
        x,
        y,
        xerr=[xerr_low, xerr_high],
        fmt="o",
        capsize=3,
    )
    plt.axvline(0, linestyle="--", linewidth=1)
    plt.yticks(y, plot_df["trait"])
    plt.xlabel("Pearson correlation with harm score")
    plt.ylabel("Trait")
    plt.title(f"Trait-harm Pearson correlations with 95% bootstrap CIs, n_boot={n_boot}")
    plt.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

    return corr_df


def plot_threshold_percentages(df, outpath):
    thresholds = np.round(np.arange(0.1, 1.01, 0.1), 1)

    rows = []
    for round_id, group in df.groupby("round"):
        scores = group["score"].values
        n = len(scores)
        row = {"round": int(round_id)}
        for t in thresholds:
            row[f">{t:.1f}"] = 100.0 * np.mean(scores > t) if n else np.nan
        rows.append(row)

    thresh_df = pd.DataFrame(rows).sort_values("round")

    plt.figure(figsize=(11, 7))
    for t in thresholds:
        col = f">{t:.1f}"
        plt.plot(thresh_df["round"], thresh_df[col], marker="o", label=col)

    plt.xlabel("CEM round")
    plt.ylabel("% of rollouts above threshold")
    plt.title("Percent of rollouts with harm score above each threshold")
    plt.ylim(-1, 101)
    plt.grid(True, alpha=0.3)
    plt.legend(title="Score threshold", ncol=2)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

    return thresh_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", required=True, help="Path to rollout JSON or JSONL file.")
    parser.add_argument("--distributions", required=True, help="Path to trait distribution JSON or JSONL file.")
    parser.add_argument("--outdir", default="cem_analysis", help="Directory to save plots and CSVs.")
    parser.add_argument("--elite-k", type=int, default=20, help="Top-k rollouts per round treated as elites if no elite flag exists.")
    parser.add_argument("--round-step", type=int, default=10, help="Show trait distributions every N rounds.")
    parser.add_argument("--n-boot", type=int, default=2000, help="Bootstrap resamples for Pearson 95% CIs.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for bootstrap.")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rollout_records = load_records(args.rollouts)
    distribution_records = load_records(args.distributions)

    if len(rollout_records) == 0:
        raise ValueError("No rollout records found.")
    if len(distribution_records) == 0:
        raise ValueError("No distribution records found.")

    dist_df = build_distribution_df(distribution_records)

    trait_names = (
        dist_df[dist_df["trait"] != "age"]["trait"]
        .drop_duplicates()
        .tolist()
    )

    score_key = infer_score_key(rollout_records)
    rollout_df = build_rollout_df(rollout_records, trait_names, score_key)
    rollout_df = add_inferred_elites(rollout_df, elite_k=args.elite_k)

    trait_columns = ["age"] + trait_names

    round_summary = plot_harm_by_round(
        rollout_df,
        outdir / "harm_by_round.png",
    )
    round_summary.to_csv(outdir / "round_summary.csv", index=False)

    plot_trait_distribution_means(
        dist_df,
        outdir / "trait_distribution_means_every_10_rounds.png",
        round_step=args.round_step,
    )

    plot_age_distribution(
        dist_df,
        outdir / "age_distribution_every_10_rounds.png",
        round_step=args.round_step,
    )

    corr_df = plot_pearson_correlations(
        rollout_df,
        trait_columns,
        outdir / "pearson_correlations_with_bootstrap_ci.png",
        outdir / "pearson_correlations.csv",
        n_boot=args.n_boot,
        seed=args.seed,
    )

    threshold_df = plot_threshold_percentages(
        rollout_df,
        outdir / "harm_threshold_percent_by_round.png",
    )
    threshold_df.to_csv(outdir / "harm_threshold_percent_by_round.csv", index=False)

    print(f"Loaded {len(rollout_df)} rollouts.")
    print(f"Loaded {len(distribution_records)} distribution rounds.")
    print(f"Detected {len(trait_names)} boolean traits plus age.")
    print(f"Using score key: {score_key}")
    print(f"Elite definition: explicit elite/is_elite flag if present, otherwise top {args.elite_k} scores per round.")
    print()
    print("Saved outputs to:", outdir.resolve())
    print("- harm_by_round.png")
    print("- trait_distribution_means_every_10_rounds.png")
    print("- age_distribution_every_10_rounds.png")
    print("- pearson_correlations_with_bootstrap_ci.png")
    print("- harm_threshold_percent_by_round.png")
    print("- round_summary.csv")
    print("- pearson_correlations.csv")
    print("- harm_threshold_percent_by_round.csv")


if __name__ == "__main__":
    main()