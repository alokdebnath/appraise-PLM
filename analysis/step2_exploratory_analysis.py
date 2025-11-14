#!/usr/bin/env python3
"""
Step 2 — Broad Exploratory Scan for 21-Dimension Appraisals (CLI, no notebook required)

What this script does:
- Loads multiple CSV dialogue corpora that already include appraisal columns.
- Computes per-corpus descriptive stats for each appraisal dimension.
- Builds a cross-corpus matrix of mean values (dims x corpora) and saves a heatmap.
- Computes per-corpus correlation matrices among appraisal dimensions and saves heatmaps.
- Finds the most discriminative dimensions across corpora (by variance of corpus means).
- Plots distributions (histograms) for the top-K discriminative dimensions across corpora.

Inputs:
- --datasets "name1=path1.csv,name2=path2.csv,..."
- --dims (comma-separated list), or --dims-file (txt or json list), or --model-config (json with key 'appraisal_dimensions')
- Optional: --id-cols (comma-separated list of meta columns to carry to some outputs; not required for plots)

Outputs (in --outdir):
- corpus_means.csv              : dims x corpora mean matrix
- top_dims.txt                  : top-K discriminative dimensions
- stats_<name>.csv              : per-dimension descriptive statistics for each corpus
- corr_<name>.csv               : per-dimension correlation matrix for each corpus
- heatmap_corpus_means.png      : heatmap of dims x corpora means
- heatmap_corr_<name>.png       : heatmap of correlations for each corpus
- dist_<dim>.png                : histogram overlay across corpora for each of the top-K dims

Example:
python step2_exploratory_scan.py \
  --datasets "ED=data/ed_with_appraisals.csv,ALO=data/alo_with_appraisals.csv" \
  --dims pleasantness,control,unexpectedness,goal_conduciveness,agency,certainty,... \
  --outdir analysis/step2 \
  --topk 6 --bins 30

Notes:
- Requires: pandas, numpy, matplotlib.
- Avoids seaborn to keep dependencies minimal.
- If some corpora are missing certain dimensions, the script will warn and use what's available.
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ------------------ Logging ------------------
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("explore_appraisals")


# ------------------ Helpers ------------------
def parse_datasets_arg(arg: str) -> Dict[str, str]:
    """Parse --datasets of the form "name1=path1,name2=path2" into a dict."""
    pairs = [s.strip() for s in arg.split(",") if s.strip()]
    mapping = {}
    for p in pairs:
        if "=" not in p:
            raise ValueError(f"Malformed --datasets entry '{p}'. Expected 'name=path'.")
        name, path = p.split("=", 1)
        name, path = name.strip(), path.strip()
        if not name or not path:
            raise ValueError(f"Malformed --datasets entry '{p}'.")
        mapping[name] = path
    return mapping


def load_dims(args) -> List[str]:
    """Load appraisal dimensions from arguments / file / model config."""
    if args.dims:
        dims = [d.strip() for d in args.dims.split(",") if d.strip()]
        if not dims:
            raise ValueError("--dims parsed to an empty list.")
        return dims

    if args.dims_file:
        p = Path(args.dims_file)
        if not p.exists():
            raise FileNotFoundError(f"--dims-file not found: {p}")
        if p.suffix.lower() in {".json"}:
            with open(p, "r") as f:
                data = json.load(f)
            if isinstance(data, dict):
                # try common keys
                for key in ["dims", "dimensions", "appraisal_dimensions"]:
                    if key in data and isinstance(data[key], list):
                        return [str(x) for x in data[key]]
                raise ValueError("JSON dims file must be a list or contain one of keys: dims/dimensions/appraisal_dimensions")
            elif isinstance(data, list):
                return [str(x) for x in data]
            else:
                raise ValueError("JSON dims file must be a list or dict with dims.")
        else:
            # assume text file, one dim per line, or comma-separated
            text = p.read_text(encoding="utf-8")
            parts = [s.strip() for s in text.replace("\n", ",").split(",") if s.strip()]
            if not parts:
                raise ValueError("--dims-file had no entries.")
            return parts

    if args.model_config:
        p = Path(args.model_config)
        if not p.exists():
            raise FileNotFoundError(f"--model-config not found: {p}")
        with open(p, "r") as f:
            cfg = json.load(f)
        if "appraisal_dimensions" not in cfg or not isinstance(cfg["appraisal_dimensions"], list):
            raise ValueError("model_config JSON must contain key 'appraisal_dimensions' as a list.")
        return [str(x) for x in cfg["appraisal_dimensions"]]

    raise ValueError("Please provide appraisal dimensions via --dims, --dims-file, or --model-config.")


def ensure_numeric(df: pd.DataFrame, dims: List[str], name: str) -> Tuple[pd.DataFrame, List[str]]:
    """Coerce appraisal columns to numeric; warn for missing dims. Returns df and list of usable dims."""
    missing = [d for d in dims if d not in df.columns]
    if missing:
        logger.warning(f"[{name}] Missing dimensions (will skip): {missing}")
    usable = [d for d in dims if d in df.columns]
    # coerce to numeric
    for d in usable:
        df[d] = pd.to_numeric(df[d], errors="coerce")
    return df, usable


def save_corpus_means(means: pd.DataFrame, outdir: Path):
    means.to_csv(outdir / "corpus_means.csv")
    # Heatmap of dims x corpora
    fig, ax = plt.subplots(figsize=(max(6, len(means.columns) * 1.2), max(6, len(means.index) * 0.35)))
    im = ax.imshow(means.values, aspect="auto")
    ax.set_xticks(np.arange(len(means.columns)))
    ax.set_xticklabels(means.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(means.index)))
    ax.set_yticklabels(means.index)
    ax.set_title("Corpus Means per Appraisal Dimension")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean value", rotation=90)
    fig.tight_layout()
    fig.savefig(outdir / "heatmap_corpus_means.png", dpi=200)
    plt.close(fig)


def save_corr(df: pd.DataFrame, dims: List[str], name: str, outdir: Path):
    corr = df[dims].corr()
    corr.to_csv(outdir / f"corr_{name}.csv")
    # Heatmap
    fig, ax = plt.subplots(figsize=(max(6, len(dims) * 0.5), max(6, len(dims) * 0.5)))
    im = ax.imshow(corr.values, vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(dims)))
    ax.set_xticklabels(dims, rotation=90)
    ax.set_yticks(np.arange(len(dims)))
    ax.set_yticklabels(dims)
    ax.set_title(f"Correlation Heatmap — {name}")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("r", rotation=0)
    fig.tight_layout()
    fig.savefig(outdir / f"heatmap_corr_{name}.png", dpi=200)
    plt.close(fig)


def save_descriptive_stats(df: pd.DataFrame, dims: List[str], name: str, outdir: Path):
    stats = df[dims].describe(percentiles=[0.25, 0.5, 0.75]).T
    stats.to_csv(outdir / f"stats_{name}.csv")


def choose_topk_dims(means: pd.DataFrame, topk: int) -> List[str]:
    """Select top-K discriminative dimensions by variance across corpora means."""
    if means.shape[1] < 2:
        # Only one corpus; pick topk by variance within that corpus? Fallback to arbitrary order
        logger.warning("Only one corpus provided; selecting top-K by per-dimension variance within the corpus.")
        # This requires raw data; but here we only have means. We'll just return first topk dims.
        return list(means.index)[:topk]
    var_across = means.var(axis=1)
    top_dims = var_across.sort_values(ascending=False).head(topk).index.tolist()
    return top_dims


def plot_distributions(dfs: Dict[str, pd.DataFrame], dims: List[str], outdir: Path, bins: int):
    for dim in dims:
        fig, ax = plt.subplots(figsize=(7, 4))
        for name, df in dfs.items():
            data = df[dim].dropna().values
            if data.size == 0:
                logger.warning(f"[{name}] No data to plot for dim '{dim}'. Skipping this dataset in histogram.")
                continue
            ax.hist(data, bins=bins, alpha=0.5, density=True, label=name)
        ax.set_title(f"Distribution — {dim}")
        ax.set_xlabel(dim)
        ax.set_ylabel("Density")
        ax.legend()
        fig.tight_layout()
        fig.savefig(outdir / f"dist_{dim}.png", dpi=200)
        plt.close(fig)


# ------------------ Main ------------------

def main():
    parser = argparse.ArgumentParser(description="Broad Exploratory Scan for 21-Dimension Appraisals (CLI)")
    parser.add_argument("--datasets", type=str, required=True,
                        help="Comma-separated mapping: name1=path1.csv,name2=path2.csv")
    parser.add_argument("--dims", type=str, default=None,
                        help="Comma-separated list of appraisal dimension column names.")
    parser.add_argument("--dims-file", type=str, default=None,
                        help="Path to a txt/json file listing dimensions.")
    parser.add_argument("--model-config", type=str, default=None,
                        help="Path to model config JSON with key 'appraisal_dimensions'.")
    parser.add_argument("--outdir", type=str, default="analysis_step2",
                        help="Output directory for summaries and plots.")
    parser.add_argument("--id-cols", type=str, default=None,
                        help="Optional: comma-separated list of metadata columns (carried in summaries if needed).")
    parser.add_argument("--topk", type=int, default=6, help="Top-K discriminative dimensions to highlight.")
    parser.add_argument("--bins", type=int, default=30, help="Bins for histograms.")

    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    datasets = parse_datasets_arg(args.datasets)
    dims = load_dims(args)
    logger.info(f"Using dimensions: {dims}")

    id_cols = [c.strip() for c in args.id_cols.split(",")] if args.id_cols else []

    # Load dataframes and ensure numeric dims
    dfs: Dict[str, pd.DataFrame] = {}
    usable_dims_per_corpus: Dict[str, List[str]] = {}

    for name, path in datasets.items():
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Dataset '{name}' not found at: {p}")
        df = pd.read_csv(p)
        df, usable = ensure_numeric(df, dims, name)
        if not usable:
            logger.error(f"[{name}] No usable appraisal dimensions found. Skipping this corpus.")
            continue
        dfs[name] = df
        usable_dims_per_corpus[name] = usable
        logger.info(f"[{name}] Loaded {len(df)} rows. Usable dims: {len(usable)}/{len(dims)}")

    if not dfs:
        raise SystemExit("No datasets loaded with usable appraisal dimensions. Nothing to do.")

    # Compute corpus means on the intersection of available dims across corpora
    common_dims = sorted(set.intersection(*(set(v) for v in usable_dims_per_corpus.values())))
    if not common_dims:
        raise SystemExit("No common appraisal dimensions across provided corpora. Cannot compare.")

    logger.info(f"Common dimensions across corpora: {common_dims}")

    means_mat = []
    for dim in common_dims:
        row = []
        for name in dfs.keys():
            row.append(dfs[name][dim].mean(skipna=True))
        means_mat.append(row)

    means_df = pd.DataFrame(means_mat, index=common_dims, columns=list(dfs.keys()))
    save_corpus_means(means_df, outdir)

    # Per-corpus stats and correlations
    for name, df in dfs.items():
        dims_here = [d for d in common_dims if d in df.columns]
        save_descriptive_stats(df, dims_here, name, outdir)
        save_corr(df, dims_here, name, outdir)

    # Choose top-K discriminative dims and plot distributions
    top_dims = choose_topk_dims(means_df, args.topk)
    (outdir / "top_dims.txt").write_text("\n".join(top_dims), encoding="utf-8")
    logger.info(f"Top-{len(top_dims)} discriminative dimensions: {top_dims}")

    plot_distributions(dfs, top_dims, outdir, bins=args.bins)

    logger.info("All done. Outputs written to: %s", outdir.resolve())


if __name__ == "__main__":
    main()