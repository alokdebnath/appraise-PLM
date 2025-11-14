#!/usr/bin/env python3
"""
Step 3 — Appraisal–Emotion Correlation and Distribution Scan

What this script does:
- Loads one dataset with appraisal columns + categorical label(s).
- Computes correlation between appraisals and label one-hots.
- Runs ANOVA/effect-size stats to see how appraisals differ by label.
- Saves box plots of appraisal distributions per label category.
- Saves heatmap of appraisal–label correlations.
- Saves heatmap of appraisal means grouped by each label.

Inputs:
- --dataset path.csv
- --dims / --dims-file / --model-config  (same as Step 2)
- --label-cols col1,col2,...   (categorical columns for labels, e.g. emotion)
- --outdir  output folder

Outputs:
- label_corr.csv
- label_effects.csv
- box_<label>_<dim>.png
- heatmap_label_corr.png
- heatmap_means_<label>.png
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import f_oneway, pointbiserialr

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("appraisal_emotion")

# ---- helpers (reuse from Step 2 if structured) ----
def load_dims(args):
    if args.dims:
        return [d.strip() for d in args.dims.split(",") if d.strip()]
    if args.dims_file:
        p = Path(args.dims_file)
        if p.suffix.lower() == ".json":
            data = json.loads(p.read_text())
            if isinstance(data, dict):
                return data.get("appraisal_dimensions", [])
            return data
        return [s.strip() for s in p.read_text().replace("\n", ",").split(",") if s.strip()]
    if args.model_config:
        cfg = json.loads(Path(args.model_config).read_text())
        return cfg.get("appraisal_dimensions", [])
    raise ValueError("Need --dims, --dims-file, or --model-config.")

def ensure_numeric(df, dims):
    for d in dims:
        df[d] = pd.to_numeric(df[d], errors="coerce")
    return df

# ---- analysis ----
def compute_label_correlations(df, dims, label_cols, outdir):
    results = []
    for label in label_cols:
        if label not in df.columns:
            logger.warning(f"Missing label column: {label}")
            continue
        cats = df[label].dropna().unique()
        if len(cats) == 2:
            # binary: point-biserial correlations
            df_enc = (df[label] == cats[1]).astype(int)
            for d in dims:
                vals = df[d].dropna()
                aligned = df_enc.loc[vals.index]
                if len(vals) > 1:
                    r, _ = pointbiserialr(aligned, vals)
                    results.append((cats[1], d, "pointbiserial", r))
        else:
            # multi-class: one-hot encode + Pearson
            dummies = pd.get_dummies(df[label])
            for cat in dummies.columns:
                for d in dims:
                    r = np.corrcoef(dummies[cat], df[d].fillna(0))[0,1]
                    results.append((cat, d, "pearson", r))
    res_df = pd.DataFrame(results, columns=["label","dim","method","r"])
    res_df.to_csv(outdir / "label_corr.csv", index=False)
    return res_df

def compute_effect_sizes(df, dims, label_cols, outdir):
    results = []
    for label in label_cols:
        cats = df[label].dropna().unique()
        for d in dims:
            groups = [df[df[label]==c][d].dropna().values for c in cats]
            if len(groups) > 1 and all(len(g)>1 for g in groups):
                f, p = f_oneway(*groups)
                ss_between = sum([len(g)*(g.mean()-df[d].mean())**2 for g in groups])
                ss_total = ((df[d].dropna() - df[d].mean())**2).sum()
                eta2 = ss_between/ss_total if ss_total>0 else np.nan
                results.append((label, d, f, p, eta2))
    res_df = pd.DataFrame(results, columns=["label","dim","F","p","eta2"])
    res_df.to_csv(outdir / "label_effects.csv", index=False)
    return res_df

def plot_distributions(df, dims, label_cols, outdir):
    for label in label_cols:
        for d in dims:
            if label not in df.columns: 
                continue
            fig, ax = plt.subplots(figsize=(7,4))
            df.boxplot(column=d, by=label, ax=ax)
            plt.xticks(rotation=90)
            ax.set_title(f"{d} by {label}")
            ax.set_ylabel(d)
            fig.suptitle("")
            fig.tight_layout()
            fig.savefig(outdir / f"box_{label}_{d}.png", dpi=200)
            plt.close(fig)

def plot_corr_heatmap(corr_df, outdir):
    if corr_df.empty:
        return
    pivot = corr_df.pivot(index="dim", columns="label", values="r").fillna(0)
    fig, ax = plt.subplots(figsize=(8,6))
    im = ax.imshow(pivot.values, aspect="auto", vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    fig.colorbar(im, ax=ax, label="Correlation")
    fig.tight_layout()
    fig.savefig(outdir / "heatmap_label_corr.png", dpi=200)
    plt.close(fig)

def plot_means_heatmap(df, dims, label_cols, outdir):
    for label in label_cols:
        if label not in df.columns:
            continue
        means = df.groupby(label)[dims].mean().T
        fig, ax = plt.subplots(figsize=(max(6, len(means.columns)*1.2), max(6, len(means.index)*0.35)))
        im = ax.imshow(means.values, aspect="auto", cmap="viridis")
        ax.set_xticks(np.arange(len(means.columns)))
        ax.set_xticklabels(means.columns, rotation=45, ha="right")
        ax.set_yticks(np.arange(len(means.index)))
        ax.set_yticklabels(means.index)
        ax.set_title(f"Average Appraisals by {label}")
        fig.colorbar(im, ax=ax, label="Mean Value")
        fig.tight_layout()
        fig.savefig(outdir / f"heatmap_means_{label}.png", dpi=200)
        plt.close(fig)

# ---- main ----
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dims", default=None)
    parser.add_argument("--dims-file", default=None)
    parser.add_argument("--model-config", default=None)
    parser.add_argument("--label-cols", required=True,
                        help="Comma-separated categorical label columns (e.g., emotion).")
    parser.add_argument("--outdir", default="analysis_step3")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    dims = load_dims(args)
    df = pd.read_csv(args.dataset)
    df = ensure_numeric(df, dims)
    label_cols = [s.strip() for s in args.label_cols.split(",")]

    corr_df = compute_label_correlations(df, dims, label_cols, outdir)
    compute_effect_sizes(df, dims, label_cols, outdir)
    plot_distributions(df, dims, label_cols, outdir)
    plot_corr_heatmap(corr_df, outdir)
    plot_means_heatmap(df, dims, label_cols, outdir)

    logger.info("Done. Outputs at %s", outdir.resolve())

if __name__ == "__main__":
    main()
