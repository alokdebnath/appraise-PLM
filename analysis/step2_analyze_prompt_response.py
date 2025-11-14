#!/usr/bin/env python3
"""
Analyze prompt–response appraisal datasets (server/headless friendly).

Keeps everything from the earlier script and adds:
  • Prompt-only / Response-only / Average / Delta summaries
  • Correlation matrices (+ PNG heatmaps) for each of the four views
  • Overlay histograms for Prompt vs Response (per dimension)
  • Effect sizes (Cohen's d) for Response vs Prompt (per dimension)
  • Distribution shifts: KL and Jensen–Shannon divergence (per dimension)
  • t-SNE projection of 21D appraisal space with Prompt/Response labels (PNG)

Assumptions & flexibility:
  • Appraisal columns are prefixed as <prompt_prefix><dim> and <response_prefix><dim>.
    Default prefixes are 'prompt_' and 'response_'.
  • Dimension names can be provided explicitly via --dims, or they will be
    inferred by stripping the prefixes from matching pairs of columns common to both roles.

Usage examples:
  python analyze_prompt_response_plus.py data/pairs.csv out/analysis_pairs
  python analyze_prompt_response_plus.py data/pairs.csv out/analysis_pairs \
      --prompt-prefix seeker_post_ --response-prefix response_post_ \
      --dims pleasantness,control,unexpectedness,...

Requires: pandas, numpy, matplotlib, scikit-learn
"""

from __future__ import annotations
import argparse
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


# --------------------- I/O helpers ---------------------

def ensure_outdir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_csv(df: pd.DataFrame, path: Path):
    df.to_csv(path, index=True)


# --------------------- Column discovery ---------------------

def infer_dimensions(df: pd.DataFrame, prompt_prefix: str, response_prefix: str, dims_arg: List[str] | None) -> List[str]:
    if dims_arg:
        return list(dims_arg)
    # infer from columns present in both prompt_ and response_
    p_dims = {c[len(prompt_prefix):] for c in df.columns if c.startswith(prompt_prefix)}
    r_dims = {c[len(response_prefix):] for c in df.columns if c.startswith(response_prefix)}
    common = sorted(list(p_dims.intersection(r_dims)))
    if not common:
        raise ValueError(
            "Could not infer dimensions. Provide --dims or check your prefixes and columns."
        )
    return common


def select_role_df(df: pd.DataFrame, dims: List[str], prefix: str) -> pd.DataFrame:
    cols = [f"{prefix}{d}" for d in dims]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")
    sub = df[cols].copy()
    # coerce to numeric
    for c in cols:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")
    # drop prefix for analysis consistency
    sub.columns = dims
    return sub


# --------------------- Stats & plots ---------------------

def describe_block(block: pd.DataFrame) -> pd.DataFrame:
    return block.describe(percentiles=[0.25, 0.5, 0.75]).T


def correlation(block: pd.DataFrame) -> pd.DataFrame:
    return block.corr()


def save_corr_heatmap(corr: pd.DataFrame, title: str, outpath: Path):
    # Basic matplotlib heatmap (no seaborn dependency)
    fig, ax = plt.subplots(figsize=(max(8, 0.45 * len(corr.columns)), max(6, 0.45 * len(corr.index))))
    im = ax.imshow(corr.values, vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(np.arange(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticks(np.arange(len(corr.index)))
    ax.set_yticklabels(corr.index)
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('r', rotation=0)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def save_overlay_hist(prompt: pd.Series, response: pd.Series, dim: str, outdir: Path, bins: int = 30):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(prompt.dropna().values, bins=bins, alpha=0.5, density=True, label='Prompt')
    ax.hist(response.dropna().values, bins=bins, alpha=0.5, density=True, label='Response')
    ax.set_title(f"Overlay Histogram — {dim}")
    ax.set_xlabel(dim)
    ax.set_ylabel('Density')
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / f"overlay_hist_{dim}.png", dpi=200)
    plt.close(fig)


def save_bar(values: pd.Series, title: str, ylabel: str, outpath: Path):
    fig, ax = plt.subplots(figsize=(max(8, 0.4 * len(values.index)), 4))
    ax.bar(values.index, values.values)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticklabels(values.index, rotation=45, ha='right')
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


# --------------------- Effect size & divergences ---------------------

def cohens_d(x: pd.Series, y: pd.Series) -> float:
    x = x.dropna().astype(float)
    y = y.dropna().astype(float)
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return np.nan
    mx, my = x.mean(), y.mean()
    sx, sy = x.std(ddof=1), y.std(ddof=1)
    # pooled std
    sp2 = ((nx - 1) * sx * sx + (ny - 1) * sy * sy) / (nx + ny - 2)
    sp = math.sqrt(sp2) if sp2 > 0 else np.nan
    if not np.isfinite(sp) or sp == 0:
        return np.sign(my - mx) * np.inf if (my != mx) else 0.0
    return (my - mx) / sp


def hist_pmf(series: pd.Series, bins: int = 30, eps: float = 1e-9) -> Tuple[np.ndarray, np.ndarray]:
    data = series.dropna().astype(float).values
    if data.size == 0:
        return np.array([1.0]), np.array([0.0])  # degenerate PMF
    hist, edges = np.histogram(data, bins=bins, density=False)
    hist = hist.astype(float)
    hist += eps  # smoothing to avoid zeros
    pmf = hist / hist.sum()
    # use bin centers as support
    # centers = 0.5 * (edges[:-1] + edges[-1:1:-1])  # incorrect; fix below
    centers = 0.5 * (edges[:-1] + edges[1:])
    return pmf, centers


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    # assumes p and q are probability vectors with same length and >0 elements
    return float(np.sum(p * np.log(p / q)))


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)


def divergences_per_dim(prompt_df: pd.DataFrame, response_df: pd.DataFrame, bins: int = 30) -> pd.DataFrame:
    out = []
    for dim in prompt_df.columns:
        p_pmf, _ = hist_pmf(prompt_df[dim], bins=bins)
        q_pmf, _ = hist_pmf(response_df[dim], bins=bins)
        # Align lengths by padding/truncation to common length
        L = max(len(p_pmf), len(q_pmf))
        p = np.pad(p_pmf, (0, L - len(p_pmf)))
        q = np.pad(q_pmf, (0, L - len(q_pmf)))
        kl = kl_divergence(p, q)
        js = js_divergence(p, q)
        out.append((dim, kl, js))
    return pd.DataFrame(out, columns=['dimension', 'kl_prompt_to_response', 'js_sym'])


# --------------------- t-SNE ---------------------

def tsne_prompt_response(prompt_df: pd.DataFrame, response_df: pd.DataFrame, perplexity: float, learning_rate: float, random_state: int) -> Tuple[np.ndarray, np.ndarray]:
    X = np.vstack([prompt_df.values, response_df.values])
    labels = np.array(["Prompt"] * len(prompt_df) + ["Response"] * len(response_df))
    tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate, random_state=random_state, init='random')
    emb = tsne.fit_transform(X)
    return emb, labels


def save_tsne_plot(emb: np.ndarray, labels: np.ndarray, outpath: Path):
    fig, ax = plt.subplots(figsize=(6, 6))
    mask_p = labels == 'Prompt'
    mask_r = labels == 'Response'
    ax.scatter(emb[mask_p, 0], emb[mask_p, 1], alpha=0.6, label='Prompt')
    ax.scatter(emb[mask_r, 0], emb[mask_r, 1], alpha=0.6, label='Response')
    ax.set_title('t-SNE of Appraisal Vectors')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


# --------------------- Main pipeline ---------------------

def main():
    parser = argparse.ArgumentParser(description='Analyze prompt-response appraisal datasets (enhanced).')
    parser.add_argument('input_csv', type=str, help='Path to CSV with prompt/response appraisal columns')
    parser.add_argument('output_dir', type=str, help='Directory to save outputs')
    parser.add_argument('--prompt-prefix', type=str, default='prompt_', help='Prefix for prompt appraisal columns')
    parser.add_argument('--response-prefix', type=str, default='response_', help='Prefix for response appraisal columns')
    parser.add_argument('--dims', type=str, default=None, help='Comma-separated list of dimension names (without prefixes). If omitted, inferred from columns.')
    parser.add_argument('--bins', type=int, default=30, help='Number of bins for histograms/divergences')
    parser.add_argument('--tsne-perplexity', type=float, default=30.0, help='t-SNE perplexity')
    parser.add_argument('--tsne-lr', type=float, default=200.0, help='t-SNE learning rate')
    parser.add_argument('--random-state', type=int, default=42, help='Random seed for t-SNE')

    args = parser.parse_args()

    outdir = ensure_outdir(args.output_dir)

    # Load
    df = pd.read_csv(args.input_csv)

    # Determine dimensions
    dims = infer_dimensions(df, args.prompt_prefix, args.response_prefix,
                            [d.strip() for d in args.dims.split(',')] if args.dims else None)

    # Build role dataframes
    df_prompt = select_role_df(df, dims, args.prompt_prefix)
    df_response = select_role_df(df, dims, args.response_prefix)

    # ---- Summaries & correlations for Prompt / Response / Average / Delta ----
    views = {}
    views['prompt_only'] = df_prompt
    views['response_only'] = df_response
    views['average'] = (df_prompt + df_response) / 2.0
    views['delta'] = df_response - df_prompt

    for label, block in views.items():
        # Descriptive stats
        desc = describe_block(block)
        save_csv(desc, outdir / f'{label}_describe.csv')

        # Means
        means = block.mean()
        save_csv(means.to_frame('mean'), outdir / f'{label}_means.csv')
        save_bar(means, f'{label.replace("_", " ").title()} — Mean per Dimension', 'Mean', outdir / f'{label}_means.png')

        # Correlations
        corr = correlation(block)
        save_csv(corr, outdir / f'{label}_correlation.csv')
        save_corr_heatmap(corr, f'Correlation Heatmap — {label.replace("_", " ").title()}', outdir / f'{label}_correlation_heatmap.png')

    # ---- Overlay histograms per dimension (Prompt vs Response) ----
    for dim in dims:
        save_overlay_hist(df_prompt[dim], df_response[dim], dim, outdir, bins=args.bins)

    # ---- Effect sizes per dimension (Response vs Prompt) ----
    effects = []
    for dim in dims:
        d = cohens_d(df_prompt[dim], df_response[dim])
        effects.append((dim, d))
    eff_df = pd.DataFrame(effects, columns=['dimension', "cohens_d_response_vs_prompt"])
    save_csv(eff_df.set_index('dimension'), outdir / 'effect_sizes_cohens_d.csv')
    # bar plot of effect sizes
    save_bar(eff_df.set_index('dimension')["cohens_d_response_vs_prompt"], 'Effect Size (Cohen\'s d) — Response vs Prompt', "d", outdir / 'effect_sizes_cohens_d.png')

    # ---- Divergences per dimension ----
    div_df = divergences_per_dim(df_prompt, df_response, bins=args.bins)
    save_csv(div_df.set_index('dimension'), outdir / 'divergences.csv')
    # bar plots
    save_bar(div_df.set_index('dimension')['kl_prompt_to_response'], 'KL(Prompt || Response) per Dimension', 'KL', outdir / 'kl_divergence.png')
    save_bar(div_df.set_index('dimension')['js_sym'], 'Jensen–Shannon Divergence per Dimension', 'JSD', outdir / 'js_divergence.png')

    # ---- t-SNE projection ----
    try:
        emb, labels = tsne_prompt_response(df_prompt, df_response, args.tsne_perplexity, args.tsne_lr, args.random_state)
        save_tsne_plot(emb, labels, outdir / 'tsne_prompt_response.png')
    except Exception as e:
        # Don't fail the whole run if t-SNE struggles with tiny datasets or params
        (outdir / 'tsne_error.txt').write_text(str(e))

    print(f"Analysis complete. Results saved to: {outdir.resolve()}")


if __name__ == '__main__':
    main()
