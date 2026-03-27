# Gravitational Form Factor Reconstruction: Analysis & Curve Generation

Companion repository to [diffusion-gff-reconstruction](https://github.com/Herzallah15/diffusion-gff-reconstruction), containing the training code for the paper:

> **Reconstruction of gravitational form factors using generative machine learning**
> 
> Herzallah Alharazin, Julia Yu. Panteleeva
>
>  [https://arxiv.org/abs/2602.19267v2]

## Overview

This repository provides the code used to:

1. **Generate reconstructed GFF curves** from sparse lattice QCD / ChPT / DVCS data using a pretrained DDPM model (see the [main repository](https://github.com/Herzallah15/diffusion-gff-reconstruction) for training code).
2. **Analyse and visualise** the results, including extraction of the chiral low-energy constants $c_8$ and $c_9$.

The proton gravitational form factors $A(t)$, $J(t)$, and $D(t)$ are reconstructed over the full spacelike range $0 \le -t \le 2\;\mathrm{GeV}^2$ by conditioning the diffusion model on a small number of known data points. The framework yields non-parametric, model-independent reconstructions with quantified uncertainties.

## Repository Structure

```
├── README.md
├── Generate_Curves.py        # Runs the DDPM sampler for all GFF configurations
├── analysis.ipynb             # Jupyter notebook: plotting, LEC extraction, comparisons
├── plot_GFFs.py               # Reusable plotting helpers for single and multi-panel figures
└── Source/
    └── Sampling.py            # DDPM reverse sampler + conditioning + de-normalisation
```

> **Note:** The `Source/` directory also requires the diffusion model code (`DiffusionModel.py`) and the trained checkpoint (`gff_ddpm_best.pt`) from the [main repository](https://github.com/Herzallah15/diffusion-gff-reconstruction).


## File Descriptions

### `Sampling.py`

Key components:

- **`sample_ddpm`** — Full 1000-step DDPM reverse process with per-sample Gaussian jitter on the conditioning values and replacement at intermediate steps.
- **`build_conditioning`** — Converts physical $(t, \mathrm{GFF}(t))$ data points with experimental errors into the model's normalised mask, conditioning, and error vectors.
- **`denormalize_and_summarize`** — Transforms generated curves back to physical units and computes point-wise statistics (mean, median, 16th/84th percentiles).
- **`sample_GFF`** — End-to-end pipeline: load model → condition → sample → de-normalise → filter → save.

### `Generate_Curves.py`

Runs `sample_GFF` for every GFF configuration reported in the paper:

- **$A(t)$ interpolation** — 4, 3, 2, and 1 lattice points from (https://arxiv.org/abs/2310.08484), each with the EMT constraint $A(0) = 1$.
- **$J(t)$ interpolation** — 4, 3, 2, and 1 lattice points with $J(0) = 1/2$.
- **$D(t)$ interpolation** — 5 to 2 lattice points with two physicality filters: $D(t) < 0\;\forall t$ and $D(0) < 0$.
- **$D(t)$ quark / gluon decomposition** — Separate reconstructions of $D_{u+d+s}(t)$ and $D_g(t)$.
- **$D(t)$ from DVCS** — Reconstruction from DVCS data (https://arxiv.org/abs/2104.02031), with and without the extracted $D$-term.
- **ChPT extrapolation** — $A(t)$, $J(t)$, and $D(t)$ conditioned on ChPT data at low $|t|$, extrapolated to $-t = 2\;\mathrm{GeV}^2$.

### `analysis.ipynb`

Jupyter notebook that produces the paper's figures and physics results:

1. Training / fine-tuning loss curves.
2. Lattice reference curves (dipole and $z$-expansion fits from (https://arxiv.org/abs/2310.08484)).
3. Multi-panel plots of $A(t)$, $J(t)$, $D(t)$ reconstructions with varying numbers of conditioning points.
4. Extraction of the chiral LECs $c_9$ (from $A(t)$, $J(t)$) and $c_8$ (from $D(t)$).
5. Quark vs. gluon $D(t)$ decomposition.
6. DVCS-based reconstruction of $D(t)$.
7. ChPT-conditioned extrapolation plots.

### `plot_GFFs.py`

Reusable matplotlib helpers:

- **`plot_on_ax`** — Draws a reconstruction (median, mean, 68 % CI band) onto a given `Axes` object. Supports lattice error bars, EMT constraints, and reference curves.
- **`plot`** — Standalone single-figure version of the above.

## Requirements

- Python ≥ 3.9
- PyTorch ≥ 2.0
- NumPy, Matplotlib, Pandas, SymPy
- Jupyter (for `analysis.ipynb`)

Install dependencies:

```bash
pip install torch numpy matplotlib pandas sympy jupyter
```

## Usage

### 1. Generate reconstructed curves

Make sure the trained checkpoint (`gff_ddpm_best.pt`) and `DiffusionModel.py` from the [main repository](https://github.com/Herzallah15/diffusion-gff-reconstruction) are placed in the `Source/` directory alongside `Sampling.py` (which is already included here), then run:

```bash
python Generate_Curves.py
```

This produces `.pt` files containing the sampled curves, statistics, and metadata for each GFF configuration.

### 2. Analyse and plot

Open the analysis notebook:

```bash
jupyter notebook analysis.ipynb
```

The notebook loads the `.pt` output files and generates all figures shown in the paper.

## Data Sources

| Source | Reference | Used for |
|--------|-----------|----------|
| Lattice QCD (Hackett et al.) | (https://arxiv.org/abs/2310.08484) | $A(t)$, $J(t)$, $D(t)$ interpolation conditioning points |
| DVCS (Burkert et al.) | (https://arxiv.org/abs/2104.02031) | $D(t)$ from experiment |
| ChPT (Alharazin et al.) | (https://arxiv.org/abs/2602.19267) | Low-$|t|$ extrapolation data |

## Citation

If you use this code, please cite:

```bibtex
@article{Alharazin:2026gff,
    author    = {Alharazin, Herzallah and Panteleeva, Julia Yu.},
    title     = {Reconstruction of gravitational form factors using generative machine learning},
    year      = {2026},
    eprint    = {2602.19267},
    archivePrefix = {arXiv},
    primaryClass  = {hep-ph}
}
```
