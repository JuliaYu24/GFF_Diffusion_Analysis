"""
Generate_Curves.py — Batch GFF Curve Generation
================================================
Authors: H. Alharazin, J. Yu. Panteleeva

Runs the DDPM-based sampler for every gravitational form factor (GFF)
configuration reported in arXiv:2602.19267. Each block defines a set of
conditioning points {t: (central_value, error)} and calls sample_GFF()
to generate an ensemble of reconstructed curves.

Data sources:
  - Lattice QCD points:   arXiv:2310.08484 (Hackett et al.)
  - DVCS points:          arXiv:2104.02031 (Burkert et al.)
  - ChPT points:          This work (arXiv:2602.19267), extracted from (arXiv:2006.05890)

Output:
  Each call produces a .pt file containing the generated ensemble,
  summary statistics (mean, median, quantiles), and metadata.

Usage:
    python Generate_Curves.py
"""

from Source.Sampling import *


# ═══════════════════════════════════════════════════════════════════
#  A(t) — INTERPOLATION FROM LATTICE DATA
#  Conditioning: lattice points from arXiv:2310.08484 + EMT constraint A(0) = 1
#  The map format is {t : (central_value, error)}
#  Central values are averages of dipole and z-expansion fits.
# ═══════════════════════════════════════════════════════════════════

# ── A(t), 4 lattice points + A(0) ──────────────────────────────

Amap = {0: (1, 0),
        -0.18 : ( (0.8389 + 0.8363)/2, 0.020),
        -0.55 : ( (0.6007+0.5947)/2, 0.016),
        -0.8: ( (0.4935 + 0.4953)/2, 0.015),
        -1.7: ( (0.2785 + 0.2828)/2, 0.015)
       }


t_known = torch.tensor([i for i in Amap.keys()])
gff_known = torch.tensor([Amap[i][0] for i in Amap.keys()])
gff_sigma = torch.tensor([Amap[i][1] for i in Amap])

sample_GFF(n_samples = 10000, batch_size = 250, seed = 42, t_known = t_known, gff_known = gff_known, gff_sigma = gff_sigma, save_path = 'A_Interpolated')


# ── A(t), 3 lattice points + A(0) ──────────────────────────────

Amap = {0: (1, 0),
        -0.55 : ( (0.6007+0.5947)/2, 0.016),
        -0.8: ( (0.4935 + 0.4953)/2, 0.015),
        -1.7: ( (0.2785 + 0.2828)/2, 0.015)
       }


t_known = torch.tensor([i for i in Amap.keys()])
gff_known = torch.tensor([Amap[i][0] for i in Amap.keys()])
gff_sigma = torch.tensor([Amap[i][1] for i in Amap])

sample_GFF(n_samples = 10000, batch_size = 250, seed = 42, t_known = t_known, gff_known = gff_known, gff_sigma = gff_sigma,save_path = 'A_Interpolated')

# ── A(t), 2 lattice points + A(0) ──────────────────────────────

Amap = {0: (1, 0),
        -0.8: ( (0.4935 + 0.4953)/2, 0.015),
        -1.7: ( (0.2785 + 0.2828)/2, 0.015)
       }


t_known = torch.tensor([i for i in Amap.keys()])
gff_known = torch.tensor([Amap[i][0] for i in Amap.keys()])
gff_sigma = torch.tensor([Amap[i][1] for i in Amap])

sample_GFF(n_samples = 10000, batch_size = 250, seed = 42, t_known = t_known, gff_known = gff_known, gff_sigma = gff_sigma,save_path = 'A_Interpolated')

# ── A(t), 1 lattice point + A(0) ───────────────────────────────

Amap = {0: (1, 0),
        -0.8: ( (0.4935 + 0.4953)/2, 0.015)}


t_known = torch.tensor([i for i in Amap.keys()])
gff_known = torch.tensor([Amap[i][0] for i in Amap.keys()])
gff_sigma = torch.tensor([Amap[i][1] for i in Amap])

sample_GFF(n_samples = 10000, batch_size = 250, seed = 42, t_known = t_known, gff_known = gff_known, gff_sigma = gff_sigma,save_path = 'A_Interpolated')


# ═══════════════════════════════════════════════════════════════════
#  J(t) — INTERPOLATION FROM LATTICE DATA
#  Conditioning: lattice points from arXiv:2310.08484 + J(0) = 1/2
# ═══════════════════════════════════════════════════════════════════

# ── J(t), 4 lattice points + J(0) ──────────────────────────────

Jmap = {0: (0.5, 0.001),
         -0.18: ((0.4333 +0.4309)/2, 0.03),
        -0.55: ((0.3263+0.3243)/2, 0.025),
        -1.05: ((0.2367+0.2387)/2, 0.022),
        -1.7 : ((0.1656+0.1644)/2, 0.03)   
       }
gff_known = torch.tensor([Jmap[i][0] for i in Jmap.keys()])
gff_sigma = torch.tensor([Jmap[i][1] for i in Jmap])
t_known = torch.tensor([i for i in Jmap.keys()])

sample_GFF(n_samples = 10000, batch_size = 250, seed = 42, t_known = t_known, gff_known = gff_known, gff_sigma = gff_sigma,save_path = 'J_Interpolated_4')


# ── J(t), 3 lattice points + J(0) ──────────────────────────────

Jmap = {0: (0.5, 0.001),
         -0.18: ((0.4333 +0.4309)/2, 0.03),
        #-0.55: ((0.3263+0.3243)/2, 0.025),
        -1.05: ((0.2367+0.2387)/2, 0.022),
        -1.7 : ((0.1656+0.1644)/2, 0.03)   
       }
gff_known = torch.tensor([Jmap[i][0] for i in Jmap.keys()])
gff_sigma = torch.tensor([Jmap[i][1] for i in Jmap])
t_known = torch.tensor([i for i in Jmap.keys()])

sample_GFF(n_samples = 10000, batch_size = 250, seed = 42, t_known = t_known, gff_known = gff_known, gff_sigma = gff_sigma,save_path = 'J_Interpolated_3')

# ── J(t), 2 lattice points + J(0) ──────────────────────────────

Jmap = {0: (0.5, 0.001),
         -0.18: ((0.4333 +0.4309)/2, 0.03),
        #-0.55: ((0.3263+0.3243)/2, 0.025),
        -1.05: ((0.2367+0.2387)/2, 0.022),
        #-1.7 : ((0.1656+0.1644)/2, 0.03)   
       }
gff_known = torch.tensor([Jmap[i][0] for i in Jmap.keys()])
gff_sigma = torch.tensor([Jmap[i][1] for i in Jmap])
t_known = torch.tensor([i for i in Jmap.keys()])

sample_GFF(n_samples = 10000, batch_size = 250, seed = 42, t_known = t_known, gff_known = gff_known, gff_sigma = gff_sigma,save_path = 'J_Interpolated_2')

# ── J(t), 1 lattice point + J(0) ───────────────────────────────

Jmap = {0: (0.5, 0.001),
        # -0.18: ((0.4333 +0.4309)/2, 0.03),
        #-0.55: ((0.3263+0.3243)/2, 0.025),
        -1.05: ((0.2367+0.2387)/2, 0.022),
        #-1.7 : ((0.1656+0.1644)/2, 0.03)   
       }
gff_known = torch.tensor([Jmap[i][0] for i in Jmap.keys()])
gff_sigma = torch.tensor([Jmap[i][1] for i in Jmap])
t_known = torch.tensor([i for i in Jmap.keys()])

sample_GFF(n_samples = 10000, batch_size = 250, seed = 42, t_known = t_known, gff_known = gff_known, gff_sigma = gff_sigma,save_path = 'J_Interpolated_1')



# ═══════════════════════════════════════════════════════════════════
#  D(t) — INTERPOLATION WITH D(t) < 0 EVERYWHERE
#  Physicality constraint: require_negative_all filters out any
#  curve that is positive anywhere on the grid.
# ═══════════════════════════════════════════════════════════════════

# ── D(t), 5 lattice points, D(t) < 0 ───────────────────────────

Dmap = {
        -0.18: (-1.3, 0.5),
        -0.55 : (( -0.6865 -0.7105)/2, 0.15),
       -1.05 : (( -0.3142 -0.3090)/2, 0.1),
       -1.4: ((-0.2085-0.2025)/2, 0.08),
       -1.6 : (( -0.1706 -0.1708)/2, 0.08)
       }

gff_known = torch.tensor([Dmap[i][0] for i in Dmap.keys()])
gff_sigma = torch.tensor([Dmap[i][1] for i in Dmap])
t_known = torch.tensor([i for i in Dmap.keys()])

sample_GFF(n_samples = 20000, batch_size = 400, seed = 42, t_known = t_known, gff_known = gff_known,
           gff_sigma = gff_sigma,save_path = 'D/D_Interpolated_5_Points', require_negative_all = True)


# ── D(t), 4 lattice points, D(t) < 0 ───────────────────────────

Dmap = {
        -0.18: (-1.3, 0.5),
        -0.55 : (( -0.6865 -0.7105)/2, 0.15),
       -1.05 : (( -0.3142 -0.3090)/2, 0.1),
       #-1.4: ((-0.2085-0.2025)/2, 0.08),
       -1.6 : (( -0.1706 -0.1708)/2, 0.08)
       }

gff_known = torch.tensor([Dmap[i][0] for i in Dmap.keys()])
gff_sigma = torch.tensor([Dmap[i][1] for i in Dmap])
t_known = torch.tensor([i for i in Dmap.keys()])

sample_GFF(n_samples = 20000, batch_size = 400, seed = 42, t_known = t_known, gff_known = gff_known,
           gff_sigma = gff_sigma,save_path = 'D/D_Interpolated_4_Points', require_negative_all = True)

# ── D(t), 3 lattice points, D(t) < 0 ───────────────────────────

Dmap = {
        -0.18: (-1.3, 0.5),
        #-0.55 : (( -0.6865 -0.7105)/2, 0.15),
       -1.05 : (( -0.3142 -0.3090)/2, 0.1),
       #-1.4: ((-0.2085-0.2025)/2, 0.08),
       -1.6 : (( -0.1706 -0.1708)/2, 0.08)
       }

gff_known = torch.tensor([Dmap[i][0] for i in Dmap.keys()])
gff_sigma = torch.tensor([Dmap[i][1] for i in Dmap])
t_known = torch.tensor([i for i in Dmap.keys()])

sample_GFF(n_samples = 20000, batch_size = 400, seed = 42, t_known = t_known, gff_known = gff_known,
           gff_sigma = gff_sigma,save_path = 'D/D_Interpolated_3_Points', require_negative_all = True)

# ── D(t), 2 lattice points, D(t) < 0 ───────────────────────────

Dmap = {
        -0.18: (-1.3, 0.5),
        #-0.55 : (( -0.6865 -0.7105)/2, 0.15),
       #-1.05 : (( -0.3142 -0.3090)/2, 0.1),
       #-1.4: ((-0.2085-0.2025)/2, 0.08),
       -1.6 : (( -0.1706 -0.1708)/2, 0.08)
       }

gff_known = torch.tensor([Dmap[i][0] for i in Dmap.keys()])
gff_sigma = torch.tensor([Dmap[i][1] for i in Dmap])
t_known = torch.tensor([i for i in Dmap.keys()])

sample_GFF(n_samples = 20000, batch_size = 400, seed = 42, t_known = t_known, gff_known = gff_known,
           gff_sigma = gff_sigma,save_path = 'D/D_Interpolated_2_Points', require_negative_all = True)


# ═══════════════════════════════════════════════════════════════════
#  D(t) — INTERPOLATION WITH D(0) < 0 ONLY
#  Weaker physicality constraint: only require negativity at t = 0.
# ═══════════════════════════════════════════════════════════════════

# ── D(t), 5 lattice points, D(0) < 0 ───────────────────────────

Dmap = {
        -0.18: (-1.3, 0.5),
        -0.55 : (( -0.6865 -0.7105)/2, 0.15),
       -1.05 : (( -0.3142 -0.3090)/2, 0.1),
       -1.4: ((-0.2085-0.2025)/2, 0.08),
       -1.6 : (( -0.1706 -0.1708)/2, 0.08)
       }

gff_known = torch.tensor([Dmap[i][0] for i in Dmap.keys()])
gff_sigma = torch.tensor([Dmap[i][1] for i in Dmap])
t_known = torch.tensor([i for i in Dmap.keys()])

sample_GFF(n_samples = 20000, batch_size = 400, seed = 42, t_known = t_known, gff_known = gff_known,
           gff_sigma = gff_sigma,save_path = 'D/D_Cond_Interpolated_5_Points', require_negative_at_zero = True)

# ── D(t), 4 lattice points, D(0) < 0 ───────────────────────────

Dmap = {
        -0.18: (-1.3, 0.5),
        -0.55 : (( -0.6865 -0.7105)/2, 0.15),
       -1.05 : (( -0.3142 -0.3090)/2, 0.1),
       #-1.4: ((-0.2085-0.2025)/2, 0.08),
       -1.6 : (( -0.1706 -0.1708)/2, 0.08)
       }

gff_known = torch.tensor([Dmap[i][0] for i in Dmap.keys()])
gff_sigma = torch.tensor([Dmap[i][1] for i in Dmap])
t_known = torch.tensor([i for i in Dmap.keys()])

sample_GFF(n_samples = 20000, batch_size = 400, seed = 42, t_known = t_known, gff_known = gff_known,
           gff_sigma = gff_sigma,save_path = 'D/D_Cond_Interpolated_4_Points', require_negative_at_zero = True)

# ── D(t), 3 lattice points, D(0) < 0 ───────────────────────────

Dmap = {
        -0.18: (-1.3, 0.5),
        #-0.55 : (( -0.6865 -0.7105)/2, 0.15),
       -1.05 : (( -0.3142 -0.3090)/2, 0.1),
       #-1.4: ((-0.2085-0.2025)/2, 0.08),
       -1.6 : (( -0.1706 -0.1708)/2, 0.08)
       }

gff_known = torch.tensor([Dmap[i][0] for i in Dmap.keys()])
gff_sigma = torch.tensor([Dmap[i][1] for i in Dmap])
t_known = torch.tensor([i for i in Dmap.keys()])

sample_GFF(n_samples = 20000, batch_size = 400, seed = 42, t_known = t_known, gff_known = gff_known,
           gff_sigma = gff_sigma,save_path = 'D/D_Cond_Interpolated_3_Points', require_negative_at_zero = True)

# ── D(t), 2 lattice points, D(0) < 0 ───────────────────────────

Dmap = {
        -0.18: (-1.3, 0.5),
        #-0.55 : (( -0.6865 -0.7105)/2, 0.15),
       #-1.05 : (( -0.3142 -0.3090)/2, 0.1),
      #-1.4: ((-0.2085-0.2025)/2, 0.08),
       -1.6 : (( -0.1706 -0.1708)/2, 0.08)
       }

gff_known = torch.tensor([Dmap[i][0] for i in Dmap.keys()])
gff_sigma = torch.tensor([Dmap[i][1] for i in Dmap])
t_known = torch.tensor([i for i in Dmap.keys()])

sample_GFF(n_samples = 20000, batch_size = 400, seed = 42, t_known = t_known, gff_known = gff_known,
           gff_sigma = gff_sigma,save_path = 'D/D_Cond_Interpolated_2_Points', require_negative_at_zero = True)

# ═══════════════════════════════════════════════════════════════════
#  D(t) — QUARK AND GLUON DECOMPOSITION
#  Separate reconstructions of D_{u+d+s}(t) and D_g(t)
#  from lattice data in arXiv:2310.08484
# ═══════════════════════════════════════════════════════════════════

# ── D_{u+d+s}(t) — quark contribution, 2 lattice points ────────

t_poles = [-0.31, -1.68]
gff_poles = [-0.62, -0.1]
sigma_poles = [0.18, 0.05]

t_known   = torch.tensor(t_poles)
gff_known = torch.tensor(gff_poles)
gff_sigma = torch.tensor(sigma_poles)

sample_GFF(n_samples=10000, batch_size=250, seed=402,
           t_known=t_known, gff_known=gff_known, gff_sigma=gff_sigma,
           save_path='D/D_uds_Interpolation', require_negative_at_zero = True)


# ── D_g(t) — gluon contribution, 2 lattice points ──────────────

t_poles = [-0.31, -1.68]
gff_poles = [-0.5, -0.1]
sigma_poles = [0.18, 0.05]

t_known   = torch.tensor(t_poles)
gff_known = torch.tensor(gff_poles)
gff_sigma = torch.tensor(sigma_poles)

sample_GFF(n_samples=10000, batch_size=250, seed=402,
           t_known=t_known, gff_known=gff_known, gff_sigma=gff_sigma,
           save_path='D/D_g_Interpolation', require_negative_at_zero = True)

# ═══════════════════════════════════════════════════════════════════
#  D(t) — FROM DVCS EXPERIMENTAL DATA
#  Source: arXiv:2104.02031 (Burkert et al.)
# ═══════════════════════════════════════════════════════════════════

# ── D(t) from DVCS, 5 experimental points ──────────────────────

t_poles = [-0.11, -0.15, -0.20, -0.26, -0.34]
gff_poles = [-1.14, -0.99, -0.93, -0.79, -0.66]
sigma_poles = [0.1, 0.09, 0.07, 0.06, 0.06]


t_known   = torch.tensor(t_poles)
gff_known = torch.tensor(gff_poles)
gff_sigma = torch.tensor(sigma_poles)

sample_GFF(n_samples=10000, batch_size=500, seed=402,
           t_known=t_known, gff_known=gff_known, gff_sigma=gff_sigma,
           save_path='D/D_DVCS')

# ── D(t) from DVCS + extracted D-term from arXiv:2602.19267 ────
# Includes the ChPT-extracted value D(0) = −4.3 ± 0.8 as an
# additional conditioning point at t = 0.

t_poles = [0, -0.11, -0.15, -0.20, -0.26, -0.34]
gff_poles = [-4.3, -1.14, -0.99, -0.93, -0.79, -0.66]
sigma_poles = [0.8, 0.1, 0.09, 0.07, 0.06, 0.08]

t_known   = torch.tensor(t_poles)
gff_known = torch.tensor(gff_poles)
gff_sigma = torch.tensor(sigma_poles)

sample_GFF(n_samples=10000, batch_size=500, seed=402,
           t_known=t_known, gff_known=gff_known, gff_sigma=gff_sigma,
           save_path='D/D_DVCS_ChPT', require_negative_at_zero = True)

# ═══════════════════════════════════════════════════════════════════
#  A(t) — EXTRAPOLATION FROM ChPT DATA
#  Conditioning on ChPT values at low |t|, extrapolating to −t = 2 GeV²
# ═══════════════════════════════════════════════════════════════════

# ── A(t), 8 ChPT points + A(0) ─────────────────────────────────

Amap = {0.0: (1.0, 0.0),
        -0.07: (0.9344737990368793, 0.032799784090198125),
        -0.09: (0.9193792231995355, 0.04157989120981328),
        -0.11: (0.904721632101717, 0.050951515554693856),
        -0.13: (0.8904732018861509, 0.06113684115149621),
        -0.15: (0.8766099051819893, 0.07235526901970449),
        -0.17: (0.8631106810583772, 0.08480995322634824),
        -0.19: (0.8499568400341218, 0.09867889505860611),
        -0.21: (0.8371316248877014, 0.11411076355642022)}

t_known = torch.tensor([i for i in Amap.keys()])
gff_known = torch.tensor([Amap[i][0] for i in Amap.keys()])
gff_sigma = torch.tensor([Amap[i][1] for i in Amap])

sample_GFF(n_samples = 10000, batch_size = 250, seed = 42, t_known = t_known, gff_known = gff_known, gff_sigma = gff_sigma,save_path = 'A/A_Extrapolated_8_Points')

# ── A(t), 5 ChPT points + A(0) ─────────────────────────────────

Amap = {0.0: (1.0, 0.0),
        -0.07: (0.9344737990368793, 0.032799784090198125),
        -0.09: (0.9193792231995355, 0.04157989120981328),
        -0.11: (0.904721632101717, 0.050951515554693856),
        -0.13: (0.8904732018861509, 0.06113684115149621),
        -0.15: (0.8766099051819893, 0.07235526901970449)}


t_known = torch.tensor([i for i in Amap.keys()])
gff_known = torch.tensor([Amap[i][0] for i in Amap.keys()])
gff_sigma = torch.tensor([Amap[i][1] for i in Amap])

sample_GFF(n_samples = 10000, batch_size = 250, seed = 42, t_known = t_known, gff_known = gff_known, gff_sigma = gff_sigma,save_path = 'A/A_Extrapolated_5_Points')


# ── A(t), 1 ChPT point + A(0) ──────────────────────────────────

Amap = {0.0: (1.0, 0.0),
        -0.15: (0.8766099051819893, 0.07235526901970449)}

t_known = torch.tensor([i for i in Amap.keys()])
gff_known = torch.tensor([Amap[i][0] for i in Amap.keys()])
gff_sigma = torch.tensor([Amap[i][1] for i in Amap])

sample_GFF(n_samples = 10000, batch_size = 250, seed = 42, t_known = t_known, gff_known = gff_known, gff_sigma = gff_sigma,save_path = 'A/A_Extrapolated_1_Points')


# ═══════════════════════════════════════════════════════════════════
#  J(t) — EXTRAPOLATION FROM ChPT DATA
# ═══════════════════════════════════════════════════════════════════

# ── J(t), 6 ChPT points + J(0) ─────────────────────────────────

Jmap = {0.0: (0.5, 0.0),
        -0.07: (0.45293105562058694, 0.01765335741455985),
        -0.09: (0.44205555276546415, 0.023197625735253122),
        -0.11: (0.4314428465035261, 0.029372036041822952),
        -0.13: (0.4210633235481441, 0.036187351941575706),
        -0.15: (0.4108928598293451, 0.04362930604790273),
        -0.17: (0.4009114536576321, 0.0516674551948898)}

t_known = torch.tensor([i for i in Jmap.keys()])
gff_known = torch.tensor([Jmap[i][0] for i in Jmap.keys()])
gff_sigma = torch.tensor([Jmap[i][1] for i in Jmap])

sample_GFF(n_samples = 10000, batch_size = 250, seed = 42, t_known = t_known, gff_known = gff_known, gff_sigma = gff_sigma,save_path = 'J/J_Extrapolated_6_Points.pt')

# ── J(t), 4 ChPT points + J(0) ─────────────────────────────────

Jmap = {0.0: (0.5, 0.0),
        -0.07: (0.45293105562058694, 0.01765335741455985),
        -0.09: (0.44205555276546415, 0.023197625735253122),
        -0.11: (0.4314428465035261, 0.029372036041822952),
        -0.13: (0.4210633235481441, 0.036187351941575706)}

t_known = torch.tensor([i for i in Jmap.keys()])
gff_known = torch.tensor([Jmap[i][0] for i in Jmap.keys()])
gff_sigma = torch.tensor([Jmap[i][1] for i in Jmap])

sample_GFF(n_samples = 10000, batch_size = 250, seed = 42, t_known = t_known, gff_known = gff_known, gff_sigma = gff_sigma,save_path = 'J/J_Extrapolated_4_Points.pt')


# ── J(t), 1 ChPT point + J(0) ──────────────────────────────────

Jmap = {0.0: (0.5, 0.0),
        -0.13: (0.4210633235481441, 0.036187351941575706)}

t_known = torch.tensor([i for i in Jmap.keys()])
gff_known = torch.tensor([Jmap[i][0] for i in Jmap.keys()])
gff_sigma = torch.tensor([Jmap[i][1] for i in Jmap])

sample_GFF(n_samples = 10000, batch_size = 250, seed = 42, t_known = t_known, gff_known = gff_known, gff_sigma = gff_sigma,save_path = 'J/J_Extrapolated_1_Points.pt')


# ═══════════════════════════════════════════════════════════════════
#  D(t) — EXTRAPOLATION FROM ChPT DATA
# ═══════════════════════════════════════════════════════════════════

# ── D(t), 8 ChPT points ────────────────────────────────────────

Dmap = {0.0: (-3.368411412493769, 0.7506160000000001),
        -0.02: (-3.0113675843363676, 0.7508874711035897),
        -0.04: (-2.69759613672832, 0.7523570652917586),
        -0.06: (-2.417763060909726, 0.7553269135841324),
        -0.08: (-2.165638518826084, 0.7595500752347096),
        -0.1: (-1.9368093017943888, 0.7645268149488289),
        -0.12: (-1.7280073268286298, 0.7696854257111111),
        -0.14: (-1.5367280048052816, 0.7744892054118326)}

gff_known = torch.tensor([Dmap[i][0] for i in Dmap.keys()])
gff_sigma = torch.tensor([Dmap[i][1] for i in Dmap])
t_known = torch.tensor([i for i in Dmap.keys()])

sample_GFF(n_samples = 20000, batch_size = 400, seed = 42, t_known = t_known, gff_known = gff_known,
           gff_sigma = gff_sigma,save_path = 'D/D_Extrapolated_8_Points', require_negative_at_zero = True)

# ── D(t), 5 ChPT points ────────────────────────────────────────

Dmap = {0.0: (-3.368411412493769, 0.7506160000000001),
        -0.02: (-3.0113675843363676, 0.7508874711035897),
        -0.04: (-2.69759613672832, 0.7523570652917586),
        -0.06: (-2.417763060909726, 0.7553269135841324),
        -0.08: (-2.165638518826084, 0.7595500752347096)
}

gff_known = torch.tensor([Dmap[i][0] for i in Dmap.keys()])
gff_sigma = torch.tensor([Dmap[i][1] for i in Dmap])
t_known = torch.tensor([i for i in Dmap.keys()])

sample_GFF(n_samples = 20000, batch_size = 400, seed = 42, t_known = t_known, gff_known = gff_known,
           gff_sigma = gff_sigma,save_path = 'D/D_Extrapolated_5_Points', require_negative_at_zero = True)

# ── D(t), 4 ChPT points ────────────────────────────────────────
# Note: the comment says "4 points" but the map contains 7 entries.

Dmap = {0.0: (-3.368411412493769, 0.7506160000000001),
        -0.02: (-3.0113675843363676, 0.7508874711035897),
        -0.04: (-2.69759613672832, 0.7523570652917586),
        -0.06: (-2.417763060909726, 0.7553269135841324),
}

gff_known = torch.tensor([Dmap[i][0] for i in Dmap.keys()])
gff_sigma = torch.tensor([Dmap[i][1] for i in Dmap])
t_known = torch.tensor([i for i in Dmap.keys()])

sample_GFF(n_samples = 20000, batch_size = 400, seed = 42, t_known = t_known, gff_known = gff_known,
           gff_sigma = gff_sigma,save_path = 'D/D_Extrapolated_4_Points', require_negative_at_zero = True)
