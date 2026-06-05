#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HESTIA ISO Monte-Carlo / Parameter-Sweep Wrapper
=================================================

Companion to `hestia_iso_proximity_sim.py`. This file ADDS a sweep capability
without modifying the original simulation: it imports the original module and,
for each of N iterations, overrides the ISO mean radius (R) and mass (M) over
fixed percentage ranges, regenerates the ISO, rebuilds the probe survey/descent
and lander descent trajectories, and harvests the resulting numbers.

It then produces a single full-panel figure containing:
  * ALL N probe + lander trajectories overlaid (3D and 2D x-z projection),
    coloured by iteration so the spread of paths is visible at a glance.
  * ALL N LiDAR-coverage curves and lander altitude profiles, overlaid.
  * Scatter/trend panels showing how the key landing & trajectory outputs
    (touchdown surface radius, probe & lander delta-v, propellant mass,
    survey-orbit period, descent duration, max survey speed) vary with R and M.

Why a controlled sweep instead of the original random seeds:
  The original module randomises R via `(1 +/- seed2/seed)`, which can divide by
  zero and produce wild, uncontrolled multipliers. For a sweep where you want to
  SEE the effect of R and M on the trajectory, we instead step R and M smoothly
  over user-set percentage ranges around their nominal values.

Run:
    python3 hestia_iso_sweep.py                 # default N=60, +/-30% on R and M
    python3 hestia_iso_sweep.py --n 25
    python3 hestia_iso_sweep.py --r-range 0.4 --m-range 0.5
    python3 hestia_iso_sweep.py --mode random   # random within the ranges
    python3 hestia_iso_sweep.py --save-only      # don't show, just save PNGs

Requires: numpy, matplotlib (+ the original module on the path).
"""
import os
import io
import argparse
import contextlib
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import ISO_AOCS as H   # the ORIGINAL simulation (unmodified)

# Nominal baseline values (the "brief" values from the original module)
R_NOMINAL = 500.0          # ISO mean radius [m]
M_NOMINAL = 2.6e11         # ISO mass [kg]


# --------------------------------------------------------------------------- #
#  Single-iteration driver
# --------------------------------------------------------------------------- #
def run_iteration(r_mean, mass, n_pts=900, mesh_seed=42):
    """
    Run ONE proximity-ops + lander pass for a given ISO radius and mass by
    overriding the original module's globals, then harvest all metrics and
    trajectories. Returns a dict of arrays + scalars. No plots, no animation.

    The original functions read `H.ISO_MASS` / `H.ISO_RMEAN` as module globals
    at call time, so overriding them here fully propagates into the trajectory,
    delta-v, and ADCS-sizing math.
    """
    # --- override the original module's globals for this iteration --------- #
    H.ISO_RMEAN = float(r_mean)
    H.ISO_MASS = float(mass)

    # --- build the (non-uniform) ISO at this radius ------------------------ #
    iso = H.make_iso_shape(r_mean=r_mean, seed=mesh_seed)
    iso_obj = type("ISO", (), {})()
    iso_obj.mass = float(mass)
    for k, v in iso.items():
        setattr(iso_obj, k, v)

    # --- probe survey + descent trajectory --------------------------------- #
    t_p, pos_p, phase_p, site_dir, v_dir = H.build_probe_trajectory(
        iso_obj, n_pts=n_pts)

    # --- LiDAR coverage curve ---------------------------------------------- #
    lidar = H.LidarCoverage(iso_obj)
    cov_track = np.empty(len(pos_p))
    for i, p in enumerate(pos_p):
        cov_track[i] = lidar.update(p)
    final_cov = float(cov_track[-1])
    idx50 = next((i for i, c in enumerate(cov_track) if c >= 0.5), None)
    t50_min = (t_p[idx50] / 60.0) if idx50 is not None else np.nan

    # --- lander descent trajectory ----------------------------------------- #
    t_l, pos_l, r_surface = H.build_lander_trajectory(
        iso_obj, site_dir, n_pts=max(200, n_pts // 2))
    alt_l = np.linalg.norm(pos_l, axis=1) - r_surface     # altitude vs surface

    # --- delta-v budgets + propellant -------------------------------------- #
    _, pb_dv, pb_dvm, pb_mp, _ = H.deltav_budget_probe(pos_p, phase_p)
    _, lb_dv, lb_dvm, lb_mp, _ = H.deltav_budget_lander(pos_l, r_surface)

    # --- survey-orbit period and descent duration -------------------------- #
    _, _, T_orbit = H.keplerian_orbit_radius(mass, 1500.0)
    descent_dur_min = (t_p[-1] - t_p[phase_p == 0][-1]) / 60.0

    # --- max survey speed (from the analytic velocity field) --------------- #
    v_survey_max = float(np.max(v_dir))

    return dict(
        r_mean=float(r_mean), mass=float(mass),
        t_p=t_p, pos_p=pos_p, phase_p=phase_p,
        cov_track=cov_track, final_cov=final_cov, t50_min=t50_min,
        t_l=t_l, pos_l=pos_l, alt_l=alt_l, r_surface=float(r_surface),
        probe_dv=float(pb_dv), probe_dv_margin=float(pb_dvm), probe_mprop=float(pb_mp),
        lander_dv=float(lb_dv), lander_dv_margin=float(lb_dvm), lander_mprop=float(lb_mp),
        T_orbit_min=float(T_orbit / 60.0), descent_dur_min=float(descent_dur_min),
        v_survey_max=v_survey_max, r_max=float(iso_obj.r_max),
        iso_fverts=iso_obj.fverts,
        v_dir=np.asarray(v_dir, dtype=float),      # ADDED: full velocity field for plotting
    )


# --------------------------------------------------------------------------- #
#  Sweep schedule
# --------------------------------------------------------------------------- #
def build_schedule(n, r_range, m_range, mode):
    """
    Return arrays (r_values, m_values) of length n.
      mode='grid'   : R and M each swept linearly across +/- range (paired).
      mode='cross'  : alternate sweeping R (M fixed) then M (R fixed) so each
                      effect is isolated -- best for "see R vs M" separately.
      mode='random' : random uniform within +/- range (Monte-Carlo).
    """
    rlo, rhi = R_NOMINAL * (1 - r_range), R_NOMINAL * (1 + r_range)
    mlo, mhi = M_NOMINAL * (1 - m_range), M_NOMINAL * (1 + m_range)

    if mode == "grid":
        r_vals = np.linspace(rlo, rhi, n)
        m_vals = np.linspace(mlo, mhi, n)
    elif mode == "cross":
        # first half: vary R at nominal M; second half: vary M at nominal R
        nh = n // 2
        r_vals = np.concatenate([np.linspace(rlo, rhi, nh),
                                 np.full(n - nh, R_NOMINAL)])
        m_vals = np.concatenate([np.full(nh, M_NOMINAL),
                                 np.linspace(mlo, mhi, n - nh)])
    elif mode == "random":
        rng = np.random.default_rng(0)
        r_vals = rng.uniform(rlo, rhi, n)
        m_vals = rng.uniform(mlo, mhi, n)
    else:
        raise ValueError(f"unknown mode {mode!r}")
    return r_vals, m_vals


# --------------------------------------------------------------------------- #
#  Full-panel plotting
# --------------------------------------------------------------------------- #
def plot_overlays(results, r_vals, m_vals, fname_traj, fname_metrics):
    """
    Two figures:
      A) overlay figure : all trajectories + all coverage / altitude curves,
                          one on top of the other, coloured by iteration.
      B) metrics figure : landing & trajectory outputs vs R and vs M.
    """
    n = len(results)
    colors = cm.viridis(np.linspace(0, 1, n))

    # ===================================================================== #
    #  FIGURE A : trajectory + time-history overlays
    # ===================================================================== #
    figA = plt.figure(figsize=(16, 9))
    figA.suptitle(f"HESTIA ISO sweep - {n} iterations: all trajectories overlaid",
                  fontsize=15, fontweight="bold")

    ax3d = figA.add_subplot(2, 3, 1, projection="3d")
    axXZ = figA.add_subplot(2, 3, 2)
    axCov = figA.add_subplot(2, 3, 3)
    axLand3d = figA.add_subplot(2, 3, 4, projection="3d")
    axLandAlt = figA.add_subplot(2, 3, 5)
    axColbar = figA.add_subplot(2, 3, 6)

    # --- show ONE representative ISO mesh (the median-R run) as context ---- #
    rep = int(np.argsort([r["r_mean"] for r in results])[n // 2])
    tri = Poly3DCollection(results[rep]["iso_fverts"] / 1000, alpha=0.20)
    tri.set_facecolor((0.55, 0.5, 0.45)); tri.set_edgecolor((0.3, 0.28, 0.25, 0.15))
    ax3d.add_collection3d(tri)

    for k, res in enumerate(results):
        p = res["pos_p"] / 1000.0
        ax3d.plot(p[:, 0], p[:, 1], p[:, 2], lw=0.8, color=colors[k], alpha=0.7)
        # x-z projection (cleaner 2D read of the spiral-down)
        axXZ.plot(p[:, 0], p[:, 2], lw=0.8, color=colors[k], alpha=0.7)
        axCov.plot(res["t_p"] / 60.0, res["cov_track"] * 100.0,
                   lw=1.0, color=colors[k], alpha=0.8)
        pl = res["pos_l"] / 1000.0
        axLand3d.plot(pl[:, 0], pl[:, 1], pl[:, 2], lw=0.9, color=colors[k], alpha=0.7)
        axLandAlt.plot(res["t_l"] / 60.0, res["alt_l"],
                       lw=1.0, color=colors[k], alpha=0.8)

    lim = max(r["r_max"] for r in results) / 1000.0 * 5
    ax3d.set_xlim(-lim, lim); ax3d.set_ylim(-lim, lim); ax3d.set_zlim(-lim, lim)
    ax3d.set_xlabel("x [km]"); ax3d.set_ylabel("y [km]"); ax3d.set_zlabel("z [km]")
    ax3d.set_title("All probe survey+descent tracks (3D)")

    # lander panel: tight autoscale to the descent tracks (otherwise the
    # ~200 m descent is invisible inside the probe-scale +/-5 km box).
    lpts = np.vstack([r["pos_l"] for r in results]) / 1000.0
    cx, cy, cz = lpts.mean(0)
    half = float(np.max(np.abs(lpts - [cx, cy, cz]))) * 1.15 + 1e-6
    axLand3d.set_xlim(cx - half, cx + half)
    axLand3d.set_ylim(cy - half, cy + half)
    axLand3d.set_zlim(cz - half, cz + half)
    axLand3d.set_xlabel("x [km]"); axLand3d.set_ylabel("y [km]"); axLand3d.set_zlabel("z [km]")
    axLand3d.set_title("All lander descent tracks (3D, autoscaled)")

    axXZ.set_title("Probe tracks - x-z projection")
    axXZ.set_xlabel("x [km]"); axXZ.set_ylabel("z [km]")
    axXZ.grid(True, alpha=0.3); axXZ.set_aspect("equal", adjustable="datalim")

    axCov.set_title("LiDAR coverage curves (all runs)")
    axCov.set_xlabel("Time [min]"); axCov.set_ylabel("Scanned [%]")
    axCov.axhline(50, color="k", ls="--", lw=1); axCov.set_ylim(0, 100)
    axCov.grid(True, alpha=0.3)

    axLandAlt.set_title("Lander altitude profiles (all runs)")
    axLandAlt.set_xlabel("Time [min]"); axLandAlt.set_ylabel("Altitude above surface [m]")
    axLandAlt.grid(True, alpha=0.3)

    # colourbar legend mapping iteration -> (R, M)
    axColbar.axis("off")
    sm = cm.ScalarMappable(cmap="viridis",
                           norm=plt.Normalize(vmin=0, vmax=n - 1))
    cb = figA.colorbar(sm, ax=axColbar, fraction=0.5, pad=0.05)
    cb.set_label("iteration index")
    txt = "Iteration -> (R [m], M [kg]):\n" + "\n".join(
        f"{k:2d}: R={r['r_mean']:6.1f}, M={r['mass']:.3e}"
        for k, r in enumerate(results))
    axColbar.text(0.0, 0.98, txt, va="top", ha="left", fontsize=7, family="monospace",
                  transform=axColbar.transAxes)

    figA.tight_layout(rect=[0, 0, 1, 0.96])
    figA.savefig(fname_traj, dpi=130)

    # ===================================================================== #
    #  FIGURE B : landing & trajectory metrics vs R and vs M
    # ===================================================================== #
    R = np.array([r["r_mean"] for r in results])
    M = np.array([r["mass"] for r in results])
    metrics = {
        "Touchdown surface radius [m]": np.array([r["r_surface"] for r in results]),
        "Probe delta-v [m/s]":          np.array([r["probe_dv"] for r in results]),
        "Lander delta-v [m/s]":         np.array([r["lander_dv"] for r in results]),
        "Lander propellant [kg]":       np.array([r["lander_mprop"] for r in results]),
        "Survey-orbit period [min]":    np.array([r["T_orbit_min"] for r in results]),
        "Max survey speed [m/s]":       np.array([r["v_survey_max"] for r in results]),
    }

    figB, axes = plt.subplots(len(metrics), 2, figsize=(12, 3.0 * len(metrics)))
    figB.suptitle("Landing & trajectory metrics vs ISO radius (R) and mass (M)",
                  fontsize=15, fontweight="bold")

    for row, (name, vals) in enumerate(metrics.items()):
        axR = axes[row, 0]
        axM = axes[row, 1]
        axR.scatter(R, vals, c=np.arange(n), cmap="viridis", s=40, edgecolor="k", lw=0.4)
        axM.scatter(M, vals, c=np.arange(n), cmap="viridis", s=40, edgecolor="k", lw=0.4)
        # connect with a faint trend line when monotone-ish (sorted by axis)
        oR = np.argsort(R); oM = np.argsort(M)
        axR.plot(R[oR], vals[oR], color="tab:gray", lw=0.8, alpha=0.5)
        axM.plot(M[oM], vals[oM], color="tab:gray", lw=0.8, alpha=0.5)
        axR.set_ylabel(name, fontsize=9)
        axR.grid(True, alpha=0.3); axM.grid(True, alpha=0.3)
        if row == 0:
            axR.set_title("vs R [m]"); axM.set_title("vs M [kg]")
        if row == len(metrics) - 1:
            axR.set_xlabel("ISO mean radius R [m]")
            axM.set_xlabel("ISO mass M [kg]")

    figB.tight_layout(rect=[0, 0, 1, 0.97])
    figB.savefig(fname_metrics, dpi=130)

    return figA, figB


# --------------------------------------------------------------------------- #
#  Velocity overlay plotting (ADDED - non-destructive)
# --------------------------------------------------------------------------- #
def plot_velocity_overlays(results, fname_velocity):
    """
    ADDED: velocity figure. Two panels:
      LEFT  : probe speed (|v|) vs time, all runs overlaid, coloured by iteration.
      RIGHT : probe speed vs distance-from-ISO-centre, all runs overlaid,
              showing how survey/descent speed scales with range.
    Reuses the v_dir field harvested in run_iteration; no sim re-run.
    """
    n = len(results)
    colors = cm.viridis(np.linspace(0, 1, n))

    figV, (axT, axR) = plt.subplots(1, 2, figsize=(14, 6))
    figV.suptitle(f"HESTIA ISO sweep - probe velocity profiles ({n} runs overlaid)",
                  fontsize=15, fontweight="bold")

    for k, res in enumerate(results):
        v = res["v_dir"]
        m = min(len(v), len(res["t_p"]), len(res["pos_p"]))
        t = res["t_p"][:m] / 60.0
        rng_km = np.linalg.norm(res["pos_p"][:m], axis=1) / 1000.0
        axT.plot(t, v[:m], lw=0.9, color=colors[k], alpha=0.8)
        axR.plot(rng_km, v[:m], lw=0.9, color=colors[k], alpha=0.8)

    axT.set_title("Probe speed vs time")
    axT.set_xlabel("Time [min]"); axT.set_ylabel("Speed |v| [m/s]")
    axT.grid(True, alpha=0.3)

    axR.set_title("Probe speed vs range from ISO centre")
    axR.set_xlabel("Range [km]"); axR.set_ylabel("Speed |v| [m/s]")
    axR.grid(True, alpha=0.3)

    sm = cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=0, vmax=n - 1))
    cb = figV.colorbar(sm, ax=[axT, axR], fraction=0.04, pad=0.02)
    cb.set_label("iteration index")

    figV.savefig(fname_velocity, dpi=130)
    return figV


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description="Sweep the HESTIA ISO over N iterations of varying R and M.")
    ap.add_argument("--n", type=int, default=60, help="number of iterations (default 60)")
    ap.add_argument("--r-range", type=float, default=0.30,
                    help="fractional +/- range on R (default 0.30 = +/-30%%)")
    ap.add_argument("--m-range", type=float, default=0.30,
                    help="fractional +/- range on M (default 0.30 = +/-30%%)")
    ap.add_argument("--mode", choices=["grid", "cross", "random"], default="grid",
                    help="how R and M are stepped (default grid)")
    ap.add_argument("--n-pts", type=int, default=900,
                    help="trajectory sample points per run (default 900)")
    ap.add_argument("--save-only", action="store_true",
                    help="save figures without opening a window")
    args = ap.parse_args()

    r_vals, m_vals = build_schedule(args.n, args.r_range, args.m_range, args.mode)

    print(f"Running {args.n} iterations | mode={args.mode} | "
          f"R in [{r_vals.min():.1f}, {r_vals.max():.1f}] m | "
          f"M in [{m_vals.min():.3e}, {m_vals.max():.3e}] kg")

    results = []
    for k, (r, m) in enumerate(zip(r_vals, m_vals)):
        # suppress the original module's stray prints during each call
        with contextlib.redirect_stdout(io.StringIO()):
            res = run_iteration(r, m, n_pts=args.n_pts)
        results.append(res)
        print(f"  [{k+1:2d}/{args.n}] R={r:7.1f} m  M={m:.3e} kg  ->  "
              f"r_surf={res['r_surface']:7.1f} m  "
              f"probe dv={res['probe_dv']:.3f}  lander dv={res['lander_dv']:.3f}  "
              f"cov={res['final_cov']*100:4.0f}%")

    fA = "hestia_sweep_trajectories.png"
    fB = "hestia_sweep_metrics.png"
    plot_overlays(results, r_vals, m_vals, fA, fB)
    print(f"\nSaved overlay figure   : {fA}")
    print(f"Saved metrics figure   : {fB}")

    fV = "hestia_sweep_velocity.png"                       # ADDED
    plot_velocity_overlays(results, fV)                    # ADDED
    print(f"Saved velocity figure  : {fV}")                # ADDED

    if not args.save_only:
        plt.show()


if __name__ == "__main__":
    if os.environ.get("MPLBACKEND", "").lower() in ("", "agg", "template"):
        # honour an interactive backend if one is available locally
        try:
            cur = matplotlib.get_backend().lower()
            if cur in ("agg", "template"):
                for be in ("MacOSX", "QtAgg", "TkAgg"):
                    try:
                        matplotlib.use(be, force=True)
                        break
                    except Exception:
                        continue
        except Exception:
            pass
    main()