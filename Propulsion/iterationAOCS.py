#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HESTIA Interstellar-Object Proximity-Operations & Lander ADCS Simulation
=========================================================================

Group 04 - AE3200 Design Synthesis Exercise
"Exploring an Interstellar Object" - Rendezvous + Lander (Heliocentric standby), HESTIA.

This script simulates the close-proximity phase of the HESTIA mission AFTER the
spacecraft has matched the ISO's heliocentric velocity. Per the assumption set
(AS002), the probe and ISO share the same velocity vector, so the encounter is
modelled in a free-space frame co-moving with the ISO (the ISO is "still").

Two vehicles are simulated in sequence:
  1) The MOTHER PROBE  -- flies a survey orbit around the ISO, performing LiDAR
                          scans until >= 50% of the surface is mapped, then
                          descends to a close stand-off point for lander release,
                          then RETURNS to its original survey orbit and acts as
                          a comm relay for the lander.
  2) The LANDER         -- separates and descends to the surface under its own
                          ADCS / RCS. It is rendered as a visible cube and shown
                          in the probe animation after release.

ALTITUDE-FOLLOWING ORBIT (this revision)
----------------------------------------
The survey orbit, descent spiral, and post-release return no longer use a fixed
radius (ISO_RMEAN + altitude). For a strongly elongated ISO that approximation
puts the probe INSIDE the body along the long axis. Instead, at every point the
local surface radius along the current bearing is found by a Moller-Trumbore
ray/triangle intersection (surface_radius), and the commanded radius is
"local surface radius + altitude". The local circular speed uses the
instantaneous radius (v = sqrt(mu/r)), so velocity tracks the true geometry.

COLLISION / LANDING DETECTION (this revision)
---------------------------------------------
* PROBE  : signed altitude above the surface along its own bearing is computed
           for the whole trajectory; if it ever goes negative the probe has
           crashed (time + position reported).
* LANDER : signed altitude is tracked through the descent; touchdown is flagged
           the first time it falls within TOUCHDOWN_TOL of the surface, and any
           sub-surface penetration before that is reported as a hard impact.

Outputs
-------
* A LIVE matplotlib animation showing the vehicle flying around the non-uniform
  ISO, scanned patches lighting up, the descent/landing track, the released
  lander, and the probe<->lander comm link, plus live time histories.
* Static summary figures: disturbance torques, coverage curve, 3D tracks,
  thrust & cumulative-mass plots.
* A printed report of all valuable numbers, the ADCS hardware selection for both
  vehicles, and the collision/landing results.

Run with:
    python3 hestia_iso_proximity_sim.py            # mother probe, then lander
    python3 hestia_iso_proximity_sim.py --fast     # coarser time-step (quick look)
    python3 hestia_iso_proximity_sim.py --no-anim  # skip the live animation

Requires: numpy, scipy, matplotlib  (run locally, not in a headless container
                                     if you want the live animation).
"""
import random
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from scipy.spatial import ConvexHull

# ----------------------------------------------------------------------------- #
#  Physical constants
# ----------------------------------------------------------------------------- #
G          = 6.67430e-11        # gravitational constant            [m^3 kg^-1 s^-2]
AU         = 1.495978707e11     # astronomical unit                 [m]
C_LIGHT    = 2.99792458e8       # speed of light                    [m s^-1]
L_SUN      = 3.828e26           # solar luminosity                  [W]
MU_SUN     = 1.32712440018e20   # Sun gravitational parameter       [m^3 s^-2]
SIGMA_SB   = 5.670374419e-8     # Stefan-Boltzmann                  [W m^-2 K^-4]

# ----------------------------------------------------------------------------- #
#  Mission / target parameters (from the HESTIA midterm report + problem brief)
# ----------------------------------------------------------------------------- #
ISO_MASS   = 2.6e11             # ISO mass                          [kg]   (brief)
ISO_RMEAN  = 500.0              # ISO mean radius                   [m]    (brief)
R_HELIO    = 100.0 * AU         # rendezvous heliocentric distance  [m]    (report 3.4/3.8)

# --- Mother probe (HESTIA bus after kick-stage jettison) --------------------- #
PROBE_MASS   = 2000             # spacecraft wet mass at ISO        [kg]   (Table 3.10)
PROBE_SIDE   = 2.0              # cube side length                  [m]    (sec. 3.11.1)
PROBE_CD     = 1.4             # surface area coeff for SRP (cube faces, conservative)
PROBE_REFL   = 0.6             # reflectivity (MLI / coatings, 0=black 1=mirror)
PROBE_CG_OFF = 0.05            # CoP-CoG offset, fraction of side  -> 0.10 m
PROBE_RES_DIP = 0.5           # residual magnetic dipole          [A m^2] (heritage)
PROBE_ALT     = 1500          # survey altitude ABOVE LOCAL SURFACE [m]

# --- Lander (Philae-class, sec. 3.12.2) -------------------------------------- #
LANDER_MASS   = 95            # Philae mass + 10% margin          [kg]   (Table 3.10)
LANDER_SIDE   = 1.0           # compact lander envelope           [m]
LANDER_REFL   = 0.5
LANDER_CG_OFF = 0.06
LANDER_RES_DIP = 0.1          # residual magnetic dipole          [A m^2]

# --- Heliospheric environment at 100 AU (for "Sun-dominated" disturbances) --- #
B_FIELD_100AU = 1.0e-9          # interplanetary B-field ~1 nT      [T]
SOLAR_WIND_NP = 0.5e4           # proton number density ~0.005 cm^-3 -> m^-3
SOLAR_WIND_V  = 4.0e5           # solar-wind bulk speed             [m s^-1]
M_PROTON      = 1.6726e-27      # proton mass                       [kg]

SAFETY_MARGIN = 0.05            # AS001: ADCS sized with 5% margin

# --- Delta-v budget / propellant parameters ---------------------------------- #
G0            = 9.80665         # standard gravity (Isp -> exhaust vel)   [m s^-2]
RCS_ISP       = 30.0            # cold-gas GN2 RCS specific impulse        [s]
LAN_ISP       = 76
DV_MARGIN     = 0.10            # 10% delta-v margin (ECSS-style)
N_STATIONKEEP = 6              # number of survey-orbit station-keeping corrections
SK_DV_EACH    = 0.05           # delta-v per station-keeping pulse        [m/s]
DESAT_DV_BUDGET = 0.10         # cumulative RCS delta-v for wheel desats   [m/s]
ATT_DV_BUDGET   = 0.20         # cumulative delta-v-equivalent for attitude/RCS pulsing [m/s]
LANDER_REL_DV   = 0.10         # probe retreat after lander release        [m/s]

# --- Collision / landing detection ------------------------------------------- #
TOUCHDOWN_TOL = 1.5           # [m] lander within this of surface == landed
CRASH_TOL     = 0.0           # [m] probe altitude below this == crashed (sub-surface)

# --- Lander mounting geometry ------------------------------------------------ #
LANDER_MOUNT_OFFSET = np.array([0.0, 0.0, -(PROBE_SIDE / 2 + LANDER_SIDE / 2)])

# --- Visualisation scales ---------------------------------------------------- #
PROBE_VIS_SIDE  = 220.0         # rendered probe cube side          [m]  (visual only)
LANDER_VIS_SIDE = 140.0         # rendered lander cube side         [m]  (visual only)

# --- Hardcoded moments of inertia -------------------------------------------- #
PROBE_I_COMBINED = np.array([1804.73, 1804.73, 1584.95])  # [kg m^2] PLACEHOLDER
PROBE_I_POST     = np.array([1568.67, 1568.67, 1568.67])  # [kg m^2] PLACEHOLDER
PROBE_DELTA_COM  = np.array([0.0,     0.0,     0.0598])   # [m]      PLACEHOLDER
LANDER_I         = np.array([11.014,   11.724,   12.679])  # [kg m^2] PLACEHOLDER


# ============================================================================= #
#  1.  NON-UNIFORM ISO SHAPE MODEL
# ============================================================================= #
def make_iso_shape(r_mean=ISO_RMEAN, n_lat=22, n_lon=44, seed=7):
    """
    Build a lumpy, non-uniform "potato" ISO by perturbing a sphere with a sum of
    low-order spherical-harmonic-like bumps. Returns a triangulated convex-ish
    surface mesh (vertices, faces, face-centroids, face-normals, face-areas).

    The shape is intentionally irregular (elongated + dented) like 1I/'Oumuamua
    or a cometary nucleus, satisfying the "non-uniform shape" requirement.
    """
    rng = np.random.default_rng(seed)

    lat = np.linspace(-np.pi / 2, np.pi / 2, n_lat)
    lon = np.linspace(-np.pi, np.pi, n_lon)
    LON, LAT = np.meshgrid(lon, lat)

    # Base radius field: elongate along x (a/b/c axes), then add random lumps.
    a, b, c = 4.2, 0.85, 0.80          # tri-axial elongation (very non-spherical)
    x0 = a * np.cos(LAT) * np.cos(LON)
    y0 = b * np.cos(LAT) * np.sin(LON)
    z0 = c * np.sin(LAT)
    base = np.sqrt(x0**2 + y0**2 + z0**2)

    # Low-order random topography (craters / ridges)
    bumps = np.zeros_like(LAT)
    for _ in range(8):
        l = rng.integers(2, 6)
        m = rng.integers(0, l + 1)
        amp = rng.uniform(-0.16, 0.16)
        phase = rng.uniform(0, 2 * np.pi)
        bumps += amp * np.cos(m * LON + phase) * np.cos(l * LAT)

    r_field = base * (1.0 + bumps)
    r_field = np.clip(r_field, 0.55, None)

    # Scale so the *volume-equivalent mean radius* matches r_mean.
    r_field *= r_mean / np.mean(r_field)

    X = 4 * r_field * np.cos(LAT) * np.cos(LON)
    Y = r_field * np.cos(LAT) * np.sin(LON)
    Z = r_field * np.sin(LAT)

    verts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

    # Triangulate via convex hull (gives clean closed surface for rendering/coverage)
    hull = ConvexHull(verts)
    faces = hull.simplices
    fverts = verts[faces]                       # (F,3,3)

    centroids = fverts.mean(axis=1)             # (F,3)
    # Outward normals
    v0, v1, v2 = fverts[:, 0], fverts[:, 1], fverts[:, 2]
    n = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.linalg.norm(n, axis=1)
    n_unit = n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)
    # ensure outward
    outward = np.sign(np.einsum('ij,ij->i', n_unit, centroids))
    outward[outward == 0] = 1.0
    n_unit *= outward[:, None]

    return dict(verts=verts, faces=faces, fverts=fverts,
                centroids=centroids, normals=n_unit, areas=areas,
                r_mean=r_mean, r_max=np.linalg.norm(verts, axis=1).max())


# ============================================================================= #
#  1a.  RAY-SURFACE INTERSECTION  (altitude-following + collision detection)
# ============================================================================= #
def _iso_field(iso, key):
    """Read a mesh field whether `iso` is the dict or the iso_obj namespace."""
    return iso[key] if isinstance(iso, dict) else getattr(iso, key)


def surface_radius(iso, direction):
    """
    Distance from the ISO centre to its surface along `direction`, via a
    vectorised Moller-Trumbore ray/triangle intersection over all faces.

    The ray origin is the ISO centroid (0,0,0); the ray points along
    `direction`. The FARTHEST valid hit is returned (the outer surface) so a
    wrapped survey orbit clears the whole body even when the ISO is extremely
    elongated. Robust for star-convex "potato" shapes; falls back to the
    nearest-bearing centroid radius if (degenerately) no triangle is hit.
    """
    faces = _iso_field(iso, 'faces')
    verts = _iso_field(iso, 'verts')
    centroids = _iso_field(iso, 'centroids')

    d = np.asarray(direction, float)
    d = d / (np.linalg.norm(d) + 1e-15)

    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    e1 = v1 - v0
    e2 = v2 - v0
    p = np.cross(d, e2)                                   # (F,3)
    det = np.einsum('ij,ij->i', e1, p)                    # (F,)
    mask = np.abs(det) > 1e-9
    inv = np.where(mask, 1.0 / np.where(mask, det, 1.0), 0.0)

    tvec = -v0                                            # origin (0) - v0
    u = np.einsum('ij,ij->i', tvec, p) * inv
    q = np.cross(tvec, e1)                                # (F,3)
    v = (q @ d) * inv
    t = np.einsum('ij,ij->i', e2, q) * inv

    hit = mask & (u >= -1e-6) & (v >= -1e-6) & (u + v <= 1 + 1e-6) & (t > 1e-6)
    if not np.any(hit):
        cu = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
        return float(np.linalg.norm(centroids[np.argmax(cu @ d)]))
    return float(t[hit].max())


def altitude_above_surface(iso, pos):
    """
    Signed altitude of a point above the ISO surface along its OWN bearing.
    Positive = outside the body; negative = inside the body (collision).
    """
    r = np.linalg.norm(pos)
    if r < 1e-9:
        return -surface_radius(iso, [1.0, 0.0, 0.0])
    return r - surface_radius(iso, pos)


# ============================================================================= #
#  1b.  VEHICLE GEOMETRY HELPERS
# ============================================================================= #
def cube_poly_verts(center, side):
    """
    Return the 6 quadrilateral faces of an axis-aligned cube of the given side
    length centred on `center`, as a list of (4,3) vertex arrays suitable for a
    Poly3DCollection. Used to render the probe / lander bodies.
    """
    h = side / 2.0
    c = np.array([[sx, sy, sz] for sx in (-h, h) for sy in (-h, h) for sz in (-h, h)])
    c = c + np.asarray(center, dtype=float)
    F = [[0, 1, 3, 2], [4, 5, 7, 6],        # -x, +x
         [0, 1, 5, 4], [2, 3, 7, 6],        # -y, +y
         [0, 2, 6, 4], [1, 3, 7, 5]]        # -z, +z
    return [c[f] for f in F]


# RCS layout: 4 clusters of 3 thrusters each, on OPPOSING corners of the cube
# (tetrahedral pattern). Each cluster's 3 nozzles point outward along the body
# x, y, z axes -> full 6-DOF torque + translation authority with 12 thrusters.
RCS_CORNER_SIGNS = np.array([[ 1,  1,  1],
                             [ 1, -1, -1],
                             [-1,  1, -1],
                             [-1, -1,  1]], dtype=float)


def rcs_segments(center, side, nozzle_len=None):
    """Line segments for the 4x3 corner-mounted RCS clusters of a cube body."""
    h = side / 2.0
    if nozzle_len is None:
        nozzle_len = 0.45 * side
    center = np.asarray(center, dtype=float)
    segs = []
    for s in RCS_CORNER_SIGNS:
        corner = center + s * h
        for ax in range(3):
            dvec = np.zeros(3)
            dvec[ax] = s[ax] * nozzle_len
            segs.append(np.array([corner, corner + dvec]))
    return segs


def rotation_between(a, b):
    """Rotation matrix mapping unit vector a onto unit vector b (Rodrigues)."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a / np.linalg.norm(a); b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = np.linalg.norm(v)
    if s < 1e-12:
        if c > 0:
            return np.eye(3)
        p = np.array([1.0, 0.0, 0.0])
        if abs(a[0]) > 0.9:
            p = np.array([0.0, 1.0, 0.0])
        axis = np.cross(a, p); axis /= np.linalg.norm(axis)
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        return np.eye(3) + 2 * (K @ K)
    K = np.array([[0, -v[2], v[1]],
                  [v[2], 0, -v[0]],
                  [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * ((1 - c) / s**2)


# ============================================================================= #
#  2.  RIGID-BODY INERTIA OF A UNIFORM CUBE BUS
# ============================================================================= #
def cube_inertia(mass, side):
    """Principal moments of inertia of a uniform solid cube about its centre."""
    I = (1.0 / 6.0) * mass * side**2
    return np.array([I, I, I])      # I_xx = I_yy = I_zz for a cube


def probe_inertia_with_lander():
    """
    Inertia tensor of the combined probe+lander system about the PROBE CoM.
    Returns the hardcoded constant; derivation kept for reference.
    """
    return PROBE_I_COMBINED


def probe_inertia_post_release():
    """
    Inertia tensor of the probe AFTER lander deployment (lander mass removed),
    plus the small CoM shift. Returns hardcoded constants.
    """
    return PROBE_I_POST, PROBE_DELTA_COM


# ============================================================================= #
#  3.  DISTURBANCE-TORQUE MODELS (Sun-dominated, evaluated at 100 AU)
# ============================================================================= #
class DisturbanceModel:
    """
    Evaluates the magnitude of each disturbance-torque source acting on a vehicle
    in the ISO frame at the rendezvous heliocentric distance.

    Sources (all "around the Sun"; per AS006 the ISO's own contributions are
    negligible; AS001 makes the ISO gravity-gradient negligible too):
        - Solar radiation pressure (SRP) torque
        - Solar gravity-gradient torque
        - Solar-wind / charged-particle drag torque
        - Magnetic torque from residual dipole in the interplanetary field
    """

    def __init__(self, mass, side, refl, cg_off_frac, res_dipole,
                 r_helio=R_HELIO, inertia=None):
        self.mass = mass
        self.side = side
        self.area = side * side                     # one face area  [m^2]
        self.refl = refl
        self.cp_cg = cg_off_frac * side             # CoP-CoG offset [m]
        self.dipole = res_dipole
        self.r_helio = r_helio
        self.I = inertia if inertia is not None else cube_inertia(mass, side)

        # Solar flux at rendezvous distance
        self.solar_flux = L_SUN / (4 * np.pi * r_helio**2)     # [W/m^2]
        self.nsw = SOLAR_WIND_NP                                # use the 100-AU value
        self.B = B_FIELD_100AU

    def srp(self, theta):
        """Solar radiation pressure torque [N m]. theta = illuminated-face tilt."""
        P = self.solar_flux / C_LIGHT                          # radiation pressure [Pa]
        F = P * self.area * (1 + self.refl) * np.abs(np.cos(theta))
        return F * self.cp_cg

    def gravity_gradient(self, phi):
        """Solar gravity-gradient torque [N m]. phi = body tilt wrt local vertical."""
        Imax, Imin = self.I.max(), self.I.min()
        coef = 3 * MU_SUN / (2 * self.r_helio**3)
        return coef * (Imax - Imin) * np.abs(np.sin(2 * phi))

    def solar_wind(self, theta):
        """Solar-wind charged-particle drag torque [N m]."""
        Pdyn = self.nsw * M_PROTON * SOLAR_WIND_V**2           # dynamic pressure [Pa]
        F = Pdyn * self.area * np.abs(np.cos(theta))
        return F * self.cp_cg

    def magnetic(self):
        """Residual-dipole / interplanetary-field magnetic torque [N m]."""
        return self.dipole * self.B

    def evaluate_timeseries(self, t, omega_attitude=2 * np.pi / 600.0):
        """Return dict of torque time-histories as the body slowly reorients."""
        theta = omega_attitude * t                  # face tilt sweeps slowly
        phi = 0.5 * omega_attitude * t
        srp = self.srp(theta)
        gg = self.gravity_gradient(phi)
        sw = self.solar_wind(theta)
        mag = np.full_like(t, self.magnetic())
        total = np.sqrt(srp**2 + gg**2 + sw**2 + mag**2)
        return dict(SRP=srp, GravGrad=gg, SolarWind=sw,
                    Magnetic=mag, Total=total)

    def worst_case(self):
        """Worst-case magnitudes for sizing."""
        return dict(SRP=self.srp(0.0),
                    GravGrad=self.gravity_gradient(np.pi / 4),
                    SolarWind=self.solar_wind(0.0),
                    Magnetic=self.magnetic())


# ============================================================================= #
#  4.  PROXIMITY TRAJECTORY  (probe survey orbit -> descent -> return)
# ============================================================================= #
def keplerian_orbit_radius(iso_mass, alt):
    """Circular-orbit speed & period for an orbit at radius (r_mean+alt).
    (Used only for nominal pacing / period estimates; the actual orbit radius
    is altitude-following, see build_probe_trajectory.)"""
    mu = G * iso_mass
    r = ISO_RMEAN + alt
    v = np.sqrt(mu / r)
    T = 2 * np.pi * np.sqrt(r**3 / mu)
    return r, v, T


def build_probe_trajectory(iso, n_survey_orbits=3.0, survey_alt=PROBE_ALT,
                           standoff_alt=200.0, n_pts=1400):
    """
    Survey phase: a precessing near-polar circular orbit whose RADIUS follows the
    local surface (surface_radius(direction) + survey_alt) so the ground track
    walks around the ISO and covers > 50% of the surface WITHOUT ever entering
    an arbitrarily elongated body. The local circular speed uses the
    instantaneous radius, so it updates with the geometry.

    Descent phase: spiral from the (local) survey radius down to a stand-off
    altitude above the selected landing site, where the lander is released. The
    spiral radius also follows the local surface, with the altitude interpolated
    from survey_alt down to standoff_alt so it never clips terrain.

    Returns (t, pos[N,3], phase[N], site_dir, v_n)  phase: 0=survey, 1=descent.
    """
    mu = G * ISO_MASS

    # ---- survey: precessing circular orbit, altitude-following ------------ #
    n_surv = int(n_pts * 0.7)
    # pace true anomaly with a nominal period (mean radius); timing only.
    r_nom = ISO_RMEAN + survey_alt
    T_s = 2 * np.pi * np.sqrt(r_nom**3 / mu)
    t_surv = np.linspace(0, n_survey_orbits * T_s, n_surv)
    nu = 2 * np.pi * t_surv / T_s                       # true anomaly
    prec = np.linspace(0, np.pi, n_surv)                # nodal precession
    incl = np.deg2rad(85.0)                             # near-polar

    ux = np.cos(nu) * np.cos(prec) - np.sin(nu) * np.cos(incl) * np.sin(prec)
    uy = np.cos(nu) * np.sin(prec) + np.sin(nu) * np.cos(incl) * np.cos(prec)
    uz = np.sin(nu) * np.sin(incl)
    dirs_s = np.column_stack([ux, uy, uz])
    dirs_s /= np.linalg.norm(dirs_s, axis=1, keepdims=True)

    # instantaneous orbit radius = local surface radius + survey altitude
    r_local = np.array([surface_radius(iso, d) for d in dirs_s])
    r_s_arr = r_local + survey_alt
    survey = dirs_s * r_s_arr[:, None]

    # local circular speed at each instantaneous radius (updates with geometry)
    v_n = np.sqrt(mu / r_s_arr)

    # ---- descent: spiral from local survey radius to local stand-off ------ #
    n_desc = n_pts - n_surv
    # land near the +x equatorial bulge (a "flat" lit region)
    site_dir = iso['normals'][np.argmax(iso['centroids'][:, 0])] \
        if isinstance(iso, dict) else iso.normals[np.argmax(iso.centroids[:, 0])]
    site_dir = site_dir / np.linalg.norm(site_dir)

    ang = np.linspace(0, 2.5 * np.pi, n_desc)
    # spiral that ends pointing at the landing-site direction
    base = np.column_stack([np.cos(ang), np.sin(ang), 0.2 * np.sin(0.5 * ang)])
    base /= np.linalg.norm(base, axis=1, keepdims=True)

    # rotate the spiral so its first direction coincides with the survey end
    # direction -> position-continuous across the survey->descent seam.
    end_dir = survey[-1] / np.linalg.norm(survey[-1])
    base = base @ rotation_between(base[0], end_dir).T

    # blend spiral direction into the site direction toward the end
    blend = np.linspace(0, 1, n_desc)[:, None]
    dirs = (1 - blend) * base + blend * site_dir
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

    # radius follows the local surface along the spiral, interpolating the
    # ALTITUDE from survey_alt down to standoff_alt (so it never clips terrain)
    r_surf_along = np.array([surface_radius(iso, d) for d in dirs])
    alt_profile = np.linspace(survey_alt, standoff_alt, n_desc)
    r_desc = r_surf_along + alt_profile
    descent = dirs * r_desc[:, None]

    # offset the first descent sample by one survey time-step so the time axis
    # is strictly increasing (dt != 0 across the seam).
    dt0 = t_surv[-1] - t_surv[-2]
    t_desc = t_surv[-1] + np.linspace(dt0, 0.6 * T_s, n_desc)

    pos = np.vstack([survey, descent])
    t = np.concatenate([t_surv, t_desc])
    phase = np.concatenate([np.zeros(n_surv), np.ones(n_desc)])
    return t, pos, phase, site_dir, v_n


def build_probe_return(start_pos, t_start, iso, survey_alt=PROBE_ALT, n_pts=420):
    """
    Phase 2: post-release return to the survey orbit (altitude-following).

    After the lander is released at the stand-off point, the probe climbs back
    out to its original survey ALTITUDE along a smooth spiral (mirror of the
    descent), then holds a circular parking arc at the survey altitude where it
    acts as the comm relay between the lander and Earth. Both the climb and the
    parking arc use surface_radius() so they clear an elongated body.

    Returns (t, pos[N,3]) with t continuing from t_start; position is continuous
    with the descent end point (start_pos).
    """
    mu = G * ISO_MASS
    r_nom = ISO_RMEAN + survey_alt
    T_s = 2 * np.pi * np.sqrt(r_nom**3 / mu)
    r0 = np.linalg.norm(start_pos)
    u = np.asarray(start_pos, float) / r0

    # in-plane direction for the climb / parking arc
    zhat = np.array([0.0, 0.0, 1.0])
    w = np.cross(zhat, u)
    if np.linalg.norm(w) < 1e-6:
        w = np.cross(np.array([0.0, 1.0, 0.0]), u)
    w /= np.linalg.norm(w)

    n_up = max(2, int(n_pts * 0.55))
    n_park = max(2, n_pts - n_up)

    # ---- climb: smoothstep ALTITUDE from current to survey_alt ------------ #
    s = np.linspace(0, 1, n_up)
    ang_up = 1.25 * np.pi * s
    dirs_up = np.cos(ang_up)[:, None] * u + np.sin(ang_up)[:, None] * w
    dirs_up /= np.linalg.norm(dirs_up, axis=1, keepdims=True)
    alt0 = altitude_above_surface(iso, start_pos)
    alt_up = alt0 + (survey_alt - alt0) * (3 * s**2 - 2 * s**3)
    r_up = np.array([surface_radius(iso, d) for d in dirs_up]) + alt_up
    pos_up = dirs_up * r_up[:, None]
    t_up = np.linspace(0.0, 0.6 * T_s, n_up)

    # ---- parking arc at the survey altitude (comm-relay station) ---------- #
    t_park = np.linspace(0.0, 0.5 * T_s, n_park)
    ang_park = ang_up[-1] + (2 * np.pi / T_s) * t_park
    dirs_park = np.cos(ang_park)[:, None] * u + np.sin(ang_park)[:, None] * w
    dirs_park /= np.linalg.norm(dirs_park, axis=1, keepdims=True)
    r_park = np.array([surface_radius(iso, d) for d in dirs_park]) + survey_alt
    pos_park = dirs_park * r_park[:, None]

    dtp = t_park[1] - t_park[0]
    t_rel = np.concatenate([t_up, t_up[-1] + dtp + t_park])
    pos = np.vstack([pos_up, pos_park])
    dt0 = t_up[1] - t_up[0]
    t = t_start + dt0 + t_rel
    return t, pos


def build_lander_trajectory(iso, site_dir, release_alt=200.0, n_pts=500):
    """
    Lander descent: from the release stand-off point straight down to the surface
    along the landing-site direction, with a gentle braking S-curve and a small
    cross-range correction. The touchdown radius is the true LOCAL surface radius
    along the site direction (ray-cast), so it lands on the actual surface even
    for an elongated body.

    Returns (t, pos[N,3], r_surface).
    """
    # local surface radius along the site direction (true touchdown radius)
    r_surface = surface_radius(iso, site_dir)
    r0 = r_surface + release_alt                       # release above LOCAL surface
    rf = r_surface + 1.0                               # touchdown 1 m above face

    s = np.linspace(0, 1, n_pts)
    # smootherstep braking profile (slow near touchdown)
    brake = 6 * s**5 - 15 * s**4 + 10 * s**3
    r = r0 + (rf - r0) * brake

    # small lateral correction that decays to zero at touchdown
    perp = np.cross(site_dir, [0, 0, 1.0])
    if np.linalg.norm(perp) < 1e-6:
        perp = np.cross(site_dir, [0, 1.0, 0])
    perp /= np.linalg.norm(perp)
    lateral = (1 - brake) * 60.0 * np.sin(3 * np.pi * s)        # [m]

    dirs = site_dir[None, :] * r[:, None] + perp[None, :] * lateral[:, None]
    pos = dirs
    t = np.linspace(0, 90.0 * 60.0, n_pts)                      # ~1.5 h descent
    return t, pos, r_surface


# ============================================================================= #
#  5.  LIDAR SURFACE-COVERAGE MODEL
# ============================================================================= #
class LidarCoverage:
    """
    Tracks which ISO surface faces have been scanned by the probe's LiDAR.
    A face is 'scanned' when the probe is within range AND the face is visible
    (its outward normal points toward the probe within the sensor's incidence
    limit). BELA-class LiDAR range ~1050 km far exceeds our survey altitude, so
    range is never the limiter here; geometry/incidence is.
    """

    def __init__(self, iso, max_incidence_deg=70.0, max_range=1050e3):
        self.iso = iso
        self.faces = _iso_field(iso, 'faces')
        self.centroids = _iso_field(iso, 'centroids')
        self.normals = _iso_field(iso, 'normals')
        self.areas = _iso_field(iso, 'areas')
        self.scanned = np.zeros(len(self.faces), dtype=bool)
        self.cos_lim = np.cos(np.deg2rad(max_incidence_deg))
        self.max_range = max_range
        self.total_area = self.areas.sum()

    def update(self, probe_pos):
        los = probe_pos[None, :] - self.centroids       # face -> probe
        dist = np.linalg.norm(los, axis=1)
        los_u = los / (dist[:, None] + 1e-12)
        cosang = np.einsum('ij,ij->i', self.normals, los_u)
        visible = (cosang > self.cos_lim) & (dist < self.max_range)
        self.scanned |= visible
        return self.coverage()

    def coverage(self):
        return self.areas[self.scanned].sum() / self.total_area


# ============================================================================= #
#  6.  ADCS SIZING & SELECTION
# ============================================================================= #
def size_adcs(name, dist_model, I, slew_angle_deg=180.0, slew_time_s=600.0,
              pointing_req_deg=0.025, margin=SAFETY_MARGIN):
    """
    Size reaction wheels + RCS from worst-case disturbance + a representative
    slew, then pick concrete heritage hardware. Returns a results dict.
    """
    wc = dist_model.worst_case()
    T_dist = np.sqrt(sum(v**2 for v in wc.values()))     # RSS worst-case disturbance

    # ---- slew sizing (rest-to-rest, bang-bang) ---------------------------- #
    theta = np.deg2rad(slew_angle_deg)
    Imax = I.max()
    alpha = 4 * theta / slew_time_s**2                   # required ang. accel
    T_slew = Imax * alpha
    omega_peak = alpha * (slew_time_s / 2)
    h_slew = Imax * omega_peak                            # peak momentum

    # ---- momentum storage to absorb secular disturbance over an orbit ------ #
    _, _, T_orbit = keplerian_orbit_radius(ISO_MASS, 1500.0)
    h_secular = T_dist * (T_orbit / 4)                   # quarter-orbit build-up

    T_rw_req = max(T_slew, T_dist) * (1 + margin)
    h_rw_req = max(h_slew, h_secular) * (1 + margin)

    # ---- RCS sizing: must overcome disturbance + provide control authority - #
    T_rcs_req = max(10 * T_dist, T_slew) * (1 + margin)

    return dict(name=name, I=I, worst=wc, T_dist=T_dist,
                alpha=alpha, omega_peak=omega_peak,
                T_slew=T_slew, h_slew=h_slew, h_secular=h_secular,
                T_orbit=T_orbit, T_rw_req=T_rw_req, h_rw_req=h_rw_req,
                T_rcs_req=T_rcs_req, pointing_req_deg=pointing_req_deg)


# Heritage hardware catalogues (representative, deep-space-flown classes)
REACTION_WHEELS = [
    # name, max torque [N m], momentum storage [N m s], mass [kg]
    ("Honeywell HR0610",  0.055, 4.0,  3.6),
    ("Honeywell HR12",    0.20, 12.0,  4.9),
    ("Honeywell HR14",    0.20, 25.0,  7.5),
    ("Honeywell HR16",    0.30, 50.0,  9.5),
]
RCS_THRUSTERS = [
    # name, thrust [N], Isp [s], note
    ("Cold-gas N2 micro (10 mN)",  0.010, 65,  "fine prox-ops, lander"),
    ("Cold-gas GN2 (0.1 N)",       0.10,  70,  "probe prox-ops / desat"),
    ("Cold-gas GN2 (1 N)",         1.0,   70,  "probe coarse / safe-mode"),
    ("Cold-gas Xe ", 5, 30, " "),
]
STAR_TRACKERS = [
    ("Sodern Auriga (multi-head)", 0.0008, "wide+narrow FOV, deep-space heritage"),
    ("Sodern Hydra-M",             0.0006, "multi-head, high precision"),
]


def select_hardware(sized, moment_arm):
    """Pick the smallest RW / RCS / star tracker meeting the sized requirements."""
    rw = None
    for nm, tmax, hstore, mass in REACTION_WHEELS:
        if tmax >= sized["T_rw_req"] and hstore >= sized["h_rw_req"]:
            rw = (nm, tmax, hstore, mass)
            break
    if rw is None:
        rw = REACTION_WHEELS[-1]

    rcs = None
    for nm, thrust, isp, note in RCS_THRUSTERS:
        if thrust * moment_arm >= sized["T_rcs_req"]:
            rcs = (nm, thrust, isp, note)
            break
    if rcs is None:
        rcs = RCS_THRUSTERS[-1]

    st = None
    for nm, acc_deg, note in STAR_TRACKERS:
        if acc_deg < 0.3 * sized["pointing_req_deg"]:
            st = (nm, acc_deg, note)
            break
    if st is None:
        st = STAR_TRACKERS[-1]

    return dict(rw=rw, rcs=rcs, st=st, moment_arm=moment_arm)


# ============================================================================= #
#  6b.  DELTA-V BUDGET & PROPELLANT MASS  (Tsiolkovsky, RCS cold-gas)
# ============================================================================= #
def orbit_speed(alt):
    """Circular-orbit speed about the ISO at the given altitude [m/s]
    (uses ISO_RMEAN + alt; representative magnitude for budgeting)."""
    mu = G * ISO_MASS
    return np.sqrt(mu / (ISO_RMEAN + alt))


def propellant_mass(dv_total, m0, isp):
    """Tsiolkovsky propellant mass: m_prop = m0 * (1 - exp(-dv/(Isp*g0)))."""
    ve = isp * G0
    m_prop = m0 * (1.0 - np.exp(-dv_total / ve))
    return m_prop, ve


def deltav_budget_probe(pos, phase, isp=RCS_ISP, m0=PROBE_MASS):
    """
    Build the mother-probe close-proximity delta-v budget from the simulated
    trajectory (all manoeuvres by RCS, per the brief). Propellant is computed in
    TWO CHAINED STAGES so the post-release phase uses the reduced probe mass.

    Returns (budget_dict, dv_total, dv_with_margin, m_prop_total, ve).
    """
    surv = phase == 0
    desc = phase == 1

    r_surv = np.linalg.norm(pos[surv], axis=1).mean()
    alt_surv = r_surv - ISO_RMEAN
    dv_insert = orbit_speed(alt_surv)

    dv_sk = N_STATIONKEEP * SK_DV_EACH
    dv_desat = DESAT_DV_BUDGET
    dv_att = ATT_DV_BUDGET

    r_standoff = np.linalg.norm(pos[desc][-1])
    v_standoff = orbit_speed(r_standoff - ISO_RMEAN)
    dv_descent = orbit_speed(alt_surv) + v_standoff

    dv_release = LANDER_REL_DV

    budget_pre = {
        "Survey-orbit insertion": dv_insert,
        "Station-keeping (survey)": dv_sk,
        "Wheel desaturation (RCS)": dv_desat,
        "Attitude control (RCS)": dv_att,
        "Descent to stand-off": dv_descent,
        "Lander-release retreat": dv_release,
    }

    dv_pre = sum(budget_pre.values())
    dv_pre_margin = dv_pre * (1 + DV_MARGIN)
    ve = isp * G0
    m_prop_pre = m0 * (1.0 - np.exp(-dv_pre_margin / ve))

    budget = dict(budget_pre)
    m_post_release = m0 - m_prop_pre

    if np.any(phase == 2):
        dv_return = v_standoff + orbit_speed(alt_surv)
        budget["Return to survey orbit (post-release)"] = dv_return
        dv_return_margin = dv_return * (1 + DV_MARGIN)
        m_prop_post = m_post_release * (1.0 - np.exp(-dv_return_margin / ve))
    else:
        m_prop_post = 0.0
        dv_return = 0.0

    m_prop_total = m_prop_pre + m_prop_post
    dv_total = sum(budget.values())
    dv_margin = dv_total * (1 + DV_MARGIN)
    return budget, dv_total, dv_margin, m_prop_total, ve


def deltav_budget_lander(pos, r_surface, isp=LAN_ISP, m0=LANDER_MASS):
    """
    Lander descent delta-v budget from the simulated descent track.
    Returns (budget_dict, dv_total, dv_with_margin, m_prop, ve).
    """
    r0 = np.linalg.norm(pos[0])
    alt0 = r0 - r_surface
    v_release = orbit_speed(alt0)

    dv_sep = 0.05
    dv_brake = v_release
    lateral_path = np.linalg.norm(np.diff(pos, axis=0), axis=1).sum() \
        - abs(r0 - np.linalg.norm(pos[-1]))
    dv_crossrange = max(0.0, 0.02 * lateral_path / max(1.0, len(pos)))
    dv_crossrange = min(dv_crossrange, 0.10)
    dv_touchdown = 0.05

    budget = {
        "Separation push-off": dv_sep,
        "Descent braking (null orbital v)": dv_brake,
        "Cross-range / hazard avoid": dv_crossrange,
        "Touchdown null burn": dv_touchdown,
    }
    dv_total = sum(budget.values())
    dv_margin = dv_total * (1 + DV_MARGIN)
    m_prop, ve = propellant_mass(dv_margin, m0, isp)
    return budget, dv_total, dv_margin, m_prop, ve


def report_deltav(label, budget, dv_total, dv_margin, m_prop, ve, m0, isp):
    """Pretty-print a delta-v budget table and propellant result."""
    print_header(f"{label}  -  DELTA-V BUDGET & PROPELLANT")
    print(f"  Propellant: cold-gas RCS, Isp = {isp:.0f} s "
          f"(exhaust vel ve = {ve:.1f} m/s)")
    print(f"  Initial (wet) mass m0 = {m0:.2f} kg\n")
    print(f"  {'Manoeuvre':<34s}{'delta-v [m/s]':>14s}")
    print("  " + "-" * 48)
    for k, v in budget.items():
        print(f"  {k:<34s}{v:>14.4f}")
    print("  " + "-" * 48)
    print(f"  {'Sub-total':<34s}{dv_total:>14.4f}")
    print(f"  {'With ' + str(int(DV_MARGIN*100)) + '% margin':<34s}{dv_margin:>14.4f}")
    print()
    print(f"  Required propellant (Tsiolkovsky) : {m_prop:10.4f} kg")
    print(f"     -> {m_prop/m0*100:6.3f} % of initial mass")
    print(f"     -> dry mass after prox-ops      : {m0 - m_prop:10.4f} kg")


# ============================================================================= #
#  7.  PRINTED REPORT
# ============================================================================= #
def print_header(title):
    print("\n" + "=" * 78)
    print(title.center(78))
    print("=" * 78)


def report_vehicle(label, mass, side, dist_model, sized, hw, extra=None,
                   sized_post=None, hw_post=None, delta_com=None):
    print_header(f"{label}  -  CONFIGURATION & ADCS SELECTION")
    print(f"  Mass                          : {mass:10.2f} kg")
    print(f"  Body (cube) side length       : {side:10.2f} m")
    print(f"  Principal MoI  (Ixx=Iyy=Izz)  : {sized['I'][0]:10.2f} kg m^2")
    if sized_post is not None:
        print(f"  MoI post-release (probe only) : {sized_post['I'][0]:10.2f} kg m^2"
              f"  (delta = {sized_post['I'][0] - sized['I'][0]:+.2f} kg m^2,"
              f" {(sized_post['I'][0]/sized['I'][0]-1)*100:+.2f}%)")
        if delta_com is not None:
            print(f"  CoM shift at lander release   : "
                  f"[{delta_com[0]:+.4f}, {delta_com[1]:+.4f}, {delta_com[2]:+.4f}] m")
    print(f"  Heliocentric distance         : {dist_model.r_helio/AU:10.2f} AU")
    print(f"  Solar flux at station         : {dist_model.solar_flux:10.4e} W/m^2")

    print("\n  Worst-case disturbance torques (Sun-dominated, per AS006):")
    for k, v in sized["worst"].items():
        print(f"     - {k:<10s} : {v:10.4e} N m")
    print(f"     - {'RSS total':<10s} : {sized['T_dist']:10.4e} N m")

    print("\n  Representative slew (rest-to-rest):")
    print(f"     angular accel required     : {sized['alpha']:10.4e} rad/s^2")
    print(f"     peak rate                  : {sized['omega_peak']:10.4e} rad/s")
    print(f"     peak torque (slew)         : {sized['T_slew']:10.4e} N m")
    print(f"     peak momentum (slew)       : {sized['h_slew']:10.4e} N m s")

    print("\n  Momentum / torque requirements (incl. 5% margin, AS001):")
    print(f"     survey-orbit period        : {sized['T_orbit']/60:10.2f} min")
    print(f"     secular momentum build-up  : {sized['h_secular']:10.4e} N m s")
    print(f"     RW torque required         : {sized['T_rw_req']:10.4e} N m")
    print(f"     RW momentum required       : {sized['h_rw_req']:10.4e} N m s")
    print(f"     RCS control torque required: {sized['T_rcs_req']:10.4e} N m")
    print(f"     pointing requirement (TPE) : {sized['pointing_req_deg']:10.3f} deg")

    print("\n  SELECTED ADCS HARDWARE:")
    nm, tmax, hstore, m = hw["rw"]
    print(f"     Reaction wheels  : 4x {nm}  (pyramid)")
    print(f"                        Tmax={tmax:.3f} N m, H={hstore:.1f} N m s, "
          f"{m:.1f} kg each  -> margin: T x{tmax/sized['T_rw_req']:.0f}, "
          f"H x{hstore/sized['h_rw_req']:.0f}")
    nm, thr, isp, note = hw["rcs"]
    print(f"     RCS thrusters    : {nm}")
    print(f"                        thrust={thr:.3f} N, Isp={isp} s, arm={hw['moment_arm']:.2f} m")
    print(f"                        -> control torque {thr*hw['moment_arm']:.3e} N m "
          f"(req {sized['T_rcs_req']:.3e})  [{note}]")
    print(f"                        layout: 4 corner clusters x 3 nozzles = 12 thrusters")
    print(f"                                on opposing cube corners (tetrahedral pattern),")
    print(f"                                nozzles along body x/y/z -> full 6-DOF authority")
    nm, acc, note = hw["st"]
    print(f"     Attitude sensor  : {nm}  (acc {acc*3600:.1f} arcsec)  [{note}]")
    print(f"     + IMU (gyro+accel), Sun sensors (safe mode), "
          f"{'LiDAR rel-nav' if 'probe' in label.lower() else 'LiDAR/optical rel-nav'}")
    if extra:
        for line in extra:
            print(f"  {line}")

    if sized_post is not None and hw_post is not None:
        print("\n  POST-RELEASE ADCS (probe alone, lander jettisoned):")
        print(f"     Mass (probe only)          : {PROBE_MASS:10.2f} kg"
              f"  (was {PROBE_MASS + LANDER_MASS:.2f} kg combined)")
        print(f"     Inertia (Ixx=Iyy=Izz)     : {sized_post['I'][0]:10.2f} kg m^2"
              f"  (was {sized['I'][0]:.2f} kg m^2, delta {sized_post['I'][0]-sized['I'][0]:+.2f})")
        print(f"     RW torque required         : {sized_post['T_rw_req']:10.4e} N m"
              f"  (pre {sized['T_rw_req']:.4e},"
              f" delta {(sized_post['T_rw_req']/sized['T_rw_req']-1)*100:+.1f}%)")
        print(f"     RW momentum required       : {sized_post['h_rw_req']:10.4e} N m s"
              f"  (pre {sized['h_rw_req']:.4e},"
              f" delta {(sized_post['h_rw_req']/sized['h_rw_req']-1)*100:+.1f}%)")
        print(f"     RCS torque required        : {sized_post['T_rcs_req']:10.4e} N m"
              f"  (pre {sized['T_rcs_req']:.4e},"
              f" delta {(sized_post['T_rcs_req']/sized['T_rcs_req']-1)*100:+.1f}%)")
        nm_post, tmax_p, hstore_p, mass_p = hw_post["rw"]
        nm_pre, tmax_r, hstore_r, mass_r = hw["rw"]
        hw_changed = nm_post != nm_pre
        print(f"     Selected RW post-release   : 4x {nm_post}"
              f"  {'<<< HARDWARE CHANGE' if hw_changed else '(unchanged)'}")
        nm_rcs_post, thr_p, isp_p, note_p = hw_post["rcs"]
        nm_rcs_pre,  thr_r, isp_r, note_r = hw["rcs"]
        rcs_changed = nm_rcs_post != nm_rcs_pre
        print(f"     Selected RCS post-release  : {nm_rcs_post}"
              f"  {'<<< HARDWARE CHANGE' if rcs_changed else '(unchanged)'}")


# ============================================================================= #
#  7b.  COLLISION / LANDING DETECTION
# ============================================================================= #
def check_probe_collision(iso, t, pos, phase):
    """
    Compute the probe's signed altitude above the ISO surface (along its own
    bearing) for the whole trajectory and report any crash. Returns a dict with
    the altitude track and the crash verdict.
    """
    alt_track = np.array([altitude_above_surface(iso, p) for p in pos])
    min_alt = float(alt_track.min())
    crashed = min_alt < CRASH_TOL
    idx_min = int(np.argmin(alt_track))

    print_header("PROBE COLLISION CHECK")
    print(f"  Trajectory points              : {len(pos)}")
    print(f"  Min altitude above ISO surface : {min_alt:10.1f} m")
    # per-phase minima for context
    for ph, name in [(0, "survey"), (1, "descent"), (2, "return")]:
        m = alt_track[phase == ph]
        if len(m):
            print(f"     phase {ph} ({name:<7s}) min alt : {m.min():10.1f} m")
    if crashed:
        print(f"\n  *** PROBE CRASHED into the ISO at t = "
              f"{t[idx_min]/60:.1f} min (altitude {min_alt:.1f} m) ***")
        print(f"      crash position [km]: "
              f"[{pos[idx_min,0]/1000:.2f}, "
              f"{pos[idx_min,1]/1000:.2f}, {pos[idx_min,2]/1000:.2f}]")
    else:
        print(f"\n  Probe CLEARS the ISO for the entire trajectory "
              f"(margin {min_alt:.1f} m).  No collision.")
    return dict(alt_track=alt_track, min_alt=min_alt, crashed=crashed,
                idx_min=idx_min, t_min=t[idx_min])


def check_lander_touchdown(iso, t, pos, tol=TOUCHDOWN_TOL):
    """
    Track the lander's signed altitude above the surface and flag touchdown the
    first time it falls within `tol` of the surface. Reports any sub-surface
    penetration before touchdown as a hard impact. Returns a dict.
    """
    alt_track = np.array([np.linalg.norm(pos[i]) - surface_radius(iso, pos[i])
                          for i in range(len(pos))])
    landed_mask = alt_track <= tol

    print_header("LANDER TOUCHDOWN DETECTION")
    print(f"  Release altitude               : {alt_track[0]:10.1f} m")
    print(f"  Final altitude                 : {alt_track[-1]:10.2f} m")
    print(f"  Touchdown tolerance            : {tol:10.1f} m")
    result = dict(alt_track=alt_track, landed=False, idx_td=None, t_td=None,
                  hard_impact=False)

    if np.any(landed_mask):
        i_td = int(np.argmax(landed_mask))     # first index at/under tolerance
        result.update(landed=True, idx_td=i_td, t_td=t[i_td])
        print(f"\n  *** LANDER TOUCHDOWN at t = {t[i_td]/60:.1f} min, "
              f"altitude {alt_track[i_td]:.2f} m ***")
        print(f"      touchdown position [m]: "
              f"[{pos[i_td,0]:.1f}, {pos[i_td,1]:.1f}, {pos[i_td,2]:.1f}]")
        if np.any(alt_track[:i_td] < -tol):
            j = int(np.argmin(alt_track[:i_td]))
            result["hard_impact"] = True
            print(f"  WARNING: lander penetrated the surface BEFORE touchdown at "
                  f"t = {t[j]/60:.1f} min (alt {alt_track[j]:.1f} m) -> hard impact")
    else:
        print(f"\n  Lander did NOT reach the surface; min altitude "
              f"{alt_track.min():.2f} m above local surface.")
    return result


# ============================================================================= #
#  8b.  THRUST & CUMULATIVE MASS PLOTS
# ============================================================================= #
_BURN_DURATIONS = {
    # probe
    "Survey-orbit insertion":              60.0,
    "Station-keeping (survey)":           180.0,
    "Wheel desaturation (RCS)":            30.0,
    "Attitude control (RCS)":             600.0,
    "Descent to stand-off":              3600.0,
    "Lander-release retreat":              30.0,
    "Return to survey orbit (post-release)": 3600.0,
    # lander
    "Separation push-off":                 10.0,
    "Descent braking (null orbital v)":   200.0,
    "Cross-range / hazard avoid":         300.0,
    "Touchdown null burn":                 10.0,
}
_DEFAULT_BURN_DUR = 60.0


def build_thrust_profile(budget, t_starts, isp, m0, phase_t, phase_arr,
                         phase_map, dt=10.0):
    """Construct thrust [N] and cumulative propellant mass [kg] vs time [s]."""
    ve = isp * G0
    t_out = np.arange(phase_t[0], phase_t[-1] + dt, dt)
    thrust_out = np.zeros_like(t_out)
    mass_used_out = np.zeros_like(t_out)

    m_current = m0
    for label, dv in budget.items():
        dv_m = dv * (1 + DV_MARGIN)
        dur = _BURN_DURATIONS.get(label, _DEFAULT_BURN_DUR)
        t_c = t_starts.get(label, phase_t[0])
        t0b = t_c - dur / 2.0
        t1b = t_c + dur / 2.0

        dm = m_current * (1.0 - np.exp(-dv_m / ve))
        mdot = dm / dur
        F = mdot * ve

        mask = (t_out >= t0b) & (t_out < t1b)
        thrust_out[mask] += F
        m_current -= dm

    for j in range(1, len(t_out)):
        mass_used_out[j] = mass_used_out[j-1] + thrust_out[j-1] / ve * dt

    return t_out, thrust_out, mass_used_out


def _probe_burn_centres(t_p, phase_p):
    """Map each probe budget label to a representative time coordinate."""
    idx_surv = np.where(phase_p == 0)[0]
    idx_desc = np.where(phase_p == 1)[0]
    idx_ret = np.where(phase_p == 2)[0]

    t_surv_mid = float(t_p[idx_surv[len(idx_surv)//2]]) if len(idx_surv) else 0.0
    t_surv_end = float(t_p[idx_surv[-1]]) if len(idx_surv) else 0.0
    t_desc_mid = float(t_p[idx_desc[len(idx_desc)//2]]) if len(idx_desc) else t_surv_end
    t_desc_end = float(t_p[idx_desc[-1]]) if len(idx_desc) else t_surv_end
    t_ret_mid = float(t_p[idx_ret[len(idx_ret)//2]]) if len(idx_ret) else t_desc_end

    return {
        "Survey-orbit insertion":              float(t_p[idx_surv[0]]) + 30.0,
        "Station-keeping (survey)":            t_surv_mid,
        "Wheel desaturation (RCS)":            t_surv_mid + 600.0,
        "Attitude control (RCS)":              t_surv_mid - 300.0,
        "Descent to stand-off":                t_desc_mid,
        "Lander-release retreat":              t_desc_end,
        "Return to survey orbit (post-release)": t_ret_mid,
    }


def _lander_burn_centres(t_l):
    n = len(t_l)
    return {
        "Separation push-off":               float(t_l[0]) + 5.0,
        "Descent braking (null orbital v)":  float(t_l[n//5]),
        "Cross-range / hazard avoid":        float(t_l[n//2]),
        "Touchdown null burn":               float(t_l[-1]) - 5.0,
    }


_PHASE_COLORS = {0: "#d0e8ff", 1: "#ffe8c0", 2: "#d4f5d4"}
_PHASE_LABELS = {0: "Survey orbit", 1: "Descent / stand-off", 2: "Return & comm relay"}


def plot_thrust_mass(t_out, thrust, mass_used, phase_t, phase_arr,
                     vehicle_label, isp, fname):
    """Two-panel figure: thrust [mN] and cumulative propellant [g/kg] vs time."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    fig.suptitle(f"{vehicle_label}  -  RCS thrust output & propellant consumption",
                 fontsize=13)

    t_min = t_out / 60.0

    phase_interp = np.interp(t_out, phase_t, phase_arr)
    phase_block = np.round(phase_interp).astype(int)
    for ph, col in _PHASE_COLORS.items():
        mask = phase_block == ph
        if not np.any(mask):
            continue
        idx = np.where(np.diff(np.concatenate([[False], mask, [False]])))[0]
        for k in range(0, len(idx), 2):
            x0 = t_min[idx[k]] if idx[k] < len(t_min) else t_min[-1]
            x1 = t_min[min(idx[k+1], len(t_min)-1)]
            lbl = _PHASE_LABELS[ph] if k == 0 else None
            ax1.axvspan(x0, x1, color=col, alpha=0.45, label=lbl)
            ax2.axvspan(x0, x1, color=col, alpha=0.45)

    thrust_mN = thrust * 1e3
    ax1.step(t_min, thrust_mN, where="mid", color="tab:red", lw=1.8,
             label="RCS thrust (total, 12 thrusters)")
    ax1.set_ylabel("Thrust [mN]")
    ax1.set_ylim(bottom=0)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right", fontsize=8, ncol=2)

    pk = thrust_mN.max()
    if pk > 0:
        pk_t = t_min[np.argmax(thrust_mN)]
        ax1.annotate(f"peak {pk:.1f} mN",
                     xy=(pk_t, pk), xytext=(pk_t + t_min[-1]*0.03, pk * 0.85),
                     arrowprops=dict(arrowstyle="->", color="k", lw=1.0),
                     fontsize=8)

    m_total = mass_used[-1]
    if m_total < 1.0:
        m_plot = mass_used * 1e3
        m_unit = "g"
    else:
        m_plot = mass_used
        m_unit = "kg"

    ax2.plot(t_min, m_plot, color="tab:blue", lw=2.0,
             label=f"Cumulative propellant consumed [Isp = {isp:.0f} s]")
    ax2.set_xlabel("Mission elapsed time [min]")
    ax2.set_ylabel(f"Propellant used [{m_unit}]")
    ax2.set_ylim(bottom=0)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper left", fontsize=8)

    ax2.annotate(f"total: {m_plot[-1]:.3f} {m_unit}",
                 xy=(t_min[-1], m_plot[-1]),
                 xytext=(t_min[-1] * 0.75, m_plot[-1] * 0.85),
                 arrowprops=dict(arrowstyle="->", color="k", lw=1.0),
                 fontsize=8)

    fig.tight_layout()
    fig.savefig(fname, dpi=130)
    return fig


# ============================================================================= #
#  8.  STATIC SUMMARY FIGURES
# ============================================================================= #
def plot_disturbances(t, series, title, fname):
    fig, ax = plt.subplots(figsize=(9, 5))
    for k, v in series.items():
        lw = 2.5 if k == "Total" else 1.4
        ls = "-" if k == "Total" else "--"
        ax.semilogy(t / 60, np.maximum(v, 1e-18), ls, lw=lw, label=k)
    ax.set_xlabel("Time [min]")
    ax.set_ylabel("Disturbance torque magnitude [N m]")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(fname, dpi=130)
    return fig


def plot_coverage(t, cov, fname):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(t / 60, np.array(cov) * 100, lw=2, color="tab:green")
    ax.axhline(50, color="k", ls="--", lw=1, label="50% coverage target")
    ax.set_xlabel("Time [min]")
    ax.set_ylabel("ISO surface scanned [%]")
    ax.set_title("LiDAR survey coverage vs. time")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fname, dpi=130)
    return fig


def plot_track_3d(iso, pos, phase, fname, title, crash_idx=None, td_idx=None,
                  lander_pos=None):
    """
    3D track with phase colours. Optionally marks a probe crash point (red X),
    a lander touchdown point (green star), and overlays the lander track.
    """
    fverts = _iso_field(iso, 'fverts')
    r_max = _iso_field(iso, 'r_max')

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    tri = Poly3DCollection(fverts / 1000, alpha=0.55)
    tri.set_facecolor((0.55, 0.5, 0.45))
    tri.set_edgecolor((0.3, 0.28, 0.25, 0.25))
    ax.add_collection3d(tri)

    surv = phase == 0
    desc = phase == 1
    ax.plot(pos[surv, 0]/1000, pos[surv, 1]/1000, pos[surv, 2]/1000,
            color="tab:blue", lw=1.5, label="LiDAR survey orbit")
    ax.plot(pos[desc, 0]/1000, pos[desc, 1]/1000, pos[desc, 2]/1000,
            color="tab:red", lw=2.0, label="Descent / approach")
    ret = phase == 2
    if np.any(ret):
        ax.plot(pos[ret, 0]/1000, pos[ret, 1]/1000, pos[ret, 2]/1000,
                color="tab:green", lw=1.8, label="Return to survey orbit")

    if lander_pos is not None:
        ax.plot(lander_pos[:, 0]/1000, lander_pos[:, 1]/1000, lander_pos[:, 2]/1000,
                color="tab:purple", lw=1.6, label="Lander descent")

    if crash_idx is not None:
        ax.scatter([pos[crash_idx, 0]/1000], [pos[crash_idx, 1]/1000],
                   [pos[crash_idx, 2]/1000], color="red", marker="X", s=120,
                   label="PROBE CRASH", depthshade=False)
    if td_idx is not None and lander_pos is not None:
        ax.scatter([lander_pos[td_idx, 0]/1000], [lander_pos[td_idx, 1]/1000],
                   [lander_pos[td_idx, 2]/1000], color="lime", marker="*", s=180,
                   edgecolor="k", label="LANDER TOUCHDOWN", depthshade=False)

    lim = r_max / 1000 * 1.1
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    ax.set_xlabel("x [km]"); ax.set_ylabel("y [km]"); ax.set_zlabel("z [km]")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(fname, dpi=130)
    return fig


# ============================================================================= #
#  9.  LIVE ANIMATION
# ============================================================================= #
def live_animation(iso, t, pos, phase, lidar_track, cov_track, dist_series,
                   vehicle_name, color_body="tab:blue", surface_ref=None,
                   body_side_m=None, lander_overlay=None,
                   lander_side_m=LANDER_VIS_SIDE, alt_track=None,
                   crash_idx=None, td_idx=None):
    """
    Single live figure with a 3D panel (ISO + cube vehicle + RCS clusters +
    scanned faces + track + optional released lander & comm link) and three live
    time-history panels (disturbance torques, coverage/altitude, speed).

    If `alt_track` is supplied, the status line shows live altitude-above-surface
    and announces CRASH (probe) / TOUCHDOWN (lander) when those frames are
    reached.
    """
    centroids = _iso_field(iso, 'centroids')
    fverts = _iso_field(iso, 'fverts')
    faces = _iso_field(iso, 'faces')
    r_mean = _iso_field(iso, 'r_mean')
    r_max = _iso_field(iso, 'r_max')

    fig = plt.figure(figsize=(14, 7))
    fig.suptitle(f"HESTIA proximity operations - {vehicle_name}", fontsize=14)

    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    axT = fig.add_subplot(3, 2, 2)
    axC = fig.add_subplot(3, 2, 4)
    axV = fig.add_subplot(3, 2, 6)

    # ---- static ISO mesh -------------------------------------------------- #
    face_colors = np.tile(np.array([0.55, 0.5, 0.45, 0.9]), (len(faces), 1))
    tri = Poly3DCollection(fverts / 1000)
    tri.set_facecolor(face_colors)
    tri.set_edgecolor((0.3, 0.28, 0.25, 0.2))
    ax3d.add_collection3d(tri)

    lim = r_max / 1000 * 1.1
    ax3d.set_xlim(-lim, lim); ax3d.set_ylim(-lim, lim); ax3d.set_zlim(-lim, lim)
    ax3d.set_xlabel("x [km]"); ax3d.set_ylabel("y [km]"); ax3d.set_zlabel("z [km]")
    ax3d.set_title("ISO + vehicle (live; cube bodies not to scale)")

    (track_line,) = ax3d.plot([], [], [], lw=1.3, color=color_body, alpha=0.8)
    vehicle_pt = ax3d.plot([], [], [], "o", color=color_body, ms=3)[0]
    beam_line, = ax3d.plot([], [], [], color="tab:orange", lw=1.0, alpha=0.7)

    # ---- vehicle cube body + 4x3 corner RCS clusters ---------------------- #
    vis_side = (body_side_m if body_side_m is not None else PROBE_VIS_SIDE) / 1000.0
    cube_col = Poly3DCollection(cube_poly_verts(pos[0] / 1000, vis_side))
    cube_col.set_facecolor((0.82, 0.82, 0.88, 0.95))
    cube_col.set_edgecolor((0.1, 0.1, 0.1, 1.0))
    ax3d.add_collection3d(cube_col)
    rcs_col = Line3DCollection(rcs_segments(pos[0] / 1000, vis_side),
                               colors="tab:orange", linewidths=1.6)
    ax3d.add_collection3d(rcs_col)

    # ---- released lander overlay + comm link ------------------------------ #
    lander_vis = lander_side_m / 1000.0
    lander_cube = None
    lander_rcs = None
    comm_line = None
    lander_track_line = None
    if lander_overlay is not None:
        lander_cube = Poly3DCollection(cube_poly_verts(np.zeros(3), lander_vis))
        lander_cube.set_facecolor((0.75, 0.6, 0.85, 0.95))
        lander_cube.set_edgecolor((0.1, 0.1, 0.1, 1.0))
        lander_cube.set_visible(False)
        ax3d.add_collection3d(lander_cube)
        lander_rcs = Line3DCollection(rcs_segments(np.zeros(3), lander_vis),
                                      colors="tab:red", linewidths=1.2)
        lander_rcs.set_visible(False)
        ax3d.add_collection3d(lander_rcs)
        (comm_line,) = ax3d.plot([], [], [], ls="--", color="tab:green",
                                 lw=1.3, alpha=0.9)
        (lander_track_line,) = ax3d.plot([], [], [], color="tab:purple",
                                         lw=1.4, alpha=0.85)

    # ---- disturbance panel ----------------------------------------------- #
    axT.set_title("Disturbance torque magnitudes")
    axT.set_xlabel("Time [min]"); axT.set_ylabel("|T| [N m]")
    axT.set_yscale("log")
    axT.grid(True, which="both", alpha=0.3)
    tmin = max(1e-16, min(np.min(np.maximum(v, 1e-18)) for v in dist_series.values()))
    tmax = max(np.max(v) for v in dist_series.values()) * 2
    axT.set_xlim(0, t[-1] / 60); axT.set_ylim(tmin, tmax)
    dist_lines = {}
    for k in dist_series:
        lw = 2.4 if k == "Total" else 1.2
        ls = "-" if k == "Total" else "--"
        (ln,) = axT.plot([], [], ls, lw=lw, label=k)
        dist_lines[k] = ln
    axT.legend(fontsize=8, loc="upper right", ncol=2)

    # ---- coverage / altitude panel --------------------------------------- #
    is_probe = "probe" in vehicle_name.lower()
    if is_probe:
        axC.set_title("LiDAR surface coverage")
        axC.set_xlabel("Time [min]"); axC.set_ylabel("Scanned [%]")
        axC.set_xlim(0, t[-1] / 60); axC.set_ylim(0, 100)
        axC.axhline(50, color="k", ls="--", lw=1)
        (cov_line,) = axC.plot([], [], lw=2, color="tab:green")
    else:
        axC.set_title("Lander altitude above surface")
        axC.set_xlabel("Time [min]"); axC.set_ylabel("Altitude [m]")
        if alt_track is not None:
            alt = alt_track
        else:
            ref = surface_ref if surface_ref is not None else r_mean
            alt = np.linalg.norm(pos, axis=1) - ref
        axC.set_xlim(0, t[-1] / 60); axC.set_ylim(min(0, alt.min()), max(alt) * 1.05)
        axC.axhline(0, color="saddlebrown", ls="--", lw=1, label="surface")
        (cov_line,) = axC.plot([], [], lw=2, color="tab:purple")

    # ---- velocity panel --------------------------------------------------- #
    speed = np.zeros(len(t))
    if len(t) > 2:
        speed[1:-1] = np.linalg.norm(pos[2:] - pos[:-2], axis=1) / (t[2:] - t[:-2])
        speed[0], speed[-1] = speed[1], speed[-2]
    axV.set_title("Vehicle speed (rel. to ISO)")
    axV.set_xlabel("Time [min]"); axV.set_ylabel("Speed [m/s]")
    axV.set_xlim(0, t[-1] / 60); axV.set_ylim(0, max(speed.max() * 1.1, 1e-3))
    axV.grid(True, alpha=0.3)
    (vel_line,) = axV.plot([], [], lw=2, color="tab:red")

    status = fig.text(0.5, 0.02, "", ha="center", fontsize=10)

    step = max(1, len(t) // 280)
    frames = range(1, len(t), step)

    def init():
        track_line.set_data([], []); track_line.set_3d_properties([])
        vehicle_pt.set_data([], []); vehicle_pt.set_3d_properties([])
        beam_line.set_data([], []); beam_line.set_3d_properties([])
        for ln in dist_lines.values():
            ln.set_data([], [])
        cov_line.set_data([], [])
        vel_line.set_data([], [])
        return ()

    def update(i):
        p = pos[i] / 1000
        track_line.set_data(pos[:i, 0]/1000, pos[:i, 1]/1000)
        track_line.set_3d_properties(pos[:i, 2]/1000)
        vehicle_pt.set_data([p[0]], [p[1]]); vehicle_pt.set_3d_properties([p[2]])

        cube_col.set_verts(cube_poly_verts(p, vis_side))
        rcs_col.set_segments(rcs_segments(p, vis_side))

        nearest = centroids[np.argmin(np.linalg.norm(centroids - pos[i], axis=1))]
        beam_line.set_data([p[0], nearest[0]/1000], [p[1], nearest[1]/1000])
        beam_line.set_3d_properties([p[2], nearest[2]/1000])

        if is_probe:
            scanned = lidar_track[i]
            fc = face_colors.copy()
            fc[scanned] = np.array([0.15, 0.7, 0.95, 0.95])     # cyan = scanned
            tri.set_facecolor(fc)

        comm_txt = ""
        if lander_overlay is not None:
            lp = lander_overlay[i]
            if np.all(np.isfinite(lp)):
                lkm = lp / 1000
                lander_cube.set_visible(True)
                lander_rcs.set_visible(True)
                lander_cube.set_verts(cube_poly_verts(lkm, lander_vis))
                lander_rcs.set_segments(rcs_segments(lkm, lander_vis))
                comm_line.set_data([p[0], lkm[0]], [p[1], lkm[1]])
                comm_line.set_3d_properties([p[2], lkm[2]])
                fin = np.isfinite(lander_overlay[:i, 0])
                lander_track_line.set_data(lander_overlay[:i, 0][fin] / 1000,
                                           lander_overlay[:i, 1][fin] / 1000)
                lander_track_line.set_3d_properties(lander_overlay[:i, 2][fin] / 1000)
                comm_txt = (f"   |   comm link: "
                            f"{np.linalg.norm(pos[i] - lp)/1000:5.2f} km")

        for k, ln in dist_lines.items():
            ln.set_data(t[:i] / 60, np.maximum(dist_series[k][:i], 1e-18))

        if is_probe:
            cov_line.set_data(t[:i] / 60, np.array(cov_track[:i]) * 100)
            if phase[i] == 0:
                ph = "SURVEY"
            elif phase[i] == 1:
                ph = "DESCENT (to stand-off)"
            else:
                ph = "LANDER RELEASED - RETURN TO SURVEY ORBIT (comm relay)"
            crash_txt = ""
            if alt_track is not None:
                crash_txt = f"   |   alt: {alt_track[i]:7.1f} m"
                if crash_idx is not None and i >= crash_idx:
                    crash_txt += "   *** CRASHED ***"
            status.set_text(f"t = {t[i]/60:6.1f} min   |   phase: {ph}   |   "
                            f"coverage: {cov_track[i]*100:5.1f}%   |   "
                            f"speed: {speed[i]:6.3f} m/s{crash_txt}{comm_txt}")
        else:
            if alt_track is not None:
                cur_alt = alt_track[i]
                cov_line.set_data(t[:i] / 60, alt_track[:i])
            else:
                ref = surface_ref if surface_ref is not None else r_mean
                a_ = np.linalg.norm(pos[:i], axis=1) - ref
                cov_line.set_data(t[:i] / 60, a_)
                cur_alt = np.linalg.norm(pos[i]) - ref
            td_txt = ""
            if td_idx is not None and i >= td_idx:
                td_txt = "   *** TOUCHDOWN ***"
            status.set_text(f"t = {t[i]/60:6.1f} min   |   LANDER DESCENT   |   "
                            f"altitude: {cur_alt:7.1f} m   |   "
                            f"speed: {speed[i]:6.3f} m/s{td_txt}")

        vel_line.set_data(t[:i] / 60, speed[:i])
        return ()

    anim = FuncAnimation(fig, update, frames=frames, init_func=init,
                         interval=30, blit=False, repeat=False)
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    return anim


# ============================================================================= #
#  10.  MAIN
# ============================================================================= #
def main():
    ap = argparse.ArgumentParser(description="HESTIA ISO proximity-ops & lander ADCS sim")
    ap.add_argument("--fast", action="store_true", help="coarser sampling")
    ap.add_argument("--no-anim", action="store_true", help="skip live animation")
    args = ap.parse_args()

    n_pts = 800 if args.fast else 1400

    print_header("HESTIA  -  INTERSTELLAR OBJECT PROXIMITY OPERATIONS SIMULATION")
    print("  Frame: co-moving with the ISO (AS002 -> relative velocity = 0).")
    print(f"  ISO mass = {ISO_MASS:.3e} kg, mean radius = {ISO_RMEAN:.0f} m, "
          f"non-uniform 'potato' shape.")
    print(f"  Rendezvous heliocentric distance = {R_HELIO/AU:.0f} AU.")
    print("  Orbit model: ALTITUDE-FOLLOWING (radius = local surface + altitude),")
    print("  with probe collision detection and lander touchdown detection.")

    # --- build ISO --------------------------------------------------------- #
    iso = make_iso_shape()
    iso_obj = type("ISO", (), {})()
    iso_obj.mass = ISO_MASS
    for k, v in iso.items():
        setattr(iso_obj, k, v)
    print(f"  ISO mesh: {len(iso['faces'])} faces, r_max = {iso['r_max']:.0f} m, "
          f"r_min = {np.linalg.norm(iso['verts'], axis=1).min():.0f} m.")
    print(f"  Surface radius: +x(long)={surface_radius(iso_obj,[1,0,0]):.0f} m, "
          f"+y={surface_radius(iso_obj,[0,1,0]):.0f} m, "
          f"+z={surface_radius(iso_obj,[0,0,1]):.0f} m.")
    mu_iso = G * ISO_MASS
    print(f"  ISO gravitational parameter mu = {mu_iso:.4e} m^3/s^2 "
          f"(surface g ~ {mu_iso/ISO_RMEAN**2*1e6:.3f} micro-m/s^2).")

    # ====================================================================== #
    #  PHASE A : MOTHER PROBE
    # ====================================================================== #
    t_p, pos_p, phase_p, site_dir, v_dir = build_probe_trajectory(iso_obj, n_pts=n_pts)

    # phase 2 - after lander release, climb back to the survey orbit (relay)
    t_ret, pos_ret = build_probe_return(pos_p[-1], t_p[-1], iso_obj,
                                        n_pts=max(120, n_pts // 3))
    t_p = np.concatenate([t_p, t_ret])
    pos_p = np.vstack([pos_p, pos_ret])
    phase_p = np.concatenate([phase_p, 2.0 * np.ones(len(t_ret))])

    print(f"\n  Survey local speed (altitude-following): "
          f"{v_dir.min():.4f} - {v_dir.max():.4f} m/s "
          f"(varies with the {surface_radius(iso_obj,[1,0,0])/surface_radius(iso_obj,[0,1,0]):.0f}:1 "
          f"radius change along the body).")

    # LiDAR coverage over the survey (+ descent + return)
    lidar = LidarCoverage(iso_obj)
    lidar_track, cov_track = [], []
    for p in pos_p:
        cov = lidar.update(p)
        lidar_track.append(lidar.scanned.copy())
        cov_track.append(cov)
    final_cov = cov_track[-1]

    # ---- PROBE COLLISION DETECTION --------------------------------------- #
    probe_coll = check_probe_collision(iso_obj, t_p, pos_p, phase_p)

    # ---------------------------------------------------------------------- #
    #  INERTIA ACCOUNTING  (mass-moment update at lander release)
    # ---------------------------------------------------------------------- #
    I_combined = probe_inertia_with_lander()
    I_post, delta_com = probe_inertia_post_release()
    mass_combined = PROBE_MASS + LANDER_MASS

    print_header("INERTIA ACCOUNTING  -  LANDER ATTACHED vs. POST-RELEASE")
    print(f"  Lander mount offset (body frame) : "
          f"[{LANDER_MOUNT_OFFSET[0]:.3f}, "
          f"{LANDER_MOUNT_OFFSET[1]:.3f}, "
          f"{LANDER_MOUNT_OFFSET[2]:.3f}] m  (-z face, centred)")
    print(f"  Combined mass (probe+lander)     : {mass_combined:.2f} kg")
    print(f"  Probe-only mass (post-release)   : {PROBE_MASS:.2f} kg")
    print(f"  I_combined  (Ixx=Iyy=Izz) [kg m^2]: "
          f"{I_combined[0]:.4f}  {I_combined[1]:.4f}  {I_combined[2]:.4f}")
    print(f"  I_post      (Ixx=Iyy=Izz) [kg m^2]: "
          f"{I_post[0]:.4f}  {I_post[1]:.4f}  {I_post[2]:.4f}")
    dI = I_post - I_combined
    pct = dI / I_combined * 100
    print(f"  Delta I     (post - combined)    : "
          f"{dI[0]:+.4f}  {dI[1]:+.4f}  {dI[2]:+.4f}  kg m^2")
    print(f"                               pct : "
          f"{pct[0]:+.2f}%  {pct[1]:+.2f}%  {pct[2]:+.2f}%")
    print(f"  CoM shift at release (body frame): "
          f"[{delta_com[0]:+.4f}, {delta_com[1]:+.4f}, {delta_com[2]:+.4f}] m")

    # ---------------------------------------------------------------------- #
    #  DISTURBANCE MODELS  (two instances, one per inertia state)
    # ---------------------------------------------------------------------- #
    dist_probe_pre = DisturbanceModel(mass_combined, PROBE_SIDE, PROBE_REFL,
                                      PROBE_CG_OFF, PROBE_RES_DIP,
                                      inertia=I_combined)
    dist_probe_post = DisturbanceModel(PROBE_MASS, PROBE_SIDE, PROBE_REFL,
                                       PROBE_CG_OFF, PROBE_RES_DIP,
                                       inertia=I_post)

    idx_release = np.searchsorted(phase_p, 1.5)
    t_pre = t_p[:idx_release]
    t_post = t_p[idx_release:]
    ser_pre = dist_probe_pre.evaluate_timeseries(t_pre)
    ser_post = dist_probe_post.evaluate_timeseries(t_post - t_post[0])
    series_p = {k: np.concatenate([ser_pre[k], ser_post[k]]) for k in ser_pre}

    # ---------------------------------------------------------------------- #
    #  ADCS SIZING
    # ---------------------------------------------------------------------- #
    sized_p_pre = size_adcs("probe_pre", dist_probe_pre, I_combined,
                            slew_angle_deg=180, slew_time_s=600)
    sized_p_post = size_adcs("probe_post", dist_probe_post, I_post,
                             slew_angle_deg=180, slew_time_s=600)
    hw_p_pre = select_hardware(sized_p_pre, moment_arm=PROBE_SIDE / 2)
    hw_p_post = select_hardware(sized_p_post, moment_arm=PROBE_SIDE / 2)

    idx50 = next((i for i, c in enumerate(cov_track) if c >= 0.5), None)
    t50 = t_p[idx50] / 60 if idx50 is not None else None

    crash_note = (f"PROBE CRASHED at t={probe_coll['t_min']/60:.1f} min "
                  f"(min alt {probe_coll['min_alt']:.1f} m)") \
        if probe_coll['crashed'] else \
        (f"No collision: probe clears the ISO "
         f"(min alt {probe_coll['min_alt']:.1f} m).")

    report_vehicle("MOTHER PROBE (HESTIA bus)", mass_combined, PROBE_SIDE,
                   dist_probe_pre, sized_p_pre, hw_p_pre,
                   extra=[
                       f"LiDAR survey: final coverage = {final_cov*100:.1f}% "
                       f"(target >= 50%)",
                       f"50% coverage reached at t = "
                       f"{t50:.1f} min" if t50 else "50% coverage NOT reached",
                       f"Survey altitude = {PROBE_ALT:.0f} m ABOVE LOCAL SURFACE, "
                       f"descent stand-off = 200 m (lander release).",
                       "Orbit is altitude-following (radius = local surface + alt).",
                       f"Collision check: {crash_note}",
                       "Post-release: probe returns to the survey orbit and acts "
                       "as comm relay (phase 2).",
                   ],
                   sized_post=sized_p_post, hw_post=hw_p_post,
                   delta_com=delta_com)

    pb, pb_dv, pb_dvm, pb_mp, pb_ve = deltav_budget_probe(pos_p, phase_p)
    report_deltav("MOTHER PROBE (HESTIA bus)", pb, pb_dv, pb_dvm, pb_mp, pb_ve,
                  PROBE_MASS, RCS_ISP)

    # ====================================================================== #
    #  PHASE B : LANDER
    # ====================================================================== #
    t_l, pos_l, r_surface_land = build_lander_trajectory(iso_obj, site_dir,
                                                         n_pts=n_pts // 2)
    phase_l = np.ones(len(t_l))
    dist_land = DisturbanceModel(LANDER_MASS, LANDER_SIDE, LANDER_REFL,
                                 LANDER_CG_OFF, LANDER_RES_DIP,
                                 inertia=LANDER_I)
    series_l = dist_land.evaluate_timeseries(t_l, omega_attitude=2*np.pi/300)
    sized_l = size_adcs("lander", dist_land, dist_land.I,
                        slew_angle_deg=90, slew_time_s=120,
                        pointing_req_deg=0.1)
    hw_l = select_hardware(sized_l, moment_arm=LANDER_SIDE / 2)

    # ---- LANDER TOUCHDOWN DETECTION -------------------------------------- #
    lander_td = check_lander_touchdown(iso_obj, t_l, pos_l)

    final_alt = lander_td['alt_track'][-1]
    td_note = (f"TOUCHDOWN at t={lander_td['t_td']/60:.1f} min "
               f"(alt {lander_td['alt_track'][lander_td['idx_td']]:.2f} m)") \
        if lander_td['landed'] else \
        f"did NOT reach the surface (min alt {lander_td['alt_track'].min():.2f} m)"

    report_vehicle("LANDER (Philae-class)", LANDER_MASS, LANDER_SIDE,
                   dist_land, sized_l, hw_l,
                   extra=[
                       f"Local surface radius at landing site = {r_surface_land:.1f} m "
                       f"(ray-cast along site direction).",
                       f"Descent from 200 m release point to surface.",
                       f"Touchdown detection: lander {td_note}.",
                       f"Touchdown altitude residual = {final_alt:.2f} m above "
                       f"local surface (AS005: ADCS off after contact).",
                       "All descent manoeuvres by RCS only (per brief).",
                   ])

    lb, lb_dv, lb_dvm, lb_mp, lb_ve = deltav_budget_lander(pos_l, r_surface_land)
    report_deltav("LANDER (Philae-class)", lb, lb_dv, lb_dvm, lb_mp, lb_ve,
                  LANDER_MASS, RCS_ISP)

    # ====================================================================== #
    #  POST-RELEASE OPERATIONS: lander overlay on the probe timeline
    # ====================================================================== #
    lander_overlay = np.full((len(t_p), 3), np.nan)
    idx_ret = np.where(phase_p == 2)[0]
    if len(idx_ret) and len(pos_l):
        n_map = max(2, int(0.55 * len(idx_ret)))
        for j, i in enumerate(idx_ret):
            frac = min(1.0, j / (n_map - 1))
            k = int(round(frac * (len(pos_l) - 1)))
            lander_overlay[i] = pos_l[k]

        i_td = idx_ret[min(n_map - 1, len(idx_ret) - 1)]
        link_td = np.linalg.norm(pos_p[i_td] - lander_overlay[i_td])
        link_end = np.linalg.norm(pos_p[-1] - lander_overlay[-1])
        print_header("POST-RELEASE OPERATIONS  -  RETURN TO ORBIT & COMM RELAY")
        print(f"  Probe returns to the {PROBE_ALT:.0f} m survey orbit after release "
              f"(phase 2, green track).")
        print(f"  Probe-lander comm-link distance at lander touchdown : "
              f"{link_td/1000:8.3f} km")
        print(f"  Probe-lander comm-link distance on parking arc      : "
              f"{link_end/1000:8.3f} km")
        print(f"  RCS layout (both vehicles): 4 corner clusters x 3 nozzles "
              f"= 12 thrusters, opposing cube corners.")

    # ====================================================================== #
    #  STATIC SUMMARY FIGURES
    # ====================================================================== #
    print_header("GENERATING FIGURES")
    plot_disturbances(t_p, series_p,
                      "Mother probe - disturbance torques (Sun-dominated, 100 AU)",
                      "probe_disturbances.png")
    plot_coverage(t_p, cov_track, "probe_coverage.png")
    plot_track_3d(iso_obj, pos_p, phase_p, "probe_track.png",
                  "Mother probe: LiDAR survey + descent + return track",
                  crash_idx=probe_coll['idx_min'] if probe_coll['crashed'] else None,
                  td_idx=lander_td['idx_td'], lander_pos=pos_l)
    plot_disturbances(t_l, series_l,
                      "Lander - disturbance torques during descent",
                      "lander_disturbances.png")
    plot_track_3d(iso_obj, pos_l, phase_l, "lander_track.png",
                  "Lander: descent to surface",
                  td_idx=lander_td['idx_td'], lander_pos=pos_l)
    print("  Saved: probe_disturbances.png, probe_coverage.png, probe_track.png,")
    print("         lander_disturbances.png, lander_track.png")

    # ---- Thrust & cumulative mass plots ---------------------------------- #
    pb_centres = _probe_burn_centres(t_p, phase_p)
    t_th_p, thr_p, mused_p = build_thrust_profile(
        pb, pb_centres, RCS_ISP, PROBE_MASS, t_p, phase_p,
        phase_map={0: 0, 1: 1, 2: 2})
    plot_thrust_mass(t_th_p, thr_p, mused_p, t_p, phase_p,
                     "Mother Probe (HESTIA bus)", RCS_ISP,
                     "probe_thrust_mass.png")
    lb_centres = _lander_burn_centres(t_l)
    t_th_l, thr_l, mused_l = build_thrust_profile(
        lb, lb_centres, LAN_ISP, LANDER_MASS, t_l, phase_l,
        phase_map={1: 1})
    plot_thrust_mass(t_th_l, thr_l, mused_l, t_l, phase_l,
                     "Lander (Philae-class)", LAN_ISP,
                     "lander_thrust_mass.png")
    print("  Saved: probe_thrust_mass.png, lander_thrust_mass.png")

    # ====================================================================== #
    #  LIVE ANIMATIONS
    # ====================================================================== #
    anims = []
    if not args.no_anim:
        print_header("LIVE ANIMATION")
        print("  Showing MOTHER PROBE survey + descent + lander release + return (live)...")
        a1 = live_animation(iso_obj, t_p, pos_p, phase_p, lidar_track, cov_track,
                            series_p, "Mother Probe", color_body="tab:blue",
                            body_side_m=PROBE_VIS_SIDE,
                            lander_overlay=lander_overlay,
                            alt_track=probe_coll['alt_track'],
                            crash_idx=probe_coll['idx_min'] if probe_coll['crashed'] else None)
        anims.append(a1)
        print("  Showing LANDER descent (live)...")
        a2 = live_animation(iso_obj, t_l, pos_l, phase_l, [None]*len(t_l), None,
                            series_l, "Lander", color_body="tab:purple",
                            surface_ref=r_surface_land,
                            body_side_m=LANDER_VIS_SIDE,
                            alt_track=lander_td['alt_track'],
                            td_idx=lander_td['idx_td'])
        anims.append(a2)
        plt.show()
    else:
        print("\n(--no-anim) Skipping live animation; static figures saved.")
        plt.close("all")

    print_header("SIMULATION COMPLETE")
    return anims


if __name__ == "__main__":
    try:
        if matplotlib.get_backend().lower() in ("agg", "template"):
            for be in ("MacOSX", "QtAgg", "TkAgg"):
                try:
                    matplotlib.use(be, force=True)
                    break
                except Exception:
                    continue
    except Exception:
        pass
    main()