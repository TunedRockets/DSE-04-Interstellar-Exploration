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
                          descends to a close stand-off point for lander release.
  2) The LANDER         -- separates and descends to the surface under its own
                          ADCS / RCS.

Outputs
-------
* A LIVE matplotlib animation (not a gif, not a sequence of stills) showing:
    - left  3D panel : vehicle flying around the non-uniform ISO, with the
                       LiDAR-scanned surface patches lighting up, and the final
                       descent / landing track.
    - right panels   : live time histories of every disturbance-torque source.
* Static summary figures: disturbance-torque magnitudes, surface-coverage curve,
  and the 3D survey + descent track.
* A printed report of all valuable numbers and the resulting ADCS hardware
  selection for both the probe and the lander.

Run with:
    python3 hestia_iso_proximity_sim.py            # mother probe, then lander
    python3 hestia_iso_proximity_sim.py --fast     # coarser time-step (quick look)
    python3 hestia_iso_proximity_sim.py --no-anim  # skip the live animation

Requires: numpy, scipy, matplotlib  (run locally, not in a headless container).
"""

import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
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
# Report Table 3.10: spacecraft wet mass ~2188.5 kg after the kick stage is
# discarded; the bus is a 2 m cube (sec. 3.11.1). We use the on-station wet mass.
PROBE_MASS   = 1500           # spacecraft wet mass at ISO        [kg]   (Table 3.10)
PROBE_SIDE   = 2.0              # cube side length                  [m]    (sec. 3.11.1)
PROBE_CD     = 1.4              # surface area coeff for SRP (cube faces, conservative)
PROBE_REFL   = 0.6              # reflectivity (MLI / coatings, 0=black 1=mirror)
PROBE_CG_OFF = 0.05             # CoP-CoG offset, fraction of side  -> 0.10 m
PROBE_RES_DIP = 0.5             # residual magnetic dipole          [A m^2] (heritage)

# --- Lander (Philae-class, sec. 3.12.2) -------------------------------------- #
LANDER_MASS   = 88.8 * 1.10     # Philae mass + 10% margin          [kg]   (Table 3.10)
LANDER_SIDE   = 1.0             # compact lander envelope           [m]
LANDER_REFL   = 0.5
LANDER_CG_OFF = 0.06
LANDER_RES_DIP = 0.1            # residual magnetic dipole          [A m^2]

# --- Heliospheric environment at 100 AU (for "Sun-dominated" disturbances) --- #
# Per AS006 the Sun's disturbances dominate those of the ISO; we evaluate them
# at the rendezvous distance.  At 100 AU the interplanetary field is weak.
B_FIELD_100AU = 1.0e-9          # interplanetary B-field ~1 nT      [T]
SOLAR_WIND_NP = 0.5e4           # proton number density ~0.005 cm^-3 -> m^-3 (declines ~1/r^2)
SOLAR_WIND_V  = 4.0e5           # solar-wind bulk speed             [m s^-1]
M_PROTON      = 1.6726e-27      # proton mass                       [kg]

SAFETY_MARGIN = 0.05            # AS001: ADCS sized with 5% margin

# --- Delta-v budget / propellant parameters (additive) ----------------------- #
G0            = 9.80665         # standard gravity (Isp -> exhaust vel)   [m s^-2]
RCS_ISP       = 70.0            # cold-gas GN2 RCS specific impulse        [s]  (catalogue)
DV_MARGIN     = 0.10            # 10% delta-v margin (ECSS-style)
N_STATIONKEEP = 6              # number of survey-orbit station-keeping corrections
SK_DV_EACH    = 0.05           # delta-v per station-keeping pulse        [m/s]
DESAT_DV_BUDGET = 0.10         # cumulative RCS delta-v for wheel desats   [m/s]
ATT_DV_BUDGET   = 0.20         # cumulative delta-v-equivalent for attitude/RCS pulsing [m/s]
LANDER_REL_DV   = 0.10         # probe retreat after lander release        [m/s]


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
    a, b, c = 1.45, 0.85, 0.80          # tri-axial elongation (very non-spherical)
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

    X = r_field * np.cos(LAT) * np.cos(LON)
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
#  2.  RIGID-BODY INERTIA OF A UNIFORM CUBE BUS
# ============================================================================= #
def cube_inertia(mass, side):
    """Principal moments of inertia of a uniform solid cube about its centre."""
    I = (1.0 / 6.0) * mass * side**2
    return np.array([I, I, I])      # I_xx = I_yy = I_zz for a cube


# ============================================================================= #
#  3.  DISTURBANCE-TORQUE MODELS (Sun-dominated, evaluated at 100 AU)
# ============================================================================= #
class DisturbanceModel:
    """
    Evaluates the magnitude of each disturbance-torque source acting on a vehicle
    in the ISO frame at the rendezvous heliocentric distance.

    Sources (all "around the Sun", per AS006 the ISO's own contributions are
    negligible; AS001 makes the ISO gravity-gradient negligible too):
        - Solar radiation pressure (SRP) torque   (dominant beyond ~5 AU per report)
        - Solar gravity-gradient torque
        - Solar-wind / charged-particle drag torque
        - Magnetic torque from residual dipole in the interplanetary field
    """

    def __init__(self, mass, side, refl, cg_off_frac, res_dipole,
                 r_helio=R_HELIO):
        self.mass = mass
        self.side = side
        self.area = side * side                     # one face area  [m^2]
        self.refl = refl
        self.cp_cg = cg_off_frac * side             # CoP-CoG offset [m]
        self.dipole = res_dipole
        self.r_helio = r_helio
        self.I = cube_inertia(mass, side)

        # Solar flux at rendezvous distance
        self.solar_flux = L_SUN / (4 * np.pi * r_helio**2)     # [W/m^2]
        # Solar-wind dynamic pressure (density falls ~1/r^2 from 1 AU baseline)
        nsw = SOLAR_WIND_NP * (AU / r_helio) ** 2 * (AU / AU)   # already ~100AU scaled below
        self.nsw = SOLAR_WIND_NP                                # use the 100-AU value directly
        self.B = B_FIELD_100AU

    # ---- individual sources, as a function of Sun-relative attitude angle ---- #
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

    # ---- total over a representative attitude profile ------------------------ #
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
#  4.  PROXIMITY TRAJECTORY  (probe survey orbit -> descent)
# ============================================================================= #
def keplerian_orbit_radius(iso_mass, alt):
    """Circular-orbit speed & period for an orbit at radius (r_mean+alt)."""
    mu = G * iso_mass
    r = ISO_RMEAN + alt
    v = np.sqrt(mu / r)
    T = 2 * np.pi * np.sqrt(r**3 / mu)
    return r, v, T


def build_probe_trajectory(iso, n_survey_orbits=3.0, survey_alt=1500.0,
                           standoff_alt=200.0, n_pts=1400):
    """
    Survey phase: a precessing near-polar circular orbit so the ground track
    walks around the ISO and covers > 50% of the surface.
    Descent phase: spiral down from the survey orbit to a close stand-off point
    above the selected landing site, where the lander is released.
    Returns (t, pos[N,3], phase[N])  phase: 0=survey, 1=descent.
    """
    mu = G * iso.mass if hasattr(iso, 'mass') else G * ISO_MASS
    r_s, v_s, T_s = keplerian_orbit_radius(ISO_MASS, survey_alt)

    # ---- survey: precessing circular orbit -------------------------------- #
    n_surv = int(n_pts * 0.7)
    t_surv = np.linspace(0, n_survey_orbits * T_s, n_surv)
    nu = 2 * np.pi * t_surv / T_s                       # true anomaly
    # nodal precession so successive orbits scan new longitudes
    prec = np.linspace(0, np.pi, n_surv)
    incl = np.deg2rad(85.0)                             # near-polar

    x = r_s * (np.cos(nu) * np.cos(prec) - np.sin(nu) * np.cos(incl) * np.sin(prec))
    y = r_s * (np.cos(nu) * np.sin(prec) + np.sin(nu) * np.cos(incl) * np.cos(prec))
    z = r_s * (np.sin(nu) * np.sin(incl))
    survey = np.column_stack([x, y, z])

    # ---- descent: spiral from survey radius to stand-off ------------------ #
    n_desc = n_pts - n_surv
    r_end = ISO_RMEAN + standoff_alt
    # land near +x equatorial bulge (a "flat" lit region)
    site_dir = iso.normals[np.argmax(iso.centroids[:, 0])]
    r_desc = np.linspace(r_s, r_end, n_desc)
    ang = np.linspace(0, 2.5 * np.pi, n_desc)
    # spiral that ends pointing at the landing site direction
    base = np.column_stack([np.cos(ang), np.sin(ang), 0.2 * np.sin(0.5 * ang)])
    base /= np.linalg.norm(base, axis=1, keepdims=True)
    # blend spiral direction into the site direction toward the end
    blend = np.linspace(0, 1, n_desc)[:, None]
    dirs = (1 - blend) * base + blend * site_dir
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    descent = dirs * r_desc[:, None]

    t_desc = t_surv[-1] + np.linspace(0, 0.6 * T_s, n_desc)

    pos = np.vstack([survey, descent])
    t = np.concatenate([t_surv, t_desc])
    phase = np.concatenate([np.zeros(n_surv), np.ones(n_desc)])
    return t, pos, phase, site_dir


def build_lander_trajectory(iso, site_dir, release_alt=200.0, n_pts=500):
    """
    Lander descent: from the release stand-off point straight down to the surface
    along the landing-site direction, with a gentle braking S-curve and a small
    cross-range correction (its own ADCS keeping it pointed at the site).
    Returns (t, pos[N,3], r_surface) where r_surface is the local surface radius
    along the landing-site direction.
    """
    # local surface radius along the site direction (true touchdown radius)
    proj = iso.centroids @ site_dir
    face_idx = np.argmax(proj)
    r_surface = float(np.linalg.norm(iso.centroids[face_idx]))
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
    limit). BELA-class LiDAR range from the report: ~1050 km, far exceeding our
    survey altitude, so range is never the limiter here; geometry/incidence is.
    """

    def __init__(self, iso, max_incidence_deg=70.0, max_range=1050e3):
        self.iso = iso
        self.scanned = np.zeros(len(iso.faces), dtype=bool)
        self.cos_lim = np.cos(np.deg2rad(max_incidence_deg))
        self.max_range = max_range
        self.total_area = iso.areas.sum()

    def update(self, probe_pos):
        los = probe_pos[None, :] - self.iso.centroids       # face -> probe
        dist = np.linalg.norm(los, axis=1)
        los_u = los / (dist[:, None] + 1e-12)
        cosang = np.einsum('ij,ij->i', self.iso.normals, los_u)
        visible = (cosang > self.cos_lim) & (dist < self.max_range)
        self.scanned |= visible
        return self.coverage()

    def coverage(self):
        return self.iso.areas[self.scanned].sum() / self.total_area


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
    # bang-bang: accelerate for half, decelerate for half
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
]
STAR_TRACKERS = [
    ("Sodern Auriga (multi-head)", 0.0008, "wide+narrow FOV, deep-space heritage"),
    ("Sodern Hydra-M",             0.0006, "multi-head, high precision"),
]


def select_hardware(sized, moment_arm):
    """Pick the smallest RW / RCS / star tracker meeting the sized requirements."""
    # reaction wheels
    rw = None
    for nm, tmax, hstore, mass in REACTION_WHEELS:
        if tmax >= sized["T_rw_req"] and hstore >= sized["h_rw_req"]:
            rw = (nm, tmax, hstore, mass)
            break
    if rw is None:
        rw = REACTION_WHEELS[-1]

    # RCS: thrust * moment_arm must exceed required control torque
    rcs = None
    for nm, thrust, isp, note in RCS_THRUSTERS:
        if thrust * moment_arm >= sized["T_rcs_req"]:
            rcs = (nm, thrust, isp, note)
            break
    if rcs is None:
        rcs = RCS_THRUSTERS[-1]

    # star tracker meeting pointing knowledge (need << pointing requirement)
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
    """Circular-orbit speed about the ISO at the given altitude [m/s]."""
    mu = G * ISO_MASS
    return np.sqrt(mu / (ISO_RMEAN + alt))


def propellant_mass(dv_total, m0, isp=RCS_ISP):
    """
    Tsiolkovsky propellant mass for a given total delta-v and initial (wet) mass.
        m_prop = m0 * (1 - exp(-dv / (Isp*g0)))
    Returns (m_prop [kg], ve [m/s]).
    """
    ve = isp * G0
    m_prop = m0 * (1.0 - np.exp(-dv_total / ve))
    return m_prop, ve


def deltav_budget_probe(pos, phase, isp=RCS_ISP, m0=PROBE_MASS):
    """
    Build the mother-probe close-proximity delta-v budget directly from the
    simulated trajectory (all manoeuvres by RCS, per the brief).

    Components:
      - Survey-orbit insertion (match circular speed at survey altitude)
      - Station-keeping during the precessing survey
      - Reaction-wheel momentum desaturation (RCS)
      - Attitude-control RCS pulsing (slews / pointing)
      - Descent: integrated speed change along the spiral from survey to stand-off
      - Lander-release retreat burn
    Returns (budget_dict, dv_total, dv_with_margin, m_prop, ve).
    """
    surv = phase == 0
    desc = phase == 1

    # survey altitude inferred from the survey-orbit radius in the trajectory
    r_surv = np.linalg.norm(pos[surv], axis=1).mean()
    alt_surv = r_surv - ISO_RMEAN
    dv_insert = orbit_speed(alt_surv)            # capture into the survey orbit

    dv_sk = N_STATIONKEEP * SK_DV_EACH
    dv_desat = DESAT_DV_BUDGET
    dv_att = ATT_DV_BUDGET

    # descent: sum of |delta-speed| along the descent track (RCS controlled)
    dpos = np.diff(pos[desc], axis=0)
    dt = 1.0                                     # per-step; magnitudes only -> use path
    seg = np.linalg.norm(dpos, axis=1)
    # convert the radius change into an equivalent braked delta-v: the descent
    # starts at survey speed and is brought to ~0 at the stand-off point, plus
    # the path-following control effort. Use survey speed as the dominant term.
    r_standoff = np.linalg.norm(pos[desc][-1])
    v_standoff = orbit_speed(r_standoff - ISO_RMEAN)
    dv_descent = orbit_speed(alt_surv) + v_standoff   # de-orbit + re-circularise/null

    dv_release = LANDER_REL_DV

    budget = {
        "Survey-orbit insertion": dv_insert,
        "Station-keeping (survey)": dv_sk,
        "Wheel desaturation (RCS)": dv_desat,
        "Attitude control (RCS)": dv_att,
        "Descent to stand-off": dv_descent,
        "Lander-release retreat": dv_release,
    }
    dv_total = sum(budget.values())
    dv_margin = dv_total * (1 + DV_MARGIN)
    m_prop, ve = propellant_mass(dv_margin, m0, isp)
    return budget, dv_total, dv_margin, m_prop, ve


def deltav_budget_lander(pos, r_surface, isp=RCS_ISP, m0=LANDER_MASS):
    """
    Lander descent delta-v budget from the simulated descent track.

    Components:
      - Separation / departure from probe (small push-off)
      - Descent braking: null the release-point orbital speed and control the
        approach down to touchdown (smootherstep braked profile)
      - Cross-range / hazard-avoidance correction (the lateral S-curve)
      - Touchdown null burn (cancel residual velocity before contact, AS005)
    Returns (budget_dict, dv_total, dv_with_margin, m_prop, ve).
    """
    r0 = np.linalg.norm(pos[0])
    alt0 = r0 - r_surface
    v_release = orbit_speed(alt0)                # speed to null at release

    dv_sep = 0.05                                # gentle separation push-off
    dv_brake = v_release                         # cancel orbital speed during descent
    # cross-range effort: path length of the lateral excursion
    lateral_path = np.linalg.norm(np.diff(pos, axis=0), axis=1).sum() \
                   - abs(r0 - np.linalg.norm(pos[-1]))
    dv_crossrange = max(0.0, 0.02 * lateral_path / max(1.0, len(pos)))  # small
    dv_crossrange = min(dv_crossrange, 0.10)     # cap to a sensible value
    dv_touchdown = 0.05                          # final null before contact

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
    print(f"  Propellant: cold-gas GN2 RCS, Isp = {isp:.0f} s "
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


def report_vehicle(label, mass, side, dist_model, sized, hw, extra=None):
    print_header(f"{label}  -  CONFIGURATION & ADCS SELECTION")
    print(f"  Mass                          : {mass:10.2f} kg")
    print(f"  Body (cube) side length       : {side:10.2f} m")
    print(f"  Principal MoI  (Ixx=Iyy=Izz)  : {sized['I'][0]:10.2f} kg m^2")
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
    nm, acc, note = hw["st"]
    print(f"     Attitude sensor  : {nm}  (acc {acc*3600:.1f} arcsec)  [{note}]")
    print(f"     + IMU (gyro+accel), Sun sensors (safe mode), "
          f"{'LiDAR rel-nav' if 'probe' in label.lower() else 'LiDAR/optical rel-nav'}")
    if extra:
        for line in extra:
            print(f"  {line}")


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


def plot_track_3d(iso, pos, phase, fname, title):
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    tri = Poly3DCollection(iso.fverts / 1000, alpha=0.55)
    tri.set_facecolor((0.55, 0.5, 0.45))
    tri.set_edgecolor((0.3, 0.28, 0.25, 0.25))
    ax.add_collection3d(tri)
    surv = phase == 0
    desc = phase == 1
    ax.plot(pos[surv, 0]/1000, pos[surv, 1]/1000, pos[surv, 2]/1000,
            color="tab:blue", lw=1.5, label="LiDAR survey orbit")
    ax.plot(pos[desc, 0]/1000, pos[desc, 1]/1000, pos[desc, 2]/1000,
            color="tab:red", lw=2.0, label="Descent / approach")
    lim = iso.r_max / 1000 * 5
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    ax.set_xlabel("x [km]"); ax.set_ylabel("y [km]"); ax.set_zlabel("z [km]")
    ax.set_title(title)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(fname, dpi=130)
    return fig


# ============================================================================= #
#  9.  LIVE ANIMATION
# ============================================================================= #
def live_animation(iso, t, pos, phase, lidar_track, cov_track, dist_series,
                   vehicle_name, color_body="tab:blue", surface_ref=None):
    """
    Single live figure:
       - 3D panel: ISO + vehicle + scanned faces lighting up + trailing track
       - top-right: live disturbance-torque magnitudes
       - bottom-right: live surface coverage (probe) or altitude (lander)
    """
    fig = plt.figure(figsize=(14, 7))
    fig.suptitle(f"HESTIA proximity operations - {vehicle_name}", fontsize=14)

    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    axT = fig.add_subplot(2, 2, 2)
    axC = fig.add_subplot(2, 2, 4)

    # ---- static ISO mesh -------------------------------------------------- #
    face_colors = np.tile(np.array([0.55, 0.5, 0.45, 0.9]), (len(iso.faces), 1))
    tri = Poly3DCollection(iso.fverts / 1000)
    tri.set_facecolor(face_colors)
    tri.set_edgecolor((0.3, 0.28, 0.25, 0.2))
    ax3d.add_collection3d(tri)

    lim = iso.r_max / 1000 * 5
    ax3d.set_xlim(-lim, lim); ax3d.set_ylim(-lim, lim); ax3d.set_zlim(-lim, lim)
    ax3d.set_xlabel("x [km]"); ax3d.set_ylabel("y [km]"); ax3d.set_zlabel("z [km]")
    ax3d.set_title("ISO + vehicle (live)")

    (track_line,) = ax3d.plot([], [], [], lw=1.3, color=color_body, alpha=0.8)
    vehicle_pt = ax3d.plot([], [], [], "o", color=color_body, ms=8)[0]
    beam_line, = ax3d.plot([], [], [], color="tab:orange", lw=1.0, alpha=0.7)

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
        ref = surface_ref if surface_ref is not None else iso.r_mean
        alt = np.linalg.norm(pos, axis=1) - ref
        axC.set_xlim(0, t[-1] / 60); axC.set_ylim(0, max(alt) * 1.05)
        (cov_line,) = axC.plot([], [], lw=2, color="tab:purple")

    status = fig.text(0.5, 0.02, "", ha="center", fontsize=10)

    # animation step (subsample for speed)
    step = max(1, len(t) // 280)
    frames = range(1, len(t), step)

    def init():
        track_line.set_data([], []); track_line.set_3d_properties([])
        vehicle_pt.set_data([], []); vehicle_pt.set_3d_properties([])
        beam_line.set_data([], []); beam_line.set_3d_properties([])
        for ln in dist_lines.values():
            ln.set_data([], [])
        cov_line.set_data([], [])
        return ()

    def update(i):
        p = pos[i] / 1000
        # trailing track
        track_line.set_data(pos[:i, 0]/1000, pos[:i, 1]/1000)
        track_line.set_3d_properties(pos[:i, 2]/1000)
        vehicle_pt.set_data([p[0]], [p[1]]); vehicle_pt.set_3d_properties([p[2]])

        # LiDAR beam from vehicle to nearest visible surface point
        nearest = iso.centroids[np.argmin(np.linalg.norm(iso.centroids - pos[i], axis=1))]
        beam_line.set_data([p[0], nearest[0]/1000], [p[1], nearest[1]/1000])
        beam_line.set_3d_properties([p[2], nearest[2]/1000])

        # light up scanned faces (probe only)
        if is_probe:
            scanned = lidar_track[i]
            fc = face_colors.copy()
            fc[scanned] = np.array([0.15, 0.7, 0.95, 0.95])     # cyan = scanned
            tri.set_facecolor(fc)

        # disturbance traces
        for k, ln in dist_lines.items():
            ln.set_data(t[:i] / 60, np.maximum(dist_series[k][:i], 1e-18))

        # coverage / altitude trace
        if is_probe:
            cov_line.set_data(t[:i] / 60, np.array(cov_track[:i]) * 100)
            ph = "SURVEY" if phase[i] == 0 else "DESCENT (lander release)"
            status.set_text(f"t = {t[i]/60:6.1f} min   |   phase: {ph}   |   "
                            f"coverage: {cov_track[i]*100:5.1f}%")
        else:
            ref = surface_ref if surface_ref is not None else iso.r_mean
            alt = np.linalg.norm(pos[:i], axis=1) - ref
            cov_line.set_data(t[:i] / 60, alt)
            cur_alt = np.linalg.norm(pos[i]) - ref
            status.set_text(f"t = {t[i]/60:6.1f} min   |   LANDER DESCENT   |   "
                            f"altitude: {cur_alt:7.1f} m")
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

    # --- build ISO --------------------------------------------------------- #
    iso = make_iso_shape()
    iso_obj = type("ISO", (), {})()
    iso_obj.mass = ISO_MASS
    for k, v in iso.items():
        setattr(iso_obj, k, v)
    print(f"  ISO mesh: {len(iso['faces'])} faces, "
          f"axis-ratio elongation ~1.7:1, r_max = {iso['r_max']:.0f} m.")
    mu_iso = G * ISO_MASS
    print(f"  ISO gravitational parameter mu = {mu_iso:.4e} m^3/s^2 "
          f"(surface g ~ {mu_iso/ISO_RMEAN**2*1e6:.3f} micro-m/s^2).")

    # ====================================================================== #
    #  PHASE A : MOTHER PROBE
    # ====================================================================== #
    t_p, pos_p, phase_p, site_dir = build_probe_trajectory(iso_obj, n_pts=n_pts)

    # LiDAR coverage over the survey
    lidar = LidarCoverage(iso_obj)
    lidar_track, cov_track = [], []
    for p in pos_p:
        cov = lidar.update(p)
        lidar_track.append(lidar.scanned.copy())
        cov_track.append(cov)
    final_cov = cov_track[-1]

    # disturbance torques (probe)
    dist_probe = DisturbanceModel(PROBE_MASS, PROBE_SIDE, PROBE_REFL,
                                  PROBE_CG_OFF, PROBE_RES_DIP)
    series_p = dist_probe.evaluate_timeseries(t_p)

    # ADCS sizing + selection (probe)
    sized_p = size_adcs("probe", dist_probe, dist_probe.I,
                        slew_angle_deg=180, slew_time_s=600)
    hw_p = select_hardware(sized_p, moment_arm=PROBE_SIDE / 2)

    # find when 50% coverage reached
    idx50 = next((i for i, c in enumerate(cov_track) if c >= 0.5), None)
    t50 = t_p[idx50] / 60 if idx50 is not None else None

    report_vehicle("MOTHER PROBE (HESTIA bus)", PROBE_MASS, PROBE_SIDE,
                   dist_probe, sized_p, hw_p,
                   extra=[
                       f"LiDAR survey: final coverage = {final_cov*100:.1f}% "
                       f"(target >= 50%)",
                       f"50% coverage reached at t = "
                       f"{t50:.1f} min" if t50 else "50% coverage NOT reached",
                       f"Survey altitude = 1500 m, descent stand-off = 200 m "
                       f"(lander release).",
                   ])

    # delta-v budget + propellant (probe)
    pb, pb_dv, pb_dvm, pb_mp, pb_ve = deltav_budget_probe(pos_p, phase_p)
    report_deltav("MOTHER PROBE (HESTIA bus)", pb, pb_dv, pb_dvm, pb_mp, pb_ve,
                  PROBE_MASS, RCS_ISP)

    # ====================================================================== #
    #  PHASE B : LANDER
    # ====================================================================== #
    t_l, pos_l, r_surface_land = build_lander_trajectory(iso_obj, site_dir, n_pts=n_pts // 2)
    phase_l = np.ones(len(t_l))             # entirely descent
    dist_land = DisturbanceModel(LANDER_MASS, LANDER_SIDE, LANDER_REFL,
                                 LANDER_CG_OFF, LANDER_RES_DIP)
    series_l = dist_land.evaluate_timeseries(t_l, omega_attitude=2*np.pi/300)
    sized_l = size_adcs("lander", dist_land, dist_land.I,
                        slew_angle_deg=90, slew_time_s=120,
                        pointing_req_deg=0.1)
    hw_l = select_hardware(sized_l, moment_arm=LANDER_SIDE / 2)

    final_alt = np.linalg.norm(pos_l[-1]) - r_surface_land
    report_vehicle("LANDER (Philae-class)", LANDER_MASS, LANDER_SIDE,
                   dist_land, sized_l, hw_l,
                   extra=[
                       f"Local surface radius at landing site = {r_surface_land:.1f} m.",
                       f"Descent from 200 m release point to surface.",
                       f"Touchdown altitude residual = {final_alt:.1f} m above "
                       f"local surface (AS005: ADCS off after contact).",
                       "All descent manoeuvres by RCS only (per brief).",
                   ])

    # delta-v budget + propellant (lander)
    lb, lb_dv, lb_dvm, lb_mp, lb_ve = deltav_budget_lander(pos_l, r_surface_land)
    report_deltav("LANDER (Philae-class)", lb, lb_dv, lb_dvm, lb_mp, lb_ve,
                  LANDER_MASS, RCS_ISP)

    # ====================================================================== #
    #  STATIC SUMMARY FIGURES
    # ====================================================================== #
    print_header("GENERATING FIGURES")
    plot_disturbances(t_p, series_p,
                      "Mother probe - disturbance torques (Sun-dominated, 100 AU)",
                      "probe_disturbances.png")
    plot_coverage(t_p, cov_track, "probe_coverage.png")
    plot_track_3d(iso_obj, pos_p, phase_p, "probe_track.png",
                  "Mother probe: LiDAR survey + descent track")
    plot_disturbances(t_l, series_l,
                      "Lander - disturbance torques during descent",
                      "lander_disturbances.png")
    plot_track_3d(iso_obj, pos_l, phase_l, "lander_track.png",
                  "Lander: descent to surface")
    print("  Saved: probe_disturbances.png, probe_coverage.png, probe_track.png,")
    print("         lander_disturbances.png, lander_track.png")

    # ====================================================================== #
    #  LIVE ANIMATIONS
    # ====================================================================== #
    anims = []
    if not args.no_anim:
        print_header("LIVE ANIMATION")
        print("  Showing MOTHER PROBE survey + descent (live)...")
        a1 = live_animation(iso_obj, t_p, pos_p, phase_p, lidar_track, cov_track,
                            series_p, "Mother Probe", color_body="tab:blue")
        anims.append(a1)
        print("  Showing LANDER descent (live)...")
        a2 = live_animation(iso_obj, t_l, pos_l, phase_l, [None]*len(t_l), None,
                            series_l, "Lander", color_body="tab:purple",
                            surface_ref=r_surface_land)
        anims.append(a2)
        plt.show()
    else:
        print("\n(--no-anim) Skipping live animation; static figures saved.")
        plt.close("all")

    print_header("SIMULATION COMPLETE")
    return anims


if __name__ == "__main__":
    # Use an interactive backend if available so live animation actually animates.
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