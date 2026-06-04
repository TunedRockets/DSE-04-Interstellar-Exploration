"""
HESTIA  -  Heliocentric parking-orbit attitude & ADCS sizing simulation
========================================================================
Models the 2x2x2 m spacecraft cube in its highly-eccentric heliocentric
parking orbit (rp = 10 Rsun, ra = 1166 Rsun, e = 0.983).

Attitude law:
    * During the long cruise the +X face (heat-shield side) is kept pointed
      at the Sun  ->  Sun-tracking attitude.
    * At perihelion an Oberth burn is performed: the spacecraft slews so the
      thruster (-X / aft) fires along the tangential (prograde) velocity
      direction.

Disturbance torques modelled (relevant in deep space / near-Sun):
    1. Solar radiation pressure (SRP) torque   -> dominant
    2. Gravity-gradient torque (about the Sun)  -> dominant near perihelion
    3. Solar-wind / dynamic pressure torque     -> small
    4. Thruster misalignment torque (burn only) -> sizes RCS, transient
    (Magnetic torque is ~0: no planetary field in a heliocentric orbit.)

Outputs:
    * Live animation of the cube orbiting the Sun with the Sun-vector
      and velocity-vector drawn.
    * Disturbance-torque magnitudes vs true anomaly (orbit angle).
    * Printed report and an automatic ADCS hardware selection.

Run:  python3 hestia_adcs_sim.py
Author: generated for DSE-04 / HESTIA midterm follow-up
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

# ----------------------------------------------------------------------
# 1. PHYSICAL CONSTANTS
# ----------------------------------------------------------------------
G        = 6.67430e-11            # gravitational constant [m^3 kg^-1 s^-2]
MU_SUN   = 1.32712440018e20       # Sun GM [m^3 s^-2]
R_SUN    = 6.957e8                # solar radius [m]
AU       = 1.495978707e11         # astronomical unit [m]
L_SUN    = 3.828e26               # solar luminosity [W]
c        = 2.99792458e8           # speed of light [m/s]
SOLAR_CONST_1AU = 1361.0          # solar flux at 1 AU [W/m^2]
YEAR     = 3.15576e7              # seconds in a Julian year

# ----------------------------------------------------------------------
# 2. SPACECRAFT DEFINITION  (from HESTIA / RVLj midterm report)
# ----------------------------------------------------------------------
SIDE      = 2.0                   # cube edge length [m]
HALF      = SIDE / 2.0            # 1.0 m
FACE_AREA = SIDE * SIDE           # 4.0 m^2 per face

MASS = 2188.5                     # [kg]

I_CUBE = (1.0 / 6.0) * MASS * SIDE**2
I = np.array([I_CUBE, I_CUBE, I_CUBE])     # [Ixx, Iyy, Izz] [kg m^2]

REFLECTIVITY = 0.6                # effective reflectance of sunlit face (-)
CP_CM_OFFSET = 0.10               # [m]  (5% of side length, conservative)

THRUST_OBERTH      = 4.0e3        # [N] effective main-burn thrust (placeholder)
THRUST_MISALIGN    = np.deg2rad(0.5)   # thrust-vector misalignment [rad]
THRUST_LEVER       = HALF        # moment arm to CoM [m]

SC_DIPOLE = 1.0                   # residual magnetic dipole [A m^2]

# ----------------------------------------------------------------------
# 3. ORBIT DEFINITION
# ----------------------------------------------------------------------
RP = 10.0   * R_SUN               # perihelion radius [m]
RA = 1166.0 * R_SUN               # aphelion radius   [m]
A  = (RP + RA) / 2.0              # semi-major axis [m]
E  = (RA - RP) / (RA + RP)        # eccentricity (-)
P_ORB = 2.0 * np.pi * np.sqrt(A**3 / MU_SUN)   # orbital period [s]


def radius_at_true_anomaly(nu):
    """Heliocentric distance for true anomaly nu [rad]."""
    return A * (1.0 - E**2) / (1.0 + E * np.cos(nu))


def speed_at_radius(r):
    """vis-viva speed at radius r."""
    return np.sqrt(MU_SUN * (2.0 / r - 1.0 / A))


# ----------------------------------------------------------------------
# 4. DISTURBANCE TORQUE MODELS  (as functions of true anomaly)
# ----------------------------------------------------------------------
def solar_flux(r):
    """Solar irradiance at distance r [m]  ->  [W/m^2]."""
    return L_SUN / (4.0 * np.pi * r**2)


def srp_torque(r):
    S = solar_flux(r)
    P = S / c                                   # radiation pressure [Pa]
    F = P * (1.0 + REFLECTIVITY) * FACE_AREA    # force on the face [N]
    return F * CP_CM_OFFSET                      # [N m]


def gravity_gradient_torque(r):
    dI = 0.10 * I_CUBE                           # 10% inertia asymmetry
    return 3.0 * MU_SUN / (2.0 * r**3) * dI * 1.0  # worst case [N m]


def solar_wind_torque(r):
    n_1au = 7.0e6           # protons / m^3 at 1 AU
    v_sw  = 4.5e5           # m/s
    m_p   = 1.6726e-27      # kg
    n = n_1au * (AU / r)**2
    dyn_p = n * m_p * v_sw**2          # dynamic pressure [Pa]
    F = dyn_p * FACE_AREA
    return F * CP_CM_OFFSET            # [N m]


def magnetic_torque():
    B_imf = 5.0e-9                     # interplanetary field ~5 nT
    return SC_DIPOLE * B_imf           # [N m]  -> ~5e-9, negligible


def thruster_misalignment_torque():
    """Transient torque during the Oberth burn from thrust-vector misalignment."""
    F_perp = THRUST_OBERTH * np.sin(THRUST_MISALIGN)
    return F_perp * THRUST_LEVER       # [N m]


# ----------------------------------------------------------------------
# 5. SWEEP OVER THE ORBIT
# ----------------------------------------------------------------------
NU = np.linspace(0.0, 2.0 * np.pi, 1441)     # true anomaly grid (0.25 deg)
R  = radius_at_true_anomaly(NU)

T_srp = srp_torque(R)
T_gg  = gravity_gradient_torque(R)
T_sw  = solar_wind_torque(R)
T_mag = np.full_like(NU, magnetic_torque())
T_total_env = T_srp + T_gg + T_sw + T_mag    # combined environmental torque

T_thrust = thruster_misalignment_torque()    # scalar, burn-only

idx_peri = np.argmin(R)
peak = {
    "SRP":            T_srp[idx_peri],
    "GravityGradient": T_gg[idx_peri],
    "SolarWind":      T_sw[idx_peri],
    "Magnetic":       T_mag[idx_peri],
    "Environmental_total": T_total_env[idx_peri],
    "ThrustMisalign(burn)": T_thrust,
}

# ----------------------------------------------------------------------
# 6. MOMENTUM ACCUMULATION & ADCS SIZING
# ----------------------------------------------------------------------
def time_from_true_anomaly(nu):
    """Time since perihelion for true anomaly nu (vectorised)."""
    Ecc = 2.0 * np.arctan2(np.sqrt(1 - E) * np.sin(nu / 2.0),
                           np.sqrt(1 + E) * np.cos(nu / 2.0))
    Mn = Ecc - E * np.sin(Ecc)             # mean anomaly
    Mn = np.unwrap(Mn)
    return Mn * P_ORB / (2.0 * np.pi)

t_grid = time_from_true_anomaly(NU)
H_orbit = np.trapezoid(T_total_env, t_grid)        # [N m s]

nu_pass = np.linspace(np.deg2rad(-30), np.deg2rad(30), 601)
Ecc_p   = 2.0 * np.arctan2(np.sqrt(1 - E) * np.sin(nu_pass / 2.0),
                           np.sqrt(1 + E) * np.cos(nu_pass / 2.0))
M_p     = Ecc_p - E * np.sin(Ecc_p)            # signed mean anomaly
t_pass  = M_p * P_ORB / (2.0 * np.pi)          # signed time about perihelion [s]
r_pass  = radius_at_true_anomaly(nu_pass)
T_env_pass = srp_torque(r_pass) + gravity_gradient_torque(r_pass) \
             + solar_wind_torque(r_pass) + magnetic_torque()
H_peri_pass = abs(np.trapezoid(T_env_pass, t_pass))   # [N m s] over the pass
peri_pass_minutes = (t_pass.max() - t_pass.min()) / 60.0

V_REL_FLYBY = 55.7e3      # m/s  (report avg flyby speed)
R_FLYBY     = 1000e3      # m    (1000 km closest approach)
alpha_max = (3.0 * np.sqrt(3.0) / 8.0) * V_REL_FLYBY**2 / R_FLYBY**2
M_slew    = I_CUBE * alpha_max          # required slew torque [N m]
omega_max = 0.0557                       # rad/s peak turn rate (report)
H_slew    = I_CUBE * omega_max           # peak momentum during slew [N m s]

slew_angle = np.deg2rad(90.0)
slew_time  = 600.0                       # 10 minutes available [s]
M_oberth_slew = 4.0 * slew_angle * I_CUBE / slew_time**2   # bang-bang [N m]

# ----------------------------------------------------------------------
# 7. ADCS HARDWARE SELECTION LOGIC
# ----------------------------------------------------------------------
def select_adcs():
    margin = 2.0
    req_cont_torque = peak["Environmental_total"] * margin
    req_slew_torque = max(M_slew, M_oberth_slew) * margin
    req_momentum    = max(H_peri_pass, H_slew) * margin

    rw_catalogue = [   # (name, momentum [Nms], torque [Nm])
        ("Small  (e.g. RW class 12 Nms)",   12.0, 0.075),
        ("Medium (e.g. RW class 30 Nms)",   30.0, 0.20),
        ("Large  (e.g. RW class 75 Nms)",   75.0, 0.30),
        ("XL     (e.g. RW class 150 Nms)", 150.0, 0.50),
        ("XXL    (e.g. RW class 300 Nms)", 300.0, 0.75),
    ]
    per_wheel_mom_req = req_momentum / np.sqrt(3.0)   # distributed over 3 axes
    per_wheel_trq_req = req_slew_torque / np.sqrt(3.0)
    chosen = None
    for name, mom, trq in rw_catalogue:
        if mom >= per_wheel_mom_req and trq >= per_wheel_trq_req:
            chosen = (name, mom, trq)
            break
    if chosen is None:
        chosen = rw_catalogue[-1]   # largest available; flag oversizing
    rw_name, rw_momentum_cap, rw_torque_cap = chosen
    n_wheels_total = 4   # 3 active orthogonal + 1 skew redundant (pyramid)

    sel = {
        "controlled_mass_kg": MASS,
        "cube_inertia_kgm2": I_CUBE,
        "req_continuous_torque_Nm": req_cont_torque,
        "req_slew_torque_Nm": req_slew_torque,
        "req_momentum_storage_Nms": req_momentum,
        "reaction_wheels": {
            "type": "Reaction wheels (momentum exchange)",
            "wheel_class": rw_name,
            "per_wheel_torque_Nm": rw_torque_cap,
            "per_wheel_momentum_Nms": rw_momentum_cap,
            "n_wheels": n_wheels_total,
            "config": "4-wheel pyramid (3 orthogonal active + 1 skew redundant)",
        },
        "desaturation": {
            "type": "Cold-gas / RCS thrusters for momentum dumping",
            "reason": "No magnetic field in heliocentric orbit -> "
                      "magnetorquers unusable; RCS dumps RW momentum.",
            "burn_disturbance_torque_Nm": peak["ThrustMisalign(burn)"],
        },
        "attitude_determination": {
            "primary": "Multi-head star tracker (wide + narrow FOV)",
            "rate": "Gyro / IMU between star-tracker updates",
            "coarse": "Sun sensors for safe-mode & Sun-pointing",
            "rel_nav": "LiDAR for ISO final approach (rendezvous phase)",
            "excluded": "Magnetometer / horizon sensor (no planetary field)",
        },
        "pointing_budget_deg": {
            "star_tracker": 0.003,
            "gyro_drift": 0.010,
            "structural_flex": 0.020,
            "rw_jitter": 0.005,
            "residual_disturbance": 0.008,
        },
    }
    rss = np.sqrt(sum(v**2 for v in sel["pointing_budget_deg"].values()))
    sel["total_pointing_error_deg"] = rss
    return sel


# ----------------------------------------------------------------------
# 8. PRINTED REPORT
# ----------------------------------------------------------------------
def print_report(sel):
    line = "=" * 70
    print(line)
    print(" HESTIA  -  HELIOCENTRIC PARKING ORBIT  ATTITUDE / ADCS REPORT")
    print(line)
    print("\n[ ORBIT ]")
    print(f"  Perihelion              : {RP/R_SUN:8.1f} R_sun  ({RP/AU:.4f} AU)")
    print(f"  Aphelion                : {RA/R_SUN:8.1f} R_sun  ({RA/AU:.4f} AU)")
    print(f"  Semi-major axis a       : {A/AU:8.4f} AU")
    print(f"  Eccentricity e          : {E:8.4f}")
    print(f"  Orbital period          : {P_ORB/YEAR:8.3f} years")
    print(f"  Perihelion speed        : {speed_at_radius(RP)/1e3:8.2f} km/s")
    print(f"  Aphelion speed          : {speed_at_radius(RA)/1e3:8.2f} km/s")

    print("\n[ SPACECRAFT ]")
    print(f"  Cube edge               : {SIDE:.1f} m  (face area {FACE_AREA:.1f} m^2)")
    print(f"  Controlled mass         : {MASS:.1f} kg")
    print(f"  Cube inertia (each axis): {I_CUBE:.1f} kg m^2")
    print(f"  CoP-CoM offset          : {CP_CM_OFFSET:.3f} m")
    print(f"  +X (heat-shield) face   : held toward the Sun in cruise")

    print("\n[ PEAK DISTURBANCE TORQUES  (at perihelion) ]")
    for k in ["SRP", "GravityGradient", "SolarWind", "Magnetic",
              "Environmental_total", "ThrustMisalign(burn)"]:
        print(f"  {k:24s}: {peak[k]:.3e}  N m")

    print("\n[ MOMENTUM / SLEW SIZING ]")
    print(f"  H over one orbit undamped: {H_orbit:.3e}  N m s (if never dumped)")
    print(f"  H per perihelion pass    : {H_peri_pass:.3e}  N m s (sizing case)")
    print(f"  Perihelion pass duration : {peri_pass_minutes:.1f} min (+/-30 deg)")
    print(f"  Flyby peak slew torque   : {M_slew:.3e}  N m")
    print(f"  Oberth 90deg slew torque : {M_oberth_slew:.3e}  N m")
    print(f"  Peak slew momentum       : {H_slew:.3e}  N m s")

    print("\n[ SELECTED ADCS ]")
    rw = sel["reaction_wheels"]
    print(f"  Primary actuator        : {rw['type']}")
    print(f"     wheel class          : {rw['wheel_class']}")
    print(f"     wheels               : {rw['n_wheels']}  ({rw['config']})")
    print(f"     per-wheel torque      : {rw['per_wheel_torque_Nm']} N m")
    print(f"     per-wheel momentum    : {rw['per_wheel_momentum_Nms']} N m s")
    print(f"  Required slew torque    : {sel['req_slew_torque_Nm']:.3e} N m (x2 margin)")
    print(f"  Required momentum store : {sel['req_momentum_storage_Nms']:.3e} N m s (x2 margin)")
    des = sel["desaturation"]
    print(f"  Momentum dumping        : {des['type']}")
    print(f"     reason               : {des['reason']}")
    ad = sel["attitude_determination"]
    print(f"  Attitude determination  :")
    print(f"     primary              : {ad['primary']}")
    print(f"     rate                 : {ad['rate']}")
    print(f"     coarse / safe-mode    : {ad['coarse']}")
    print(f"     relative nav          : {ad['rel_nav']}")
    print(f"     excluded             : {ad['excluded']}")
    print(f"  Pointing budget (3-sigma RSS):")
    for k, v in sel["pointing_budget_deg"].items():
        print(f"     {k:22s}: {v:.3f} deg")
    print(f"     {'TOTAL (RSS)':22s}: {sel['total_pointing_error_deg']:.3f} deg")
    print(line)


# ----------------------------------------------------------------------
# 9. STATIC PLOTS OF DISTURBANCE TORQUES vs ORBIT ANGLE
# ----------------------------------------------------------------------
def plot_disturbances():
    deg = np.degrees(NU)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9))

    ax1.semilogy(deg, T_srp, label="SRP")
    ax1.semilogy(deg, T_gg, label="Gravity gradient")
    ax1.semilogy(deg, T_sw, label="Solar wind")
    ax1.semilogy(deg, T_mag, label="Magnetic (IMF)")
    ax1.semilogy(deg, T_total_env, "k--", lw=2, label="Environmental total")
    ax1.axvline(0, color="orange", ls=":", alpha=0.7)
    ax1.axvline(360, color="orange", ls=":", alpha=0.7)
    ax1.set_xlabel("True anomaly [deg]   (0 / 360 = perihelion)")
    ax1.set_ylabel("Disturbance torque [N m]")
    ax1.set_title("HESTIA disturbance torques vs orbit angle (log scale)")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(loc="upper right")

    ax2b = ax2.twinx()
    ax2.plot(deg, T_total_env, "k-", lw=2, label="Environmental total torque")
    ax2b.plot(deg, R / R_SUN, "tab:red", lw=1.2, alpha=0.7,
              label="Heliocentric distance")
    ax2.set_xlabel("True anomaly [deg]")
    ax2.set_ylabel("Total environmental torque [N m]")
    ax2b.set_ylabel("Distance [R_sun]", color="tab:red")
    ax2b.tick_params(axis="y", labelcolor="tab:red")
    ax2.set_title("Total environmental torque & distance vs orbit angle")
    ax2.grid(True, alpha=0.3)
    l1, lb1 = ax2.get_legend_handles_labels()
    l2, lb2 = ax2b.get_legend_handles_labels()
    ax2.legend(l1 + l2, lb1 + lb2, loc="upper right")

    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------
# 10. ORBIT + ATTITUDE ANIMATION  (2D in-plane, LIVE)
# ----------------------------------------------------------------------
def run_animation(n_orbits=2.0, n_frames=360, do_oberth=True, peri_slow=6.0):
    """2D in-plane LIVE animation: clearest way to show the eccentric orbit,
    the heat-shield-to-Sun cruise attitude, and the prograde Oberth burn.
    Returns (anim, fig); keep a reference alive while the window is open.

    peri_slow controls how much the *playback* dwells at perihelion. The
    spacecraft physically whips through perihelion in minutes, so a grid that
    is uniform in time (mean anomaly) barely shows it. Instead we sample the
    mean anomaly with extra frame density near each perihelion (M = 2*pi*k),
    which makes the animation slow down there without distorting the physics
    (positions/speeds are still the true Keplerian values at each frame).
    peri_slow=1.0 recovers the original uniform-in-time playback; larger
    values dwell longer at perihelion."""
    from matplotlib.patches import Polygon, Circle

    # Uniform-in-time mean-anomaly endpoints, then warp the SAMPLING so frames
    # bunch up near every perihelion. The warp acts on the fractional phase
    # within each orbit: phi in [0,1), perihelion at phi=0 and phi=1.
    M_lo, M_hi = 0.0, 2 * np.pi * n_orbits
    u = np.linspace(0.0, 1.0, n_frames)          # uniform parametric frames
    M_lin = M_lo + u * (M_hi - M_lo)             # uniform-in-time mean anomaly
    orbit_phase = M_lin / (2 * np.pi)            # perihelion at integer values
    k = np.floor(orbit_phase)                    # which orbit
    phi = orbit_phase - k                        # phase within orbit, [0,1)
    # Symmetric warp about perihelion (phi=0 and phi=1). s in [-1,1], 0 at
    # perihelion. Raising |s| to a power >1 expands time spent near s=0.
    s = 2.0 * phi - 1.0                          # -1 at peri(start)..+1 at peri(end)
    s_warp = np.sign(s) * np.abs(s) ** (1.0 / peri_slow)
    phi_warp = 0.5 * (s_warp + 1.0)
    M = 2 * np.pi * (k + phi_warp)               # warped mean-anomaly samples
    Ecc = M.copy()
    for _ in range(60):
        Ecc = Ecc - (Ecc - E * np.sin(Ecc) - M) / (1 - E * np.cos(Ecc))
    nu = 2 * np.arctan2(np.sqrt(1 + E) * np.sin(Ecc / 2),
                        np.sqrt(1 - E) * np.cos(Ecc / 2))
    r = radius_at_true_anomaly(nu)
    pos = np.column_stack([r * np.cos(nu), r * np.sin(nu)])

    vdir = np.column_stack([-np.sin(nu), E + np.cos(nu)])
    vdir /= np.linalg.norm(vdir, axis=1, keepdims=True)

    burn = np.zeros(n_frames, dtype=bool)
    if do_oberth:
        # Final perihelion within range: M closest to 2*pi*k_final. With the
        # warped sampling, frames cluster here, so the burn arc spans more of
        # them -> the prograde-burn attitude is clearly visible.
        k_final = int(np.floor(n_orbits - 1e-9))
        M_peri = 2 * np.pi * k_final
        frame_at_peri = int(np.argmin(np.abs(M - M_peri)))
        # widen the flagged window proportionally to the perihelion dwell
        half = max(3, int(round(3 * peri_slow)))
        lo = max(0, frame_at_peri - half)
        hi = min(n_frames, frame_at_peri + half + 1)
        burn[lo:hi] = True

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.set_aspect("equal")
    lim = RA * 1.1
    ax.set_xlim(-lim, lim*0.3); ax.set_ylim(-lim*0.7, lim*0.7)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_title("HESTIA cube in heliocentric parking orbit (orbit plane)\n"
                 "red = heat-shield face, gold = Sun line, cyan = velocity")

    nu_full = np.linspace(0, 2*np.pi, 1000)
    r_full = radius_at_true_anomaly(nu_full)
    ax.plot(r_full*np.cos(nu_full), r_full*np.sin(nu_full),
            color="gray", lw=0.8, alpha=0.6)
    ax.add_patch(Circle((0, 0), 20*R_SUN, color="orange", zorder=5))
    ax.text(0, 25*R_SUN, "Sun", ha="center", color="darkorange")

    cube_patch = Polygon(np.zeros((4, 2)), closed=True,
                         facecolor="steelblue", edgecolor="k", zorder=6)
    shield_patch = Polygon(np.zeros((2, 2)), closed=False,
                           edgecolor="red", lw=4, zorder=7)
    ax.add_patch(cube_patch); ax.add_patch(shield_patch)
    sun_line, = ax.plot([], [], color="gold", lw=2)
    vel_line, = ax.plot([], [], color="cyan", lw=2)
    txt = ax.text(0.02, 0.97, "", transform=ax.transAxes, va="top",
                  fontfamily="monospace",
                  bbox=dict(boxstyle="round", fc="white", alpha=0.8))

    cube_draw = 9e9   # drawn half-size of the cube (exaggerated)

    def update(i):
        c = pos[i]
        dist = np.linalg.norm(c)
        sun_dir = -c / dist
        xb = vdir[i] if burn[i] else sun_dir
        yb = np.array([-xb[1], xb[0]])      # 90 deg in-plane
        sc = cube_draw * (1.0 + 2.5 * dist / RA)   # grow with distance
        corners = np.array([c + sc*( xb+yb), c + sc*( xb-yb),
                            c + sc*(-xb-yb), c + sc*(-xb+yb)])
        cube_patch.set_xy(corners)
        shield_patch.set_xy(np.array([c + sc*(xb+yb), c + sc*(xb-yb)]))
        L = RA * 0.22
        sun_line.set_data([c[0], c[0]+sun_dir[0]*L], [c[1], c[1]+sun_dir[1]*L])
        vel_line.set_data([c[0], c[0]+vdir[i][0]*L], [c[1], c[1]+vdir[i][1]*L])
        state = "OBERTH BURN (prograde)" if burn[i] else "cruise: shield->Sun"
        txt.set_text(f"r  = {dist/R_SUN:7.1f} Rsun ({dist/AU:5.2f} AU)\n"
                     f"nu = {np.degrees(nu[i])%360:6.1f} deg\n"
                     f"v  = {speed_at_radius(dist)/1e3:6.1f} km/s\n"
                     f"{state}")
        return cube_patch, shield_patch, sun_line, vel_line, txt

    anim = animation.FuncAnimation(fig, update, frames=n_frames,
                                   interval=40, blit=False, repeat=True)
    return anim, fig


# ----------------------------------------------------------------------
# 11. MAIN
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sel = select_adcs()
    print_report(sel)
    plot_disturbances()
    # Keep a reference to the animation so it isn't garbage-collected
    # while the live window is open.
    anim, anim_fig = run_animation(n_orbits=2.5, n_frames=320,
                                   do_oberth=True, peri_slow=6.0)
    plt.show()   # show the live animation + disturbance plots interactively