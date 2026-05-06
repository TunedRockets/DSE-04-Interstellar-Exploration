#!/usr/bin/env python3
"""
Heliocentric orbit simulator with configurable Keplerian elements and optional perturbations.

Models included:
  - Solar point-mass gravity
  - Planetary gravity from analytic circular/elliptic ephemeris approximation
  - Solar radiation pressure (cannonball model)
  - First-order post-Newtonian general relativity correction from the Sun
  - Solar oblateness J2
  - Solar wind / corona drag placeholder model
  - Simple thermal re-radiation recoil placeholder model

This is intended for mission-design experimentation and sensitivity studies, not final navigation.
For precision work, replace the simple planetary ephemerides with SPICE/Horizons states.

Dependencies:
  pip install numpy scipy matplotlib

Example:
  python heliocentric_orbit_simulator.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Tuple
import math
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# -----------------------------
# Constants, SI units
# -----------------------------

AU = 1.495978707e11                      # m
R_SUN = 6.957e8                           # m
MU_SUN = 1.32712440018e20                 # m^3/s^2
C = 299_792_458.0                         # m/s
SOLAR_FLUX_1_AU = 1361.0                  # W/m^2
P_SRP_1_AU = SOLAR_FLUX_1_AU / C          # N/m^2 for perfect absorber; Cr scales this
J2_SUN = 2.2e-7                           # approximate solar J2
OMEGA_SUN_POLE_RA = math.radians(286.13)  # rough IAU solar pole right ascension
OMEGA_SUN_POLE_DEC = math.radians(63.87)  # rough IAU solar pole declination

DAY = 86400.0
YEAR = 365.25 * DAY


# -----------------------------
# Data classes
# -----------------------------

@dataclass
class Spacecraft:
    mass_kg: float = 500.0
    area_m2: float = 10.0
    cr: float = 1.5                         # SRP reflectivity coefficient, about 1 to 2
    cd: float = 2.2                         # drag coefficient
    absorptivity: float = 0.7
    emissivity: float = 0.8
    thermal_recoil_efficiency: float = 0.0   # 0 disables crude thermal recoil

    @property
    def area_to_mass(self) -> float:
        return self.area_m2 / self.mass_kg


@dataclass
class PerturbationFlags:
    sun_gravity: bool = True
    solar_radiation_pressure: bool = True
    planetary_gravity: bool = True
    general_relativity: bool = True
    solar_j2: bool = True
    solar_wind_drag: bool = False
    thermal_recoil: bool = False


@dataclass
class OrbitElements:
    """Classical heliocentric Keplerian elements."""
    semi_major_axis_m: float = 0.112 * AU
    eccentricity: float = 0.983
    inclination_rad: float = math.radians(5.0)
    raan_rad: float = math.radians(0.0)
    arg_periapsis_rad: float = math.radians(0.0)
    true_anomaly_rad: float = math.radians(0.0)


@dataclass
class Planet:
    name: str
    mu: float                     # m^3/s^2
    a: float                      # m
    e: float                      # eccentricity, simple approximation
    period: float                 # seconds
    phase0: float = 0.0           # rad at t=0
    inclination: float = 0.0      # rad, optional simple tilt


@dataclass
class SimulationConfig:
    spacecraft: Spacecraft = field(default_factory=Spacecraft)
    perturbations: PerturbationFlags = field(default_factory=PerturbationFlags)
    planets: Tuple[Planet, ...] = field(default_factory=tuple)
    t_span_s: Tuple[float, float] = (0.0, 3652.5 * DAY)
    max_step_s: float = 2.5 * DAY
    rtol: float = 1e-10
    atol: float = 1e-3

    # Corona / solar-wind drag knobs. Very simplified.
    # Density model: rho = rho_ref * (r_ref / r)^density_power
    corona_density_ref_kg_m3: float = 1e-16
    corona_density_ref_radius_m: float = 10.0 * R_SUN
    corona_density_power: float = 6.0
    solar_wind_speed_m_s: float = 400_000.0


# -----------------------------
# Basic astrodynamics utilities
# -----------------------------

def norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))


def rotation_matrix_3(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rotation_matrix_1(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def elements_to_state(elements: OrbitElements, mu: float = MU_SUN) -> np.ndarray:
    """Convert classical orbital elements to inertial Cartesian state [r, v]."""
    a = elements.semi_major_axis_m
    e = elements.eccentricity
    i = elements.inclination_rad
    raan = elements.raan_rad
    argp = elements.arg_periapsis_rad
    nu = elements.true_anomaly_rad

    if not (0.0 <= e < 1.0):
        raise ValueError("This starter script supports elliptical orbits only: 0 <= e < 1.")

    p = a * (1.0 - e * e)
    r_pf = np.array([
        p * math.cos(nu) / (1.0 + e * math.cos(nu)),
        p * math.sin(nu) / (1.0 + e * math.cos(nu)),
        0.0,
    ])
    v_pf = math.sqrt(mu / p) * np.array([-math.sin(nu), e + math.cos(nu), 0.0])

    q = rotation_matrix_3(raan) @ rotation_matrix_1(i) @ rotation_matrix_3(argp)
    r = q @ r_pf
    v = q @ v_pf
    return np.concatenate([r, v])


def state_to_orbital_summary(y: np.ndarray, mu: float = MU_SUN) -> Dict[str, float]:
    """Return osculating a, e, perihelion, aphelion from a Cartesian state."""
    r = y[:3]
    v = y[3:]
    rmag = norm(r)
    vmag = norm(v)
    energy = 0.5 * vmag * vmag - mu / rmag
    a = -mu / (2.0 * energy)
    h = np.cross(r, v)
    e_vec = np.cross(v, h) / mu - r / rmag
    e = norm(e_vec)
    return {
        "a_m": a,
        "e": e,
        "q_m": a * (1.0 - e),
        "Q_m": a * (1.0 + e),
        "r_m": rmag,
        "v_m_s": vmag,
    }


# -----------------------------
# Approximate ephemerides
# -----------------------------

def default_planets() -> Tuple[Planet, ...]:
    """Very rough heliocentric ephemeris data. Good for perturbation experiments only."""
    return (
        Planet("Mercury", 2.2032e13, 0.387098 * AU, 0.2056, 87.969 * DAY, phase0=0.0, inclination=math.radians(7.0)),
        Planet("Venus",   3.24859e14, 0.723332 * AU, 0.0068, 224.701 * DAY, phase0=1.0, inclination=math.radians(3.4)),
        Planet("Earth",   3.986004418e14, 1.000000 * AU, 0.0167, 365.256 * DAY, phase0=2.0, inclination=0.0),
        Planet("Mars",    4.282837e13, 1.523679 * AU, 0.0934, 686.980 * DAY, phase0=2.7, inclination=math.radians(1.85)),
        Planet("Jupiter", 1.26686534e17, 5.2044 * AU, 0.0489, 4332.59 * DAY, phase0=1.5, inclination=math.radians(1.3)),
        Planet("Saturn",  3.7931187e16, 9.5826 * AU, 0.0565, 10759.22 * DAY, phase0=4.0, inclination=math.radians(2.5)),
    )


def solve_kepler(mean_anomaly: float, eccentricity: float, tol: float = 1e-12) -> float:
    """Solve M = E - e sin(E)."""
    m = (mean_anomaly + math.pi) % (2.0 * math.pi) - math.pi
    e = eccentricity
    E = m if e < 0.8 else math.pi
    for _ in range(30):
        f = E - e * math.sin(E) - m
        fp = 1.0 - e * math.cos(E)
        step = f / fp
        E -= step
        if abs(step) < tol:
            break
    return E


def planet_state_simple(planet: Planet, t: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Approximate heliocentric planet position and velocity in a tilted orbit.
    Does not include RAAN/argp; adequate only as a lightweight perturbation model.
    """
    n = 2.0 * math.pi / planet.period
    M = planet.phase0 + n * t
    E = solve_kepler(M, planet.e)
    a, e = planet.a, planet.e

    x = a * (math.cos(E) - e)
    y = a * math.sqrt(1.0 - e * e) * math.sin(E)
    r = a * (1.0 - e * math.cos(E))

    vx = -a * n * math.sin(E) / (1.0 - e * math.cos(E))
    vy = a * n * math.sqrt(1.0 - e * e) * math.cos(E) / (1.0 - e * math.cos(E))

    rot = rotation_matrix_1(planet.inclination)
    return rot @ np.array([x, y, 0.0]), rot @ np.array([vx, vy, 0.0])


# -----------------------------
# Perturbation accelerations
# -----------------------------

def accel_sun_gravity(r: np.ndarray) -> np.ndarray:
    rmag = norm(r)
    return -MU_SUN * r / rmag**3


def accel_srp(r: np.ndarray, sc: Spacecraft) -> np.ndarray:
    """
    Solar radiation pressure, cannonball model.
    Direction is away from the Sun.
    """
    rmag = norm(r)
    pressure = P_SRP_1_AU * (AU / rmag) ** 2
    return pressure * sc.cr * sc.area_to_mass * (r / rmag)


def accel_planets(r_sc: np.ndarray, t: float, planets: Iterable[Planet]) -> np.ndarray:
    """
    Third-body planetary perturbation in heliocentric frame:
      mu_p * [(r_p - r_sc)/|r_p - r_sc|^3 - r_p/|r_p|^3]
    The indirect term accounts for acceleration of the heliocentric frame.
    """
    a_total = np.zeros(3)
    for planet in planets:
        r_p, _ = planet_state_simple(planet, t)
        delta = r_p - r_sc
        a_total += planet.mu * (delta / norm(delta) ** 3 - r_p / norm(r_p) ** 3)
    return a_total


def accel_gr_sun(r: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    First post-Newtonian Schwarzschild correction for a test particle around the Sun.
    Approximation valid when solar spin and higher multipoles are negligible.
    """
    rmag = norm(r)
    v2 = float(np.dot(v, v))
    rv = float(np.dot(r, v))
    return (MU_SUN / (C**2 * rmag**3)) * ((4.0 * MU_SUN / rmag - v2) * r + 4.0 * rv * v)


def solar_spin_axis_unit() -> np.ndarray:
    return np.array([
        math.cos(OMEGA_SUN_POLE_DEC) * math.cos(OMEGA_SUN_POLE_RA),
        math.cos(OMEGA_SUN_POLE_DEC) * math.sin(OMEGA_SUN_POLE_RA),
        math.sin(OMEGA_SUN_POLE_DEC),
    ])


def accel_solar_j2(r: np.ndarray) -> np.ndarray:
    """Solar J2 acceleration for arbitrary spin-axis orientation."""
    rmag = norm(r)
    s = solar_spin_axis_unit()
    z = float(np.dot(r, s))
    factor = 1.5 * J2_SUN * MU_SUN * R_SUN**2 / rmag**5
    return factor * ((5.0 * z**2 / rmag**2 - 1.0) * r - 2.0 * z * s)


def corona_density(rmag: float, cfg: SimulationConfig) -> float:
    """Simple radial power-law density model for the extended corona."""
    return cfg.corona_density_ref_kg_m3 * (cfg.corona_density_ref_radius_m / rmag) ** cfg.corona_density_power


def accel_solar_wind_drag(r: np.ndarray, v: np.ndarray, cfg: SimulationConfig) -> np.ndarray:
    """
    Very simplified drag relative to a radial solar wind.
    Near the Sun, actual plasma density/speed vary strongly with latitude, solar cycle,
    and magnetic topology, so treat this as a sensitivity knob.
    """
    rmag = norm(r)
    rhat = r / rmag
    wind_v = cfg.solar_wind_speed_m_s * rhat
    v_rel = v - wind_v
    v_rel_mag = norm(v_rel)
    rho = corona_density(rmag, cfg)
    return -0.5 * rho * cfg.spacecraft.cd * cfg.spacecraft.area_to_mass * v_rel_mag * v_rel


def accel_thermal_recoil(r: np.ndarray, sc: Spacecraft) -> np.ndarray:
    """
    Crude absorbed-solar-power recoil model.
    Real thermal recoil depends on geometry, attitude, conduction, emissivity maps,
    and time lag. This placeholder applies a small radial acceleration away from the Sun.
    """
    if sc.thermal_recoil_efficiency == 0.0:
        return np.zeros(3)
    rmag = norm(r)
    flux = SOLAR_FLUX_1_AU * (AU / rmag) ** 2
    absorbed_power_per_mass = flux * sc.absorptivity * sc.area_to_mass
    return sc.thermal_recoil_efficiency * absorbed_power_per_mass / C * (r / rmag)


# -----------------------------
# Dynamics and integration
# -----------------------------

def rhs(t: float, y: np.ndarray, cfg: SimulationConfig) -> np.ndarray:
    r = y[:3]
    v = y[3:]
    flags = cfg.perturbations
    a = np.zeros(3)

    if flags.sun_gravity:
        a += accel_sun_gravity(r)
    if flags.solar_radiation_pressure:
        a += accel_srp(r, cfg.spacecraft)
    if flags.planetary_gravity:
        a += accel_planets(r, t, cfg.planets)
    if flags.general_relativity:
        a += accel_gr_sun(r, v)
    if flags.solar_j2:
        a += accel_solar_j2(r)
    if flags.solar_wind_drag:
        a += accel_solar_wind_drag(r, v, cfg)
    if flags.thermal_recoil:
        a += accel_thermal_recoil(r, cfg.spacecraft)

    return np.concatenate([v, a])


def perihelion_event(_t: float, y: np.ndarray, _cfg: SimulationConfig) -> float:
    """Detect local radial-velocity zero crossings. Direction +1 means perihelion."""
    r = y[:3]
    v = y[3:]
    return float(np.dot(r, v) / norm(r))


perihelion_event.direction = 1
perihelion_event.terminal = False


def solar_collision_event(_t: float, y: np.ndarray, _cfg: SimulationConfig) -> float:
    return norm(y[:3]) - R_SUN


solar_collision_event.direction = -1
solar_collision_event.terminal = True


def integrate_orbit(elements: OrbitElements, cfg: SimulationConfig):
    y0 = elements_to_state(elements)

    def f(t, y):
        return rhs(t, y, cfg)

    def ev_peri(t, y):
        return perihelion_event(t, y, cfg)

    def ev_sun(t, y):
        return solar_collision_event(t, y, cfg)

    ev_peri.direction = 1
    ev_peri.terminal = False
    ev_sun.direction = -1
    ev_sun.terminal = True

    return solve_ivp(
        f,
        cfg.t_span_s,
        y0,
        method="DOP853",
        rtol=cfg.rtol,
        atol=cfg.atol,
        max_step=cfg.max_step_s,
        dense_output=False,
        events=[ev_peri, ev_sun],
    )


# -----------------------------
# Plotting and diagnostics
# -----------------------------

def print_summary(sol, label: str = "simulation") -> None:
    y0 = sol.y[:, 0]
    yf = sol.y[:, -1]
    s0 = state_to_orbital_summary(y0)
    sf = state_to_orbital_summary(yf)
    r = np.linalg.norm(sol.y[:3, :], axis=0)

    print(f"\n--- {label} ---")
    print(f"Steps: {sol.t.size}")
    print(f"Integrated duration: {sol.t[-1] / DAY:.3f} days")
    print(f"Initial a: {s0['a_m'] / AU:.6f} AU, e: {s0['e']:.8f}, q: {s0['q_m'] / R_SUN:.3f} R_sun")
    print(f"Final   a: {sf['a_m'] / AU:.6f} AU, e: {sf['e']:.8f}, q: {sf['q_m'] / R_SUN:.3f} R_sun")
    print(f"Minimum simulated radius: {r.min() / R_SUN:.3f} R_sun")
    if sol.t_events and len(sol.t_events[0]):
        print("Perihelion event times, days:", np.round(sol.t_events[0] / DAY, 6))
    if sol.status == 1:
        print("Integration stopped by event, likely solar-radius crossing.")


def plot_orbit(sol, planets: Iterable[Planet] = ()) -> None:
    xyz = sol.y[:3, :] / AU

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(xyz[0], xyz[1], xyz[2], label="spacecraft")
    ax.scatter([0], [0], [0], s=80, label="Sun")

    # Draw planet positions at final time only, to avoid clutter.
    tf = float(sol.t[-1])
    for planet in planets:
        r_p, _ = planet_state_simple(planet, tf)
        r_p_au = r_p / AU
        ax.scatter([r_p_au[0]], [r_p_au[1]], [r_p_au[2]], s=20, label=planet.name)

    max_abs = np.max(np.abs(xyz))
    if ref_xyz is not None:
        max_abs = max(max_abs, float(np.max(np.abs(ref_xyz))))
    lim = max(0.1, max_abs * 1.1)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_xlabel("x [AU]")
    ax.set_ylabel("y [AU]")
    ax.set_zlabel("z [AU]")
    ax.set_title("Heliocentric trajectory")
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.show()


def plot_radius(sol) -> None:
    r_rsun = np.linalg.norm(sol.y[:3, :], axis=0) / R_SUN
    plt.figure(figsize=(9, 4))
    plt.plot(sol.t / DAY, r_rsun)
    plt.axhline(6.0, linestyle="--", linewidth=1, label="6 R_sun")
    plt.xlabel("Time [days]")
    plt.ylabel("Heliocentric distance [R_sun]")
    plt.title("Solar distance over time")
    plt.legend()
    plt.tight_layout()
    plt.show()


def make_reference_twobody_solution(elements: OrbitElements, cfg: SimulationConfig):
    """Integrate the unperturbed two-body orbit from the same initial state."""
    ref_cfg = SimulationConfig(
        spacecraft=cfg.spacecraft,
        perturbations=PerturbationFlags(
            sun_gravity=True,
            solar_radiation_pressure=False,
            planetary_gravity=False,
            general_relativity=False,
            solar_j2=False,
            solar_wind_drag=False,
            thermal_recoil=False,
        ),
        planets=(),
        t_span_s=cfg.t_span_s,
        max_step_s=cfg.max_step_s,
        rtol=cfg.rtol,
        atol=cfg.atol,
    )
    return solve_ivp(
        lambda t, y: rhs(t, y, ref_cfg),
        ref_cfg.t_span_s,
        elements_to_state(elements),
        method="DOP853",
        rtol=ref_cfg.rtol,
        atol=ref_cfg.atol,
        max_step=ref_cfg.max_step_s,
        dense_output=True,
    )


def animate_orbit(
    sol,
    planets: Iterable[Planet] = (),
    reference_sol=None,
    show_deviation_vector: bool = True,
    trail_points: int = 250,
    frame_stride: int = 2,
    interval_ms: int = 30,
    save_path: str | None = None,
) -> None:
    """
    Live animated orbit viewer.

    Controls while the animation window is focused:
      - Space: pause / resume
      - Left/right arrows: step backward / forward while paused

    Set save_path="orbit_animation.mp4" or "orbit_animation.gif" to export.
    MP4 export requires ffmpeg; GIF export requires pillow.
    """
    xyz = sol.y[:3, :] / AU
    frames = np.arange(0, sol.t.size, max(1, frame_stride))

    ref_xyz = None
    deviation_km = None
    if reference_sol is not None:
        ref_state = reference_sol.sol(sol.t)
        ref_xyz = ref_state[:3, :] / AU
        deviation_km = np.linalg.norm(sol.y[:3, :] - ref_state[:3, :], axis=0) / 1000.0

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    max_abs = np.max(np.abs(xyz))
    planet_positions_all = {}
    for planet in planets:
        pts = np.array([planet_state_simple(planet, float(t))[0] / AU for t in sol.t])
        planet_positions_all[planet.name] = pts
        max_abs = max(max_abs, float(np.max(np.abs(pts))))

    lim = max(0.1, max_abs * 1.1)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_xlabel("x [AU]")
    ax.set_ylabel("y [AU]")
    ax.set_zlabel("z [AU]")
    ax.set_title("Live heliocentric orbit")

    ax.scatter([0], [0], [0], s=120, label="Sun")
    orbit_line, = ax.plot([], [], [], linewidth=1.0, alpha=0.35, label="perturbed trajectory")
    trail_line, = ax.plot([], [], [], linewidth=2.0, label="perturbed trail")
    satellite = ax.scatter([], [], [], s=45, label="perturbed satellite")

    ref_orbit_line = None
    ref_trail_line = None
    ref_satellite = None
    deviation_line = None
    if reference_sol is not None and ref_xyz is not None:
        ref_orbit_line, = ax.plot([], [], [], linewidth=1.0, alpha=0.25, linestyle="--", label="initial two-body orbit")
        ref_trail_line, = ax.plot([], [], [], linewidth=1.5, linestyle="--", label="two-body trail")
        ref_satellite = ax.scatter([], [], [], s=30, marker="x", label="two-body satellite")
        if show_deviation_vector:
            deviation_line, = ax.plot([], [], [], linewidth=1.5, linestyle=":", label="deviation vector")

    time_text = ax.text2D(0.03, 0.95, "", transform=ax.transAxes)

    planet_artists = {}
    for planet in planets:
        artist = ax.scatter([], [], [], s=25, label=planet.name)
        planet_artists[planet.name] = artist

    orbit_line.set_data(xyz[0], xyz[1])
    orbit_line.set_3d_properties(xyz[2])
    if ref_xyz is not None and ref_orbit_line is not None:
        ref_orbit_line.set_data(ref_xyz[0], ref_xyz[1])
        ref_orbit_line.set_3d_properties(ref_xyz[2])
    ax.legend(loc="upper right", fontsize=8)

    paused = {"value": False, "frame": 0}

    def set_scatter_3d(scatter, point):
        scatter._offsets3d = ([point[0]], [point[1]], [point[2]])

    def draw_frame(frame_index: int):
        i = int(frames[frame_index])
        paused["frame"] = frame_index

        start = max(0, i - trail_points)
        trail_line.set_data(xyz[0, start:i + 1], xyz[1, start:i + 1])
        trail_line.set_3d_properties(xyz[2, start:i + 1])
        set_scatter_3d(satellite, xyz[:, i])

        if ref_xyz is not None:
            ref_trail_line.set_data(ref_xyz[0, start:i + 1], ref_xyz[1, start:i + 1])
            ref_trail_line.set_3d_properties(ref_xyz[2, start:i + 1])
            set_scatter_3d(ref_satellite, ref_xyz[:, i])
            if deviation_line is not None:
                deviation_line.set_data([ref_xyz[0, i], xyz[0, i]], [ref_xyz[1, i], xyz[1, i]])
                deviation_line.set_3d_properties([ref_xyz[2, i], xyz[2, i]])

        for planet in planets:
            p = planet_positions_all[planet.name][i]
            set_scatter_3d(planet_artists[planet.name], p)

        radius_rsun = np.linalg.norm(sol.y[:3, i]) / R_SUN
        if deviation_km is not None:
            time_text.set_text(
                f"t = {sol.t[i] / DAY:.2f} days | r = {radius_rsun:.2f} R_sun | "
                f"deviation = {deviation_km[i]:.3e} km"
            )
        else:
            time_text.set_text(f"t = {sol.t[i] / DAY:.2f} days | r = {radius_rsun:.2f} R_sun")

        artists = [trail_line, satellite, time_text, *planet_artists.values()]
        if ref_xyz is not None:
            artists.extend([ref_trail_line, ref_satellite])
            if deviation_line is not None:
                artists.append(deviation_line)
        return artists

    def update(frame_index: int):
        if paused["value"]:
            frame_index = paused["frame"]
        return draw_frame(frame_index)

    def on_key(event):
        if event.key == " ":
            paused["value"] = not paused["value"]
        elif event.key == "right" and paused["value"]:
            paused["frame"] = min(paused["frame"] + 1, len(frames) - 1)
            draw_frame(paused["frame"])
            fig.canvas.draw_idle()
        elif event.key == "left" and paused["value"]:
            paused["frame"] = max(paused["frame"] - 1, 0)
            draw_frame(paused["frame"])
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("key_press_event", on_key)

    animation = FuncAnimation(
        fig,
        update,
        frames=len(frames),
        interval=interval_ms,
        blit=False,
        repeat=True,
    )

    if save_path:
        if save_path.lower().endswith(".gif"):
            animation.save(save_path, writer="pillow", fps=max(1, int(1000 / interval_ms)))
        else:
            animation.save(save_path, fps=max(1, int(1000 / interval_ms)))
        print(f"Saved animation to {save_path}")

    plt.tight_layout()
    plt.show()


def plot_deviation_from_reference(sol, reference_sol) -> None:
    """Plot position deviation from the initial unperturbed two-body orbit."""
    ref_state = reference_sol.sol(sol.t)
    dr_km = np.linalg.norm(sol.y[:3, :] - ref_state[:3, :], axis=0) / 1000.0

    plt.figure(figsize=(9, 4))
    plt.semilogy(sol.t / DAY, np.maximum(dr_km, 1e-12))
    plt.xlabel("Time [days]")
    plt.ylabel("Deviation from initial two-body orbit [km]")
    plt.title("Perturbation-driven deviation from initial orbit")
    plt.tight_layout()
    plt.show()


def compare_with_without_perturbations(elements: OrbitElements, cfg: SimulationConfig) -> None:
    """Small helper to estimate perturbation growth relative to a pure two-body solution."""
    cfg_nominal = cfg
    cfg_twobody = SimulationConfig(
        spacecraft=cfg.spacecraft,
        perturbations=PerturbationFlags(
            sun_gravity=True,
            solar_radiation_pressure=False,
            planetary_gravity=False,
            general_relativity=False,
            solar_j2=False,
            solar_wind_drag=False,
            thermal_recoil=False,
        ),
        planets=(),
        t_span_s=cfg.t_span_s,
        max_step_s=cfg.max_step_s,
        rtol=cfg.rtol,
        atol=cfg.atol,
    )

    sol_nom = integrate_orbit(elements, cfg_nominal)
    sol_two = integrate_orbit(elements, cfg_twobody)

    # Interpolate two-body solution at nominal times for a crude difference plot.
    sol_two_dense = solve_ivp(
        lambda t, y: rhs(t, y, cfg_twobody),
        cfg_twobody.t_span_s,
        elements_to_state(elements),
        method="DOP853",
        rtol=cfg.rtol,
        atol=cfg.atol,
        max_step=cfg.max_step_s,
        dense_output=True,
    )
    y_two_at_nom = sol_two_dense.sol(sol_nom.t)
    dr_km = np.linalg.norm(sol_nom.y[:3, :] - y_two_at_nom[:3, :], axis=0) / 1000.0

    print_summary(sol_nom, "with selected perturbations")
    print_summary(sol_two, "pure two-body")

    plt.figure(figsize=(9, 4))
    plt.semilogy(sol_nom.t / DAY, np.maximum(dr_km, 1e-12))
    plt.xlabel("Time [days]")
    plt.ylabel("Position difference vs two-body [km]")
    plt.title("Perturbation-driven trajectory divergence")
    plt.tight_layout()
    plt.show()


# -----------------------------
# Main editable scenario
# -----------------------------

if __name__ == "__main__":
    # Example: very eccentric heliocentric orbit with perihelion near 6 solar radii.
    # For an ellipse: q = a(1-e). Choose q=6 R_sun and e=0.88.
    q = 10.0 * R_SUN
    e = 0.983
    a = q / (1.0 - e)

    elements = OrbitElements(
        semi_major_axis_m=a,
        eccentricity=e,
        inclination_rad=math.radians(3.0),
        raan_rad=math.radians(15.0),
        arg_periapsis_rad=math.radians(45.0),
        true_anomaly_rad=math.radians(0.0),  # start at perihelion
    )

    spacecraft = Spacecraft(
        mass_kg=500.0,
        area_m2=8.0,
        cr=1.4,
        cd=2.2,
        absorptivity=0.75,
        emissivity=0.85,
        thermal_recoil_efficiency=0.0,
    )

    flags = PerturbationFlags(
        sun_gravity=True,
        solar_radiation_pressure=True,
        planetary_gravity=True,
        general_relativity=True,
        solar_j2=True,
        solar_wind_drag=True,
        thermal_recoil=True,
    )

    cfg = SimulationConfig(
        spacecraft=spacecraft,
        perturbations=flags,
        planets=default_planets(),
        t_span_s=(0.0, 3652.5 * DAY),
        max_step_s= 1 * DAY,  # tighten near-Sun trajectories
        rtol=1e-10,
        atol=1e-2,
        corona_density_ref_kg_m3=1e-16,
        corona_density_ref_radius_m=10.0 * R_SUN,
        corona_density_power=6.0,
        solar_wind_speed_m_s=400_000.0,
    )

    sol = integrate_orbit(elements, cfg)
    ref_sol = make_reference_twobody_solution(elements, cfg)
    ref_state = ref_sol.sol(sol.t)
    deviation_km = np.linalg.norm(sol.y[:3, :] - ref_state[:3, :], axis=0) / 1000.0

    print("\n--- Deviation summary ---")
    print(f"Max deviation: {np.max(deviation_km):.6e} km")
    print(f"Min deviation: {np.min(deviation_km):.6e} km")
    print(f"Final deviation: {deviation_km[-1]:.6e} km")

    print_summary(sol, "configured perturbed run")
    print_summary(ref_sol, "initial unperturbed two-body orbit")

    # Static diagnostics.
    plot_radius(sol)
    plot_deviation_from_reference(sol, ref_sol)

    # Live orbit viewer. Close the animation window to end the script.
    # Controls: Space = pause/resume, arrow keys = step while paused.
    animate_orbit(
        sol,
        cfg.planets,
        reference_sol=ref_sol,
        show_deviation_vector=True,
        trail_points=350,
        frame_stride=2,
        interval_ms=25,
        save_path=None,  # Example: "orbit_animation.gif" or "orbit_animation.mp4"
    )

    # Uncomment for a static 3D plot instead of, or in addition to, the animation.
    # plot_orbit(sol, cfg.planets)

    # Uncomment for a sensitivity comparison against pure two-body motion.
    # compare_with_without_perturbations(elements, cfg)
