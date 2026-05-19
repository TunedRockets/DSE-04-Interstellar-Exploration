import copy
import math
from scipy.optimize import minimize_scalar

from src2.utilities import DAY, YEAR
from src2.orbit import *

import numpy as np
from tqdm import tqdm
import matplotlib as mpl
# mpl.use('TkAgg')
import matplotlib.pyplot as plt
from proper_oberth_sizer import *

# ============================================================
#                         CONSTANTS
# ============================================================

g0 = 9.81
sigma = 5.670374419e-8
R_universal = 8.314462618

F_1AU = 1361
R_sun = 6.96e5

# Earth constants
mu_earth = 3.986004418e14   # m^3/s^2
R_earth = 6371e3            # m

LEO_altitude = 200_000
r_LEO = R_earth + LEO_altitude

LEO_velocity = np.sqrt(mu_earth / r_LEO)
escape_velocity = np.sqrt(2 * mu_earth / r_LEO)

# ============================================================
#                    ORBITAL ENERGY UTILITIES
# ============================================================

def dv_to_vinf(dv, v_circ, v_esc):
    """
    Convert impulsive periapsis burn delta-v from circular orbit
    into resulting hyperbolic excess velocity.
    """

    v_post = v_circ + dv

    if v_post <= v_esc:
        return 0.0

    return np.sqrt(v_post**2 - v_esc**2)


def vinf_to_required_dv(vinf, v_circ, v_esc):
    """
    Required impulsive periapsis delta-v to achieve target v∞.
    """

    return np.sqrt(vinf**2 + v_esc**2) - v_circ


def combine_vinf_and_dv(vinf_initial, dv, v_esc):
    """
    Apply impulsive periapsis burn onto existing hyperbolic trajectory.
    """

    v_peri_initial = np.sqrt(vinf_initial**2 + v_esc**2)

    v_peri_final = v_peri_initial + dv

    return np.sqrt(v_peri_final**2 - v_esc**2)


# ============================================================
#                    PROPULSION UTILITIES
# ============================================================

def get_prop_mass_with_end_mass(delta_v, end_mass, Isp):

    return (
        np.exp(delta_v / (g0 * Isp))
        * end_mass
        - end_mass
    )


def get_prop_mass_with_start_mass(delta_v, start_mass, Isp):

    return (
        start_mass
        * (
            1
            - np.exp(
                -delta_v / (g0 * Isp)
            )
        )
    )


def get_required_thrust(
        delta_v,
        Isp,
        start_mass,
        burn_time):

    prop_mass = get_prop_mass_with_start_mass(
        delta_v,
        start_mass,
        Isp
    )

    mdot = prop_mass / burn_time

    return mdot * Isp * g0


def get_required_electric_power(
        thrust,
        Isp,
        efficiency):

    ve = Isp * g0

    return (
        thrust
        * ve
        / (2 * efficiency)
    )


def get_dv(start_mass, end_mass, Isp):

    return Isp * g0 * np.log(start_mass / end_mass)


# ============================================================
#                         STAGE
# ============================================================

class Stage():

    def __init__(
        self,
        Isp,
        max_prop_mass,
        dry_mass
    ):

        self.Isp = Isp
        self.max_prop_mass = max_prop_mass
        self.dry_mass = dry_mass

        self.total_mass = (
            max_prop_mass + dry_mass
        )

    def get_total_dv(self, payload_mass):

        start_mass = (
            self.max_prop_mass
            + self.dry_mass
            + payload_mass
        )

        end_mass = (
            self.dry_mass
            + payload_mass
        )

        return get_dv(
            start_mass,
            end_mass,
            self.Isp
        )

    def get_remaining_dv(
        self,
        remaining_prop_mass,
        payload_mass
    ):

        start_mass = (
            remaining_prop_mass
            + self.dry_mass
            + payload_mass
        )

        end_mass = (
            self.dry_mass
            + payload_mass
        )

        return get_dv(
            start_mass,
            end_mass,
            self.Isp
        )


# ============================================================
#                         VEHICLE
# ============================================================

class Vehicle():

    def __init__(self, StageList):

        self.StageList = StageList

    def get_total_dv(self, payload_mass=0):

        total_dv = 0

        upper_mass = payload_mass

        for stage in reversed(self.StageList):

            start_mass = (
                stage.dry_mass
                + stage.max_prop_mass
                + upper_mass
            )

            end_mass = (
                stage.dry_mass
                + upper_mass
            )

            stage_dv = get_dv(
                start_mass,
                end_mass,
                stage.Isp
            )

            total_dv += stage_dv

            upper_mass += (
                stage.dry_mass
                + stage.max_prop_mass
            )

        return total_dv


# ============================================================
#                         LAUNCHER
# ============================================================

class Launcher():

    def __init__(
        self,
        LEO_payload,
        UpperStage,
        LEO_payload_altitude=200_000
    ):

        self.LEO_payload = LEO_payload
        self.UpperStage = UpperStage
        self.LEO_payload_altitude = LEO_payload_altitude

        r = R_earth + LEO_payload_altitude

        self.ref_LEO_velocity = np.sqrt(mu_earth / r)

        self.ref_escape_velocity = np.sqrt(
            2 * mu_earth / r
        )

    def get_vinf_performance(self, payload_mass):

        remaining_prop_mass = min(
            self.LEO_payload - payload_mass,
            self.UpperStage.max_prop_mass
        )

        if remaining_prop_mass <= 0:
            return 0.0

        dv = self.UpperStage.get_remaining_dv(
            remaining_prop_mass,
            payload_mass
        )

        return dv_to_vinf(
            dv,
            self.ref_LEO_velocity,
            self.ref_escape_velocity
        )

    def get_C3_performance(self, payload_mass):

        vinf = self.get_vinf_performance(payload_mass)

        return vinf**2

    def plot_vinf(
        self,
        ax,
        n=2000,
        label=None,
        vinf_threshold=0,
        kickstage=None
    ):

        payloads = np.linspace(
            100,
            self.LEO_payload * 0.99,
            n
        )

        base_payloads = payloads.copy()

        vinf_vals = np.array([
            self.get_vinf_performance(m)
            for m in base_payloads
        ])

        if kickstage is not None:
            vinf_vals = np.array([
                self.get_vinf_performance(m+kickstage.total_mass)
                for m in base_payloads
            ])

            kick_dv = np.array([
                kickstage.get_total_dv(m)
                for m in base_payloads
            ])

            vinf_vals = np.array([
                combine_vinf_and_dv(
                    vinf,
                    dv,
                    self.ref_escape_velocity
                )
                for vinf, dv in zip(vinf_vals, kick_dv)
            ])

        mask = vinf_vals > vinf_threshold

        ax.plot(
            vinf_vals[mask],
            base_payloads[mask],
            label=label if label else "Launcher"
        )

        ax.set_ylabel("Payload Mass [kg]")
        ax.set_xlabel("V∞ [m/s]")
        ax.grid(True)

    def plot_C3(
        self,
        ax,
        n=2000,
        label=None,
        vinf_threshold=0,
        kickstage=None
    ):

        payloads = np.linspace(
            100,
            self.LEO_payload * 0.99,
            n
        )

        base_payloads = payloads.copy()

        vinf_vals = np.array([
            self.get_vinf_performance(m)
            for m in base_payloads
        ])

        if kickstage is not None:
            vinf_vals = np.array([
                self.get_vinf_performance(m+kickstage.total_mass)
                for m in base_payloads
            ])

            kick_dv = np.array([
                kickstage.get_total_dv(m)
                for m in base_payloads
            ])

            vinf_vals = np.array([
                combine_vinf_and_dv(
                    vinf,
                    dv,
                    self.ref_escape_velocity
                )
                for vinf, dv in zip(vinf_vals, kick_dv)
            ])

        c3_vals = vinf_vals**2

        mask = vinf_vals > vinf_threshold

        ax.plot(
            c3_vals[mask],
            base_payloads[mask],
            label=label if label else "Launcher"
        )

        ax.set_ylabel("Payload Mass [kg]")
        ax.set_xlabel("C3 [m²/s²]")
        ax.grid(True)


# ============================================================
# REFERENCE STAGES
# ============================================================

Helios = Stage(
    Isp=375,
    max_prop_mass=14_000,
    dry_mass=2_000
)

StarshipUpper = Stage(
    Isp=380,
    max_prop_mass=1_200_000,
    dry_mass=85_000
)

CentaurV = Stage(
    Isp=451,
    max_prop_mass=54_000,
    dry_mass=5_400
)

Ariane64Upper = Stage(
    Isp=457,
    max_prop_mass=31_000,
    dry_mass=4_500
)

FalconHeavyUpper = Stage(
    Isp=348,
    max_prop_mass=109_000,
    dry_mass=10_000
)

SLS_ICPS = Stage(
    Isp=465,
    max_prop_mass=27_200,
    dry_mass=3_500
)

NewGlennUpper = Stage(
    Isp=450,
    max_prop_mass=160_000,
    dry_mass=12_000
)

Star48BV = Stage(
    Isp=292,
    max_prop_mass=2010,
    dry_mass=120
)

Star63 = Stage(
    Isp=298,
    max_prop_mass=11000,
    dry_mass=900
)

Orion38 = Stage(
    Isp=289,
    max_prop_mass=770,
    dry_mass=70
)

ESC_A = Stage(
    Isp=444,            # HM7B vacuum specific impulse
    max_prop_mass=14700, # kg (LOX/LH2 propellant load)
    dry_mass=4500        # kg (structure + engine + avionics)
)

ESCB = Stage(
    Isp=457,
    max_prop_mass=28_000,
    dry_mass=4_000
)

VegaC_AVUM_plus = Stage(
    Isp=315,          # s (typical hypergolic AVUM/AVUM+ range ~315–320 s)
    max_prop_mass=2200,  # kg (approx prop load)
    dry_mass=160      # kg (structure + engine + avionics, approx)
)

VegaC_Zefiro9 = Stage(
    Isp=295,
    max_prop_mass=3500,
    dry_mass=900
)

Astris = Stage(
    Isp=320,          # estimated effective vacuum Isp (BERTA class hypergolic, optimistic modern design)
    max_prop_mass=2000,   # kg (assumed mission-configurable tanked kick stage)
    dry_mass=350       # kg (structure + tanks + avionics, ESA modern lightweight kick stage estimate)
)


# ============================================================
# LAUNCHERS
# ============================================================

Ariane64_Launcher = Launcher(
    LEO_payload=21_000,
    UpperStage=Ariane64Upper
)

Ariane62_Launcher = Launcher(
    LEO_payload=10_500,
    UpperStage=Ariane64Upper
)

FalconHeavy_Expendable = Launcher(
    LEO_payload=63_800,
    UpperStage=FalconHeavyUpper
)

FalconHeavy_Reusable = Launcher(
    LEO_payload=50_000,
    UpperStage=FalconHeavyUpper
)

SLS_Block1_ICPS = Launcher(
    LEO_payload=95_000,
    UpperStage=SLS_ICPS
)

Starship_SuperHeavy = Launcher(
    LEO_payload=150_000,
    UpperStage=StarshipUpper
)

Vulcan = Launcher(
    LEO_payload=27_200,
    UpperStage=CentaurV
)

SLS_CentaurV = Launcher(
    LEO_payload=105_000,
    UpperStage=CentaurV
)

Falcon9 = Launcher(
    LEO_payload=22_800,
    UpperStage=FalconHeavyUpper
)

NewGlennLauncher = Launcher(
    LEO_payload=45_000,
    UpperStage=NewGlennUpper
)

# ============================================================
# SPACECRAFT SIZING
# ============================================================

def size_spacecraft(
    bus_mass,
    launchers,
    kickstages,
    total_dv,
    rdvz_dv,
    tolerance=0.001,
    verbose=False
):

    initial_mass, payload_mass, required_prop, _, _, _, _, _ =run_configuration(bus_mass, print_results=False)
    needed_excess_dv = float(total_dv - rdvz_dv)



    Space_Craft = Stage(
        320,
        required_prop,
        initial_mass-required_prop
    )

    wet_mass = initial_mass

    dry_mass = initial_mass - required_prop

    prop_mass = required_prop

    prop_mass_fraction = prop_mass / wet_mass

    viable = []

    for launcher, launcher_name in launchers:

        for kick, kick_name in kickstages:

            m = wet_mass

            if kick is not None:
                if m+kick.total_mass > launcher.LEO_payload:
                    continue

                v_inf_launcher = launcher.get_vinf_performance(
                    m + kick.total_mass
                )

                kick_dv = kick.get_total_dv(m)

                total_vinf = combine_vinf_and_dv(
                    v_inf_launcher,
                    kick_dv,
                    launcher.ref_escape_velocity
                )

            else:

                total_vinf = launcher.get_vinf_performance(m)

            margin = total_vinf - needed_excess_dv

            if margin > 0:

                viable.append({
                    "launcher_name": launcher_name,
                    "kick_name": kick_name,
                    "launcher": launcher,
                    "kickstage": kick,
                    "margin_m_s": margin
                })

    viable.sort(
        key=lambda x: x["margin_m_s"],
        reverse=True
    )

    if verbose:

        print("\n========== SPACECRAFT SIZING ==========")

        print(f"Bus mass:      {bus_mass:.1f} kg")
        # print(f"Tank mass:     {tank_mass:.1f} kg")
        print(f"Prop mass:     {prop_mass:.1f} kg")
        print(f"Wet mass:      {wet_mass:.1f} kg")
        print(f"Prop fraction: {prop_mass_fraction:.4f}")

        print("\n========== VIABLE COMBINATIONS ==========")

        if not viable:

            print("No viable combinations found.")

        else:

            for v in viable:

                print(
                    f"{v['launcher_name']:25s} + "
                    f"{v['kick_name']:20s} | "
                    f"margin: {v['margin_m_s']:.1f} m/s"
                )

    return {
        "wet_mass": wet_mass,
        "dry_mass": dry_mass,
        "prop_mass": prop_mass,
        "prop_fraction": prop_mass_fraction,
        "viable_combinations": viable,
        "spacecraft": Space_Craft,
        "payload_mass": payload_mass
    }


# ============================================================
# PLOTTING
# ============================================================
def plot_launcher_wetmass_feasibility(
    bus_mass_range,
    launchers,
    kickstages,
    total_dv,
    rdvz_dv,
    vertical_wetmass=None,
    vertical_label=None,
    vertical_color="red"
):
    """
    Similar to plot_launcher_busmass_feasibility(),
    but x-axis is spacecraft TOTAL WET MASS.

    Parameters
    ----------
    vertical_wetmass : float or None
        Optional wet mass value [kg] where a vertical
        reference line will be drawn.

    vertical_label : str or None
        Optional label for the vertical line.

    vertical_color : str
        Matplotlib color for the vertical line.
    """

    launcher_names = [name for _, name in launchers]

    kickstage_names = [name for _, name in kickstages]

    kickstage_name_map = {
        stage_obj: name
        for stage_obj, name in kickstages
    }

    # =========================================================
    # STORAGE
    # =========================================================
    availability = {}

    wet_masses = np.zeros(len(bus_mass_range))

    for lname in launcher_names:

        availability[(lname, "Direct")] = np.zeros(
            len(bus_mass_range),
            dtype=bool
        )

        for ks in kickstage_names:

            availability[(lname, ks)] = np.zeros(
                len(bus_mass_range),
                dtype=bool
            )

    # =========================================================
    # EVALUATION
    # =========================================================
    for j, bus_mass in enumerate(bus_mass_range):

        result = size_spacecraft(
            bus_mass,
            launchers,
            kickstages,
            total_dv,
            rdvz_dv,
            verbose=False
        )

        wet_masses[j] = result["wet_mass"]

        for v in result["viable_combinations"]:

            lname = v["launcher_name"]

            if v["kickstage"] is None:

                availability[(lname, "Direct")][j] = True

            else:

                ks = kickstage_name_map[v["kickstage"]]

                availability[(lname, ks)][j] = True

    # =========================================================
    # SORT BY WET MASS
    # =========================================================
    sort_idx = np.argsort(wet_masses)

    wet_masses = wet_masses[sort_idx]

    for key in availability:
        availability[key] = availability[key][sort_idx]

    # =========================================================
    # COLORS
    # =========================================================
    cmap = plt.get_cmap("tab20")

    mode_colors = {
        "Direct": "#006400"
    }

    for i, ks in enumerate(kickstage_names):
        mode_colors[ks] = cmap(i)

    # =========================================================
    # PLOT
    # =========================================================
    fig, ax = plt.subplots(figsize=(15, 8))

    launcher_height = 0.8

    subbar_height = (
        launcher_height / max(len(kickstage_names), 1)
    )

    for i, lname in enumerate(launcher_names):

        base_y = i - launcher_height / 2

        direct = availability[(lname, "Direct")]

        # =====================================================
        # 1. KICKSTAGES
        # =====================================================
        for m, ks in enumerate(kickstage_names):

            y = base_y + m * subbar_height

            feasible = np.logical_and(
                availability[(lname, ks)],
                ~direct
            )

            start = None

            for j, val in enumerate(feasible):

                if val and start is None:
                    start = j

                end = (
                    (not val or j == len(feasible) - 1)
                    and start is not None
                )

                if end:

                    end_j = (
                        j if val and j == len(feasible) - 1
                        else j - 1
                    )

                    x0 = wet_masses[start]
                    x1 = wet_masses[end_j]

                    ax.broken_barh(
                        [(x0, x1 - x0)],
                        (y, subbar_height * 0.9),
                        facecolors=mode_colors[ks],
                        alpha=0.7,
                        zorder=1
                    )

                    start = None

        # =====================================================
        # 2. DIRECT
        # =====================================================
        start = None

        for j, val in enumerate(direct):

            if val and start is None:
                start = j

            end = (
                (not val or j == len(direct) - 1)
                and start is not None
            )

            if end:

                end_j = (
                    j if val and j == len(direct) - 1
                    else j - 1
                )

                x0 = wet_masses[start]
                x1 = wet_masses[end_j]

                ax.broken_barh(
                    [(x0, x1 - x0)],
                    (i - launcher_height / 2, launcher_height),
                    facecolors="#006400",
                    alpha=0.9,
                    zorder=10
                )

                start = None

    # =========================================================
    # OPTIONAL VERTICAL LINE
    # =========================================================
    if vertical_wetmass is not None:

        ax.axvline(
            vertical_wetmass,
            color=vertical_color,
            linestyle="--",
            linewidth=2
        )

        if vertical_label is not None:

            ax.text(
                vertical_wetmass+100,
                len(launchers) - 0.3,
                vertical_label,
                rotation=90,
                color=vertical_color,
                va="top",
                ha="left"
            )

    # =========================================================
    # LABELS
    # =========================================================
    ax.set_yticks(range(len(launcher_names)))
    ax.set_yticklabels(launcher_names)

    ax.set_xlabel("Spacecraft Wet Mass [kg]")
    ax.set_ylabel("Launcher")

    ax.set_title(
        "Launcher / Kickstage Feasibility vs Spacecraft Wet Mass"
    )

    ax.grid(
        True,
        axis="x",
        linestyle="--",
        alpha=0.4
    )

    # =========================================================
    # LEGEND
    # =========================================================
    legend_handles = [
        Patch(color="#006400", label="Direct Injection")
    ]

    for ks in kickstage_names:

        legend_handles.append(
            Patch(color=mode_colors[ks], label=ks)
        )

    ax.legend(
        handles=legend_handles,
        loc="upper right"
    )

    ax.set_xlim(
        wet_masses[0],
        wet_masses[-1]
    )

    ax.set_ylim(
        -0.5,
        len(launchers) - 0.5
    )

    plt.tight_layout()
    plt.show()

def plot_available_launchers_vs_bus_mass(
    bus_mass_range,
    launchers,
    kickstages,
    total_dv,
    rdvz_dv
):

    counts = []
    payload_masses = []

    for bus_mass in bus_mass_range:

        result = size_spacecraft(
            bus_mass,
            launchers,
            kickstages,
            total_dv,
            rdvz_dv,
            verbose=False
        )

        counts.append(
            len(result["viable_combinations"])
        )
        payload_masses.append(result["payload_mass"])

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(payload_masses, counts)

    ax.set_xlabel("Payload Mass [kg]")
    ax.set_ylabel("Number of Viable Launchers")

    ax.set_title(
        "Launcher Availability vs Spacecraft Payload Mass"
    )

    ax.grid(True)

    plt.tight_layout()
    plt.show()


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def plot_launcher_busmass_feasibility(
    bus_mass_range,
    launchers,
    kickstages,
    total_dv,
    rdvz_dv
):

    launcher_names = [name for _, name in launchers]

    kickstage_names = [name for _, name in kickstages]

    kickstage_name_map = {
        stage_obj: name
        for stage_obj, name in kickstages
    }

    mode_names = kickstage_names  # only kickstages drawn in sub-bars

    # =========================================================
    # STORAGE
    # =========================================================
    availability = {}

    payload_masses = np.zeros(len(bus_mass_range))

    for lname in launcher_names:
        availability[(lname, "Direct")] = np.zeros(len(bus_mass_range), dtype=bool)
        for ks in kickstage_names:
            availability[(lname, ks)] = np.zeros(len(bus_mass_range), dtype=bool)

    # =========================================================
    # EVALUATION
    # =========================================================
    for j, bus_mass in enumerate(bus_mass_range):

        result = size_spacecraft(
            bus_mass,
            launchers,
            kickstages,
            total_dv,
            rdvz_dv,
            verbose=False
        )

        payload_masses[j] = result["payload_mass"]

        for v in result["viable_combinations"]:

            lname = v["launcher_name"]

            if v["kickstage"] is None:
                availability[(lname, "Direct")][j] = True
            else:
                ks = kickstage_name_map[v["kickstage"]]
                availability[(lname, ks)][j] = True

    # =========================================================
    # SORT BY PAYLOAD MASS (IMPORTANT)
    # =========================================================
    sort_idx = np.argsort(payload_masses)

    payload_masses = payload_masses[sort_idx]

    for key in availability:
        availability[key] = availability[key][sort_idx]

    # =========================================================
    # COLORS
    # =========================================================
    cmap = plt.get_cmap("tab20")

    mode_colors = {
        "Direct": "#006400"
    }

    for i, ks in enumerate(kickstage_names):
        mode_colors[ks] = cmap(i)

    # =========================================================
    # PLOT
    # =========================================================
    fig, ax = plt.subplots(figsize=(15, 8))

    launcher_height = 0.8
    subbar_height = launcher_height / max(len(kickstage_names), 1)

    for i, lname in enumerate(launcher_names):

        base_y = i - launcher_height / 2

        direct = availability[(lname, "Direct")]

        # =====================================================
        # 1. KICKSTAGES (masked by direct)
        # =====================================================
        for m, ks in enumerate(kickstage_names):

            y = base_y + m * subbar_height

            feasible = np.logical_and(
                availability[(lname, ks)],
                ~direct
            )

            start = None

            for j, val in enumerate(feasible):

                if val and start is None:
                    start = j

                end = (not val or j == len(feasible) - 1) and start is not None

                if end:

                    end_j = j if val and j == len(feasible) - 1 else j - 1

                    x0 = payload_masses[start]
                    x1 = payload_masses[end_j]

                    ax.broken_barh(
                        [(x0, x1 - x0)],
                        (y, subbar_height * 0.9),
                        facecolors=mode_colors[ks],
                        alpha=0.7,
                        zorder=1
                    )

                    start = None

        # =====================================================
        # 2. DIRECT (FULL OVERLAY)
        # =====================================================
        start = None

        for j, val in enumerate(direct):

            if val and start is None:
                start = j

            end = (not val or j == len(direct) - 1) and start is not None

            if end:

                end_j = j if val and j == len(direct) - 1 else j - 1

                x0 = payload_masses[start]
                x1 = payload_masses[end_j]

                ax.broken_barh(
                    [(x0, x1 - x0)],
                    (i - launcher_height / 2, launcher_height),
                    facecolors="#006400",
                    alpha=0.9,
                    zorder=10
                )

                start = None

    # =========================================================
    # LABELS
    # =========================================================
    ax.set_yticks(range(len(launcher_names)))
    ax.set_yticklabels(launcher_names)

    ax.set_xlabel("Payload Mass [kg]")
    ax.set_ylabel("Launcher")
    ax.set_title("Launcher / Kickstage Feasibility vs Payload Mass")

    ax.grid(True, axis="x", linestyle="--", alpha=0.4)

    # =========================================================
    # LEGEND
    # =========================================================
    legend_handles = [
        Patch(color="#006400", label="Direct Injection")
    ]

    for ks in kickstage_names:
        legend_handles.append(
            Patch(color=mode_colors[ks], label=ks)
        )

    ax.legend(handles=legend_handles, loc="upper right")

    ax.set_xlim(payload_masses[0], payload_masses[-1])
    ax.set_ylim(-0.5, len(launchers) - 0.5)

    plt.tight_layout()
    plt.show()


# ============================================================
# V-INF PERFORMANCE COMPARISON PLOT
# ============================================================

def plot_vinf_comparison(
    launchers,
    kickstages,
    payload_min=100,
    payload_max=15000,
    n=1500
):

    fig, ax = plt.subplots(figsize=(12, 8))

    payloads = np.linspace(payload_min, payload_max, n)

    for launcher, launcher_name in launchers:

        for kick, kick_name in kickstages:

            vinf_vals = []

            for m in payloads:

                # ------------------------------------------------
                # NO KICKSTAGE
                # ------------------------------------------------
                if kick is None:

                    vinf = launcher.get_vinf_performance(m)

                # ------------------------------------------------
                # WITH KICKSTAGE
                # ------------------------------------------------
                else:

                    # Skip impossible masses
                    if m + kick.total_mass > launcher.LEO_payload:
                        vinf_vals.append(np.nan)
                        continue

                    launcher_vinf = launcher.get_vinf_performance(
                        m + kick.total_mass
                    )

                    kick_dv = kick.get_total_dv(m)

                    vinf = combine_vinf_and_dv(
                        launcher_vinf,
                        kick_dv,
                        launcher.ref_escape_velocity
                    )

                vinf_vals.append(vinf)

            vinf_vals = np.array(vinf_vals)

            # Remove non-escape trajectories
            vinf_vals[vinf_vals <= 0] = np.nan

            # Label formatting
            if kick is None:
                label = launcher_name
            else:
                label = f"{launcher_name} + {kick_name}"

            ax.plot(
                vinf_vals,
                payloads,
                label=label,
                linewidth=2
            )

    # =========================================================
    # HORIZONTAL PAYLOAD LINES
    # =========================================================

    payload_lines = [1000, 3000, 5000, 10000]

    for p in payload_lines:

        ax.axhline(
            p,
            color="gray",
            linestyle="--",
            linewidth=1
        )

        ax.text(
            500,
            p + 80,
            f"{p} kg",
            color="gray"
        )

    # =========================================================
    # VERTICAL V-INF LINES
    # =========================================================

    vinf_lines = [10000, 15000]

    for v in vinf_lines:

        ax.axvline(
            v,
            color="red",
            linestyle=":",
            linewidth=1.5
        )

        ax.text(
            v + 150,
            payload_min + 300,
            f"{v/1000:.0f} km/s",
            rotation=90,
            color="red"
        )

    # =========================================================
    # FORMATTING
    # =========================================================

    ax.set_xlabel(r"$V_\infty$ [m/s]")
    ax.set_ylabel("Payload Mass [kg]")

    ax.set_title(
        r"Launcher + Kickstage Hyperbolic Excess Velocity Performance"
    )

    ax.grid(True)

    ax.set_xlim(left=0)
    ax.set_ylim(payload_min, payload_max)

    ax.legend(
        fontsize=9,
        loc="best"
    )

    plt.tight_layout()
    plt.show()




# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    launchers = [

        (Ariane64_Launcher, "Ariane 64"),
        (Ariane62_Launcher, "Ariane 62"),

        (FalconHeavy_Expendable,
         "Falcon Heavy (Expendable)"),

        (FalconHeavy_Reusable,
         "Falcon Heavy (Reusable)"),

        (SLS_Block1_ICPS,
         "SLS Block 1 (ICPS)"),

        (Starship_SuperHeavy,
         "Starship + Super Heavy"),

        (Vulcan,
         "Vulcan Centaur"),

        (SLS_CentaurV,
         "SLS + Centaur V"),

        (Falcon9,
         "Falcon 9"),

        (NewGlennLauncher,
         "New Glenn")
    ]

    kickstages = [

        (Helios, "Helios"),

        (Star63, "Star63"),

        (Star48BV, "Star48BV"),

        (ESCB, "ESC-B (Ariane 6 Second Stage)"),

        (ESC_A, "ESC-A (Ariane 5 Second Stage)"),

        (Orion38, "Orion38"),

        (VegaC_Zefiro9, "VegaC Zefiro9"),

        (VegaC_AVUM_plus, "VegaC AVUM +"),

        (Astris, "Astris"),

        # (CentaurV, "Centaur V"),

        # (Ariane64Upper, "Ariane 64 Upper"),

        # (NewGlennUpper, "New Glenn S2"),

        (None, "")
    ]

    plot_vinf_comparison(
        launchers,
        kickstages,
        payload_min=1,
        payload_max=20000,
        n=2000
    )



    # total_dv = 32500
    insert_dv = 7500
    rdvz_dv =24_000
    # rdvz_dv = total_dv - insert_dv
    total_dv=insert_dv+rdvz_dv


    print("Rendezvous Delta V", rdvz_dv)

    bus_mass_range = np.linspace(0, 5000, 1000)
    # plot_available_launchers_vs_bus_mass(
    #     bus_mass_range,
    #     launchers,
    #     kickstages,
    #     total_dv,
    #     rdvz_dv
    # )


    # plot_launcher_busmass_feasibility(
    #     bus_mass_range,
    #     launchers,
    #     kickstages,
    #     total_dv,
    #     rdvz_dv
    # )

    plot_launcher_wetmass_feasibility(
        bus_mass_range,
        launchers,
        kickstages,
        total_dv,
        rdvz_dv,
        vertical_wetmass=11240.2,
        vertical_color='black',
        # vertical_label='Updated Mass Budget'
    )