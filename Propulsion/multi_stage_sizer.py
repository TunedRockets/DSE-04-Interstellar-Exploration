import copy
import math
from scipy.optimize import minimize_scalar

from src2.utilities import DAY, YEAR
from src2.orbit import *

import numpy as np
from tqdm import tqdm
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

# ============================================================
#                         CONSTANTS
# ============================================================

g0 = 9.81
sigma = 5.670374419e-8
R_universal = 8.314462618

F_1AU = 1361
R_sun = 6.96e5

LEO_velocity = 7790 # m/s
escape_velocity = np.sqrt(2)*LEO_velocity
# Earth constants
mu_earth = 3.986004418e14   # m^3/s^2
R_earth = 6371e3            # m






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
    return Isp*g0*np.log(start_mass/end_mass)

class Stage():
    def __init__(self, Isp, max_prop_mass, dry_mass, ref_LEO_velocity=LEO_velocity, ref_escape_velocity=escape_velocity):
        self.Isp = Isp
        self.max_prop_mass = max_prop_mass
        self.dry_mass = dry_mass
        self.max_dv = get_dv(max_prop_mass+dry_mass, dry_mass, Isp)
        self.ref_LEO_velocity = ref_LEO_velocity
        self.ref_escape_velocity = ref_escape_velocity
        self.max_excess_dv = self.max_dv + self.ref_LEO_velocity - self.ref_escape_velocity
        self.total_mass = max_prop_mass + dry_mass
    def get_remaining_dv(self, remaining_prop_mass, extra_mass):
        remaining_dv = get_dv(remaining_prop_mass+extra_mass+self.dry_mass, extra_mass+self.dry_mass, self.Isp)
        return remaining_dv
    def get_total_dv(self, extra_mass):
        remaining_prop_mass = self.max_prop_mass
        remaining_dv = get_dv(remaining_prop_mass+extra_mass+self.dry_mass, extra_mass+self.dry_mass, self.Isp)
        return remaining_dv
    def get_excess_dv(self, remaining_prop_mass, extra_mass):
        remaining_dv = get_dv(remaining_prop_mass, extra_mass, self.Isp)
        excess_dv = remaining_dv + self.ref_LEO_velocity - self.ref_escape_velocity
        return excess_dv

class Vehicle():
    def __init__(self, StageList):
        self.StageList = StageList

    def get_total_dv(self, payload_mass=0):
        """
        Computes total stacked delta-v of the vehicle.

        Parameters
        ----------
        payload_mass : float
            Mass carried above the uppermost stage [kg]

        Returns
        -------
        total_dv : float
            Total ideal rocket delta-v [m/s]
        """

        total_dv = 0

        # Mass above current stage
        upper_mass = payload_mass

        # Work from top stage downward
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

            # Entire stage becomes payload for lower stages
            upper_mass += (
                stage.dry_mass
                + stage.max_prop_mass
            )

        return total_dv

    def get_total_excess_dv(self, payload_mass=0):
        """
        Computes total hyperbolic excess delta-v capability.

        Uses:
            excess_dv = total_dv + v_LEO - v_escape

        Parameters
        ----------
        payload_mass : float
            Payload mass above uppermost stage [kg]

        Returns
        -------
        excess_dv : float
            Hyperbolic excess delta-v [m/s]
        """

        total_dv = self.get_total_dv(payload_mass)

        return (
            total_dv
            + LEO_velocity
            - escape_velocity
        )



class Launcher():
    def __init__(self, LEO_payload: float, UpperStage: Stage, LEO_payload_altitude=200_000):
        self.LEO_payload = LEO_payload
        self.UpperStage = UpperStage
        self.LEO_payload_altitude = LEO_payload_altitude
        # Orbital radius
        r = R_earth + LEO_payload_altitude

        # Circular orbital velocity
        self.ref_LEO_velocity = np.sqrt(mu_earth / r)

        # Local escape velocity
        self.ref_escape_velocity = np.sqrt(2 * mu_earth / r)
        self.UpperStage.ref_LEO_velocity = self.ref_LEO_velocity
        self.UpperStage.ref_escape_velocity = self.ref_escape_velocity
    def get_vinf_performance(self, payload_mass):
        remaining_prop_mass = self.LEO_payload - payload_mass
        excess_dv = self.UpperStage.get_excess_dv(remaining_prop_mass, payload_mass)
        return excess_dv
    def get_C3_performance(self, payload_mass):
        excess_dv = self.get_vinf_performance(payload_mass)
        return excess_dv**2

    def plot_vinf(self, ax, n=2000, label=None, vinf_threshold=0, kickstage: Stage = None):
        payloads = np.linspace(100, self.LEO_payload * 0.99, n)

        # base payload mass (do NOT modify payloads in-place)
        base_payloads = payloads.copy()

        vinf_vals = np.array([
            self.get_vinf_performance(m)
            for m in base_payloads
        ])

        if kickstage is not None:
            # treat kickstage as additional delta-v capability
            kick_dv = np.array([
                kickstage.get_excess_dv(kickstage.max_prop_mass, m)
                for m in base_payloads
            ])
            vinf_vals = vinf_vals + kick_dv

        mask = vinf_vals > vinf_threshold

        ax.plot(
            vinf_vals[mask],
            base_payloads[mask],
            label=label if label else f"{self.UpperStage.__class__.__name__}"
        )

        ax.set_ylabel("Payload Mass [kg]")
        ax.set_xlabel("V∞ (Excess Velocity) [m/s]")
        ax.grid(True)

    def plot_C3(self, ax, n=2000, label=None, vinf_threshold=0, kickstage: Stage = None):
        payloads = np.linspace(100, self.LEO_payload * 0.99, n)

        base_payloads = payloads.copy()

        vinf_vals = np.array([
            self.get_vinf_performance(m)
            for m in base_payloads
        ])

        if kickstage is not None:
            kick_dv = np.array([
                kickstage.get_excess_dv(kickstage.max_prop_mass, m)
                for m in base_payloads
            ])
            vinf_vals = vinf_vals + kick_dv

        c3_vals = vinf_vals ** 2
        mask = vinf_vals > vinf_threshold

        ax.plot(
            c3_vals[mask],
            base_payloads[mask],
            label=label if label else f"{self.UpperStage.__class__.__name__}"
        )

        ax.set_ylabel("Payload Mass [kg]")
        ax.set_xlabel("C3 [m²/s²]")
        ax.grid(True)




# ============================================================
# REFERENCE STAGES
# ============================================================

# ------------------------------------------------------------
# Helios Kick Stage (Impulse Space)
# ------------------------------------------------------------
# Public numbers are limited, so these are approximate estimates
# based on published delta-v capability and methalox performance.
#
# https://www.impulsespace.com/helios
#
# Assumptions:
# - Vacuum methalox stage
# - ~375 s Isp (vacuum methalox)
# - ~8:1 mass ratio estimate
#
# Public website states:
# "Delta-V 3 to 9 km/s depending on payload mass"

Helios = Stage(
    Isp=375,
    max_prop_mass=14000,   # kg (estimated)
    dry_mass=2000          # kg (estimated)
)


# ------------------------------------------------------------
# SpaceX Starship Upper Stage
# ------------------------------------------------------------
#
# Dry mass:
# https://space.skyrocket.de/doc_lau/super-heavy-starship.htm
#
# Propellant:
# https://space.skyrocket.de/doc_lau/super-heavy-starship.htm
#
# Isp:
#
#
# Sources indicate:
# - 1200 t propellant
# - 85 t dry mass
# - 380 s Raptor Vacuum Isp

StarshipUpper = Stage(
    Isp=380,               # s
    max_prop_mass=1_200_000,  # kg
    dry_mass=85_000        # kg
)


# ------------------------------------------------------------
# Centaur V
# ------------------------------------------------------------
#
# https://en.wikipedia.org/wiki/Centaur_(rocket_stage)
#
# ULA published values:
# - ~54 t propellant
# - ~5.4 t dry
# - RL10C Isp ~451 s

CentaurV = Stage(
    Isp=451,               # s
    max_prop_mass=54_000,  # kg
    dry_mass=5_400         # kg
)


# ------------------------------------------------------------
# Ariane 64 Upper Stage (ESC-A / Vinci based)
# ------------------------------------------------------------
#
# https://en.wikipedia.org/wiki/Ariane_6
#
# Public values:
# - ~31 t propellant
# - ~4.5 t dry mass estimate
# - Vinci vacuum Isp ~457 s

Ariane64Upper = Stage(
    Isp=457,               # s
    max_prop_mass=31_000,  # kg
    dry_mass=4_500         # kg
)


# ------------------------------------------------------------
# Falcon Heavy / Falcon 9 Upper Stage
# ------------------------------------------------------------
#
# https://www.thespacereview.com/article/3980/1
#
# Values:
# - 109 t propellant
# - 10 t dry mass
# - Merlin Vacuum Isp 348 s
#
# Falcon Heavy uses the standard Falcon 9 upper stage.

FalconHeavyUpper = Stage(
    Isp=348,               # s
    max_prop_mass=109_000, # kg
    dry_mass=10_000        # kg
)


# ------------------------------------------------------------
# SLS ICPS (Interim Cryogenic Propulsion Stage)
# ------------------------------------------------------------
#
# https://en.wikipedia.org/wiki/Interim_Cryogenic_Propulsion_Stage
#
# Based on Delta IV upper stage:
# - ~27.2 t propellant
# - ~3.5 t dry mass
# - RL10B-2 Isp 465 s

SLS_ICPS = Stage(
    Isp=465,               # s
    max_prop_mass=27_200,  # kg
    dry_mass=3_500         # kg
)


# ------------------------------------------------------------
# New Glenn Upper Stage
# ------------------------------------------------------------
#
# Public values are incomplete.
#
# Approximate estimates based on:
# - BE-3U vacuum engine
# - Hydrolox upper stage
# - Payload performance disclosures
#
# https://en.wikipedia.org/wiki/New_Glenn
#
# Estimated:
# - ~160 t propellant
# - ~12 t dry mass
# - BE-3U Isp 450 s

NewGlennUpper = Stage(
    Isp=450,               # s
    max_prop_mass=160_000, # kg (estimated)
    dry_mass=12_000        # kg (estimated)
)

# ============================================================
# LAUNCHER OBJECTS
# ============================================================

# ------------------------------------------------------------
# Ariane 64
# ------------------------------------------------------------
Ariane64_Launcher = Launcher(
    LEO_payload=21_000,          # kg to LEO (A64 ~20–21 t class)
    UpperStage=Ariane64Upper
)

# ------------------------------------------------------------
# Ariane 62
# ------------------------------------------------------------
# Reduced boosters → lower performance (~10–12 t LEO)
Ariane62_Launcher = Launcher(
    LEO_payload=10_500,          # kg (approx)
    UpperStage=Ariane64Upper
)

# ------------------------------------------------------------
# Falcon Heavy (Expendable)
# ------------------------------------------------------------
# Full 3-core expendable performance ~63–64 t
FalconHeavy_Expendable = Launcher(
    LEO_payload=63_800,          # kg
    UpperStage=FalconHeavyUpper
)

# ------------------------------------------------------------
# Falcon Heavy (Partially Reusable)
# ------------------------------------------------------------
# Realistic NASA/SpaceX mission class (~50 t)
FalconHeavy_Reusable = Launcher(
    LEO_payload=50_000,          # kg
    UpperStage=FalconHeavyUpper
)

# ------------------------------------------------------------
# SLS Block 1 (ICPS)
# ------------------------------------------------------------
SLS_Block1_ICPS = Launcher(
    LEO_payload=95_000,          # kg (SLS Block 1 to LEO class)
    UpperStage=SLS_ICPS
)

# ------------------------------------------------------------
# Starship + Super Heavy
# ------------------------------------------------------------
Starship_SuperHeavy = Launcher(
    LEO_payload=150_000,         # kg (fully expendable upper bound)
    UpperStage=StarshipUpper
)

# ------------------------------------------------------------
# Vulcan Centaur (VC4S / heavy config approximation)
# ------------------------------------------------------------
# Conservative LEO estimate (upper stage optimized for GTO)
Vulcan = Launcher(
    LEO_payload=27_200,          # kg (upper-bound optimistic LEO class)
    UpperStage=CentaurV
)

# ------------------------------------------------------------
# SLS with Centaur V (hypothetical architecture)
# ------------------------------------------------------------
# Heavy lift core + high-energy upper stage
SLS_CentaurV = Launcher(
    LEO_payload=105_000,         # kg (slightly higher due to better upper stage)
    UpperStage=CentaurV
)

# ------------------------------------------------------------
# Falcon 9
# ------------------------------------------------------------
Falcon9 = Launcher(
    LEO_payload=22_800,          # kg expendable upper bound
    UpperStage=FalconHeavyUpper   # same physical stage model
)


def show_v_inf():
    # ============================================================
    # VINFINITY COMPARISON PLOT
    # ============================================================

    fig, ax = plt.subplots(figsize=(12, 7))

    launchers = [
        (Ariane64_Launcher, "Ariane 64"),
        (Ariane62_Launcher, "Ariane 62"),
        (FalconHeavy_Expendable, "Falcon Heavy (Expendable)"),
        (FalconHeavy_Reusable, "Falcon Heavy (Reusable)"),
        (SLS_Block1_ICPS, "SLS Block 1 (ICPS)"),
        (Starship_SuperHeavy, "Starship + Super Heavy"),
        (Vulcan, "Vulcan Centaur"),
        (SLS_CentaurV, "SLS + Centaur V"),
        (Falcon9, "Falcon 9"),
    ]

    for launcher, label in launchers:
        launcher.plot_vinf(ax, label=label)

    ax.set_title("Launcher v∞ Performance vs Payload Mass")
    ax.set_ylabel("Payload Mass [kg]")
    ax.set_xlabel("v∞ [m/s]")

    ax.grid(True)
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.show()

def show_v_inf_with_helios():
    fig, ax = plt.subplots(figsize=(12, 7))

    launchers = [
        (Ariane64_Launcher, "Ariane 64"),
        (Ariane62_Launcher, "Ariane 62"),
        (FalconHeavy_Expendable, "Falcon Heavy (Expendable)"),
        (FalconHeavy_Reusable, "Falcon Heavy (Reusable)"),
        (SLS_Block1_ICPS, "SLS Block 1 (ICPS)"),
        (Starship_SuperHeavy, "Starship + Super Heavy"),
        (Vulcan, "Vulcan Centaur"),
        (SLS_CentaurV, "SLS + Centaur V"),
        (Falcon9, "Falcon 9"),
    ]

    for launcher, label in launchers:
        launcher.plot_vinf(
            ax,
            label=label + " + Helios",
            kickstage=Helios   # <<< key change
        )

    ax.set_title("Launcher v∞ Performance with Helios Kick Stage")
    ax.set_ylabel("Payload Mass [kg]")
    ax.set_xlabel("v∞ [m/s]")

    ax.grid(True)
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.show()



def size_hypergolic(
    bus_mass,
    launchers,
    kickstages,
    total_dv,
    rdvz_dv,
    tolerance=0.001,
    verbose=False
):
    assumed_spacecraft_isp = 320  # s
    needed_excess_dv = float(total_dv-rdvz_dv)
    structural_fraction = 0.12

    # ============================================================
    # Spacecraft sizing (prop + tank iteration)
    # ============================================================
    required_prop = get_prop_mass_with_end_mass(
        rdvz_dv, bus_mass, assumed_spacecraft_isp
    )

    tank_mass = structural_fraction * required_prop
    previous_prop_mass = 0

    while abs(required_prop - previous_prop_mass) / required_prop > tolerance:
        previous_prop_mass = required_prop

        required_prop = get_prop_mass_with_end_mass(
            rdvz_dv,
            bus_mass + tank_mass,
            assumed_spacecraft_isp
        )

        tank_mass = structural_fraction * required_prop

    Space_Craft = Stage(
        assumed_spacecraft_isp,
        required_prop,
        tank_mass + bus_mass
    )

    wet_mass = Space_Craft.total_mass
    dry_mass = bus_mass + tank_mass
    prop_mass = required_prop
    prop_mass_fraction = prop_mass / wet_mass

    # ============================================================
    # Launcher + kickstage feasibility sweep
    # ============================================================
    viable = []

    for launcher, launcher_name in launchers:
        for kick, kick_name in kickstages:


            best_margin = -1e9

            m=wet_mass
            if kick is not None:
                v_inf_launcher = launcher.get_vinf_performance(m+kick.total_mass)
                kick_dv = kick.get_total_dv(
                    m
                )
            else:
                v_inf_launcher = launcher.get_vinf_performance(m)
                kick_dv = 0




            total_vinf = v_inf_launcher + kick_dv
            margin = total_vinf - needed_excess_dv

            if margin > best_margin:
                best_margin = margin

            if best_margin > 0:
                viable.append({
                    "launcher_name": launcher_name,
                    "kick_name": kick_name,
                    "launcher": launcher,
                    "kickstage": kick,
                    "margin_m_s": best_margin
                })

    viable.sort(key=lambda x: x["margin_m_s"], reverse=True)

    # ============================================================
    # Output
    # ============================================================
    if verbose:
        print("\n========== SPACECRAFT SIZING ==========")
        print(f"Bus mass:             {bus_mass:.1f} kg")
        print(f"Tank mass:            {tank_mass:.1f} kg")
        print(f"Prop mass:            {prop_mass:.1f} kg")
        print(f"Wet mass:             {wet_mass:.1f} kg")
        print(f"Prop fraction:        {prop_mass_fraction:.4f}")

        print("\n========== VIABLE LAUNCH STACKS ==========")

        if not viable:
            print("No viable launcher–kickstage combinations found.")
        else:
            for v in viable:
                print(
                    f"{v['launcher_name']:25s} + "
                    f"{v['kick_name']:15s} | "
                    f"margin: {v['margin_m_s']:.1f} m/s"
                )

    return {
        "wet_mass": wet_mass,
        "dry_mass": dry_mass,
        "prop_mass": prop_mass,
        "prop_fraction": prop_mass_fraction,
        "viable_combinations": viable,
        "spacecraft": Space_Craft
    }

def plot_available_launchers_vs_bus_mass(
    launchers,
    kickstages,
    total_dv,
    rdvz_dv,
    bus_mass_range=np.linspace(10, 1000, 80)
):
    counts = []

    for bus_mass in bus_mass_range:

        result = size_hypergolic(
            bus_mass,
            launchers,
            kickstages,
            total_dv,
            rdvz_dv,
            verbose=False
        )

        counts.append(len(result["viable_combinations"]))

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(bus_mass_range, counts)

    ax.set_xlabel("Bus Mass [kg]")
    ax.set_ylabel("Number of Viable Launchers")
    ax.set_title("Launcher Availability vs Spacecraft Bus Mass")
    ax.grid(True)

    plt.tight_layout()
    plt.show()
def plot_launcher_busmass_feasibility(
    bus_mass_range,
    launchers,
    kickstages,
    total_dv,
    rdvz_dv
):
    import numpy as np
    import matplotlib.pyplot as plt

    launcher_names = [name for _, name in launchers]

    # color grid:
    # 0 = not viable (red)
    # 1 = kickstage only (blue)
    # 2 = direct viable (green)
    colors = np.zeros((len(launchers), len(bus_mass_range)))

    for j, bus_mass in enumerate(bus_mass_range):

        # -----------------------------
        # compute feasibility ONCE
        # -----------------------------
        result = size_hypergolic(
            bus_mass,
            launchers,
            kickstages,
            total_dv,
            rdvz_dv,
            verbose=False
        )

        viable = result["viable_combinations"]

        viable_with_kick = set()
        viable_without_kick = set()

        for v in viable:
            lname = v["launcher_name"]
            if v["kickstage"] is None:
                viable_without_kick.add(lname)
            else:
                viable_with_kick.add(lname)

        # -----------------------------
        # classify each launcher
        # -----------------------------
        for i, (_, lname) in enumerate(launchers):

            if lname in viable_without_kick:
                colors[i, j] = 2  # green (best case)

            elif lname in viable_with_kick:
                colors[i, j] = 1  # blue (needs kickstage)

            else:
                colors[i, j] = 0  # red

    # -----------------------------
    # PLOT
    # -----------------------------
    fig, ax = plt.subplots(figsize=(12, 6))

    cmap = plt.cm.colors.ListedColormap(["black", "blue", "green"])

    im = ax.imshow(
        colors,
        aspect="auto",
        origin="lower",
        extent=[
            bus_mass_range[0],
            bus_mass_range[-1],
            -0.5,
            len(launchers) - 0.5
        ],
        cmap=cmap,
        interpolation="nearest"
    )

    ax.set_yticks(range(len(launchers)))
    ax.set_yticklabels(launcher_names)

    ax.set_xlabel("Bus Mass [kg]")
    ax.set_title("Launcher Feasibility vs Bus Mass")

    # legend
    from matplotlib.patches import Patch
    legend = [
        Patch(color="black", label="Not viable"),
        Patch(color="blue", label="Viable (kickstage required)"),
        Patch(color="green", label="Viable (no kickstage)")
    ]
    ax.legend(handles=legend, loc="upper right")

    plt.tight_layout()
    plt.show()




if __name__ == "__main__":
    # show_v_inf()
    # show_v_inf_with_helios()

    launchers = [
        (Ariane64_Launcher, "Ariane 64"),
        (Ariane62_Launcher, "Ariane 62"),
        (FalconHeavy_Expendable, "Falcon Heavy (Expendable)"),
        (FalconHeavy_Reusable, "Falcon Heavy (Reusable)"),
        (SLS_Block1_ICPS, "SLS Block 1 (ICPS)"),
        (Starship_SuperHeavy, "Starship + Super Heavy"),
        (Vulcan, "Vulcan Centaur"),
        (SLS_CentaurV, "SLS + Centaur V"),
        (Falcon9, "Falcon 9"),
    ]

    kickstages = [(Helios, "Helios"), (None, (""))]

    total_dv = float(19300)  # m/s
    rdvz_dv = float(3000) # m/s

    size_hypergolic(100, launchers, kickstages, total_dv, rdvz_dv,  verbose=True)
    # plot_available_launchers_vs_bus_mass(launchers, kickstages, total_dv, rdvz_dv)
    bus_mass_range = np.linspace(100, 1000, 100)

    # plot_launcher_busmass_feasibility(
    #     bus_mass_range,
    #     launchers,
    #     kickstages,
    #     total_dv,
    #     rdvz_dv
    # )