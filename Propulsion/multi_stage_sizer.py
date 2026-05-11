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
    assumed_spacecraft_isp = 320  # MMH N2H4
    assumed_spacecraft_isp = 3000 # electric

    needed_excess_dv = float(total_dv - rdvz_dv)

    structural_fraction = 0.12

    required_prop = get_prop_mass_with_end_mass(
        rdvz_dv,
        bus_mass,
        assumed_spacecraft_isp
    )

    tank_mass = structural_fraction * required_prop

    previous_prop_mass = 0

    while (
        abs(required_prop - previous_prop_mass)
        / required_prop
        > tolerance
    ):

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

    viable = []

    for launcher, launcher_name in launchers:

        for kick, kick_name in kickstages:

            m = wet_mass

            if kick is not None:

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
        print(f"Tank mass:     {tank_mass:.1f} kg")
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
        "spacecraft": Space_Craft
    }


# ============================================================
# PLOTTING
# ============================================================

def plot_available_launchers_vs_bus_mass(
    bus_mass_range,
    launchers,
    kickstages,
    total_dv,
    rdvz_dv
):

    counts = []

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

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(bus_mass_range, counts)

    ax.set_xlabel("Bus Mass [kg]")
    ax.set_ylabel("Number of Viable Launchers")

    ax.set_title(
        "Launcher Availability vs Spacecraft Bus Mass"
    )

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

    launcher_names = [name for _, name in launchers]

    # 0 = not viable
    # 1 = viable with kickstage
    # 2 = viable without kickstage
    colors = np.zeros((len(launchers), len(bus_mass_range)))

    for j, bus_mass in enumerate(bus_mass_range):

        result = size_spacecraft(
            bus_mass,
            launchers,
            kickstages,
            total_dv,
            rdvz_dv,
            verbose=False
        )

        viable = result["viable_combinations"]

        # Build best-state map
        launcher_state = {
            lname: 0
            for lname in launcher_names
        }

        for v in viable:

            lname = v["launcher_name"]

            # direct injection
            if v["kickstage"] is None:

                launcher_state[lname] = max(
                    launcher_state[lname],
                    2
                )

            # kickstage-assisted
            else:

                launcher_state[lname] = max(
                    launcher_state[lname],
                    1
                )

        # Fill color matrix
        for i, lname in enumerate(launcher_names):

            colors[i, j] = launcher_state[lname]

    # =========================================================
    # PLOT
    # =========================================================

    fig, ax = plt.subplots(figsize=(13, 7))

    cmap = mpl.colors.ListedColormap([
        "black",   # 0
        "blue",    # 1
        "green"    # 2
    ])

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
        interpolation="nearest",
        vmin=0,
        vmax=2
    )

    ax.set_yticks(range(len(launchers)))
    ax.set_yticklabels(launcher_names)

    ax.set_xlabel("Bus Mass [kg]")

    ax.set_title(
        "Launcher Feasibility vs Bus Mass"
    )

    from matplotlib.patches import Patch

    legend = [
        Patch(color="black", label="Not viable"),
        Patch(color="blue", label="Viable with kickstage"),
        Patch(color="green", label="Directly viable")
    ]

    ax.legend(
        handles=legend,
        loc="upper right"
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

        # (CentaurV, "Centaur V"),

        # (Ariane64Upper, "Ariane 64 Upper"),

        # (NewGlennUpper, "New Glenn S2"),

        (None, "")
    ]

    total_dv = 19300

    rdvz_dv = 4000

    print("Rendezvous Delta V", rdvz_dv)

    bus_mass_range = np.linspace(0, 10000, 100)
    plot_available_launchers_vs_bus_mass(
        bus_mass_range,
        launchers,
        kickstages,
        total_dv,
        rdvz_dv
    )


    plot_launcher_busmass_feasibility(
        bus_mass_range,
        launchers,
        kickstages,
        total_dv,
        rdvz_dv
    )