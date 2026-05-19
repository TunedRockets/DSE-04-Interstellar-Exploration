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

# ============================================================
#                     MISSION INPUTS
# ============================================================

# ------------------------------------------------------------
# Mission
# ------------------------------------------------------------

total_dv = 24_000
plane_change_delta_v = 3_000
oberth_delta_v = 4_000

rendezvous_delta_v = (
        total_dv
        - plane_change_delta_v
        - oberth_delta_v
)

# payload_dry_mass = 1000  # kg


# ------------------------------------------------------------
# Electric propulsion system
# ------------------------------------------------------------

ion_isp = 4150  # s - based on NEXT thruster performance, average
ion_thruster_thrust = 0.237  # N - based on NEXT thruster performance, higher value (lower 0.236)
ion_efficiency = 0.70  # - based on NEXT thruster performance


# ------------------------------------------------------------
# Reactor / Brayton
# ------------------------------------------------------------

brayton_hot_side_temperature = 1673  # K
radiator_emissivity = 0.9
brayton_eta_fraction = 0.6

# ------------------------------------------------------------
# Solar flyby
# ------------------------------------------------------------

perihelion_distance = 10 * R_sun
solar_absorptivity = 0.95

# ============================================================
#                    ORBIT DEFINITIONS
# ============================================================

lon_per = 40

aphelion = 5.4507 * AU

semi_major_axis = (
                          aphelion + perihelion_distance
                  ) / 2

eccentricity = (
                       aphelion - perihelion_distance
               ) / (
                       aphelion + perihelion_distance
               )

origin = orbit_from_ephemeris(
    semi_major_axis,
    eccentricity,
    m.radians(1.303),
    m.radians(100.46457166),
    m.radians(lon_per),
    m.radians(100.464),
    SGP_SUN
)

transfer_orbit = copy.deepcopy(origin)

transfer_orbit.e = 1.0562986320414216
transfer_orbit.a = -123686841.89123283

# ============================================================
#                      BURN WINDOWS
# ============================================================

# plane_change_burn_time = 13709184      # s
# oberth_burn_time = 12566               # s
# rendezvous_burn_time = 2 * YEAR

plane_change_burn_time = origin.max_impulsive_burn_time(np.pi, 5)
oberth_burn_time = origin.max_impulsive_burn_time(0, 10)
rendezvous_burn_time = 2 * YEAR


# ============================================================
#                     BASIC UTILITIES
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


# ============================================================
#                THERMAL ROCKET PERFORMANCE
# ============================================================

def get_exhaust_velocity_thermal(
        T_chamber,
        molecular_mass):
    gamma = 1.4

    return np.sqrt(
        (2 * gamma / (gamma - 1))
        * (R_universal / molecular_mass)
        * T_chamber
    )


def get_thermal_isp(
        T_chamber,
        molecular_mass):
    return (
            get_exhaust_velocity_thermal(
                T_chamber,
                molecular_mass
            )
            / g0
    )


def get_thermal_power_required(
        mdot,
        T_chamber,
        T_inlet,
        molecular_mass):
    cp = (
            (7 / 2)
            * (R_universal / molecular_mass)
    )

    return (
            mdot
            * cp
            * (T_chamber - T_inlet)
    )


# ============================================================
#                   PUMP / PRESSURE MODEL
# ============================================================

def get_chamber_pressure(
        tank_pressure,
        electric_pump_power,
        mdot,
        pump_efficiency,
        propellant_density):
    if mdot <= 0:
        return tank_pressure

    delta_p = (
            electric_pump_power
            * propellant_density
            * pump_efficiency
            / mdot
    )

    return tank_pressure + delta_p


def get_required_pump_power(
        desired_pressure,
        tank_pressure,
        mdot,
        pump_efficiency,
        propellant_density):
    delta_p = max(
        0,
        desired_pressure - tank_pressure
    )

    return (
            mdot
            * delta_p
            / (
                    propellant_density
                    * pump_efficiency
            )
    )


# ============================================================
#                   BRAYTON OPTIMIZER
# ============================================================

def optimize_brayton_radiator(
        electric_power,
        T_hot,
        emissivity=0.9,
        T_space=3,
        eta_fraction=0.6):
    def radiator_area(T_cold):

        if (
                T_cold <= T_space
                or T_cold >= T_hot
        ):
            return np.inf

        eta = (
                eta_fraction
                * (1 - T_cold / T_hot)
        )

        if eta <= 0:
            return np.inf

        thermal_power = electric_power / eta

        waste_heat = (
                thermal_power
                - electric_power
        )

        return (
                waste_heat
                / (
                        emissivity
                        * sigma
                        * (
                                T_cold ** 4
                                - T_space ** 4
                        )
                )
        )

    result = minimize_scalar(
        radiator_area,
        bounds=(300, T_hot - 1),
        method='bounded'
    )

    T_cold = result.x

    eta = (
            eta_fraction
            * (1 - T_cold / T_hot)
    )

    thermal_power = electric_power / eta

    waste_heat = (
            thermal_power
            - electric_power
    )

    area = radiator_area(T_cold)

    return (
        T_cold,
        eta,
        thermal_power,
        waste_heat,
        area
    )


# ============================================================
#                 SOLAR THERMAL SYSTEM
# ============================================================

def get_solar_flux(distance):
    return (
            F_1AU
            * (AU / distance) ** 2
    )


def get_heatshield_area(
        required_thermal_power,
        solar_flux,
        absorptivity):
    usable_flux = (
            solar_flux
            * absorptivity
    )

    return (
            required_thermal_power
            / usable_flux
    )
def run_configuration(
        spacecraft_dry_mass,
        print_results=False):

    # ============================================================
    # CONSTANTS
    # ============================================================

    hypergolic_isp = 330
    hypergolic_structural_fraction = 0.10

    ion_thruster_mass = 14
    ppu_mass_per_kw = 7
    feed_system_fraction = 0.1

    reactor_specific_mass = 1.3      # kg/kWth
    radiator_areal_density = 5       # kg/m²

    # ============================================================
    # RENDEZVOUS (NO KICKSTAGE ATTACHED)
    # ============================================================

    rendezvous_propellant = (
        get_prop_mass_with_end_mass(
            rendezvous_delta_v,
            spacecraft_dry_mass,
            ion_isp
        )
    )

    spacecraft_mass_after_sep = (
        spacecraft_dry_mass
        + rendezvous_propellant
    )

    # ============================================================
    # ITERATIVE SOLVE
    # ============================================================

    #
    # Unknown coupled quantities:
    #
    # - kickstage propellant
    # - plane change propellant
    # - reactor mass
    # - radiator mass
    #

    kickstage_propellant = 1000.0
    kickstage_dry_mass = 100.0

    previous_total_mass = 0

    for _ in range(100):

        # --------------------------------------------------------
        # Mass BEFORE plane change
        # --------------------------------------------------------

        mass_before_plane_change = (
            spacecraft_mass_after_sep
            + kickstage_propellant
            + kickstage_dry_mass
        )

        # --------------------------------------------------------
        # Plane change propellant
        # --------------------------------------------------------

        plane_change_propellant = (
            get_prop_mass_with_end_mass(
                plane_change_delta_v,
                mass_before_plane_change,
                ion_isp
            )
        )

        spacecraft_propellant = (
            rendezvous_propellant
            + plane_change_propellant
        )

        # --------------------------------------------------------
        # Full mass during plane change
        # --------------------------------------------------------

        full_plane_change_mass = (
            spacecraft_dry_mass
            + spacecraft_propellant
            + kickstage_propellant
            + kickstage_dry_mass
        )

        # --------------------------------------------------------
        # Electric propulsion sizing
        # --------------------------------------------------------

        plane_change_thrust = (
            get_required_thrust(
                plane_change_delta_v,
                ion_isp,
                full_plane_change_mass,
                plane_change_burn_time
            )
        )

        rendezvous_thrust = (
            get_required_thrust(
                rendezvous_delta_v,
                ion_isp,
                spacecraft_mass_after_sep,
                rendezvous_burn_time
            )
        )

        plane_change_power = (
            get_required_electric_power(
                plane_change_thrust,
                ion_isp,
                ion_efficiency
            )
        )

        rendezvous_power = (
            get_required_electric_power(
                rendezvous_thrust,
                ion_isp,
                ion_efficiency
            )
        )

        reactor_electric_power = max(
            plane_change_power,
            rendezvous_power
        )

        required_thruster_count = math.ceil(
            max(
                plane_change_thrust,
                rendezvous_thrust
            )
            / ion_thruster_thrust
        )

        # --------------------------------------------------------
        # Reactor / radiator
        # --------------------------------------------------------

        (
            radiator_Tcold,
            brayton_efficiency,
            reactor_thermal_power,
            reactor_waste_heat,
            radiator_area
        ) = optimize_brayton_radiator(
            reactor_electric_power,
            brayton_hot_side_temperature,
            radiator_emissivity,
            3,
            brayton_eta_fraction
        )

        # --------------------------------------------------------
        # Dry masses
        # --------------------------------------------------------

        ion_system_mass = (
            required_thruster_count
            * ion_thruster_mass
            + ppu_mass_per_kw
            * (reactor_electric_power / 1e3)
        )

        ion_system_mass *= (
            1 + feed_system_fraction
        )

        reactor_mass = (
            reactor_thermal_power / 1e3
        ) * reactor_specific_mass

        radiator_mass = (
            radiator_area
            * radiator_areal_density
        )

        spacecraft_bus_mass = (
            ion_system_mass
            + reactor_mass
            + radiator_mass
        )

        payload_remaining = (
            spacecraft_dry_mass
            - spacecraft_bus_mass
        )

        # --------------------------------------------------------
        # Recompute kickstage
        # --------------------------------------------------------

        spacecraft_wet_mass_before_oberth = (
            spacecraft_dry_mass
            + spacecraft_propellant
        )

        new_kickstage_propellant = (
            get_prop_mass_with_end_mass(
                oberth_delta_v,
                spacecraft_wet_mass_before_oberth,
                hypergolic_isp
            )
        )

        # --------------------------------------------------------
        # Heatshield
        # --------------------------------------------------------

        l = 7
        d = 4

        heatshield_area = l * d

        heatshield_areal_density = 17.57

        heatshield_mass = (
            heatshield_area
            * heatshield_areal_density
        )

        new_kickstage_dry_mass = (
            hypergolic_structural_fraction
            * new_kickstage_propellant
            + heatshield_mass
        )

        # --------------------------------------------------------
        # Convergence check
        # --------------------------------------------------------

        total_mass = (
            spacecraft_wet_mass_before_oberth
            + new_kickstage_propellant
            + new_kickstage_dry_mass
        )

        if abs(total_mass - previous_total_mass) < 1e-3:
            kickstage_propellant = new_kickstage_propellant
            kickstage_dry_mass = new_kickstage_dry_mass
            break

        previous_total_mass = total_mass

        kickstage_propellant = new_kickstage_propellant
        kickstage_dry_mass = new_kickstage_dry_mass

    # ============================================================
    # FINAL MASSES
    # ============================================================

    launch_mass = (
        spacecraft_wet_mass_before_oberth
        + kickstage_propellant
        + kickstage_dry_mass
    )

    if print_results:

        print("\n" + "=" * 80)
        print("SPACECRAFT")
        print("=" * 80)

        print("Dry mass:",
              spacecraft_dry_mass)

        print("Plane change propellant:",
              plane_change_propellant)

        print("Rendezvous propellant:",
              rendezvous_propellant)

        print("Total spacecraft propellant:",
              spacecraft_propellant)

        print()

        print("Ion system mass:",
              ion_system_mass)

        print("Reactor mass:",
              reactor_mass)

        print("Radiator mass:",
              radiator_mass)

        print()

        print("Payload remaining:",
              payload_remaining)

        print("\n" + "=" * 80)
        print("KICKSTAGE")
        print("=" * 80)

        print("Kickstage propellant:",
              kickstage_propellant)

        print("Kickstage dry mass:",
              kickstage_dry_mass)

        print("Heatshield mass:",
              heatshield_mass)

        print()

        print("Launch mass:",
              launch_mass)

    return (
        launch_mass,
        payload_remaining,
        kickstage_propellant,
        spacecraft_propellant,
        kickstage_dry_mass,
        radiator_mass,
        reactor_mass,
        heatshield_mass
    )


if __name__ == "__main__":

    dry_masses = np.linspace(1, 10000, 10000)

    launch_masses_thermal = []
    remaining_masses_thermal = []

    launch_masses_hypergolic = []
    remaining_masses_hypergolic = []

    kick_stage_propellants_thermal = []
    kick_stage_propellants_hypergolic = []

    spacecraft_propellants_thermal = []
    spacecraft_propellants_hypergolic = []

    kick_stage_dry_masses_thermal = []
    kick_stage_dry_masses_hypergolic = []

    radiator_masses_thermal = []
    radiator_masses_hypergolic = []

    reactor_masses_thermal = []
    reactor_masses_hypergolic = []

    # ============================================================
    # RUN SWEEP
    # ============================================================

    for dry_mass in tqdm(dry_masses, desc="Running mass sweep"):


        (
            launch_mass_hypergolic,
            remaining_mass_margin_hypergolic,
            kick_prop_hypergolic,
            spacecraft_prop_hypergolic,
            kick_dry_hypergolic,
            radiator_hypergolic,
            reactor_hypergolic,
            heatshield_calculated_mass
        ) = run_configuration(
            dry_mass,
            print_results=False
        )

        m_wet_actual_spacecraft = launch_mass_hypergolic - kick_dry_hypergolic - kick_prop_hypergolic + 0.02 * spacecraft_prop_hypergolic
        m_dry_actual_spacecraft = m_wet_actual_spacecraft - 1.02 * spacecraft_prop_hypergolic
        m_payload = 126.30  # kg (full set for rendezvous with 10% margin)
        m_structure_without_tanks_and_ADCS_and_TTC = remaining_mass_margin_hypergolic - m_payload
        launch_mass_hypergolic = launch_mass_hypergolic + 0.02 * kick_prop_hypergolic + 0.02 * spacecraft_prop_hypergolic

        if dry_mass == 100 or dry_mass == 1000 or dry_mass == 2700 or dry_mass == 3000 or dry_mass == 3300 or dry_mass == 5000 or dry_mass == 10000 or int(
                math.ceil(m_wet_actual_spacecraft / 100.0)) * 100 == 3000 or int(
                math.ceil(m_wet_actual_spacecraft / 100.0)) * 100 == 3300:
            # print("\n\n" + "=" * 80)
            # print("DETAILED SUMMARY FOR DRY MASS =", dry_mass, "kg, THERMAL OPTION")
            # print("=" * 80)

            # run_configuration(
            #     dry_mass,
            #     thermal=True,
            #     print_results=True
            # )
            print("\n\n" + "=" * 80)
            print("DETAILED SUMMARY FOR DRY MASS (spacecraft + kick stage) =", dry_mass, "kg, NON-THERMAL OPTION")
            print("spacecraft total (wet) mass without kick stage = ", m_wet_actual_spacecraft)
            print("spacecraft dry mass = ", m_dry_actual_spacecraft)
            print("spacecraft propellant mass =", spacecraft_prop_hypergolic, "kg, s/c total mass fraction = ",
                  spacecraft_prop_hypergolic / m_wet_actual_spacecraft)
            print("radiator mass =", radiator_hypergolic, "kg, s/c total mass fraction = ",
                  radiator_hypergolic / m_wet_actual_spacecraft)
            print("reactor mass =", reactor_hypergolic, "kg, s/c total mass fraction = ",
                  reactor_hypergolic / m_wet_actual_spacecraft)
            print("heatshield mass =", heatshield_calculated_mass, "kg, s/c total mass fraction = ",
                  heatshield_calculated_mass / m_wet_actual_spacecraft)
            print("remaining mass margin =", remaining_mass_margin_hypergolic, "kg, launch mass fraction = ",
                  remaining_mass_margin_hypergolic / m_wet_actual_spacecraft)
            print("structural mass without tanks and ADCS and TTC =", m_structure_without_tanks_and_ADCS_and_TTC,
                  "kg, s/c total mass fraction = ",
                  m_structure_without_tanks_and_ADCS_and_TTC / m_wet_actual_spacecraft)
            # print("ADCS and TTC mass = ", remaining_mass_margin_hypergolic - m_structure_without_tanks_and_ADCS_and_TTC, "kg, s/c total mass fraction = ", (remaining_mass_margin_hypergolic - m_structure_without_tanks_and_ADCS_and_TTC) /m_wet_actual_spacecraft)

            print('\n')
            print("KICK STAGE", '\n')
            print("launch mass hypergolic =", launch_mass_hypergolic, "kg")
            print("kickstage propellant mass =", kick_prop_hypergolic, "kg, launch mass fraction = ",
                  kick_prop_hypergolic / launch_mass_hypergolic)
            print("kickstage propellant mass margin =", 0.02 * kick_prop_hypergolic)
            print("kickstage dry mass =", kick_dry_hypergolic, "kg, launch mass fraction = ",
                  kick_dry_hypergolic / launch_mass_hypergolic)
            x_h = remaining_mass_margin_hypergolic
            print("=" * 80)

            run_configuration(
                dry_mass,
                print_results=False
            )


        # --------------------------------------------------------
        # Hypergolic
        # --------------------------------------------------------

        launch_masses_hypergolic.append(launch_mass_hypergolic)
        remaining_masses_hypergolic.append(remaining_mass_margin_hypergolic)

        kick_stage_propellants_hypergolic.append(kick_prop_hypergolic)
        spacecraft_propellants_hypergolic.append(spacecraft_prop_hypergolic)

        kick_stage_dry_masses_hypergolic.append(kick_dry_hypergolic)
        radiator_masses_hypergolic.append(radiator_hypergolic)
        reactor_masses_hypergolic.append(reactor_hypergolic)

    # ============================================================
    # CONVERT TO ARRAYS
    # ============================================================


    launch_masses_hypergolic = np.array(launch_masses_hypergolic)
    remaining_masses_hypergolic = np.array(remaining_masses_hypergolic)

    kick_stage_propellants_hypergolic = np.array(kick_stage_propellants_hypergolic)

    spacecraft_propellants_hypergolic = np.array(spacecraft_propellants_hypergolic)

    kick_stage_dry_masses_hypergolic = np.array(kick_stage_dry_masses_hypergolic)

    radiator_masses_hypergolic = np.array(radiator_masses_hypergolic)

    reactor_masses_hypergolic = np.array(reactor_masses_hypergolic)

    # ============================================================
    # FILTER VALID SOLUTIONS
    # ============================================================

    max_launch_mass = 20_000  # kg (20 tons)


    hypergolic_mask = (
            (remaining_masses_hypergolic >= 0)
            & (launch_masses_hypergolic <= max_launch_mass)
    )
    # ============================================================
    # SINGLE COMBINED PLOT
    # ============================================================

    fig, ax = plt.subplots(figsize=(14, 8))



    # ------------------------------------------------------------
    # HYPERGOLIC
    # ------------------------------------------------------------

    x_h = remaining_masses_hypergolic[hypergolic_mask]

    ax.plot(
        x_h,
        launch_masses_hypergolic[hypergolic_mask],
        linestyle="-",
        linewidth=3,
        label="Hypergolic Total Launch Mass"
    )

    ax.plot(
        x_h,
        kick_stage_propellants_hypergolic[hypergolic_mask],
        linestyle="-",
        label="Hypergolic Kickstage Propellant"
    )

    ax.plot(
        x_h,
        spacecraft_propellants_hypergolic[hypergolic_mask],
        linestyle="-",
        label="Hypergolic Spacecraft Propellant"
    )

    ax.plot(
        x_h,
        kick_stage_dry_masses_hypergolic[hypergolic_mask],
        linestyle="-",
        label="Hypergolic Kickstage Dry Mass"
    )

    ax.plot(
        x_h,
        radiator_masses_hypergolic[hypergolic_mask],
        linestyle="-",
        label="Hypergolic Radiator Mass"
    )

    ax.plot(
        x_h,
        reactor_masses_hypergolic[hypergolic_mask],
        linestyle="-",
        label="Hypergolic Reactor Mass"
    )

    # ============================================================
    # FORMATTING
    # ============================================================

    ax.set_xlabel("Remaining Allowable Dry Mass [kg]")
    ax.set_ylabel("Mass [kg]")

    ax.set_title(
        "Launch Mass and Component Breakdown\n"
        "vs Remaining Allowable Dry Mass"
    )

    ax.axhline(0, color="black", linewidth=1)
    ax.axvline(0, color="black", linewidth=1)

    ax.grid(True)

    ax.legend(
        fontsize=9,
        ncol=2
    )

    plt.tight_layout()
    plt.show()