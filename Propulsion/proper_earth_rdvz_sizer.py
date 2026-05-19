import copy
import math

from scipy.optimize import minimize_scalar

from src2.utilities import YEAR
from src2.orbit import *

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from Power.powerinsizeout import reactor

# ============================================================
#                         CONSTANTS
# ============================================================

g0 = 9.81
sigma = 5.670374419e-8

# ============================================================
#                     MISSION INPUTS
# ============================================================

rendezvous_delta_v = 3000  # m/s

# ============================================================
#                ELECTRIC PROPULSION SYSTEM
# ============================================================

ion_isp = 4150
ion_thruster_thrust = 0.237
ion_efficiency = 0.70

# ============================================================
#                 REACTOR / BRAYTON
# ============================================================

brayton_hot_side_temperature = 1673
radiator_emissivity = 0.9
brayton_eta_fraction = 0.6

# ============================================================
#                     BURN WINDOW
# ============================================================

rendezvous_burn_time = 2 * YEAR

# ============================================================
#                     BASIC UTILITIES
# ============================================================

def get_prop_mass_with_end_mass(delta_v, end_mass, Isp):
    return np.exp(delta_v / (g0 * Isp)) * end_mass - end_mass


def get_prop_mass_with_start_mass(delta_v, start_mass, Isp):
    return start_mass * (1 - np.exp(-delta_v / (g0 * Isp)))


def get_required_thrust(delta_v, Isp, start_mass, burn_time):
    prop_mass = get_prop_mass_with_start_mass(delta_v, start_mass, Isp)
    mdot = prop_mass / burn_time
    return mdot * Isp * g0


def get_required_electric_power(thrust, Isp, efficiency):
    ve = Isp * g0
    return thrust * ve / (2 * efficiency)

# ============================================================
#                   BRAYTON OPTIMIZER
# ============================================================

def optimize_brayton_radiator(
    electric_power,
    T_hot,
    emissivity=0.9,
    T_space=3,
    eta_fraction=0.6
):

    def radiator_area(T_cold):

        if T_cold <= T_space or T_cold >= T_hot:
            return np.inf

        eta = eta_fraction * (1 - T_cold / T_hot)
        if eta <= 0:
            return np.inf

        thermal_power = electric_power / eta
        waste_heat = thermal_power - electric_power

        return waste_heat / (
            emissivity * sigma * (T_cold**4 - T_space**4)
        )

    result = minimize_scalar(
        radiator_area,
        bounds=(300, T_hot - 1),
        method='bounded'
    )

    T_cold = result.x

    eta = eta_fraction * (1 - T_cold / T_hot)
    thermal_power = electric_power / eta
    waste_heat = thermal_power - electric_power
    area = radiator_area(T_cold)

    return T_cold, eta, thermal_power, waste_heat, area

# ============================================================
#                    MAIN CONFIGURATION
# ============================================================

def run_configuration(spacecraft_dry_mass, print_results=False):

    ion_thruster_mass = 14
    ppu_mass_per_kw = 7
    feed_system_fraction = 0.1
    radiator_areal_density = 5  # kg/m²

    spacecraft_propellant = get_prop_mass_with_end_mass(
        rendezvous_delta_v,
        spacecraft_dry_mass,
        ion_isp
    )

    spacecraft_wet_mass = spacecraft_dry_mass + spacecraft_propellant

    rendezvous_thrust = get_required_thrust(
        rendezvous_delta_v,
        ion_isp,
        spacecraft_wet_mass,
        rendezvous_burn_time
    )

    reactor_electric_power = get_required_electric_power(
        rendezvous_thrust,
        ion_isp,
        ion_efficiency
    )

    required_thruster_count = math.ceil(
        rendezvous_thrust / ion_thruster_thrust
    )

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

    ion_system_mass = (
        required_thruster_count * ion_thruster_mass
        + ppu_mass_per_kw * (reactor_electric_power / 1e3)
    )
    ion_system_mass *= (1 + feed_system_fraction)

    anhong_reactor_mass, anhong_reactor_mass_fuel = reactor(
        reactor_electric_power
    )

    reactor_mass = anhong_reactor_mass + anhong_reactor_mass_fuel

    radiator_mass = radiator_area * radiator_areal_density

    spacecraft_bus_mass = ion_system_mass + reactor_mass + radiator_mass

    payload_remaining = spacecraft_dry_mass - spacecraft_bus_mass

    launch_mass = spacecraft_wet_mass

    return (
        launch_mass,
        payload_remaining,
        spacecraft_propellant,
        radiator_mass,
        reactor_mass,
        ion_system_mass
    )

# ============================================================
#                         MAIN
# ============================================================

if __name__ == "__main__":

    dry_masses = np.linspace(50, 3000, 1000)

    launch_masses = []
    remaining_masses = []
    propellant_masses = []
    radiator_masses = []
    reactor_masses = []

    last_printed_bin = -1

    for dry_mass in tqdm(dry_masses):

        (
            launch_mass,
            remaining_mass,
            spacecraft_propellant,
            radiator_mass,
            reactor_mass,
            ion_system_mass
        ) = run_configuration(dry_mass)

        launch_mass_bin = int(launch_mass // 100)

        # ========================================================
        # PRINT BREAKDOWN PER BIN
        # ========================================================
        if launch_mass_bin != last_printed_bin:

            last_printed_bin = launch_mass_bin

            m_payload = 126.30  # kg

            m_wet_actual_spacecraft = dry_mass + spacecraft_propellant
            m_dry_actual_spacecraft = dry_mass

            structures_mass = 0.20 * m_dry_actual_spacecraft

            remaining_internal_mass = (
                m_dry_actual_spacecraft
                - (
                    m_payload
                    + ion_system_mass
                    + reactor_mass
                    + radiator_mass
                    + structures_mass
                )
            )

            landing_mass = 0.5 * remaining_internal_mass
            adcs_mass = 0.25 * remaining_internal_mass
            ttc_mass = 0.25 * remaining_internal_mass

            print("\n" + "=" * 110)
            print("BUS SUBSYSTEM BUDGET ALLOCATION (RENDEZVOUS ONLY)")
            print("=" * 110)


            def row(name, frac, mass, source=""):
                print(f"{name:<45} {frac:>10}   {mass:>12.1f} kg   {source}")


            print("\n--- PRIMARY SYSTEMS ---")
            row("Scientific Payload Rendezvous (with margin)", "5.8%", 126.3, "Selected previously")
            row("Propulsion (incl. tanks)", "51.9%", spacecraft_propellant, "Model output")
            row("Power System (Fission Reactor)", "6.3%", reactor_mass, "")
            row("Reactor Radiator", "0.2%", radiator_mass, "")

            print("\n--- SECONDARY SYSTEMS ---")

            structures_mass = 0.20 * dry_mass

            remaining_internal_mass = (
                    dry_mass
                    - (
                            126.3
                            + ion_system_mass
                            + reactor_mass
                            + radiator_mass
                            + structures_mass
                    )
            )

            landing_mass = 2 / 4 * remaining_internal_mass
            adcs_mass = 1 / 4 * remaining_internal_mass
            ttc_mass = 1 / 4 * remaining_internal_mass

            row("Structures (Without Tanks)", "9.4%", structures_mass, "20% of dry mass")
            row("Landing System", "8.6%", landing_mass, "2/4 of remaining")
            row("ADCS", "4.3%", adcs_mass, "1/4 of remaining")
            row("TT&C", "4.3%", ttc_mass, "1/4 of remaining")

            print("\n--- SPACECRAFT MASS SUMMARY ---")

            spacecraft_dry_mass = dry_mass
            spacecraft_wet_mass = dry_mass + spacecraft_propellant

            row("Spacecraft Dry Mass", "47.1%", spacecraft_dry_mass, "")
            row("Spacecraft Propellant Mass", "52.9%", spacecraft_propellant, "")

            prop_margin = 0.02 * spacecraft_propellant
            row("Spacecraft Propellant Mass Margin", "1.1%", prop_margin, "2% margin")

            print("\n--- TOTALS ---")

            row("Spacecraft Wet Mass", "100%", spacecraft_wet_mass, "")

            print("\n--- TOTAL INJECTION ---")

            total_injected = spacecraft_wet_mass
            row("Total Injected Spacecraft Mass", "-", total_injected, "")

            print("=" * 110)

        launch_masses.append(launch_mass)
        remaining_masses.append(remaining_mass)
        propellant_masses.append(spacecraft_propellant)
        radiator_masses.append(radiator_mass)
        reactor_masses.append(reactor_mass)

    # ========================================================
    # PLOT
    # ========================================================

    fig, ax = plt.subplots(figsize=(14, 8))

    ax.plot(remaining_masses, launch_masses, label="Total Wet Mass")
    ax.plot(remaining_masses, propellant_masses, label="Propellant")
    ax.plot(remaining_masses, reactor_masses, label="Reactor")
    ax.plot(remaining_masses, radiator_masses, label="Radiator")

    ax.set_xlabel("Remaining Allowable Dry Mass [kg]")
    ax.set_ylabel("Mass [kg]")
    ax.set_title("Rendezvous-Only Spacecraft Sizing")
    ax.grid(True)
    ax.legend()

    plt.tight_layout()
    plt.show()