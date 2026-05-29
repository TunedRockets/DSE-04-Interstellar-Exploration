import copy
import math

from src2.utilities import YEAR, DAY
from src2.orbit import *

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from Power.powerinsizeout import solar

# ============================================================
#                         CONSTANTS
# ============================================================

g0 = 9.81

# ============================================================
#                     MISSION INPUTS
# ============================================================

rendezvous_delta_v = 50  # m/s

# ============================================================
#                ELECTRIC PROPULSION SYSTEM
# ============================================================

ion_isp = 4150
ion_thruster_thrust = 0.237  # N
ion_efficiency = 0.70

# ============================================================
#                 SOLAR ELECTRIC PROPULSION
# ============================================================

solar_flux_1AU = 1361              # W/m²
solar_array_efficiency = 0.30
solar_array_specific_power = 120   # W/kg
solar_degradation = 0.77

mission_distance_AU = 1.0

extra_needed_power = 2370.4  # W

# ============================================================
#                     BURN WINDOW
# ============================================================

rendezvous_burn_time = 10 * DAY

# ============================================================
#                     BASIC UTILITIES
# ============================================================

def pct(x, total):
    return 100.0 * x / total if total > 0 else 0.0


def get_prop_mass_with_end_mass(delta_v, end_mass, Isp):
    return np.exp(delta_v / (g0 * Isp)) * end_mass - end_mass


def get_prop_mass_with_start_mass(delta_v, start_mass, Isp):
    return start_mass * (1 - np.exp(-delta_v / (g0 * Isp)))


def get_required_thrust(delta_v, Isp, start_mass, burn_time):

    prop_mass = get_prop_mass_with_start_mass(
        delta_v,
        start_mass,
        Isp
    )

    mdot = prop_mass / burn_time

    return mdot * Isp * g0


def get_required_electric_power(thrust, Isp, efficiency):

    ve = Isp * g0
    return thrust * ve / (2 * efficiency)


# ============================================================
#                    MAIN CONFIGURATION
# ============================================================

def run_configuration(spacecraft_dry_mass, print_results=False):

    ion_thruster_mass = 14
    ppu_mass_per_kw = 7
    feed_system_fraction = 0.1

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

    required_electric_power = (
        get_required_electric_power(
            rendezvous_thrust,
            ion_isp,
            ion_efficiency
        )
        + extra_needed_power
    )

    required_thruster_count = math.ceil(
        rendezvous_thrust / ion_thruster_thrust
    )


    solar_array_area, solar_array_mass = solar(required_electric_power)

    power_management_mass = 0.15 * solar_array_mass

    ion_system_mass = (
        required_thruster_count * ion_thruster_mass
        + ppu_mass_per_kw * (required_electric_power / 1e3)
    )

    ion_system_mass *= (1 + feed_system_fraction)

    spacecraft_bus_mass = (
        ion_system_mass
        + solar_array_mass
        + power_management_mass
    )

    payload_remaining = spacecraft_dry_mass - spacecraft_bus_mass

    return (
        spacecraft_wet_mass,
        payload_remaining,
        spacecraft_propellant,
        solar_array_mass,
        solar_array_area,
        ion_system_mass,
        required_electric_power
    )

# ============================================================
#                         MAIN
# ============================================================

if __name__ == "__main__":

    dry_masses = np.linspace(50, 3000, 1000)

    launch_masses = []
    remaining_masses = []
    propellant_masses = []
    solar_array_masses = []
    solar_array_areas = []
    ion_system_masses = []
    electric_powers = []

    last_printed_bin = -1

    for dry_mass in tqdm(dry_masses):

        (
            launch_mass,
            remaining_mass,
            spacecraft_propellant,
            solar_array_mass,
            solar_array_area,
            ion_system_mass,
            required_electric_power
        ) = run_configuration(dry_mass)

        launch_mass_bin = int(launch_mass // 50)

        if launch_mass_bin != last_printed_bin:

            last_printed_bin = launch_mass_bin

            m_payload = 96.3
            structures_mass = 0.20 * dry_mass

            spacecraft_wet_mass = dry_mass + spacecraft_propellant

            remaining_internal_mass = (
                dry_mass
                - (
                    m_payload
                    + ion_system_mass
                    + solar_array_mass
                    + structures_mass
                )
            )

            landing_mass = 259.0
            adcs_mass = 0.5 * (remaining_internal_mass - landing_mass)
            ttc_mass = 0.5 * (remaining_internal_mass - landing_mass)

            prop_margin = 0.02 * spacecraft_propellant

            print("\n" + "=" * 110)
            print("BUS SUBSYSTEM BUDGET ALLOCATION")
            print("=" * 110)

            def row(name, mass, source=""):
                print(
                    f"{name:<45} "
                    f"{pct(mass, spacecraft_wet_mass):>10.2f}%   "
                    f"{mass:>12.1f} kg   "
                    f"{source}"
                )

            print("\n--- PRIMARY SYSTEMS ---")

            row("Scientific Payload Flyby", m_payload, "Fixed payload")
            row("Propulsion (incl. tanks)", spacecraft_propellant, "Rocket equation")
            row("Solar Arrays", solar_array_mass, "SEP power generation")
            row("Ion Propulsion System", ion_system_mass, "Thrusters + PPU")

            print("\n--- SECONDARY SYSTEMS ---")

            row("Structures (Without Tanks)", structures_mass, "20% dry mass")
            row("Impactor System", landing_mass, "Mission allocation")
            row("ADCS", adcs_mass, "Split remainder")
            row("TT&C", ttc_mass, "Split remainder")

            print("\n--- SPACECRAFT SUMMARY ---")

            row("Spacecraft Dry Mass", dry_mass, "")
            row("Spacecraft Propellant Mass", spacecraft_propellant, "")
            row("Spacecraft Propellant Margin", prop_margin, "2% margin")

            print("\n--- TOTALS ---")

            row("Spacecraft Wet Mass", spacecraft_wet_mass, "")

            print("\n--- SEP PERFORMANCE ---")

            print(f"Required Electric Power : {required_electric_power/1000:.2f} kW")
            print(f"Solar Array Area        : {solar_array_area:.2f} m²")
            print(f"Thruster Count          : {math.ceil(required_electric_power / 5000)}")

            print("=" * 110)

        launch_masses.append(launch_mass)
        remaining_masses.append(remaining_mass)
        propellant_masses.append(spacecraft_propellant)
        solar_array_masses.append(solar_array_mass)
        solar_array_areas.append(solar_array_area)
        ion_system_masses.append(ion_system_mass)
        electric_powers.append(required_electric_power / 1000)

    # ========================================================
    # PLOTS
    # ========================================================

    fig, ax = plt.subplots(figsize=(14, 8))

    ax.plot(remaining_masses, launch_masses, label="Wet Mass")
    ax.plot(remaining_masses, propellant_masses, label="Propellant")
    ax.plot(remaining_masses, solar_array_masses, label="Solar Arrays")
    ax.plot(remaining_masses, ion_system_masses, label="Ion System")

    ax.set_xlabel("Remaining Dry Mass [kg]")
    ax.set_ylabel("Mass [kg]")
    ax.set_title("SEP Rendezvous Sizing")
    ax.grid(True)
    ax.legend()

    plt.tight_layout()
    plt.show()