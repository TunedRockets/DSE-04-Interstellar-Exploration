import math
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from Power.powerinsizeout import solar
from Propulsion.proper_flyby_sizer_electric import extra_needed_power

# ============================================================
#                         CONSTANTS
# ============================================================

g0 = 9.81

# ============================================================
#                     MISSION INPUTS
# ============================================================

rendezvous_delta_v = 50  # m/s

# ============================================================
#                CHEMICAL PROPULSION SYSTEM
# ============================================================

chem_isp = 330  # s (storable chemical)

tank_structural_fraction = 0.12
engine_dry_mass_fraction = 0.03


# ============================================================
#                     UTILITIES
# ============================================================

def pct(x, total):
    return 100.0 * x / total if total > 0 else 0.0


def get_prop_mass(delta_v, dry_mass, Isp):

    return dry_mass * (
        np.exp(delta_v / (g0 * Isp)) - 1
    )




# ============================================================
#                    MAIN CONFIGURATION
# ============================================================

# ------------------------------------------------------------
# FIXED SUBSYSTEMS
# ------------------------------------------------------------

payload_mass = 96.3
impactor_mass = 259.0

def run_configuration(spacecraft_dry_mass):

    # --------------------------------------------------------
    # PROPULSION SIZING
    # --------------------------------------------------------

    spacecraft_propellant = get_prop_mass(
        rendezvous_delta_v,
        spacecraft_dry_mass,
        chem_isp
    )

    spacecraft_wet_mass = (
        spacecraft_dry_mass
        + spacecraft_propellant
    )

    engine_mass = (
        engine_dry_mass_fraction
        * spacecraft_wet_mass
    )

    tank_mass = (
        tank_structural_fraction
        * spacecraft_propellant
    )

    propulsion_system_mass = (
        engine_mass
        + tank_mass
    )

    # --------------------------------------------------------
    # POWER SYSTEM
    # --------------------------------------------------------

    area, power_system_mass = solar(extra_needed_power)

    # --------------------------------------------------------
    # STRUCTURES
    # --------------------------------------------------------

    structures_mass = (
        0.20
        * spacecraft_dry_mass
    )

    # --------------------------------------------------------
    # MASS CLOSURE
    # --------------------------------------------------------

    remaining_internal_mass = (
        spacecraft_dry_mass
        - (
            payload_mass
            + impactor_mass
            + propulsion_system_mass
            + power_system_mass
            + structures_mass
        )
    )

    adcs_mass = 0.5 * remaining_internal_mass
    ttc_mass = 0.5 * remaining_internal_mass

    # --------------------------------------------------------
    # MARGIN
    # --------------------------------------------------------

    prop_margin = (
        0.02
        * spacecraft_propellant
    )

    return (
        spacecraft_wet_mass,
        spacecraft_propellant,
        propulsion_system_mass,
        power_system_mass,
        structures_mass,
        impactor_mass,
        adcs_mass,
        ttc_mass,
        prop_margin
    )

# ============================================================
#                         MAIN
# ============================================================

if __name__ == "__main__":

    dry_masses = np.linspace(
        50,
        3000,
        1000
    )

    launch_masses = []
    propellant_masses = []
    propulsion_masses = []
    power_masses = []
    structures_masses = []
    impactor_masses = []
    adcs_masses = []
    ttc_masses = []

    last_printed_bin = -1

    for dry_mass in tqdm(dry_masses):

        (
            wet_mass,
            propellant_mass,
            propulsion_mass,
            power_system_mass,
            structures_mass,
            impactor_mass,
            adcs_mass,
            ttc_mass,
            prop_margin
        ) = run_configuration(dry_mass)

        launch_mass_bin = int(
            wet_mass // 50
        )

        if launch_mass_bin != last_printed_bin:

            last_printed_bin = launch_mass_bin

            spacecraft_wet_mass = wet_mass

            print("\n" + "=" * 110)
            print("BUS SUBSYSTEM BUDGET ALLOCATION (CHEMICAL)")
            print("=" * 110)

            def row(name, mass):

                print(
                    f"{name:<45} "
                    f"{pct(mass, spacecraft_wet_mass):>10.2f}%   "
                    f"{mass:>12.1f} kg"
                )

            # ------------------------------------------------
            # PRIMARY SYSTEMS
            # ------------------------------------------------

            print("\n--- PRIMARY SYSTEMS ---")

            row(
                "Scientific Payload Flyby",
                payload_mass
            )

            row(
                "Propulsion (incl. tanks & engines)",
                propulsion_mass
            )

            row(
                "Power System",
                power_system_mass
            )

            # ------------------------------------------------
            # SECONDARY SYSTEMS
            # ------------------------------------------------

            print("\n--- SECONDARY SYSTEMS ---")

            row(
                "Impactor System",
                impactor_mass
            )

            row(
                "Structures",
                structures_mass
            )

            row(
                "ADCS",
                adcs_mass
            )

            row(
                "TT&C",
                ttc_mass
            )

            # ------------------------------------------------
            # SPACECRAFT SUMMARY
            # ------------------------------------------------

            print("\n--- SPACECRAFT SUMMARY ---")

            row(
                "Spacecraft Dry Mass",
                dry_mass
            )

            row(
                "Spacecraft Propellant Mass",
                propellant_mass
            )

            row(
                "Propellant Margin",
                prop_margin
            )

            # ------------------------------------------------
            # TOTALS
            # ------------------------------------------------

            print("\n--- TOTALS ---")

            row(
                "Spacecraft Wet Mass",
                spacecraft_wet_mass
            )

            print("\n--- POWER SYSTEM ---")

            print(
                f"Required Electrical Power : "
                f"{extra_needed_power/1000:.2f} kW"
            )


            print("=" * 110)

        launch_masses.append(
            wet_mass
        )

        propellant_masses.append(
            propellant_mass
        )

        propulsion_masses.append(
            propulsion_mass
        )

        power_masses.append(
            power_system_mass
        )

        structures_masses.append(
            structures_mass
        )

        impactor_masses.append(
            impactor_mass
        )

        adcs_masses.append(
            adcs_mass
        )

        ttc_masses.append(
            ttc_mass
        )

    # ========================================================
    # PLOT
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(14, 8)
    )

    ax.plot(
        propellant_masses,
        launch_masses,
        label="Wet Mass"
    )

    ax.plot(
        propellant_masses,
        propulsion_masses,
        label="Propulsion"
    )

    ax.plot(
        propellant_masses,
        power_masses,
        label="Power System"
    )

    ax.plot(
        propellant_masses,
        structures_masses,
        label="Structures"
    )

    ax.plot(
        propellant_masses,
        impactor_masses,
        label="Impactor"
    )

    ax.set_xlabel(
        "Propellant Mass [kg]"
    )

    ax.set_ylabel(
        "Mass [kg]"
    )

    ax.set_title(
        "Chemical Rendezvous Spacecraft Sizing"
    )

    ax.grid(True)

    ax.legend()

    plt.tight_layout()

    plt.show()