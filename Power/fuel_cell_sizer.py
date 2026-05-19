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

ion_isp = 4150                      # s - based on NEXT thruster performance, average
ion_thruster_thrust = 0.237         # N - based on NEXT thruster performance, higher value (lower 0.236)
ion_efficiency = 0.70               # - based on NEXT thruster performance

# ------------------------------------------------------------
# Solar flyby
# ------------------------------------------------------------

perihelion_distance = 10 * R_sun
solar_absorptivity = 0.95

# Powergen
select = 2
selection = ["fuel_cell", "reactor", "rtg"]
selection = selection[select]


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

plane_change_burn_time = origin.max_impulsive_burn_time(np.pi,5)
oberth_burn_time = origin.max_impulsive_burn_time(0,10)
rendezvous_burn_time = 2*YEAR


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



def run_configuration(dry_mass_assumption, print_results=False):

    # ============================================================
    #                   ELECTRIC PROP STAGE
    # ============================================================

    # Final spacecraft mass after all maneuvers

    final_spacecraft_mass = dry_mass_assumption


    # ------------------------------------------------------------
    # Rendezvous propellant
    # ------------------------------------------------------------

    rendezvous_propellant = (
        get_prop_mass_with_end_mass(
            rendezvous_delta_v,
            final_spacecraft_mass,
            ion_isp
        )
    )

    spacecraft_mass_before_rendezvous = (
        final_spacecraft_mass
        + rendezvous_propellant
    )


    # ------------------------------------------------------------
    # Plane change propellant
    # ------------------------------------------------------------

    plane_change_propellant = (
        get_prop_mass_with_end_mass(
            plane_change_delta_v,
            spacecraft_mass_before_rendezvous,
            ion_isp
        )
    )


    # ------------------------------------------------------------
    # Spacecraft total wet mass
    # ------------------------------------------------------------

    spacecraft_total_mass = (
        final_spacecraft_mass
        + rendezvous_propellant
        + plane_change_propellant
    )


    # ============================================================
    #            HYPERGOLIC KICKSTAGE OPTION
    # ============================================================

    hypergolic_isp = 330  # s (N2O4/MMH optimistic vacuum engine)

    hypergolic_propellant_density = 1300  # kg/m^3 (mixture effective)

    hypergolic_chamber_pressure = 30e5  # 30 bar typical storable engine

    # Hypergolic propellant mass for Oberth burn
    hypergolic_propellant = get_prop_mass_with_end_mass(
        oberth_delta_v,
        spacecraft_total_mass,
        hypergolic_isp
    )

    hypergolic_kickstage_initial_mass = spacecraft_total_mass + hypergolic_propellant
    hypergolic_kickstage_final_mass = spacecraft_total_mass

    # Mass flow
    hypergolic_mdot = hypergolic_propellant / oberth_burn_time

    # Thrust
    hypergolic_thrust = hypergolic_mdot * hypergolic_isp * g0

    # No external thermal power required (chemical engine)
    hypergolic_thermal_power = 0

    # Tank volume estimate
    hypergolic_tank_volume = hypergolic_propellant / hypergolic_propellant_density

    # Delta-v verification (sanity check)
    hypergolic_achieved_dv = hypergolic_isp * g0 * math.log(
        hypergolic_kickstage_initial_mass / hypergolic_kickstage_final_mass
    )

    oberth_propellant_hypergolic = hypergolic_propellant
    kickstage_initial_mass_hypergolic = hypergolic_kickstage_initial_mass
    kickstage_final_mass_hypergolic = hypergolic_kickstage_final_mass


    # ============================================================
    #               ELECTRIC PROPULSION SIZING
    # ============================================================

    # ------------------------------------------------------------
    # Plane change
    # ------------------------------------------------------------

    plane_change_thrust_hypergolic = (
        get_required_thrust(
            plane_change_delta_v,
            ion_isp,
            kickstage_initial_mass_hypergolic,
            plane_change_burn_time
        )
    )

    plane_change_power_hypergolic = (
        get_required_electric_power(
            plane_change_thrust_hypergolic,
            ion_isp,
            ion_efficiency
        )
    )


    # ------------------------------------------------------------
    # Rendezvous
    # ------------------------------------------------------------

    rendezvous_thrust = (
        get_required_thrust(
            rendezvous_delta_v,
            ion_isp,
            spacecraft_mass_before_rendezvous,
            rendezvous_burn_time
        )
    )

    rendezvous_power = (
        get_required_electric_power(
            rendezvous_thrust,
            ion_isp,
            ion_efficiency
        )
    )


    # ------------------------------------------------------------
    # Max electric propulsion load
    # ------------------------------------------------------------

    required_electric_power_hypergolic = max(
        plane_change_power_hypergolic,
        rendezvous_power
    )

    required_thruster_count_hypergolic = math.ceil(
        max(
            plane_change_thrust_hypergolic,
            rendezvous_thrust
        )
        / ion_thruster_thrust
    )


    # ============================================================
    #                Power System SIZING
    # ============================================================


    # Fuel Cells

    fuel_cell_BOP_power_density = 12000/118 # W/kg
    fuel_cell_reactants_specific_energy = 3661*0.7 # Wh/kg

    # print(required_electric_power_hypergolic / 1e3)
    fuel_cell_mass = required_electric_power_hypergolic / fuel_cell_BOP_power_density
    # print(required_electric_power_hypergolic * plane_change_burn_time / 3600 / 1e6)
    fuel_cell_reactants_mass = required_electric_power_hypergolic * plane_change_burn_time / 3600 / fuel_cell_reactants_specific_energy

    # Reactor

    brayton_cycle_efficiency = 0.13446458166714917

    # Uranium-235 fission energy
    energy_one_fission = 169.1 * 10**(6) *  1.602176634 * 10**(-19) # MeV * J/eV source https://web.archive.org/web/20190505175631/http://www.kayelaby.npl.co.uk/atomic_and_nuclear_physics/4_7/4_7_1.html
    energy_fission_mol = energy_one_fission * 6.02214076* 10**(23) # J per Mol https://www.nist.gov/si-redefinition/meet-constants
    kg_per_mol_u235 = 235/1000 # g per mol / 1000
    u235_specific_energy = energy_fission_mol / kg_per_mol_u235 # J per kg


    # (elec power / brayton ) /  u235_specific_energy = kg per second (of pure u235)
    # minimum haleu mass = kg per sec / 0.2 , this is also haleu mass rate https://world-nuclear.org/information-library/nuclear-fuel-cycle/conversion-enrichment-and-fabrication/high-assay-low-enriched-uranium-haleu
    # fuel mass to sustain mission = burn_time * haleu mass rate

    # elec power / brayton / u235_specific energy / 0.2 * burn time = fuel mass

    reactor_fuel_equivalent_specific_energy = brayton_cycle_efficiency * u235_specific_energy * 0.2 # Wh/kg

    reactor_BOP_power_density = 100000* brayton_cycle_efficiency /100 # W/kg 


    reactor_fuel_mass = (plane_change_power_hypergolic * plane_change_burn_time + rendezvous_power* rendezvous_burn_time) / reactor_fuel_equivalent_specific_energy
    # print("fuelamsss;", reactor_fuel_mass)
    reactor_fuel_mass = reactor_fuel_mass / 0.1 # burn up mass https://beyondnerva.wordpress.com/fission-power-systems/systems-for-nuclear-auxiliary-power-snap/snap-10-10a-and-snapshot/
    # print("fuelamsss;", reactor_fuel_mass)

    reactor_mass = required_electric_power_hypergolic / reactor_BOP_power_density


    # Rtg
    rtg_power_density = 296/56
    # 1/2*(296-296*0.7)*20*365*24*3600 + 296*0.7*20*365*24*3600 = 156418560000, 43 MWh
    rtg_mass = required_electric_power_hypergolic / rtg_power_density


    if print_results:
        # ============================================================
        #               ELECTRIC PROPULSION SYSTEM
        # ============================================================

        print()
        print("=" * 80)
        print("ION PROPULSION SYSTEM")
        print("=" * 80)

        print("Ion engine Isp:",
              ion_isp,
              "s")

        print("Thruster unit thrust:",
              ion_thruster_thrust,
              "N")

        print("Required thruster count (Hypergolic Option):",
              required_thruster_count_hypergolic)

        print()

        print("Plane change burn time:",
              plane_change_burn_time,
              "s (",
              plane_change_burn_time / DAY,
              "days )")

        print("Plane change thrust (Hypergolic Option):",
              plane_change_thrust_hypergolic,
              "N")

        print("Plane change electric power (Hypergolic Option):",
              plane_change_power_hypergolic / 1e3,
              "kW")

        print()

        print("Rendezvous burn time:",
              rendezvous_burn_time,
              "s (",
              rendezvous_burn_time / DAY,
              "days )")

        print("Rendezvous thrust:",
              rendezvous_thrust,
              "N")

        print("Rendezvous electric power:",
              rendezvous_power / 1e3,
              "kW")


        # ============================================================
        #                   Power Generation
        # ============================================================

        print()
        print("=" * 80)
        print("Power Generation")
        print("=" * 80)

        print("Required electric output (Hypergolic Option):",
              required_electric_power_hypergolic / 1e3,
              "kW")


        # ============================================================
        #        FINAL KICKSTAGE COMPARISON REPORT (THERMAL vs HYPERGOLIC)
        # ============================================================

        def print_kickstage_summary(name,
                                    isp,
                                    prop_mass,
                                    mdot,
                                    thrust,
                                    burn_time,
                                    initial_mass,
                                    final_mass,
                                    achieved_dv,
                                    extra_power=0):

            print()
            print("=" * 80)
            print(f"{name} KICKSTAGE SUMMARY")
            print("=" * 80)

            print("Isp:", isp, "s")
            print("Propellant mass:", prop_mass, "kg")
            print("Initial mass:", initial_mass, "kg")
            print("Final mass:", final_mass, "kg")
            print("Mass ratio:", initial_mass / final_mass)

            print()
            print("Mass flow:", mdot, "kg/s")
            print("Thrust:", thrust, "N")

            print()
            print("Burn time:", burn_time, "s (", burn_time / 3600, "hours )")

            print()
            print("Achieved delta-v:", achieved_dv, "m/s")
            print("Target delta-v:", oberth_delta_v, "m/s")
            print("Delta-v error:", achieved_dv - oberth_delta_v, "m/s")


            if extra_power > 0:
                print("Auxiliary power load:", extra_power / 1e3, "kW")


            # ------------------------------------------------------------
            # HYPERGOLIC KICKSTAGE REPORT
            # ------------------------------------------------------------

            print_kickstage_summary(
                name="HYPERGOLIC (MMH/N2O4)",
                isp=hypergolic_isp,
                prop_mass=hypergolic_propellant,
                mdot=hypergolic_mdot,
                thrust=hypergolic_thrust,
                burn_time=oberth_burn_time,
                initial_mass=hypergolic_kickstage_initial_mass,
                final_mass=hypergolic_kickstage_final_mass,
                achieved_dv=hypergolic_achieved_dv,
                extra_power=0
            )


            # ============================================================
            #                      MASS SUMMARY
            # ============================================================

            print()
            print("=" * 80)
            print("MASS SUMMARY")
            print("=" * 80)

            print("Payload dry mass:",
                  remaining_mass_margin_thermal,
                  "kg")

            print()

            print("Plane change propellant:",
                  plane_change_propellant,
                  "kg")

            print("Rendezvous propellant:",
                  rendezvous_propellant,
                  "kg")

            print("Electric propulsion total propellant:",
                  plane_change_propellant
                  + rendezvous_propellant,
                  "kg")

            print()


            print("Oberth propellant (Hypergolic Option):",
                  oberth_propellant_hypergolic,
                  "kg")

            print()

            print("Spacecraft wet mass after kickstage separation:",
                  spacecraft_total_mass,
                  "kg")


            print("Combined initial launch mass (Kickstage Option):",
                  kickstage_initial_mass_hypergolic,
                  "kg")
            # print("Payload dry mass:",
            #       payload_dry_mass,
            #       "kg")

            print()

    # ============================================================
    #               DRY MASS MODEL ADDITION
    # ============================================================

    # ------------------------------------------------------------
    # Ion propulsion dry mass model
    # ------------------------------------------------------------

    ion_thruster_mass = 14          # kg per thruster (NEXT-class estimate)
    ppp_mass_per_kw = 7             # kg/kW (power processing + harness)
    feed_system_fraction = 0.1

    ion_system_mass_hypergolic = (
        required_thruster_count_hypergolic * ion_thruster_mass
        + ppp_mass_per_kw * (required_electric_power_hypergolic / 1e3)
    )

    ion_system_mass_hypergolic *= (1 + feed_system_fraction)


    # ------------------------------------------------------------
    # power gen mass model
    # ------------------------------------------------------------

    if selection == "fuel_cell":
        power_gen_mass_hypergolic = fuel_cell_mass + fuel_cell_reactants_mass
    elif selection == "reactor":
        power_gen_mass_hypergolic = reactor_mass # + reactor_fuel_mass
    elif selection == "rtg":
        power_gen_mass_hypergolic = rtg_mass

    # ------------------------------------------------------------
    # Hypergolic kickstage dry mass
    # ------------------------------------------------------------

    hypergolic_structural_fraction = 0.1

    kickstage_dry_mass_hypergolic = (
        hypergolic_structural_fraction * hypergolic_propellant
    )


    # ============================================================
    #               TOTAL DRY MASS BUDGET
    # ============================================================

    system_dry_mass_hypergolic = (
        0
        + ion_system_mass_hypergolic
        + power_gen_mass_hypergolic
        + kickstage_dry_mass_hypergolic
    )


    # ============================================================
    #               PAYLOAD MARGIN ANALYSIS
    # ============================================================

    payload_remaining_hypergolic = dry_mass_assumption - system_dry_mass_hypergolic

    # payload_remaining_thermal =  margin_loss_thermal - dry_mass_assumption
    # payload_remaining_hypergolic = margin_loss_hypergolic - dry_mass_assumption

    if print_results:
        # ============================================================
        #               PRINT RESULTS
        # ============================================================

        print()
        print("=" * 80)
        print("SYSTEM DRY MASS BUDGET")
        print("=" * 80)

        print("\n--- ION PROPULSION ---")
        print("Hypergolic architecture:", ion_system_mass_hypergolic, "kg")

        print("\n--- Power Generator ---")
        print("Hypergolic architecture:", power_gen_mass_hypergolic, "kg")

        print("\n--- KICKSTAGE DRY MASS ---")
        print("Hypergolic architecture:", kickstage_dry_mass_hypergolic, "kg")

        print("\n--- TOTAL DRY MASS ---")
        print("Hypergolic architecture:", system_dry_mass_hypergolic, "kg")

        print("\n--- TOTAL LAUNCH MASS ---")
        print("Hypergolic architecture:", kickstage_initial_mass_hypergolic, "kg")

        print("\n--- PAYLOAD IMPACT ---")
        # print("Thermal margin loss:", margin_loss_thermal, "kg")
        # print("Hypergolic margin loss:", margin_loss_hypergolic, "kg")

        # print("Thermal remaining payload:", payload_remaining_thermal, "kg")
        # print("Hypergolic remaining payload:", payload_remaining_hypergolic, "kg")
    # ============================================================
    # UPDATED RETURN STATEMENT
    # ============================================================
    else:
        return (
            kickstage_initial_mass_hypergolic,
            payload_remaining_hypergolic,
            oberth_propellant_hypergolic,
            plane_change_propellant + rendezvous_propellant,
            kickstage_dry_mass_hypergolic,
            power_gen_mass_hypergolic
        )

if __name__ == "__main__":

    dry_masses = np.linspace(1, 10000, 10000)

    launch_masses_hypergolic = []
    remaining_masses_hypergolic = []

    kick_stage_propellants_hypergolic = []

    spacecraft_propellants_hypergolic = []

    kick_stage_dry_masses_hypergolic = []

    power_gen_masses_hypergolic = []

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
            power_gen_hypergolic 
        ) = run_configuration(
            dry_mass,
            print_results=False
        )

        if dry_mass == 100 or dry_mass == 1000 or dry_mass == 2700 or dry_mass == 3000 or dry_mass == 3300 or dry_mass == 5000 or dry_mass == 10000:
            # print("\n\n" + "=" * 80)
            # print("DETAILED SUMMARY FOR DRY MASS =", dry_mass, "kg, THERMAL OPTION")
            # print("=" * 80)
            
            # run_configuration(
            #     dry_mass,
            #     thermal=True,
            #     print_results=True
            # )
            m_wet_actual_spacecraft = launch_mass_hypergolic - kick_dry_hypergolic - kick_prop_hypergolic
            print("\n\n" + "=" * 80)
            print("DETAILED SUMMARY FOR DRY MASS (spacecraft + kick stage) =", dry_mass, "kg, NON-THERMAL OPTION")
            print("spacecraft total (wet) mass without kick stage = ", m_wet_actual_spacecraft)
            print("spacecraft propellant mass =", spacecraft_prop_hypergolic, "kg, s/c total mass fraction = ", spacecraft_prop_hypergolic / m_wet_actual_spacecraft)
            print("power generator mass =", power_gen_hypergolic, "kg, s/c total mass fraction = ", power_gen_hypergolic / m_wet_actual_spacecraft)
            print('\n')
            print("KICK STAGE", '\n')
            print("launch mass hypergolic =", launch_mass_hypergolic, "kg")
            print("kickstage propellant mass =", kick_prop_hypergolic, "kg, launch mass fraction = ", kick_prop_hypergolic / launch_mass_hypergolic)
            print("kickstage dry mass =", kick_dry_hypergolic, "kg, launch mass fraction = ", kick_dry_hypergolic / launch_mass_hypergolic)
        
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
        power_gen_masses_hypergolic.append(power_gen_hypergolic)

    # ============================================================
    # CONVERT TO ARRAYS
    # ============================================================

    launch_masses_hypergolic = np.array(launch_masses_hypergolic)
    remaining_masses_hypergolic = np.array(remaining_masses_hypergolic)

    kick_stage_propellants_hypergolic = np.array(kick_stage_propellants_hypergolic)

    spacecraft_propellants_hypergolic = np.array(spacecraft_propellants_hypergolic)

    kick_stage_dry_masses_hypergolic = np.array(kick_stage_dry_masses_hypergolic)

    power_gen_masses_hypergolic = np.array(power_gen_masses_hypergolic)

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
    # THERMAL
    # ------------------------------------------------------------

    # x_t = remaining_masses_thermal[thermal_mask]
    #
    # ax.plot(
    #     x_t,
    #     launch_masses_thermal[thermal_mask],
    #     linestyle=":",
    #     linewidth=3,
    #     label="Thermal Total Launch Mass"
    # )
    #
    # ax.plot(
    #     x_t,
    #     kick_stage_propellants_thermal[thermal_mask],
    #     linestyle=":",
    #     label="Thermal Kickstage Propellant"
    # )
    #
    # ax.plot(
    #     x_t,
    #     spacecraft_propellants_thermal[thermal_mask],
    #     linestyle=":",
    #     label="Thermal Spacecraft Propellant"
    # )
    #
    # ax.plot(
    #     x_t,
    #     kick_stage_dry_masses_thermal[thermal_mask],
    #     linestyle=":",
    #     label="Thermal Kickstage Dry Mass"
    # )
    #
    # ax.plot(
    #     x_t,
    #     radiator_masses_thermal[thermal_mask],
    #     linestyle=":",
    #     label="Thermal Radiator Mass"
    # )
    #
    # ax.plot(
    #     x_t,
    #     reactor_masses_thermal[thermal_mask],
    #     linestyle=":",
    #     label="Thermal Reactor Mass"
    # )

    # ------------------------------------------------------------
    # HYPERGOLIC
    # ------------------------------------------------------------

    x_h = remaining_masses_hypergolic[hypergolic_mask]

    ax.plot(
        x_h,
        launch_masses_hypergolic[hypergolic_mask],
        linestyle="--",
        linewidth=3,
        label="Hypergolic Total Launch Mass"
    )

    ax.plot(
        x_h,
        kick_stage_propellants_hypergolic[hypergolic_mask],
        linestyle="--",
        label="Hypergolic Kickstage Propellant"
    )

    ax.plot(
        x_h,
        spacecraft_propellants_hypergolic[hypergolic_mask],
        linestyle="--",
        label="Hypergolic Spacecraft Propellant"
    )

    ax.plot(
        x_h,
        kick_stage_dry_masses_hypergolic[hypergolic_mask],
        linestyle="--",
        label="Hypergolic Kickstage Dry Mass"
    )

    ax.plot(
        x_h,
        power_gen_masses_hypergolic[hypergolic_mask],
        linestyle="--",
        label="Hypergolic Power Gen Mass"
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