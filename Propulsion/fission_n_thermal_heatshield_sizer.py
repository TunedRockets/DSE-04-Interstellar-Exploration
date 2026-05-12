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
# Solar thermal kick stage
# ------------------------------------------------------------

# Water
thermal_prop_molecular_mass = 18 / 1000  # kg/mol

# Propellant density
# Water: ~1000 kg/m^3
# Liquid hydrogen: ~70 kg/m^3
# Liquid methane: ~420 kg/m^3

propellant_density = 1000  # kg/m^3

solar_thermal_chamber_temperature = 1673  # K

tank_pressure = 5e5  # Pa

max_chamber_pressure = 50e5  # 50 bar

pump_efficiency = 0.70


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
                    T_cold**4
                    - T_space**4
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
        * (AU / distance)**2
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

def run_configuration(dry_mass_assumption, thermal, print_results=False):

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
    #                   SOLAR THERMAL STAGE
    # ============================================================

    thermal_isp = get_thermal_isp(
        solar_thermal_chamber_temperature,
        thermal_prop_molecular_mass
    )

    oberth_propellant_thermal = (
        get_prop_mass_with_end_mass(
            oberth_delta_v,
            spacecraft_total_mass,
            thermal_isp
        )
    )

    kickstage_initial_mass_thermal = (
        spacecraft_total_mass
        + oberth_propellant_thermal
    )

    kickstage_final_mass_thermal = spacecraft_total_mass

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

    plane_change_thrust_thermal = (
        get_required_thrust(
            plane_change_delta_v,
            ion_isp,
            kickstage_initial_mass_thermal,
            plane_change_burn_time
        )
    )
    plane_change_thrust_hypergolic = (
        get_required_thrust(
            plane_change_delta_v,
            ion_isp,
            kickstage_initial_mass_hypergolic,
            plane_change_burn_time
        )
    )

    plane_change_power_thermal = (
        get_required_electric_power(
            plane_change_thrust_thermal,
            ion_isp,
            ion_efficiency
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
    # Reactor sized by max electric propulsion load
    # ------------------------------------------------------------

    required_reactor_electric_power_thermal = max(
        plane_change_power_thermal,
        rendezvous_power
    )

    required_reactor_electric_power_hypergolic = max(
        plane_change_power_hypergolic,
        rendezvous_power
    )

    required_thruster_count_thermal = math.ceil(
        max(
            plane_change_thrust_thermal,
            rendezvous_thrust
        )
        / ion_thruster_thrust
    )

    required_thruster_count_hypergolic = math.ceil(
        max(
            plane_change_thrust_hypergolic,
            rendezvous_thrust
        )
        / ion_thruster_thrust
    )


    # ============================================================
    #              SOLAR THERMAL OBERTH SIZING
    # ============================================================

    oberth_mdot_thermal = (
        oberth_propellant_thermal
        / oberth_burn_time
    )

    oberth_thrust_thermal = (
        oberth_mdot_thermal
        * thermal_isp
        * g0
    )

    required_thermal_power = (
        get_thermal_power_required(
            oberth_mdot_thermal,
            solar_thermal_chamber_temperature,
            20,
            thermal_prop_molecular_mass
        )
    )

    solar_flux = get_solar_flux(
        perihelion_distance
    )

    required_heatshield_area = (
        get_heatshield_area(
            required_thermal_power,
            solar_flux,
            solar_absorptivity
        )
    )


    # ============================================================
    #                     PUMP SIZING
    # ============================================================

    pump_power_required = (
        get_required_pump_power(
            max_chamber_pressure,
            tank_pressure,
            oberth_mdot_thermal,
            pump_efficiency,
            propellant_density
        )
    )

    available_pump_power = (
        required_reactor_electric_power_thermal
    )

    achievable_pressure = get_chamber_pressure(
        tank_pressure,
        available_pump_power,
        oberth_mdot_thermal,
        pump_efficiency,
        propellant_density
    )

    actual_chamber_pressure = min(
        achievable_pressure,
        max_chamber_pressure
    )


    # ============================================================
    #                REACTOR / RADIATOR SIZING
    # ============================================================

    (
        radiator_Tcold_thermal,
        brayton_efficiency_thermal,
        reactor_thermal_power_thermal,
        reactor_waste_heat_thermal,
        radiator_area_thermal
    ) = optimize_brayton_radiator(
        required_reactor_electric_power_thermal,
        brayton_hot_side_temperature,
        radiator_emissivity,
        3,
        brayton_eta_fraction
    )

    (
        radiator_Tcold_hypergolic,
        brayton_efficiency_hypergolic,
        reactor_thermal_power_hypergolic,
        reactor_waste_heat_hypergolic,
        radiator_area_hypergolic
    ) = optimize_brayton_radiator(
        required_reactor_electric_power_hypergolic,
        brayton_hot_side_temperature,
        radiator_emissivity,
        3,
        brayton_eta_fraction
    )


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

        print("Required thruster count (Thermal Option):",
              required_thruster_count_thermal)

        print("Required thruster count (Hypergolic Option):",
              required_thruster_count_hypergolic)

        print()

        print("Plane change burn time:",
              plane_change_burn_time,
              "s (",
              plane_change_burn_time / DAY,
              "days )")

        print("Plane change thrust (Thermal Option):",
              plane_change_thrust_thermal,
              "N")

        print("Plane change electric power (Thermal Option):",
              plane_change_power_thermal / 1e3,
              "kW")

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
        #                   REACTOR SYSTEM
        # ============================================================

        print()
        print("=" * 80)
        print("FISSION REACTOR")
        print("=" * 80)

        print("Required electric output (Thermal Option):",
              required_reactor_electric_power_thermal / 1e3,
              "kW")

        print("Required electric output (Hypergolic Option):",
              required_reactor_electric_power_hypergolic / 1e3,
              "kW")


        print("Required thermal power (Hypergolic Option):",
              reactor_thermal_power_hypergolic / 1e3,
              "kW")

        print("Brayton efficiency (Hypergolic Option):",
              brayton_efficiency_hypergolic)

        print("Waste heat (Hypergolic Option):",
              reactor_waste_heat_hypergolic / 1e3,
              "kW")


        # ============================================================
        #                      RADIATOR SYSTEM
        # ============================================================

        print()
        print("=" * 80)
        print("RADIATOR")
        print("=" * 80)

        print("Radiator cold side temperature (Thermal Option):",
              radiator_Tcold_thermal,
              "K")
        print("Radiator cold side temperature (Hypergolic Option):",
              radiator_Tcold_hypergolic,
              "K")

        print("Radiator area (Thermal Option):",
              radiator_area_thermal,
              "m^2")
        print("Radiator area (Hypergolic Option):",
              radiator_area_hypergolic,
              "m^2")


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
                                    tank_pressure_bar,
                                    chamber_pressure_bar,
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

            print()
            print("Tank pressure:", tank_pressure_bar, "bar")
            print("Chamber pressure:", chamber_pressure_bar, "bar")

            if extra_power > 0:
                print("Auxiliary power load:", extra_power / 1e3, "kW")


            # ------------------------------------------------------------
            # THERMAL KICKSTAGE REPORT
            # ------------------------------------------------------------

            print_kickstage_summary(
                name="SOLAR THERMAL",
                isp=thermal_isp,
                prop_mass=oberth_propellant_thermal,
                mdot=oberth_mdot_thermal,
                thrust=oberth_thrust_thermal,
                burn_time=oberth_burn_time,
                initial_mass=kickstage_initial_mass_thermal,
                final_mass=kickstage_final_mass_thermal,
                achieved_dv=thermal_isp * g0 * math.log(
                    kickstage_initial_mass_thermal / kickstage_final_mass_thermal
                ),
                tank_pressure_bar=tank_pressure / 1e5,
                chamber_pressure_bar=actual_chamber_pressure / 1e5,
                extra_power=0
            )


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
                tank_pressure_bar=hypergolic_chamber_pressure / 1e5,
                chamber_pressure_bar=hypergolic_chamber_pressure / 1e5,
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

            print("Oberth propellant (Thermal Option):",
                  oberth_propellant_thermal,
                  "kg")

            print("Oberth propellant (Hypergolic Option):",
                  oberth_propellant_hypergolic,
                  "kg")

            print()

            print("Spacecraft wet mass after kickstage separation:",
                  spacecraft_total_mass,
                  "kg")

            print("Combined initial launch mass (Thermal Option):",
                  kickstage_initial_mass_thermal,
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

    ion_system_mass_thermal = (
        required_thruster_count_thermal * ion_thruster_mass
        + ppp_mass_per_kw * (required_reactor_electric_power_thermal / 1e3)
    )

    ion_system_mass_hypergolic = (
        required_thruster_count_hypergolic * ion_thruster_mass
        + ppp_mass_per_kw * (required_reactor_electric_power_hypergolic / 1e3)
    )

    ion_system_mass_thermal *= (1 + feed_system_fraction)
    ion_system_mass_hypergolic *= (1 + feed_system_fraction)


    # ------------------------------------------------------------
    # Reactor mass model
    # ------------------------------------------------------------

    reactor_specific_mass = 130/100  # kg/kW thermal (SNAP)

    reactor_mass_thermal = (
        reactor_thermal_power_thermal / 1e3
    ) * reactor_specific_mass

    reactor_mass_hypergolic = (
        reactor_thermal_power_hypergolic / 1e3
    ) * reactor_specific_mass


    # ------------------------------------------------------------
    # Radiator mass model
    # ------------------------------------------------------------

    radiator_areal_density = 3  # kg/m^2

    radiator_mass_thermal = radiator_area_thermal * radiator_areal_density
    radiator_mass_hypergolic = radiator_area_hypergolic * radiator_areal_density


    # ------------------------------------------------------------
    # Solar-thermal kickstage dry mass
    # ------------------------------------------------------------

    tank_structural_fraction = 0.1
    pump_specific_power_mass = 2  # kg/kW pump system

    kickstage_dry_mass_thermal = (
        tank_structural_fraction * oberth_propellant_thermal
        + pump_specific_power_mass * pump_power_required / 1e3
    )

    heatshield_areal_density = 8  # kg/m^2 (high-temp heat shield)

    heatshield_mass = heatshield_areal_density * required_heatshield_area

    kickstage_dry_mass_thermal += heatshield_mass


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

    system_dry_mass_thermal = (
        0
        + ion_system_mass_thermal
        + reactor_mass_thermal
        + radiator_mass_thermal
        + kickstage_dry_mass_thermal
    )

    system_dry_mass_hypergolic = (
        0
        + ion_system_mass_hypergolic
        + reactor_mass_hypergolic
        + radiator_mass_hypergolic
        + kickstage_dry_mass_hypergolic
    )


    # ============================================================
    #               PAYLOAD MARGIN ANALYSIS
    # ============================================================

    payload_remaining_thermal = dry_mass_assumption - system_dry_mass_thermal
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
        print("Thermal architecture:", ion_system_mass_thermal, "kg")
        print("Hypergolic architecture:", ion_system_mass_hypergolic, "kg")

        print("\n--- REACTOR ---")
        print("Thermal architecture:", reactor_mass_thermal, "kg")
        print("Hypergolic architecture:", reactor_mass_hypergolic, "kg")

        print("\n--- RADIATORS ---")
        print("Thermal architecture:", radiator_mass_thermal, "kg")
        print("Hypergolic architecture:", radiator_mass_hypergolic, "kg")

        print("\n--- KICKSTAGE DRY MASS ---")
        print("Thermal architecture:", kickstage_dry_mass_thermal, "kg")
        print("Hypergolic architecture:", kickstage_dry_mass_hypergolic, "kg")

        print("\n--- TOTAL DRY MASS ---")
        print("Thermal architecture:", system_dry_mass_thermal, "kg")
        print("Hypergolic architecture:", system_dry_mass_hypergolic, "kg")

        print("\n--- TOTAL LAUNCH MASS ---")
        print("Thermal architecture:", kickstage_initial_mass_thermal, "kg")
        print("Hypergolic architecture:", kickstage_initial_mass_hypergolic, "kg")

        print("\n--- PAYLOAD IMPACT ---")
        # print("Thermal margin loss:", margin_loss_thermal, "kg")
        # print("Hypergolic margin loss:", margin_loss_hypergolic, "kg")

        # print("Thermal remaining payload:", payload_remaining_thermal, "kg")
        # print("Hypergolic remaining payload:", payload_remaining_hypergolic, "kg")
    # ============================================================
    # UPDATED RETURN STATEMENT
    # ============================================================

    if thermal:
        return (
            kickstage_initial_mass_thermal,
            payload_remaining_thermal,
            oberth_propellant_thermal,
            plane_change_propellant + rendezvous_propellant,
            kickstage_dry_mass_thermal,
            radiator_mass_thermal,
            reactor_mass_thermal
        )
    else:
        return (
            kickstage_initial_mass_hypergolic,
            payload_remaining_hypergolic,
            oberth_propellant_hypergolic,
            plane_change_propellant + rendezvous_propellant,
            kickstage_dry_mass_hypergolic,
            radiator_mass_hypergolic,
            reactor_mass_hypergolic
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
            launch_mass_thermal,
            remaining_mass_margin_thermal,
            kick_prop_thermal,
            spacecraft_prop_thermal,
            kick_dry_thermal,
            radiator_thermal,
            reactor_thermal
        ) = run_configuration(
            dry_mass,
            thermal=True,
            print_results=False
        )

        (
            launch_mass_hypergolic,
            remaining_mass_margin_hypergolic,
            kick_prop_hypergolic,
            spacecraft_prop_hypergolic,
            kick_dry_hypergolic,
            radiator_hypergolic,
            reactor_hypergolic
        ) = run_configuration(
            dry_mass,
            thermal=False,
            print_results=False
        )

        if dry_mass == 100 or dry_mass ==1000 or dry_mass == 5000 or dry_mass == 10000:
            # print("\n\n" + "=" * 80)
            # print("DETAILED SUMMARY FOR DRY MASS =", dry_mass, "kg, THERMAL OPTION")
            # print("=" * 80)
            
            # run_configuration(
            #     dry_mass,
            #     thermal=True,
            #     print_results=True
            # )

            print("\n\n" + "=" * 80)
            print("DETAILED SUMMARY FOR DRY MASS =", dry_mass, "kg, NON-THERMAL OPTION")
            print("launch mass hypergolic =", launch_mass_hypergolic, "kg")
            print("kickstage propellant mass =", kick_prop_hypergolic, "kg, launch mass fraction = ", kick_prop_hypergolic / launch_mass_hypergolic)
            print("spacecraft propellant mass =", spacecraft_prop_hypergolic, "kg, launch mass fraction = ", spacecraft_prop_hypergolic / launch_mass_hypergolic)
            print("kickstage dry mass =", kick_dry_hypergolic, "kg, launch mass fraction = ", kick_dry_hypergolic / launch_mass_hypergolic)
            print("radiator mass =", radiator_hypergolic, "kg, launch mass fraction = ", radiator_hypergolic / launch_mass_hypergolic)
            print("reactor mass =", reactor_hypergolic, "kg, launch mass fraction = ", reactor_hypergolic / launch_mass_hypergolic)
            print("=" * 80)

            run_configuration(
                dry_mass,
                thermal=False,
                print_results=True
            )


        # --------------------------------------------------------
        # Thermal
        # --------------------------------------------------------

        launch_masses_thermal.append(launch_mass_thermal)
        remaining_masses_thermal.append(remaining_mass_margin_thermal)

        kick_stage_propellants_thermal.append(kick_prop_thermal)
        spacecraft_propellants_thermal.append(spacecraft_prop_thermal)

        kick_stage_dry_masses_thermal.append(kick_dry_thermal)
        radiator_masses_thermal.append(radiator_thermal)
        reactor_masses_thermal.append(reactor_thermal)

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

    launch_masses_thermal = np.array(launch_masses_thermal)
    remaining_masses_thermal = np.array(remaining_masses_thermal)

    launch_masses_hypergolic = np.array(launch_masses_hypergolic)
    remaining_masses_hypergolic = np.array(remaining_masses_hypergolic)

    kick_stage_propellants_thermal = np.array(kick_stage_propellants_thermal)
    kick_stage_propellants_hypergolic = np.array(kick_stage_propellants_hypergolic)

    spacecraft_propellants_thermal = np.array(spacecraft_propellants_thermal)
    spacecraft_propellants_hypergolic = np.array(spacecraft_propellants_hypergolic)

    kick_stage_dry_masses_thermal = np.array(kick_stage_dry_masses_thermal)
    kick_stage_dry_masses_hypergolic = np.array(kick_stage_dry_masses_hypergolic)

    radiator_masses_thermal = np.array(radiator_masses_thermal)
    radiator_masses_hypergolic = np.array(radiator_masses_hypergolic)

    reactor_masses_thermal = np.array(reactor_masses_thermal)
    reactor_masses_hypergolic = np.array(reactor_masses_hypergolic)

    # ============================================================
    # FILTER VALID SOLUTIONS
    # ============================================================

    max_launch_mass = 20_000  # kg (20 tons)

    thermal_mask = (
            (remaining_masses_thermal >= 0)
            & (launch_masses_thermal <= max_launch_mass)
    )

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
        radiator_masses_hypergolic[hypergolic_mask],
        linestyle="--",
        label="Hypergolic Radiator Mass"
    )

    ax.plot(
        x_h,
        reactor_masses_hypergolic[hypergolic_mask],
        linestyle="--",
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