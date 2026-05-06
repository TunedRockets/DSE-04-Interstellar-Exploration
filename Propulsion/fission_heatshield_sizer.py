import numpy as np
from scipy.optimize import minimize_scalar
from src2.utilities import DAY, YEAR


# ------------------ Propulsion ------------------

def get_prop_mass_with_end_mass(delta_v, end_mass, Isp):
    return np.exp(delta_v / (9.81 * Isp)) * end_mass - end_mass

def get_prop_mass_with_start_mass(delta_v, start_mass, Isp):
    return start_mass * (1 - np.exp(-delta_v / (9.81 * Isp)))

def get_required_thrust(delta_v, Isp, start_mass, burn_time):
    prop_mass = get_prop_mass_with_start_mass(delta_v, start_mass, Isp)
    return Isp * 9.81 * prop_mass / burn_time

def get_required_power(delta_v, Isp, start_mass, burn_time, efficiency=0.7):
    T_req = get_required_thrust(delta_v, Isp, start_mass, burn_time)
    return T_req * Isp * 9.81 / (2 * efficiency)


# ------------------ Brayton Radiator Optimizer ------------------

def optimize_brayton_radiator(electric_power,
                             T_hot=1673,          # 1400°C
                             emissivity=0.9,
                             T_space=3,
                             eta_fraction=0.6):   # real efficiency fraction of Carnot
    """
    Returns:
        T_cold [K]
        efficiency [-]
        thermal_power [W]
        waste_heat [W]
        radiator_area [m^2]
    """

    sigma = 5.670374419e-8

    def radiator_area_for_Tc(T_cold):
        if T_cold <= T_space or T_cold >= T_hot:
            return np.inf

        # Realistic Brayton efficiency (fraction of Carnot)
        eta = eta_fraction * (1 - T_cold / T_hot)

        if eta <= 0:
            return np.inf

        thermal_power = electric_power / eta
        waste_heat = thermal_power - electric_power

        area = waste_heat / (emissivity * sigma * (T_cold**4 - T_space**4))
        return area

    result = minimize_scalar(
        radiator_area_for_Tc,
        bounds=(300, T_hot - 1),
        method='bounded'
    )

    T_cold_opt = result.x
    eta_opt = eta_fraction * (1 - T_cold_opt / T_hot)
    thermal_power = electric_power / eta_opt
    waste_heat = thermal_power - electric_power
    area = radiator_area_for_Tc(T_cold_opt)

    return T_cold_opt, eta_opt, thermal_power, waste_heat, area


# ------------------ Helper for printing ------------------

def print_maneuver(name, delta_v, burn_time, start_mass, Isp, power):
    print(f"------------------  {name} Maneuver  ------------------")
    print("Delta V: ", delta_v, " m/s")
    print("Burn time: ", burn_time, " s (", burn_time / DAY, " days)")
    print("Thrust Required: ",
          get_required_thrust(delta_v, Isp, start_mass, burn_time), " N")

    prop_mass = get_prop_mass_with_start_mass(delta_v, start_mass, Isp)
    print("Propellant mass: ", prop_mass, " kg")
    print("Power required: ", power / 1000, " kW")

    # Radiator sizing
    T_cold, eta, P_th, Q_waste, A_rad = optimize_brayton_radiator(power)

    print("---- Brayton System ----")
    print("Radiator cold temp:", T_cold, " K")
    print("Cycle efficiency:", eta)
    print("Reactor thermal power:", P_th / 1e6, " MW")
    print("Waste heat:", Q_waste / 1e6, " MW")
    print("Radiator area:", A_rad, " m^2")
    print()


# ------------------ Mission Setup ------------------

total_dv = 20_000  # m/s
plane_change_delta_v = 1_000  # m/s
Isp = 2000
dry_mass = 3000  # kg

oberth_dv = (total_dv - plane_change_delta_v) / 2
rdvz_dv = (total_dv - plane_change_delta_v - oberth_dv)

plane_change_max_burn_time = 40 * DAY
oberth_max_burn_time = 10 * DAY
rdvz_max_burn_time = 60 * DAY


# ------------------ Mass Budget ------------------

total_prop_mass = get_prop_mass_with_end_mass(total_dv, dry_mass, Isp)

print("Total propellant mass: ", total_prop_mass, " kg")
print("Total spacecraft mass: ", total_prop_mass + dry_mass, " kg")


# ------------------ Maneuver Calculations ------------------

# Plane change
plane_change_prop_mass = get_prop_mass_with_start_mass(
    plane_change_delta_v, total_prop_mass, Isp)
plane_change_req_power = get_required_power(
    plane_change_delta_v, Isp, total_prop_mass, plane_change_max_burn_time)

mass_after_plane_change = total_prop_mass - plane_change_prop_mass


# Oberth
oberth_prop_mass = get_prop_mass_with_start_mass(
    oberth_dv, mass_after_plane_change, Isp)
oberth_required_power = get_required_power(
    oberth_dv, Isp, mass_after_plane_change, oberth_max_burn_time)

mass_after_oberth = mass_after_plane_change - oberth_prop_mass


# Rendezvous
rdvz_prop_mass = get_prop_mass_with_start_mass(
    rdvz_dv, mass_after_oberth, Isp)
rdvz_required_power = get_required_power(
    rdvz_dv, Isp, mass_after_oberth, rdvz_max_burn_time)

mass_after_rdvz = mass_after_oberth - rdvz_prop_mass


print()

# ------------------ Output ------------------

print_maneuver("Plane Change",
               plane_change_delta_v,
               plane_change_max_burn_time,
               total_prop_mass,
               Isp,
               plane_change_req_power)

print_maneuver("Oberth",
               oberth_dv,
               oberth_max_burn_time,
               mass_after_plane_change,
               Isp,
               oberth_required_power)

print_maneuver("Rendezvous",
               rdvz_dv,
               rdvz_max_burn_time,
               mass_after_oberth,
               Isp,
               rdvz_required_power)

# ------------------ Oberth solar thermal augmentation ------------------

steady_power = max(plane_change_req_power, rdvz_required_power)

power_deficit = max(0, oberth_required_power - steady_power)

print("------------------  Oberth Solar Thermal Augmentation  ------------------")
print("Oberth required power:", oberth_required_power/1000, "kW")
print("Reactor available power:", steady_power/1000, "kW")
print("Power deficit:", power_deficit/1000, "kW")

# Solar constants
F_1AU = 1361  # W/m^2 solar constant at Earth
AU = 1.496e11
R_sun = 6.96e8

r = 10 * R_sun

# Solar flux at 10 solar radii
solar_flux = F_1AU * (AU / r)**2

print("Solar flux at 10 solar radii:", solar_flux/1000, "kW/m^2")

# Heatshield at 1400°C (assumed max working temp)
T_hot = 1673  # K

# If you're using a thermal engine, you do NOT use emissivity here.
# Instead we assume absorbed thermal power ~ incident flux.

absorptivity = 0.2  # Parker solar probe

usable_flux = solar_flux * absorptivity

# Required area to supply missing power
heatshield_area = power_deficit / usable_flux if power_deficit > 0 else 0

print("Required heatshield (collector) area:", heatshield_area, "m^2")
print()