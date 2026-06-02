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

# Powergen
select = 2
selection = ["fuel_cell", "reactor", "rtg"]
selection = selection[select]


# ============================================================
#                Power System SIZING
# ============================================================


def fuelcell(power_elec, burn_time=41366655.26322658):
    # Fuel Cells

    fuel_cell_BOP_power_density = 12000/118 # W/kg
    fuel_cell_reactants_specific_energy = 3661*0.7 # Wh/kg

    # print(required_electric_power_hypergolic / 1e3)
    fuel_cell_mass = power_elec / fuel_cell_BOP_power_density
    # print(required_electric_power_hypergolic * plane_change_burn_time / 3600 / 1e6)
    fuel_cell_reactants_mass = power_elec * burn_time / 3600 / fuel_cell_reactants_specific_energy
    return fuel_cell_mass, fuel_cell_reactants_mass

def reactor(power_elec, burn_time=41366655.26322658):
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


    reactor_fuel_mass = power_elec*burn_time*2.5 / reactor_fuel_equivalent_specific_energy
    # print("fuelamsss;", reactor_fuel_mass)
    reactor_fuel_mass = reactor_fuel_mass / 0.1 # burn up mass https://beyondnerva.wordpress.com/fission-power-systems/systems-for-nuclear-auxiliary-power-snap/snap-10-10a-and-snapshot/
    # print("fuelamsss;", reactor_fuel_mass)

    reactor_mass = power_elec / reactor_BOP_power_density
    thermal_power = power_elec/brayton_cycle_efficiency
    return reactor_mass, reactor_fuel_mass, thermal_power

def reactor_thermal(power_thermal, burn_time=41366655.26322658):


    # Uranium-235 fission energy
    energy_one_fission = 169.1 * 10**(6) *  1.602176634 * 10**(-19) # MeV * J/eV source https://web.archive.org/web/20190505175631/http://www.kayelaby.npl.co.uk/atomic_and_nuclear_physics/4_7/4_7_1.html
    energy_fission_mol = energy_one_fission * 6.02214076* 10**(23) # J per Mol https://www.nist.gov/si-redefinition/meet-constants
    kg_per_mol_u235 = 235/1000 # g per mol / 1000
    u235_specific_energy = energy_fission_mol / kg_per_mol_u235 # J per kg


    # (elec power / brayton ) /  u235_specific_energy = kg per second (of pure u235)
    # minimum haleu mass = kg per sec / 0.2 , this is also haleu mass rate https://world-nuclear.org/information-library/nuclear-fuel-cycle/conversion-enrichment-and-fabrication/high-assay-low-enriched-uranium-haleu
    # fuel mass to sustain mission = burn_time * haleu mass rate

    # elec power / brayton / u235_specific energy / 0.2 * burn time = fuel mass

    reactor_fuel_equivalent_specific_energy = u235_specific_energy * 0.2 # Wh/kg
    reactor_BOP_power_density = 100000 / 100 # W/kg


    reactor_fuel_mass = power_thermal*burn_time*2.5 / reactor_fuel_equivalent_specific_energy
    # print("fuelamsss;", reactor_fuel_mass)
    reactor_fuel_mass = reactor_fuel_mass / 0.1 # burn up mass https://beyondnerva.wordpress.com/fission-power-systems/systems-for-nuclear-auxiliary-power-snap/snap-10-10a-and-snapshot/
    # print("fuelamsss;", reactor_fuel_mass)

    reactor_mass = power_thermal / reactor_BOP_power_density
    return reactor_mass, reactor_fuel_mass

def reactor_energyver(power_elec, power_energy):
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

    reactor_BOP_power_density = 100000 * brayton_cycle_efficiency /100 # W/kg


    reactor_fuel_mass = power_energy / reactor_fuel_equivalent_specific_energy
    # print("fuelamsss;", reactor_fuel_mass)
    reactor_fuel_mass = reactor_fuel_mass / 0.1 # burn up mass https://beyondnerva.wordpress.com/fission-power-systems/systems-for-nuclear-auxiliary-power-snap/snap-10-10a-and-snapshot/
    # print("fuelamsss;", reactor_fuel_mass)

    reactor_mass = power_elec / reactor_BOP_power_density
    return reactor_mass, reactor_fuel_mass

def rtgsize(power_elec):
    # Rtg
    rtg_power_density = 296/56
    # 1/2*(296-296*0.7)*20*365*24*3600 + 296*0.7*20*365*24*3600 = 156418560000, 43 MWh
    rtg_mass = power_elec / rtg_power_density
    return rtg_mass, None

def solar(power_elec):

    solar_efficiency = 0.27
    solar_constant = 1361  # W/m²
    degradation = 0.77
    distance_au = 1.0

    areal_density = 4.0  # kg/m² (ROSA-class assumption)

    flux = solar_constant / (distance_au ** 2)

    effective_eff = solar_efficiency * degradation

    area = power_elec / (flux * effective_eff)

    mass = area * areal_density

    return area, mass
