import math
from dataclasses import dataclass
import numpy as np


@dataclass
class Material:
    E: float
    density: float


@dataclass
class CableSystem:
    N: int
    R_anchor: float
    E: float
    area: float
    pretension: float = 0.0


def cable_geometry(L, R):
    Lc = math.sqrt(L**2 + R**2)
    sin_theta = R / Lc
    cos_theta = L / Lc
    return Lc, sin_theta, cos_theta

def gamma_axial_compression(radius, thickness):
    """
    Abbott Aerospace / NASA SP-8007 knockdown factor

    Parameters
    ----------
    radius : m
    thickness : m

    Returns
    -------
    gamma : float
    """

    rt = radius / thickness

    if rt >= 1500:
        raise ValueError(
            "Equation only valid for r/t < 1500."
        )

    phi = (1.0 / 16.0) * math.sqrt(rt)

    gamma = 1.0 - 0.907 * (1.0 - math.exp(-phi))

    return gamma

def shell_buckling_force(radius, thickness, E):

    gamma = gamma_axial_compression(
        radius,
        thickness,
    )

    sigma_cr = gamma * E * thickness / radius

    area = 2.0 * math.pi * radius * thickness

    return sigma_cr * area


def euler_buckling_force(radius, thickness, length, E, K=1.0):

    I = math.pi * radius**3 * thickness

    return math.pi**2 * E * I / (K * length) ** 2


def axial_frequency(radius, thickness, length, E, tip_mass):

    area = 2 * math.pi * radius * thickness

    k = E * area / length

    return (1 / (2 * math.pi)) * math.sqrt(k / tip_mass)


def lateral_frequency(radius, thickness, length, E, tip_mass, cable_sys):

    I = math.pi * radius**3 * thickness

    k_mast = 3 * E * I / length**3
    k_cables = cable_lateral_stiffness(cable_sys, length)

    k_total = k_mast + k_cables

    return (1 / (2 * math.pi)) * math.sqrt(k_total / tip_mass)

def cable_lateral_stiffness(system, L):
    Lc, s, c = cable_geometry(L, system.R_anchor)

    k_one = (system.E * system.area / Lc) * (s**2)

    angle = 2*np.pi / system.N

    return np.cos(angle)*2 * k_one

def cable_load_share(k_cables, k_mast):
    return k_cables / (k_cables + k_mast + 1e-9) # indeterminate problem

def effective_loads(lateral_load, axial_load, L, cable_sys, E, I):
    k_mast = 3 * E * I / L**3
    k_cables = cable_lateral_stiffness(cable_sys, L)

    share = cable_load_share(k_cables, k_mast)

    Lc, s, c = cable_geometry(L, cable_sys.R_anchor)

    # reduced lateral load on mast
    lateral_mast = lateral_load * (1 - share)

    # cable vertical component increases compression
    axial_from_cables = lateral_load * share * (s / (c + 1e-9))

    axial_total = axial_load + axial_from_cables

    return axial_total, lateral_mast, k_mast + k_cables

def effective_axial_load(axial_load, lateral_load, cable_sys, L):

    Lc, s, c = cable_geometry(L, cable_sys.R_anchor)

    axial_extra = lateral_load * (s / (c + 1e-9))

    return axial_load + axial_extra


def bending_stress(radius, thickness, length, lateral_force):

    I = math.pi * radius**3 * thickness

    M = lateral_force * length

    return M * radius / I


def boom_mass(radius, thickness, length, density):

    area = 2 * math.pi * radius * thickness

    return area * length * density


def optimize_boom(
    length,
    tip_mass,
    materials,
    axial_load,
    lateral_load,
    min_axial_freq,
    min_lateral_freq,
    cable_sys,
    K=1.0,
    radius_range=np.linspace(0.1, 1.0/2, 1000),
    thiclness_range=np.linspace(1/1000, 0.5, 1000),
):

    best = None
    for mat_name, material in materials.items():

        for radius in radius_range:

            for thickness in thiclness_range:
                try:
                    mass = boom_mass(
                        radius,
                        thickness,
                        length,
                        material.density,
                    )

                    f_ax = axial_frequency(
                        radius,
                        thickness,
                        length,
                        material.E,
                        tip_mass,
                    )

                    f_lat = lateral_frequency(
                        radius,
                        thickness,
                        length,
                        material.E,
                        tip_mass,
                        cable_sys
                    )

                    F_shell = shell_buckling_force(
                        radius,
                        thickness,
                        material.E,
                    )

                    F_euler = euler_buckling_force(
                        radius,
                        thickness,
                        length,
                        material.E,
                        K,
                    )

                    _, lateral_mast, _ = effective_loads(
                        lateral_load,
                        axial_load,
                        length,
                        cable_sys,
                        material.E,
                        math.pi * radius ** 3 * thickness
                    )

                    sigma_bend = bending_stress(radius, thickness, length, lateral_mast)

                    if f_ax < min_axial_freq:
                        continue

                    if f_lat < min_lateral_freq:
                        continue

                    axial_eff = effective_axial_load(axial_load, lateral_load, cable_sys, length)

                    if min(F_shell, F_euler) < axial_eff:
                        continue

                    # crude yield check
                    if sigma_bend > 150e6:
                        continue

                except:
                    continue

                if best is None or mass < best["mass"]:
                    SFs = safety_factors(
                        radius, thickness, length,
                        material, tip_mass,
                        axial_load, lateral_load, K, min_axial_freq,
                        min_lateral_freq,
                        cable_sys
                    )

                    worst_SF = min(SFs.values())
                    best = {
                        "material": mat_name,
                        "radius": radius,
                        "thickness": thickness,
                        "mass": mass,
                        "f_axial": f_ax,
                        "f_lateral": f_lat,
                        "buckling_load": min(F_shell, F_euler),
                        "bending_stress": sigma_bend,
                        "safety_factors": SFs,
                        "worst_SF": worst_SF,
                    }

    return best

def safety_factors(radius, thickness, length, material, tip_mass, axial_load, lateral_load, K, min_axial_freq, min_lateral_freq, cable_system):

    f_ax = axial_frequency(radius, thickness, length, material.E, tip_mass)
    f_lat = lateral_frequency(radius, thickness, length, material.E, tip_mass, cable_system)

    F_shell = shell_buckling_force(radius, thickness, material.E)
    F_euler = euler_buckling_force(radius, thickness, length, material.E, K)

    sigma_bend = bending_stress(radius, thickness, length, lateral_load)

    area = 2 * math.pi * radius * thickness
    axial_stress = axial_load / area

    return {
        "SF_axial_freq": f_ax / min_axial_freq,
        "SF_lateral_freq": f_lat / min_lateral_freq,
        "SF_shell_buckling": F_shell / axial_load,
        "SF_euler_buckling": F_euler / axial_load,
        "SF_bending": 150e6 / sigma_bend,
        "SF_axial_stress": 150e6 / axial_stress,
    }

if __name__ == "__main__":
    materials = {
        # "Aluminum 7075-T6": Material(E=71e9, density=2810),
        # "Aluminum 6061-T6": Material(E=69e9, density=2700),
        # "Titanium Ti-6Al-4V": Material(E=114e9, density=4430),
        # "Steel 17-4PH": Material(E=200e9, density=7800),
        "CFRP (quasi-iso)": Material(E=140e9, density=1600),
    }

    mass = 489.9
    SF = 1.5
    boom_length = 10

    test_mass = mass*SF

    cable_radius = 2.2/100 # m
    cable_density = 1600
    cable_E = 140e9
    number_of_cables = 6
    cable_base_radius = 4.6/2
    cable_sys = CableSystem(number_of_cables, cable_base_radius, cable_E, np.pi*cable_radius**2)

    cable_length, _, _ = cable_geometry(boom_length, cable_base_radius)

    cable_mass = number_of_cables*cable_length*cable_density*np.pi*cable_radius**2
    print(f"Cable mass: {cable_mass} kg")

    result = optimize_boom(
        length=boom_length,
        tip_mass=test_mass,
        materials=materials,
        axial_load=test_mass * 9.81 * 6,
        lateral_load=test_mass * 9.81 * 1.8,
        min_axial_freq=20,
        min_lateral_freq=6,
        cable_sys=cable_sys,
        K=1.0,
    )

    print(result)

    beam_mass = result["mass"]

    print(f"Total mass: {cable_mass+beam_mass} kg")