from dataclasses import dataclass
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    return Lc, R/Lc, L/Lc

def gamma_axial_compression(radius, thickness):
    rt = radius / thickness
    if rt >= 1500:
        raise ValueError
    phi = (1.0 / 16.0) * math.sqrt(rt)
    return 1.0 - 0.907 * (1.0 - math.exp(-phi))

def shell_buckling_force(radius, thickness, E):
    gamma = gamma_axial_compression(radius, thickness)
    sigma_cr = gamma * E * thickness / radius
    area = 2.0 * math.pi * radius * thickness
    return sigma_cr * area

def euler_buckling_force(radius, thickness, length, E, K=1.0):
    I = math.pi * radius**3 * thickness
    return math.pi**2 * E * I / (K * length) ** 2

def axial_frequency(radius, thickness, length, E, tip_mass):
    area = 2 * math.pi * radius * thickness
    k = E * area / length
    return (1/(2*math.pi))*math.sqrt(k/tip_mass)

def cable_lateral_stiffness(system, L):
    Lc, s, c = cable_geometry(L, system.R_anchor)
    k_one = (system.E * system.area / Lc) * (s**2)
    angle = 2*np.pi / system.N
    return np.cos(angle)*2*k_one

def lateral_frequency(radius, thickness, length, E, tip_mass, cable_sys):
    I = math.pi * radius**3 * thickness
    k_mast = 3 * E * I / length**3
    k_total = k_mast + cable_lateral_stiffness(cable_sys, length)
    return (1/(2*math.pi))*math.sqrt(k_total/tip_mass)

def effective_axial_load(axial_load, lateral_load, cable_sys, L):
    _, s, c = cable_geometry(L, cable_sys.R_anchor)
    return axial_load + lateral_load * (s/(c+1e-9))

def boom_mass(radius, thickness, length, density):
    return (2*math.pi*radius*thickness)*length*density

def optimize_boom(length, tip_mass, materials, axial_load, lateral_load,
                  min_axial_freq, min_lateral_freq, cable_sys, K=1.0):
    best = None
    radius_range=np.linspace(0.1,0.5,120)
    thickness_range=np.linspace(0.001,0.05,120)

    for mat_name, material in materials.items():
        for radius in radius_range:
            for thickness in thickness_range:
                try:
                    mass = boom_mass(radius, thickness, length, material.density)
                    f_ax = axial_frequency(radius, thickness, length, material.E, tip_mass)
                    f_lat = lateral_frequency(radius, thickness, length, material.E, tip_mass, cable_sys)

                    if f_ax < min_axial_freq or f_lat < min_lateral_freq:
                        continue

                    buckling=min(
                        shell_buckling_force(radius, thickness, material.E),
                        euler_buckling_force(radius, thickness, length, material.E, K)
                    )

                    if buckling < effective_axial_load(axial_load, lateral_load, cable_sys, length):
                        continue

                    if best is None or mass < best["mass"]:
                        best={"mass":mass,
                              'thickness':thickness,
                              'radius':radius,}
                except Exception:
                    pass
    return best

materials = {"CFRP (quasi-iso)": Material(E=140e9, density=1600)}

mass = 489.9
SF = 1.2
test_mass = mass * SF

cable_radius = 1.5/100
cable_density = 1600
cable_E = 300e9
number_of_cables = 6
cable_base_radius = 4.6/2
cable_sys = CableSystem(number_of_cables, cable_base_radius, cable_E, np.pi*cable_radius**2)

lengths = np.arange(2, 15, 0.1)
masses = []

res = optimize_boom(
        length=10,
        tip_mass=test_mass,
        materials=materials,
        axial_load=test_mass * 9.81 * 6,
        lateral_load=test_mass * 9.81 * 2.0,
        min_axial_freq=20,
        min_lateral_freq=6,
        cable_sys=cable_sys,
        K=1.0,
    )

print(res)

for L in lengths:
    res = optimize_boom(
        length=L,
        tip_mass=test_mass,
        materials=materials,
        axial_load=test_mass * 9.81 * 6,
        lateral_load=test_mass * 9.81 * 2.0,
        min_axial_freq=20,
        min_lateral_freq=6,
        cable_sys=cable_sys,
        K=1.0,
    )
    cable_length, _, _ = cable_geometry(L, cable_sys.R_anchor)
    cable_mass= cable_length*cable_density*np.pi*cable_radius**2*number_of_cables
    masses.append(np.nan if res is None else (res["mass"]+cable_mass))

plt.figure(figsize=(6,4))
plt.plot(lengths, masses)
plt.xlabel("Boom Length (m)")
plt.ylabel("Optimized Boom Mass (kg)")
# plt.title("Optimized Boom Mass vs Boom Length")
plt.grid(True)
plt.show()
