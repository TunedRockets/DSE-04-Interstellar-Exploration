from ReactoPy.CycloPy import size_power, max_radiator_temp, max_reactor_temp, BAR
import matplotlib.pyplot as plt
import numpy as np


steps = 50
elec_sweep = np.linspace(10000,30000,steps//10)
rad_pressure_sweep = np.linspace(0.1*BAR, 2.5*BAR, steps)

mass_grid = np.zeros((steps//10,steps))

def power_mass_parts_with_powerreq():
    mass_list = np.zeros(steps)
    mass_reactor_list = np.zeros(steps)
    mass_radiator_list = np.zeros(steps)
    mass_brayton_list = np.zeros(steps)

    for i, W_elec in enumerate(elec_sweep):
        mass, reactor_mass, radiator_mass, brayton_system_mass, thermal_power, radiator_area = size_power(W_elec, T3=max_reactor_temp, max_T1=max_radiator_temp, rad_pressure=2.5*BAR, verbose=False, plot=False)
        mass_list[i] = mass
        mass_reactor_list[i] = reactor_mass
        mass_radiator_list[i] = radiator_mass
        mass_brayton_list[i] = brayton_system_mass 
        print(W_elec)

    fig, ax = plt.subplots(1,1,figsize=(12,8))

    ax.plot(elec_sweep/1000, mass_list, color="red", label="Total")
    ax.plot(elec_sweep/1000, mass_reactor_list, color="green", linestyle="--", label="Reactor")
    ax.plot(elec_sweep/1000, mass_radiator_list, color="magenta", linestyle="--", label="Radiator")
    ax.plot(elec_sweep/1000, mass_brayton_list, color="blue", linestyle="--", label="Brayton")
    ax.set_xlabel("Electrical Power Required (kW)")
    ax.set_ylabel("Total Mass of Power System (kg)")
    ax.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig("powersenschart.pdf")

for i, W_elec in enumerate(elec_sweep):
    for j, rad_pressure in enumerate(rad_pressure_sweep):
        mass, reactor_mass, radiator_mass, brayton_system_mass, thermal_power, radiator_area = size_power(W_elec, T3=max_reactor_temp, max_T1=max_radiator_temp, rad_pressure=rad_pressure, verbose=False, plot=False)
        mass_grid[i,j] = mass
        if j % 10 == 0:
            print(i, j, mass)

fig, ax = plt.subplots(1,1,figsize=(8,8))

X, Y = np.meshgrid(rad_pressure_sweep, elec_sweep)

pcm = ax.pcolormesh(
    X,
    Y,
    mass_grid,
    shading="auto",
    cmap="viridis"
)

plt.rc("font", size=12)
cbar = fig.colorbar(pcm, ax=ax)
cbar.set_label("Mass (kg)")
ax.set_xlabel("Radiator Pressure (Pa)")
ax.set_ylabel("Electrical Power (W)")

plt.show()
