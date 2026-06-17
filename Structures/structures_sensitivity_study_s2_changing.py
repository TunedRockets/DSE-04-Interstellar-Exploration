import numpy as np
import matplotlib.pyplot as plt

# S1 = 3.5 # stand-off between 1st and 2nd bumper [cm]
S1 = 3 # distance between 1st (outer) and 2nd (inner) bumper [cm]
S2 = 0 # stand-off between 2nd (inner) bumper and rear wall [cm]
t_w = 0.15 # equipment cover plate (rear wall) thickness [cm]
sigma_y_ksi = 37.7 # yield strength of equipment cover plate [ksi], here for R_p0.2 = 260 MPa
t_ob = 0.03 # outer bumper thickness [cm]
t_b = 0.15 # inner bumper thickness [cm]

# fit factors
K_3D = 0.4 # general
K_3S = 1.4 # general
K_MLI = 3 # general
K_S2 = 0.1 # general
K_tw = 1.5 # fit factor 
K_CFRP = 0.75
K_cable = 0.35
alpha = 1/2
beta = 2/3
gamma = 1/3
delta = 4/3 # for theta >=65deg or theta <=45deg 
epsilon = 8/3 # for theta >=65deg or theta <=45deg

# theta = np.radians(0) # angle of impact, 0 degrees for normal incidence...?

# densities [g/cm^3]
rho_p = 2.7
rho_b = 2.7
rho_AD_MLI = 0.0447 # = m_a_MLI bcs AREA DENSITY!!!
rho_ob = 2.7
rho_w = 2.7
rho_honeycomb1 = 0.018 # polyamide - https://www.sciencedirect.com/science/article/pii/S0094576524003874
rho_honeycomb2 = 0.370 # Al - https://www.sciencedirect.com/science/article/pii/S0094576524003874
rho_honeycomb3 = 0.037  # modified Alu...?

t_eq_MLI = rho_AD_MLI/rho_ob

v = np.arange(0.1, 30.1, 0.1) # km/s
# print(v)


def calculate_d_c(v, S1, S2, vt1, vt2, theta, t_ob, t_b, t_w):
    return ((((t_w**alpha+t_b)/K_3S)*(sigma_y_ksi/40)**(1/2)+t_ob+K_MLI*t_eq_MLI)/(0.6*np.cos(theta)**delta*rho_p**(1/2)*v**(2/3)))**(18/19)

def calculate_d_c_larger_than_vt2(v, S1, S2, vt1, vt2, theta, t_ob, t_b, t_w):
    return (1.155*(S1**(1/3)*(t_b+K_tw*t_w)**(2/3)+(K_S2*S2**beta)*(t_w**gamma)*np.cos(theta)**(-epsilon))*(sigma_y_ksi/70)**(1/3))/((K_3D**(2/3))*(rho_p**(1/3))*(rho_ob**(1/9))*(v**(2/3))*(np.cos(theta)**delta))

# velocities [km/s] vs type of material 

# TYPE I: basically Al bumper: Al H/C SP / Al bumper / MLI+Al H/C SP / MLI+Al bumper
vt11 = 3
vt21 = 7
# TYPE II: standalone MLI as structure wall
vt12 = 4
vt22 = 10

def calculate_d_c_failure_rear_wall_in_shatter_veloc_regime(v, S1, S2, vt1, vt2, theta, t_ob, t_b, t_w):
    return calculate_d_c(vt1, S1, S2, vt1, vt2, theta, t_ob, t_b, t_w) + (calculate_d_c_larger_than_vt2(vt2, S1, S2, vt1, vt2, theta, t_ob, t_b, t_w) - calculate_d_c(vt1, S1, S2, vt1, vt2, theta, t_ob, t_b, t_w)) / (vt2 - vt1) * (v - vt1)

# 1. Ensure your functions are defined to take (v, S1, constant)
# def calculate_d_c(v, s, c): ...
conditions1 = [v < vt11, (v >= vt11) & (v <= vt21), v > vt21]
conditions2 = [v < vt12, (v >= vt12) & (v <= vt22), v > vt22]

# 2. Use a list of functions directly
funcs = [
    calculate_d_c,              # for v < vt1
    calculate_d_c_failure_rear_wall_in_shatter_veloc_regime, # for vt1 <= v <= vt2
    calculate_d_c_larger_than_vt2               # for v > vt2
]

# 3. Calculate using the 'args' parameter to pass S1 and the constant

# S1 = 3.5 cm
theta = 0

# type 1
y_S2_0_theta0  = np.piecewise(v, conditions1, funcs, 3, 0, vt11, vt21, theta, t_ob, t_b, t_w)
y_S2_10_theta0 = np.piecewise(v, conditions1, funcs, 3, 10, vt11, vt21, theta, t_ob, t_b, t_w)
y_S2_20_theta0 = np.piecewise(v, conditions1, funcs, 3, 20, vt11, vt21, theta, t_ob, t_b, t_w)
y_S2_30_theta0 = np.piecewise(v, conditions1, funcs, 3, 30, vt11, vt21, theta, t_ob, t_b, t_w)
y_S2_40_theta0 = np.piecewise(v, conditions1, funcs, 3, 40, vt11, vt21, theta, t_ob, t_b, t_w)

# type 2
y_S2_0_1_theta0  = np.piecewise(v, conditions2, funcs, 3, 0, vt12, vt22, theta, t_ob, t_b, t_w)
y_S2_10_1_theta0 = np.piecewise(v, conditions2, funcs, 3, 10, vt12, vt22, theta, t_ob, t_b, t_w)
y_S2_20_1_theta0 = np.piecewise(v, conditions2, funcs, 3, 20, vt12, vt22, theta, t_ob, t_b, t_w)
y_S2_30_1_theta0 = np.piecewise(v, conditions2, funcs, 3, 30, vt12, vt22, theta, t_ob, t_b, t_w)


# theta = 45

# # type 1
# y_S2_0_2_theta45  = np.piecewise(v, conditions1, funcs, 3.5, 0, vt11, vt21, theta=np.radians(45))
# y_S2_10_2_theta45 = np.piecewise(v, conditions1, funcs, 3.5, 10, vt11, vt21, theta=np.radians(45))
# y_S2_20_2_theta45 = np.piecewise(v, conditions1, funcs, 3.5, 20, vt11, vt21, theta=np.radians(45))
# y_S2_30_2_theta45 = np.piecewise(v, conditions1, funcs, 3.5, 30, vt11, vt21, theta=np.radians(45))

# # type 2
# y_S2_0_3_theta45  = np.piecewise(v, conditions2, funcs, 3.5, 0, vt12, vt22, theta=np.radians(45))
# y_S2_10_3_theta45 = np.piecewise(v, conditions2, funcs, 3.5, 10, vt12, vt22, theta=np.radians(45))
# y_S2_20_3_theta45 = np.piecewise(v, conditions2, funcs, 3.5, 20, vt12, vt22, theta=np.radians(45))
# y_S2_30_3_theta45 = np.piecewise(v, conditions2, funcs, 3.5, 30, vt12, vt22, theta=np.radians(45))

# S1 = 4 cm
# theta = 0
# # type 1
# y_S2_0_theta0_S1_4  = np.piecewise(v, conditions1, funcs, 4, 0, vt11, vt21, theta=np.radians(0))
# y_S2_10_theta0_S1_4 = np.piecewise(v, conditions1, funcs, 4, 10, vt11, vt21, theta=np.radians(0))
# y_S2_20_theta0_S1_4 = np.piecewise(v, conditions1, funcs, 4, 20, vt11, vt21, theta=np.radians(0))
# y_S2_30_theta0_S1_4 = np.piecewise(v, conditions1, funcs, 4, 30, vt11, vt21, theta=np.radians(0))
# y_S2_40_theta0_S1_4 = np.piecewise(v, conditions1, funcs, 4, 40, vt11, vt21, theta=np.radians(0))

# def plot_ballistic_limit_1x(v, y1, y2, y3, y4, y5 = None, labels=None):
#     # S2 = 0
#     plt.plot(v, y1)
#     plt.plot(v, y2)
#     plt.plot(v, y3)
#     plt.plot(v, y4)
#     plt.plot(v, y5) if y5 is not None else None
#     # plt.ylim(0, 1)
#     plt.xlabel('Velocity (km/s)')
#     plt.ylabel('Critical Diameter (cm)')
#     plt.title('Ballistic Limit vs Velocity for a certain S2')
#     # plt.legend(labels if labels else ['type 1, theta = 0deg', 'type 2, theta = 0deg', 'type 1, theta = 45deg', 'type 2, theta = 45deg', 'type 1, theta = 0deg', 'type 2, theta = 0deg', 'type 1, theta = 45deg', 'type 2, theta = 45deg'])
#     plt.grid()
#     # plt.show()

def plot_ballistic_limit_5x(v, y1, y2, y3, y4, y5 = None, labels=None):
    # S2 = 0
    plt.plot(v, y1)
    plt.plot(v, y2)
    plt.plot(v, y3)
    plt.plot(v, y4)
    plt.plot(v, y5) if y5 is not None else None
    # plt.ylim(0, 1)
    plt.xlabel('Velocity (km/s)')
    plt.ylabel('Critical Diameter (cm)')
    plt.title('Ballistic Limit vs Velocity for a certain S2')
    # plt.legend(labels if labels else ['type 1, theta = 0deg', 'type 2, theta = 0deg', 'type 1, theta = 45deg', 'type 2, theta = 45deg', 'type 1, theta = 0deg', 'type 2, theta = 0deg', 'type 1, theta = 45deg', 'type 2, theta = 45deg'])
    plt.grid()
    # plt.show()

plt.plot(v, y_S2_0_theta0)
plt.plot(v, y_S2_10_theta0)
plt.plot(v, y_S2_20_theta0)
plt.plot(v, y_S2_0_1_theta0)
plt.plot(v, y_S2_10_1_theta0)
plt.plot(v, y_S2_20_1_theta0)
# plt.plot(v, y_S2_0_2_theta45)
# plt.plot(v, y_S2_10_2_theta45)
# plt.plot(v, y_S2_20_2_theta45)
# plt.plot(v, y_S2_0_3_theta45)
# plt.plot(v, y_S2_10_3_theta45)
# plt.plot(v, y_S2_20_3_theta45)
# plt.ylim(0, 1)
plt.xlabel('Velocity (km/s)')
plt.ylabel('Critical Diameter (cm)')
plt.title('Ballistic Limit vs Velocity')
plt.legend(['S2 = 0 cm, type 1, theta = 0deg', 'S2 = 10 cm, type 1, theta = 0deg',
             'S2 = 20 cm, type 1, theta = 0deg', 'S2 = 0 cm, type 2, theta = 0deg',
               'S2 = 10 cm, type 2, theta = 0deg', 'S2 = 20 cm, type 2, theta = 0deg',
                 'S2 = 0 cm, type 1, theta = 45deg', 'S2 = 10 cm, type 1, theta = 45deg',
                   'S2 = 20 cm, type 1, theta = 45deg', 'S2 = 0 cm, type 2, theta = 45deg',
                     'S2 = 10 cm, type 2, theta = 45deg', 'S2 = 20 cm, type 2, theta = 45deg'])
plt.grid()
plt.show()


# # S2 = 0
# plot_ballistic_limit_5x(v, y_S2_0_theta0, y_S2_0_1_theta0, y_S2_0_2_theta45, y_S2_0_3_theta45, labels=['S2 = 0 cm, type 1, theta = 0deg', 'S2 = 0 cm, type 2, theta = 0deg', 'S2 = 0 cm, type 1, theta = 45deg', 'S2 = 0 cm, type 2, theta = 45deg'])

# # S2 = 10
# plot_ballistic_limit_5x(v, y_S2_10_theta0, y_S2_10_1_theta0, y_S2_10_2_theta45, y_S2_10_3_theta45, labels=['S2 = 10 cm, type 1, theta = 0deg', 'S2 = 10 cm, type 2, theta = 0deg', 'S2 = 10 cm, type 1, theta = 45deg', 'S2 = 10 cm, type 2, theta = 45deg'])

# # S2 = 20
# plot_ballistic_limit_5x(v, y_S2_20_theta0, y_S2_20_1_theta0, y_S2_20_2_theta45, y_S2_20_3_theta45, labels=['S2 = 20 cm, type 1, theta = 0deg', 'S2 = 20 cm, type 2, theta = 0deg', 'S2 = 20 cm, type 1, theta = 45deg', 'S2 = 20 cm, type 2, theta = 45deg'])

# # S2 = 30
# plot_ballistic_limit(v, y_S2_30_theta0, y_S2_30_1_theta0, y_S2_30_2_theta45, y_S2_30_3_theta45, labels=['S2 = 30 cm, type 1, theta = 0deg', 'S2 = 30 cm, type 2, theta = 0deg', 'S2 = 30 cm, type 1, theta = 45deg', 'S2 = 30 cm, type 2, theta = 45deg'])


# S2 = {0, 10, 20, 30}, type 1, theta = 0deg, S1 = 3.5 cm
plot_ballistic_limit_5x(v, y_S2_0_theta0, y_S2_10_theta0, y_S2_20_theta0, y_S2_30_theta0, y_S2_40_theta0, labels=['S2 = 0 cm, S1 = 3.5 cm', 'S2 = 10 cm, S1 = 3.5 cm', 'S2 = 20 cm, S1 = 3.5 cm', 'S2 = 30 cm, S1 = 3.5 cm', 'S2 = 40 cm, S1 = 3.5 cm', 'S2 = 0 cm, S1 = 4 cm', 'S2 = 10 cm, S1 = 4 cm', 'S2 = 20 cm, S1 = 4 cm', 'S2 = 30 cm, S1 = 4 cm', 'S2 = 40 cm, S1 = 4 cm'])

# S2 = {0, 10, 20, 30}, type 1, theta = 0deg, S1 = 4 cm
# plot_ballistic_limit_5x(v, y_S2_0_theta0_S1_4, y_S2_10_theta0_S1_4, y_S2_20_theta0_S1_4, y_S2_30_theta0_S1_4, y_S2_40_theta0_S1_4, labels=['S2 = 0 cm, S1 = 4 cm', 'S2 = 10 cm, S1 = 4 cm', 'S2 = 20 cm, S1 = 4 cm', 'S2 = 30 cm, S1 = 4 cm', 'S2 = 40 cm, S1 = 4 cm'])
plt.legend(['S2 = 0 cm, S1 = 3 cm', 'S2 = 10 cm, S1 = 3 cm', 'S2 = 20 cm, S1 = 3 cm', 'S2 = 30 cm, S1 = 3 cm', 'S2 = 40 cm, S1 = 3 cm', 'S2 = 0 cm, S1 = 4 cm', 'S2 = 10 cm, S1 = 4 cm', 'S2 = 20 cm, S1 = 4 cm', 'S2 = 30 cm, S1 = 4 cm', 'S2 = 40 cm, S1 = 4 cm'])
plt.show()


# NEED TO QUANTIFY!!! the mass increase for +10cm of S2

v_max = [10, 15, 20, 25, 30] # km/s
v_max_value = 10
# full_mass_list = []

# for v_max_value in v_max:

v_list = np.arange(1.0, v_max_value+0.1, 1.0) #km/s
# Find where the value is close to 100
indices = np.where(np.isclose(v_list, v_max_value))

# Extract the first match
if indices[0].size > 0:
    print(indices[0][0])  # Output: 91

conditions_optimum = [v_list < vt11, (v_list >= vt11) & (v_list <= vt21), v_list > vt21]

l = 2 # m
# t_ob_list = [0.03, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1, 1.5, 2, 5, 10] # cm
t_ob_list = [0.03]
# t_b_list = [0.1, 0.15, 0.2, 0.3, 0.5, 1, 0.75, 1.5, 2, 5, 10]
t_b_list = [0.1]
# t_w_list = [0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
t_w_list = np.arange(0.05, 0.55, 0.005) # cm
# t_w_list = [0.1]
# t_b_list = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1, 0.75, 1.5, 2, 5, 10]
# t_w_list = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1, 1.5, 2, 5, 10, 30, 50]
# t_w_list = [0.5, 0.75, 1, 1.5, 2, 5, 10]


# S1_list = np.arange(0.05, 10.5, 0.05) # cm
# S1_list = [3] # cm
# S1_list=[0.5, 1, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 10, 20] # cm
# S2_list=[0.5, 1, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7.5,10, 20, 30, 40, 50, 60]
# S2_list = [0.5]
S2_list = np.arange(0.05, 20.5, 0.05) # cm
# theta_value = np.radians(0)
theta=0

critical_diameter_list = [0.01, 0.015, 0.02, 0.03]

def calculate_mass_for_half_whole_and_margin_sc_shielding(S1, S2, t_ob, t_b, t_w, rho_ob, rho_b, side_length, honeycomb1, honeycomb2):
    mass = (t_ob * rho_ob + t_b * rho_b + t_w * rho_ob) # in cm*g/cm^3 = g/cm^2
    mass = 10*mass # convert to kg/m^2
    mass = mass * 3*side_length**2 
    if honeycomb1:
        mass+= S1*side_length**2*rho_honeycomb1*10 # assumed honeycomb 
    if honeycomb2:
        mass+= S2*side_length**2*rho_honeycomb2*10 # assumed honeycomb 
    mass_half = mass
    mass_whole = mass * 2
    mass_with_margin = mass_whole*1.1 # add 10% margin for the structure of the shield - fixing components/rods for the plate
    return mass_half, mass_whole, mass_with_margin

def calculate_mass_for_whole_sc_shielding(S1, S2, t_ob, t_b, t_w, rho_ob, rho_b, rho_w, side_length):
    mass = (t_ob/100 * rho_ob*1000 + t_b/100 * rho_b*1000 + t_w/100 * rho_w*1000)*side_length**2 # in kg
    mass+= S1/100*side_length**2*rho_honeycomb3*1000
    mass+= S2/100*side_length**2*rho_honeycomb3*1000 # assumed honeycomb
    # mass = mass*6
    # mass = 10*mass # convert to kg/m^2
    # mass = mass * 6*side_length**2 
    # mass*= 2 # add margin for the structure of the shield - fixing components/rods for the plate
    return mass

print(calculate_mass_for_whole_sc_shielding(3, 0.5, 0.03, 0.1, 0.1, rho_ob, rho_b, rho_w, l))

# for S1_element in S1_list:
#         for S2_element in S2_list:
#             for t_ob_element in t_ob_list:
#                 for t_b_element in t_b_list:
#                     for t_w_element in t_w_list:
#                         full_mass_element = calculate_mass_for_whole_sc_shielding(S1_element, S2_element, t_ob_element, t_b_element, t_w_element, rho_ob, rho_b, rho_w, l)
#                         full_mass_list.append(full_mass_element)

# # for S1_element in S1_list:
# #     for t_w_element in t_w_list:
# #         full_mass_element = calculate_mass_for_whole_sc_shielding(S1_element, S2_element, t_ob_element, t_b_element, t_w_element, rho_ob, rho_b, rho_w, l)
# #         full_mass_list.append(full_mass_element)

# sc = plt.scatter(t_w_list, S1_list, c=full_mass_list, cmap='viridis', edgecolor='k')
# # plt.scatter(t_w_list, S1_list, c=full_mass_list, cmap='viridis', s=50)
# plt.xlabel('t_w')
# plt.ylabel('S1')
# plt.title('S1 vs t_w with mass as color')
# cbar = plt.colorbar(sc)
# cbar.set_label('mass')

# 1. Create an empty 2D grid to hold the mass values (Rows = S2, Columns = t_w)
mass_grid = np.zeros((len(S2_list), len(t_w_list)))

S1_element = 3 # cm
# S2_element = 0.5 # cm
t_ob_element = 0.03 # cm
t_b_element = 0.1 # cm

v_max = 10
v_list = np.arange(1.0, v_max+0.1, 1.0) #km/s
# Find where the value is close to 100
indices = np.where(np.isclose(v_list, v_max))
target_value_dc = 0.2 # cm, this is the critical diameter we want to be above for the whole velocity range

# 2. Populate the grid using the indices
for i, S2_element in enumerate(S2_list):
    for j, t_w_element in enumerate(t_w_list):
        y_element = np.piecewise(v_list, conditions_optimum, funcs, S1_element, S2_element, vt11, vt21, theta, t_ob_element, t_b_element, t_w_element)
        if y_element[indices[0][0]]>=target_value_dc:
            mass_grid[i, j] = calculate_mass_for_whole_sc_shielding(
            S1_element, S2_element, t_ob_element, t_b_element, t_w_element, 
            rho_ob, rho_b, rho_w, l)
        else:
            mass_grid[i, j] = np.nan  # or some other value to indicate it's not valid

# 3. Plot using pcolormesh for a perfect 2D heatmap
plt.figure(figsize=(8, 6))
mesh = plt.pcolormesh(t_w_list, S2_list, mass_grid, shading='auto', cmap='viridis')
plt.xlabel('t_w (cm)')
plt.ylabel('S2 (cm)')
plt.title('Honeycomb panel mass (space debris of 2mm diameter, 10km/s velocity) for \n changing S2 and t_w for fixed S1 = 3 cm, t_ob = 0.03 cm, t_b = 0.1 cm')

cbar = plt.colorbar(mesh)
cbar.set_label('Panel mass (kg)')

contours = plt.contour(
    t_w_list, S2_list, mass_grid, levels=[27.5, 30, 35, 40, 45, 60, 65], colors="white", linewidths=0.8
)
# Add inline text labels to the contour lines
plt.clabel(contours, inline=True, fontsize=8, fmt="%.1f kg", colors="white")

# Optional: because t_w jumps from 10 to 50, a log scale on X might look cleaner
# plt.xscale('log') 

specific_tw = [0.1, 0.4]
specific_S2 = [0.5, 0.5]
point_labels = ["Regular Honeycomb Panel", "Baseplate (unmodified) Honeycomb Panel"]
point_colors = ["orange", "yellow"]

for tw, s2, label, color in zip(
    specific_tw, specific_S2, point_labels, point_colors
):
    plt.scatter(
        tw,
        s2,
        color=color,
        edgecolor="black",
        s=120,  # Slightly larger size
        zorder=5,  # Force points to render on top of lines
        label=label,  # Feeds directly into plt.legend()
    )

    # Optional: Keep the text coordinates floating just above the dots
    # plt.text(
    #     tw,
    #     s2 + 0.3,
    #     # f"({tw}, {s2})",
    #     color="white",
    #     weight="bold",
    #     fontsize=9,
    #     ha="center",
    # )

# Plot the points as bright red, larger dots with a black border
# plt.scatter(
#     specific_tw, 
#     specific_S2, 
#     color='orange', 
#     edgecolor='black', 
#     s=100,                  # Size of the marker
#     zorder=5,               # Ensures the dots sit ON TOP of the heatmap
#     label='Target Points'   # Label for a legend
# )

# Optional: Add text labels next to the dots so you know which is which
# for tw, s2 in zip(specific_tw, specific_S2):
#     plt.text(
#         tw, s2 + 0.5,       # Slightly offset the text vertically so it doesn't overlap
#         f'({tw}, {s2})', 
#         color='orange', 
#         weight='bold', 
#         fontsize=9,
#         ha='center'         # Horizontally center the text over the dot
#     )

plt.legend(loc="upper right", framealpha=0.9)
plt.show()

# # Custom stuff:
# # Comet Interceptor: S1 = 5cm, S2 = 2.5cm, t_ob = 0.03 cm, t_b = 0.15 cm, t_w = 0.15 cm, theta = 0deg
# y_Comet_Interceptor = np.piecewise(v, conditions1, funcs, 5, 2.5, vt11, vt21, 0, 0.03, 0.15, 0.15)
# y_ours = np.piecewise(v, conditions1, funcs, 3, 0.5, vt11, vt21, 0, 0.03, 0.1, 0.1)
# y_1 = np.piecewise(v, conditions1, funcs, 5, 2.5, vt11, vt21, 0, 0.03, 1.5, 4)
# mass_Comet_Interceptor_half, mass_Comet_Interceptor_whole, mass_Comet_Interceptor_with_margin = calculate_mass_for_half_whole_and_margin_sc_shielding(5, 2.5, 0.03, 0.15, 0.15, rho_ob, rho_b, l, True, True)
# mass_ours_half, mass_ours_whole, mass_ours_with_margin = calculate_mass_for_half_whole_and_margin_sc_shielding(3, 0.5, 0.03, 0.1, 0.1, rho_ob, rho_b, l, True, True)
# print(f"Comet Interceptor: mass for 1/2 m = {round(mass_Comet_Interceptor_half, 2)} kg, mass for m = {round(mass_Comet_Interceptor_whole, 2)} kg, mass for 1.1m = {round(mass_Comet_Interceptor_with_margin, 2)} kg")
# print(f"Proposed Design: mass for 1/2 m = {round(mass_ours_half, 2)} kg, mass for m = {round(mass_ours_whole, 2)} kg, mass for 1.1m = {round(mass_ours_with_margin, 2)} kg")
# plt.plot(v, y_Comet_Interceptor)
# plt.plot(v, y_ours)
# plt.plot(v, y_1)
# plt.xlabel('Velocity (km/s)')
# plt.ylabel('Critical Diameter (cm)')
# plt.title('Critical Space Debris Diameter vs Velocity for Comet Interceptor vs Proposed Design')
# plt.legend(['Comet Interceptor', 'Proposed Design', 'baseplate'])
# plt.grid()
# plt.show()