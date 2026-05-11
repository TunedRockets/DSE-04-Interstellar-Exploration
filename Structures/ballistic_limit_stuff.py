import numpy as np
import matplotlib.pyplot as plt

S1 = 3.5 # stand-off between 1st and 2nd bumper [cm]
S2 = 0 # stand-off between 2nd bumper and rear wall [cm]
d = 1 # diameter of projectile [cm]
t_w = 0.472 # equipment cover plate thickness
sigma_y_ksi = 37.7 # yield strength of equipment cover plate [ksi], here for R_p0.2 = 260 MPa
t_ob = 0.041
t_b = 0.041

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

theta = 0 # angle of impact, 0 degrees for normal incidence...?

# densities [g/cm^3]
rho_p = 2.7
rho_b = 2.7
rho_AD_MLI = 0.0447 # = m_a_MLI bcs AREA DENSITY!!!
rho_ob = 2.7

t_eq_MLI = rho_AD_MLI/rho_ob

v = np.arange(0, 18.1, 0.1) # km/s
# print(v)


def calculate_d_c(v, S1, S2, vt1, vt2):
    return ((((t_w**alpha+t_b)/K_3S)*(sigma_y_ksi/40)**(1/2)+t_ob+K_MLI*t_eq_MLI)/(0.6*np.cos(theta)**delta*rho_p**(1/2)*v**(2/3)))**(18/19)

def calculate_d_c_larger_than_vt2(v, S1, S2, vt1, vt2):
    return (1.155*(S1**(1/3)*(t_b+K_tw*t_w)**(2/3)+(K_S2*S2**beta)*(t_w**gamma)*np.cos(theta)**(-epsilon))*(sigma_y_ksi/70)**(1/3))/((K_3D**(2/3))*(rho_p**(1/3))*(rho_ob**(1/9))*(v**(2/3))*(np.cos(theta)**delta))

# velocities [km/s] vs type of material 

# TYPE I: Al H/C SP / Al bumper / MLI+Al H/C SP / MLI+Al bumper
vt11 = 3
vt21 = 7
# TYPE II: standalone MLI as structure wall
vt12 = 4
vt22 = 10

def calculate_d_c_failure_rear_wall_in_shatter_veloc_regime(v, S1, S2, vt1, vt2):
    return calculate_d_c(vt1, S1, S2, vt1, vt2) + (calculate_d_c_larger_than_vt2(vt2, S1, S2, vt1, vt2) - calculate_d_c(vt1, S1, S2, vt1, vt2)) / (vt2 - vt1) * (v - vt1)

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
y_S2_0  = np.piecewise(v, conditions1, funcs, S1, 0, vt11, vt21)
y_S2_10 = np.piecewise(v, conditions1, funcs, S1, 10, vt11, vt21)
y_S2_20 = np.piecewise(v, conditions1, funcs, S1, 20, vt11, vt21)

y_S2_0_1  = np.piecewise(v, conditions2, funcs, S1, 0, vt12, vt22)
y_S2_10_1 = np.piecewise(v, conditions2, funcs, S1, 10, vt12, vt22)
y_S2_20_1 = np.piecewise(v, conditions2, funcs, S1, 20, vt12, vt22)

plt.plot(v, y_S2_0)
plt.plot(v, y_S2_10)
plt.plot(v, y_S2_20)
plt.plot(v, y_S2_0_1)
plt.plot(v, y_S2_10_1)
plt.plot(v, y_S2_20_1)
# plt.ylim(0, 1)
plt.xlabel('Velocity (km/s)')
plt.ylabel('Critical Diameter (cm)')
plt.title('Ballistic Limit vs Velocity')
plt.legend(['S2 = 0 cm, type 1', 'S2 = 10 cm, type 1', 'S2 = 20 cm, type 1', 'S2 = 0 cm, type 2', 'S2 = 10 cm, type 2', 'S2 = 20 cm, type 2'])
plt.grid()
plt.show()