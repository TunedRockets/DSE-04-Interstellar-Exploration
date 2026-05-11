import numpy as np
import matplotlib.pyplot as plt

# S1 = 3.5 # stand-off between 1st and 2nd bumper [cm]
S1 = 4 # distance between 1st (outer) and 2nd (inner) bumper [cm]
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

t_eq_MLI = rho_AD_MLI/rho_ob

v = np.arange(0, 30.1, 0.1) # km/s
# print(v)


def calculate_d_c(v, S1, S2, vt1, vt2, theta):
    return ((((t_w**alpha+t_b)/K_3S)*(sigma_y_ksi/40)**(1/2)+t_ob+K_MLI*t_eq_MLI)/(0.6*np.cos(theta)**delta*rho_p**(1/2)*v**(2/3)))**(18/19)

def calculate_d_c_larger_than_vt2(v, S1, S2, vt1, vt2, theta):
    return (1.155*(S1**(1/3)*(t_b+K_tw*t_w)**(2/3)+(K_S2*S2**beta)*(t_w**gamma)*np.cos(theta)**(-epsilon))*(sigma_y_ksi/70)**(1/3))/((K_3D**(2/3))*(rho_p**(1/3))*(rho_ob**(1/9))*(v**(2/3))*(np.cos(theta)**delta))

# velocities [km/s] vs type of material 

# TYPE I: basically Al bumper: Al H/C SP / Al bumper / MLI+Al H/C SP / MLI+Al bumper
vt11 = 3
vt21 = 7
# TYPE II: standalone MLI as structure wall
vt12 = 4
vt22 = 10

def calculate_d_c_failure_rear_wall_in_shatter_veloc_regime(v, S1, S2, vt1, vt2, theta):
    return calculate_d_c(vt1, S1, S2, vt1, vt2, theta) + (calculate_d_c_larger_than_vt2(vt2, S1, S2, vt1, vt2, theta) - calculate_d_c(vt1, S1, S2, vt1, vt2, theta)) / (vt2 - vt1) * (v - vt1)

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
# theta = 0

# type 1
y_S2_0_theta0  = np.piecewise(v, conditions1, funcs, 3.5, 0, vt11, vt21, theta=np.radians(0))
y_S2_10_theta0 = np.piecewise(v, conditions1, funcs, 3.5, 10, vt11, vt21, theta=np.radians(0))
y_S2_20_theta0 = np.piecewise(v, conditions1, funcs, 3.5, 20, vt11, vt21, theta=np.radians(0))
y_S2_30_theta0 = np.piecewise(v, conditions1, funcs, 3.5, 30, vt11, vt21, theta=np.radians(0))
y_S2_40_theta0 = np.piecewise(v, conditions1, funcs, 3.5, 40, vt11, vt21, theta=np.radians(0))

# type 2
y_S2_0_1_theta0  = np.piecewise(v, conditions2, funcs, 3.5, 0, vt12, vt22, theta=np.radians(0))
y_S2_10_1_theta0 = np.piecewise(v, conditions2, funcs, 3.5, 10, vt12, vt22, theta=np.radians(0))
y_S2_20_1_theta0 = np.piecewise(v, conditions2, funcs, 3.5, 20, vt12, vt22, theta=np.radians(0))
y_S2_30_1_theta0 = np.piecewise(v, conditions2, funcs, 3.5, 30, vt12, vt22, theta=np.radians(0))


# theta = 45

# type 1
y_S2_0_2_theta45  = np.piecewise(v, conditions1, funcs, 3.5, 0, vt11, vt21, theta=np.radians(45))
y_S2_10_2_theta45 = np.piecewise(v, conditions1, funcs, 3.5, 10, vt11, vt21, theta=np.radians(45))
y_S2_20_2_theta45 = np.piecewise(v, conditions1, funcs, 3.5, 20, vt11, vt21, theta=np.radians(45))
y_S2_30_2_theta45 = np.piecewise(v, conditions1, funcs, 3.5, 30, vt11, vt21, theta=np.radians(45))

# type 2
y_S2_0_3_theta45  = np.piecewise(v, conditions2, funcs, 3.5, 0, vt12, vt22, theta=np.radians(45))
y_S2_10_3_theta45 = np.piecewise(v, conditions2, funcs, 3.5, 10, vt12, vt22, theta=np.radians(45))
y_S2_20_3_theta45 = np.piecewise(v, conditions2, funcs, 3.5, 20, vt12, vt22, theta=np.radians(45))
y_S2_30_3_theta45 = np.piecewise(v, conditions2, funcs, 3.5, 30, vt12, vt22, theta=np.radians(45))

# S1 = 4 cm
# theta = 0
# type 1
y_S2_0_theta0_S1_4  = np.piecewise(v, conditions1, funcs, 4, 0, vt11, vt21, theta=np.radians(0))
y_S2_10_theta0_S1_4 = np.piecewise(v, conditions1, funcs, 4, 10, vt11, vt21, theta=np.radians(0))
y_S2_20_theta0_S1_4 = np.piecewise(v, conditions1, funcs, 4, 20, vt11, vt21, theta=np.radians(0))
y_S2_30_theta0_S1_4 = np.piecewise(v, conditions1, funcs, 4, 30, vt11, vt21, theta=np.radians(0))
y_S2_40_theta0_S1_4 = np.piecewise(v, conditions1, funcs, 4, 40, vt11, vt21, theta=np.radians(0))

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
plt.plot(v, y_S2_0_2_theta45)
plt.plot(v, y_S2_10_2_theta45)
plt.plot(v, y_S2_20_2_theta45)
plt.plot(v, y_S2_0_3_theta45)
plt.plot(v, y_S2_10_3_theta45)
plt.plot(v, y_S2_20_3_theta45)
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
plot_ballistic_limit_5x(v, y_S2_0_theta0_S1_4, y_S2_10_theta0_S1_4, y_S2_20_theta0_S1_4, y_S2_30_theta0_S1_4, y_S2_40_theta0_S1_4, labels=['S2 = 0 cm, S1 = 4 cm', 'S2 = 10 cm, S1 = 4 cm', 'S2 = 20 cm, S1 = 4 cm', 'S2 = 30 cm, S1 = 4 cm', 'S2 = 40 cm, S1 = 4 cm'])
plt.legend(['S2 = 0 cm, S1 = 3.5 cm', 'S2 = 10 cm, S1 = 3.5 cm', 'S2 = 20 cm, S1 = 3.5 cm', 'S2 = 30 cm, S1 = 3.5 cm', 'S2 = 40 cm, S1 = 3.5 cm', 'S2 = 0 cm, S1 = 4 cm', 'S2 = 10 cm, S1 = 4 cm', 'S2 = 20 cm, S1 = 4 cm', 'S2 = 30 cm, S1 = 4 cm', 'S2 = 40 cm, S1 = 4 cm'])
plt.show()


# Custom stuff:
# Comet Interceptor: S1 = 5cm, S2 = 2.5cm, t_ob = 0.03 cm, t_b = 0.15 cm, t_w = 0.15 cm, theta = 0deg
y_Comet_Interceptor = np.piecewise(v, conditions1, funcs, 5, 2.5, vt11, vt21, theta=np.radians(0))
plt.plot(v, y_Comet_Interceptor)
plt.xlabel('Velocity (km/s)')
plt.ylabel('Critical Diameter (cm)')
plt.title('Ballistic Limit vs Velocity for Comet Interceptor')
plt.grid()
plt.show()

# NEED TO QUANTIFY!!! the mass increase for +10cm of S2
