import numpy as np

k = 4 # for simply supported plate
E = 70e9 # Young's modulus for aluminum in Pascals
nu = 0.33 # Poisson's ratio for aluminum
t = 0.01 # thickness of the plate in meters
b = 2 # width of the plate in meters 

n_max = 6

sigma_cr = k*np.pi**2*E/(12*(1-nu**2))*(t/b)**2

m = 2000 # mass in kg
g = 9.81 # acceleration due to gravity in m/s^2
F = m*g
sigma_actual = F/t*b 

D = E/100
L = 2
P_cr = np.pi**2*D/L**2
F_cr = P_cr/g/n_max

print(f"Critical buckling stress [Pa]: {sigma_cr:.2f} Pa")
print(f"Actual stress [Pa]: {sigma_actual:.2f} Pa")
print(f"Actual stress [MPa]: {sigma_actual/1e6:.2f} MPa")
print(f"Critical buckling stress [MPa]: {sigma_cr/1e6:.2f} MPa")
print(f"Critical buckling load [N]: {P_cr:.2f} N")
print(f"Maximum load [N]: {F_cr:.2f} N")
