import numpy as np
# import sympy as sp

rho = 1600
E = 140e9
r1 = 0.106
r2 = 0.107
I = np.pi/4*(r2**2-r1**2)
l = 10
A1 = np.pi*r2**2
A2 = np.pi*(r2**2-r1**2)
m_boom = rho*A2*l
print(m_boom)
mu = 46.7

# matrix = np.array([[np.cosh(alpha*l)+np.cos(alpha*l), -mu*l*alpha*np.cosh(alpha*l)+mu*l*alpha*np.cos(alpha*l)+np.sinh(alpha*l)-np.sin(alpha*l)], [np.sinh(alpha*l)+np.sin(alpha*l), -mu*l*alpha*np.sinh(alpha*l)+mu*l*alpha*sin(alpha*l)-np.cos(alpha*l)-np.cosh(alpha*l)]])

# def matrix_fct(alpha):
#     return np.array([[np.cosh(alpha*l)+np.cos(alpha*l), -mu*l*alpha*np.cosh(alpha*l)+mu*l*alpha*np.cos(alpha*l)+np.sinh(alpha*l)-np.sin(alpha*l)], [np.sinh(alpha*l)+np.sin(alpha*l), -mu*l*alpha*np.sinh(alpha*l)+mu*l*alpha*sin(alpha*l)-np.cos(alpha*l)-np.cosh(alpha*l)]])
# # Define the symbolic variable
# x = sp.Symbol('x')

# # Define the matrix with the variable
# M = matrix_fct(x)

# # Compute the determinant
# determinant = M.det()
# print(f"Determinant equation: {determinant} = 0")

# # Solve for x
# solutions = sp.solve(determinant, x)
# print("Solutions for x:", solutions)




# import numpy as np
# import sympy as sp

# # 1. Physical Constants
# rho = 1600
# E = 140e9
# r1 = 0.106
# r2 = 0.107
# I = np.pi / 4 * (r2**2 - r1**2)
# l = 10
# A1 = np.pi * r2**2
# A2 = np.pi * (r2**2 - r1**2)
# m_boom = rho * A2 * l
# mu = 46.7

# # 2. Symbolic Formulation
# x = sp.Symbol('x')

# M = sp.Matrix([
#     [
#         sp.cosh(x*l) + sp.cos(x*l), 
#         -mu*l*x*sp.cosh(x*l) + mu*l*x*sp.cos(x*l) + sp.sinh(x*l) - sp.sin(x*l)
#     ], 
#     [
#         sp.sinh(x*l) + sp.sin(x*l), 
#         -mu*l*x*sp.sinh(x*l) + mu*l*x*sp.sin(x*l) - sp.cos(x*l) - sp.cosh(x*l)
#     ]
# # ])

# # Compute and simplify the determinant
# determinant = sp.simplify(M.det())

# # 3. Solve for alpha and calculate frequencies
# guesses = [0.1, 0.5, 1.0]
# print(f"{'Mode':<5} | {'Alpha (x)':<15} | {'Omega 1 (rad/s)':<18} | {'Omega 2 (rad/s)':<18}")
# print("-" * 65)

# for i, guess in enumerate(guesses, 1):
#     try:
#         # Find root numerically as a float
#         alpha_val = float(sp.nsolve(determinant, x, guess))
#         alpha_times_l = alpha_val * l
        
#         # Calculate omega values using your formulas
#         omega1 = (alpha_times_l**2) * np.sqrt((E * I) / (rho * A1 * l**2))
#         omega2 = (alpha_times_l**2) * np.sqrt((E * I) / (rho * A2 * l**2))
        
#         print(f"n={i}   | {alpha_val:<15.6f} | {omega1:<18.4f} | {omega2:<18.4f}")
#     except ValueError:
#         print(f"Mode {i}: No root found near guess {guess}")

alpha = 0.0563 # for mu = 46.7
alpha_2 = 0.038555 # for mu = 94.7
alpha_times_l = alpha*l
omega1 = alpha_times_l**2*np.sqrt(E*I/(rho*A1*l**2))
omega2 = alpha_times_l**2*np.sqrt(E*I/(rho*A2*l**2))
omega1_2 = (alpha_2*l)**2*np.sqrt(E*I/(rho*A1*l**2))
omega2_2 = (alpha_2*l)**2*np.sqrt(E*I/(rho*A2*l**2))
print(omega1, omega2)
print(omega1_2, omega2_2)