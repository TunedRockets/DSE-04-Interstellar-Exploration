''' 
Figure out the massof stuff via N2 convergence.

Stealing bits and pieces from the other code

'''
import math as m
from Power.powerinsizeout import reactor
from ReactoPy.CycloPy import size_power, radiator_areal_density
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm
# from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Pool
import os
from pathlib import Path
import jkat


# import psutil
#
# p = psutil.Process()
#
# # P-cores only (typical mapping for 13650HX)
# p.cpu_affinity(list(range(0, 12)))
# ==== consts =====

static_mass = 50+100+200
'''[kg] mass of scientific payload, antenna, bus and oter non-varying things'''
static_power_draw = 1600
'''[w] static power draw of non-propulsion equipment'''
static_area = (2.2**2)*m.pi + 2*2
'''[m^2] static exposed area of bus, antenna, etc.'''

# ion system: (http://large.stanford.edu/courses/2025/ph240/tuckey1/docs/nasa-nov17.pdf)
Isp_ion = 4220
'''[s] ion drive isp'''
dV_inclination = 3000


'''[m/s] dv for the inclination change maneuver'''
dV_rdvz = 17_000

'''[m/s] dv for the rendezvous'''
dV_ion = dV_rdvz + dV_inclination
'''[m/s] total dv required by ion system'''
Me_ion = 15 + 36 # NEXT thruster mass
'''[kg] ion engine mass'''
P_ion = 7400
'''[w] power per ion engine'''
F_ion = 0.235
'''[N] thrust per ion engine'''
T_max_inclination = 86_000*365*1.31
'''max time spent on inclination burn'''
R = 8.31446261815324
M_xenon = 0.131293
R_xenon = R/M_xenon
propellant_margin = 2/100

xenon_tank_pressure = 187*1e5
xenon_tank_temp = 273.15+20
xenon_density=xenon_tank_pressure/(R_xenon*xenon_tank_temp)

T_max_inclination = 600*jkat.DAY  # changed from Andres estimate to more pessimistic value


a_min_ion = dV_inclination/T_max_inclination
'''[m/s^2] minimum acceleration of the ion engines'''
l_ion = 0.05
'''[-] ion tank mass fraction'''

# boost system:
Isp_boost = 330
'''[s] boost drive isp'''
dV_boost = 4_000

'''[m/s] total dv required by boost system'''
Me_boost = 100
'''[kg] boost engine mass'''
l_boost = 0.05
'''[-] boost tank mass fraction'''

# heat shield:
rho_heat = 152 # reverse engineers from parker solar probe numbers
'''[kg/m^3] heat shield density'''
t_heat = 0.11
'''[m] heat shield thickness'''
A_heat_margin = 1.1
'''Heat shield area margin (for overhang, etc.)'''


# reactor:
Psp_nuke = 134
'''[w/kg] reactor power density'''

# Stefan–Boltzmann constant
sigma = 5.670374419e-8  # W/m^2/K^4

# Inputs
# T_cold = 1298.0679247865448  # K
areal_density = radiator_areal_density           # kg/m^2
# emissivity = 0.9

# Power areal density (W/m^2)
# q = emissivity * sigma * T_cold**4

# Specific power (W/kg)
# rad_specific_power = q / areal_density


def dv2mf(dV:float, isp:float, m1:float, l:float)->float:
    '''dv [in km/s], specific impulse, non-tank-mass, 
    structural mass fraction to fuel mass'''
    ve = 9.80665 * isp
    e = m.exp(dV/ve)
    mf = m1*(e-1)/(1+l-l*e) # fuel mass
    return mf

@staticmethod
class Hestia():
    '''this is the design to configure, as a class,
    each variable has a method to set itself, which is run through
    every iteration. once iterations have converged it will terminate'''

    def __init__(
            self,
            dV_inclination=dV_inclination,
            dV_rdvz=dV_rdvz,
            dV_boost=dV_boost,
            verbose=False,
            convergence_tolerance=1e-8
    ):
        self.dV_inclination = dV_inclination
        self.dV_rdvz = dV_rdvz
        self.dV_boost = dV_boost
        self.verbose = verbose
        self.convergence_tolerance = convergence_tolerance

        # the varying variables 
        self.Mass_ion = 51
        '''the ion engines and tanks (not fuel)'''
        self.Mass_ion_fuel = 200
        '''fuel mass of xenon'''
        self.Area_heatshield = 21
        '''area of the heat shield'''
        self.Mass_boost = 106
        '''the boost stage, engines and tanks (not heat shield, or fuel)'''
        self.Mass_boost_fuel = 200
        '''fuel mass of MON/MMH or w/e we're using'''
        self.Mass_power_truss = 58
        '''mass of nuke, truss, and radiators'''
        self.Power_provided = 0
        '''power provided by the nuke truss'''
        self.Number_ions = 1
        '''Number of ion engines'''

    def __repr__(self) -> str:
        return (
            '--- Hestia configuration: ---\n'
            f'payload mass: {self.upper_stage_pl_mass:6.1f} kg\n'
            f'ion dry mass: {self.upper_stage_dry_mass:6.1f} kg\n'
            f'ion wet mass: {self.upper_stage_wet_mass:6.1f} kg\n'
            '---\n'
            f'boost sys mass: {self.lower_stage_pl_mass:6.1f} kg\n'
            f'boost dry mass: {self.lower_stage_dry_mass:6.1f} kg\n'
            f'boost wet mass: {self.lower_stage_wet_mass:6.1f} kg\n'
            '---\n'
            f'{self.Number_ions} ion engines\n'
            f'inclination burn time: {self.inclination_burn_time/86_000:3.2f} days\n'
            f'rendezvous burn time: {self.rdvz_burn_time/86_000:3.2f} days\n'
            f'{self.Power_provided:6.1f} W used from reactor with mass {self.Mass_power_truss:6.1f} kg\n'
            '---\n'
            f'total heat shield area of {self.Area_heatshield:3.3f} m^2, with mass {self.Mass_heatshield:6.1f} kg'
        )


    def _converge(self, max_iter:int=1000):
        '''run the convergence'''
        try:
            for _ in range(max_iter):

                if self._iterate():
                    if self.verbose:
                        print('\n!!! conversion finished !!!\n\n\n\n')
                        print(self)
                    return
            else:
                raise TimeoutError("Did not converge in time")
        except(ValueError,ArithmeticError,TypeError): raise ValueError("Error in divergence!")  


    def _iterate(self)->bool:
        '''runs through all iteration methods'''

        var_dict = self.__dict__.copy()
        if self.verbose:
            print("\n====== New iteration =====\n")
        mydir = dir(self)
        myfuncs = [fn for fn in mydir if callable(getattr(self, fn))]
        myfuncs = [fn for fn in myfuncs if fn.startswith('size')]
        for fn in myfuncs: # all the sizing funcs
            fn_call = getattr(self,fn)
            fn_call() # call function

        # check for convergence
        converged = True
        for key, value in var_dict.items():

            if abs(value - self.__dict__[key]) > self.convergence_tolerance:
                converged = False
        return converged

    # property methods are fine as long as there are no side-effects

    @property
    def upper_stage_pl_mass(self):
        '''mass of upper stage w/o propulsion system'''
        return static_mass + self.Mass_power_truss
    
    @property
    def upper_stage_dry_mass(self):
        '''mass of upper stage w/o propulsion system'''
        return self.upper_stage_pl_mass + self.Mass_ion

    @property
    def upper_stage_wet_mass(self):
        '''mass of upper stage w/o propulsion system'''
        return self.upper_stage_dry_mass + self.Mass_ion_fuel

    @property
    def lower_stage_pl_mass(self):
        '''mass of lower stage w/o prop system (upper stage and heat shield)'''
        return self.upper_stage_wet_mass + self.Mass_heatshield
    
    @property
    def lower_stage_dry_mass(self):
        '''mass of upper stage w/o propulsion system'''
        return self.lower_stage_pl_mass + self.Mass_boost

    @property
    def lower_stage_wet_mass(self):
        '''mass of upper stage w/o propulsion system'''
        return self.lower_stage_dry_mass + self.Mass_boost_fuel

    @property
    def total_mass(self):
        return self.lower_stage_wet_mass

    @property
    def Mass_heatshield(self):
        '''mass of heat shield'''
        return self.Area_heatshield * t_heat * rho_heat

    @property
    def inclination_burn_time(self):
        '''pessemistic estimate of burn time'''
        return self.dV_inclination/(self.Number_ions*F_ion/self.lower_stage_wet_mass)
    
    @property
    def rdvz_burn_time(self):
        '''pessemistic estimate of burn time'''
        return self.dV_rdvz/(self.Number_ions*F_ion/self.upper_stage_wet_mass)


    def size_ion_system(self):
        '''size the ion system and figure out number of engines and power draw'''

        # get no. engines and their mass:
        F_need = self.lower_stage_wet_mass*a_min_ion
        self.Number_ions = m.ceil(F_need/F_ion)
        # set new ion mass:
        self.Mass_ion = (l_ion*self.Mass_ion_fuel) + self.Number_ions*Me_ion


        m_rdzv = dv2mf(self.dV_rdvz, Isp_ion, self.upper_stage_pl_mass+ self.Number_ions*Me_ion, l_ion)

        m_plane = dv2mf(self.dV_inclination, Isp_ion, self.lower_stage_dry_mass + ((1+l_ion) * m_rdzv) + self.Number_ions * Me_ion, l_ion)

        mf = m_plane + m_rdzv
        mf = (1+propellant_margin)*mf
        self.Mass_ion_fuel = mf
        ion_fuel_tank_volume = mf/xenon_density
        if self.verbose:
            print(f"ion engine number: {self.Number_ions}"  )
            print(f"Xenon tank: {self.Mass_ion_fuel} kg, ", f"{ion_fuel_tank_volume} m3")

    def size_boost_system(self):
        '''size boost fuel tank and rest'''

        m1 = self.lower_stage_pl_mass + Me_boost
        mf = dv2mf(self.dV_boost, Isp_boost, m1, l_boost)
        self.Mass_boost_fuel = mf
        if self.verbose:
            print(f'boost fuel: {self.Mass_boost_fuel:5.1f} kg, total wet mass: {self.lower_stage_wet_mass:5.1f} kg')

    def size_power_system(self):
        '''uses only simple power density, include better system later'''

        Preq = static_power_draw + self.Number_ions*P_ion # needed power

        mass, reactor_mass, radiator_mass, brayton_system_mass, thermal_power, radiator_area = size_power(Preq)

        self.Mass_power_truss = mass
        if self.verbose:
            print(f'reactor truss weight: {self.Mass_power_truss:5.1f} kg, generating: {Preq:5.1f} W')
            print(f'thermal power: {thermal_power:5.1f} W')
            print(f'electric power: {Preq:5.1f} W')
            print(f'radiator mass: {radiator_mass:5.1f} kg')
            print(f'reactor mass: {reactor_mass:5.1f} kg')
            print(f'radiator area: {radiator_mass/areal_density:5.1f} m2')

        self.Power_provided = Preq

    def size_heat_shield(self):
        '''uses very simple model, include better system later'''

        A = static_area

        # cylinder mass to area:
        # V = pi*r*r*h
        # A = 2*r*h
        # V = m/rho
        # ==>
        # A = 2*h*sqrt(V/(pi*h))
        # ==> 
        A_fn = lambda mass,h,rho: 2*m.sqrt(mass*h/(m.pi*rho))

        # VERY APPROXIAMTE VALUES!!! CHANGE

        A += A_fn(self.Mass_power_truss,9,7000)
        # power truss approximated as cylinder half ariane6 fairing
        # with density of steel (average of reactor + truss)

        A += A_fn(self.Mass_boost_fuel,3, 1000) # 3 m cyliner of fuel
        A += A_fn(self.Mass_ion_fuel,2,xenon_density) # 1 m cyliner of xenon

        A *= A_heat_margin # margin

        self.Area_heatshield = A
        self.Mass_heatshield
        if self.verbose:
            print(f'heat shield area is: {A:3.2f} m^2 with a mass of {self.Mass_heatshield:6.1f} kg')


def _single_run(args):
    """Worker function (must be top-level for multiprocessing)."""
    i, j, k, dv_inc, dv_rdvz, dv_boost = args

    sc = Hestia(
        dV_inclination=dv_inc,
        dV_rdvz=dv_rdvz,
        dV_boost=dv_boost,
        convergence_tolerance=0.001
    )
    try:
        sc._converge()
        total_mass = sc.lower_stage_wet_mass
    except:
        total_mass = np.nan
    # print()
    # print("Result completed!")
    # print("i", i)
    # print("j", j)
    # print("k", k)
    # print()
    # print()
    # print("Result completed!")
    # print("dV Inclination: ", dv_inc)
    # print("dV Rendezvous: ", dv_rdvz)
    # print("dV Boost", dv_boost)
    # print("Total mass: ", total_mass)
    # print()

    return i, j, k, total_mass

# from concurrent.futures import ProcessPoolExecutor, as_completed

def generate_mass_database(dVs_incl, dVs_rdvz, dVs_boost):

    jobs = [
        (i, j, k, dv_inc, dv_rdvz, dv_boost)
        for i, dv_inc in enumerate(dVs_incl)
        for j, dv_rdvz in enumerate(dVs_rdvz)
        for k, dv_boost in enumerate(dVs_boost)
    ]

    masses = np.zeros((len(dVs_incl), len(dVs_rdvz), len(dVs_boost)))

    # pbar = tqdm(total=len(jobs), desc="Mass DB")

    with Pool() as p:
        results = tqdm(p.imap_unordered(_single_run, jobs,5), desc="Jobs completed: ", total=len(jobs))

        for result in results:
            i, j, k, mass = result
            # print("Result appended!")
            masses[i, j,k] = mass
            # pbar.update(1)

    print("Run complete!")

    # pbar.close()

    data = {
        "dV_inclination": dVs_incl,
        "dV_rdvz": dVs_rdvz,
        "dV_boost": dVs_boost,
        "mass": masses,
    }
    path = Path(__file__).parent / "mass_database.pkl"
    with open(path, "wb") as f:
        pickle.dump(data, f)

    return data

import plotly.graph_objects as go

def plot_mass_database(data):
    """
    Slice on inclination.
    Axes:
        X = rendezvous ΔV
        Y = boost ΔV
        Z = mass
    Slider:
        inclination ΔV
    """

    X, Y = np.meshgrid(
        data["dV_rdvz"],
        data["dV_boost"],
        indexing="ij"
    )

    frames = []

    for i in range(len(data["dV_inclination"])):

        Z = data["mass"][i, :, :]   # (rdvz, boost)

        frames.append(
            go.Frame(
                data=[
                    go.Surface(
                        x=X,
                        y=Y,
                        z=Z
                    )
                ],
                name=str(i)
            )
        )

    fig = go.Figure(
        data=[
            go.Surface(
                x=X,
                y=Y,
                z=data["mass"][0, :, :]
            )
        ],
        frames=frames
    )

    steps = [
        dict(
            method="animate",
            args=[
                [str(i)],
                dict(
                    mode="immediate",
                    frame=dict(duration=0, redraw=True),
                    transition=dict(duration=0)
                )
            ],
            label=f"{data['dV_inclination'][i]:.0f} m/s"
        )
        for i in range(len(data["dV_inclination"]))
    ]

    fig.update_layout(
        title="Mass vs ΔV (inclination slice)",
        scene=dict(
            xaxis_title="Rendezvous ΔV [m/s]",
            yaxis_title="Boost ΔV [m/s]",
            zaxis_title="Mass [kg]"
        ),
        sliders=[
            dict(
                active=0,
                currentvalue={"prefix": "Inclination ΔV: "},
                pad={"t": 50},
                steps=steps
            )
        ]
    )

    fig.show()

def plot_mass_database_2(data):
    """
    Slice on rendezvous ΔV.
    Axes:
        X = inclination ΔV
        Y = boost ΔV
        Z = mass
    Slider:
        rendezvous ΔV
    """

    X, Y = np.meshgrid(
        data["dV_inclination"],
        data["dV_boost"],
        indexing="ij"
    )

    frames = []

    for j in range(len(data["dV_rdvz"])):

        Z = data["mass"][:, j, :]   # (inclination, boost)

        frames.append(
            go.Frame(
                data=[
                    go.Surface(
                        x=X,
                        y=Y,
                        z=Z
                    )
                ],
                name=str(j)
            )
        )

    fig = go.Figure(
        data=[
            go.Surface(
                x=X,
                y=Y,
                z=data["mass"][:, 0, :]
            )
        ],
        frames=frames
    )

    steps = [
        dict(
            method="animate",
            args=[
                [str(j)],
                dict(
                    mode="immediate",
                    frame=dict(duration=0, redraw=True),
                    transition=dict(duration=0)
                )
            ],
            label=f"{data['dV_rdvz'][j]:.0f} m/s"
        )
        for j in range(len(data["dV_rdvz"]))
    ]

    fig.update_layout(
        title="Mass vs ΔV (rendezvous slice)",
        scene=dict(
            xaxis_title="Inclination ΔV [m/s]",
            yaxis_title="Boost ΔV [m/s]",
            zaxis_title="Mass [kg]"
        ),
        sliders=[
            dict(
                active=0,
                currentvalue={"prefix": "Rendezvous ΔV: "},
                pad={"t": 50},
                steps=steps
            )
        ]
    )

    fig.show()



from scipy.interpolate import RegularGridInterpolator
import pickle

path = Path(__file__).parent / "mass_database.pkl"
def load_mass_database(filename=path):
    """
    Load a precomputed Hestia mass database.

    Returns
    -------
    dict with:
        dV_inclination : np.ndarray
        dV_rdvz        : np.ndarray
        dV_boost       : np.ndarray
        mass           : np.ndarray (3D grid)
    """

    with open(filename, "rb") as f:
        data = pickle.load(f)

    required_keys = ["dV_inclination", "dV_rdvz", "dV_boost", "mass"]

    missing = [k for k in required_keys if k not in data]
    if missing:
        raise KeyError(f"Missing keys in database file: {missing}")

    # Ensure numpy arrays (pickle sometimes preserves weird types)
    data["dV_inclination"] = np.asarray(data["dV_inclination"])
    data["dV_rdvz"] = np.asarray(data["dV_rdvz"])
    data["dV_boost"] = np.asarray(data["dV_boost"])
    data["mass"] = np.asarray(data["mass"])

    # Basic sanity check
    expected_shape = (
        len(data["dV_inclination"]),
        len(data["dV_rdvz"]),
        len(data["dV_boost"])
    )

    if data["mass"].shape != expected_shape:
        raise ValueError(
            f"Mass array shape mismatch.\n"
            f"Expected {expected_shape}, got {data['mass'].shape}"
        )

    return data

class MassInterpolator:

    def __init__(self, filename=path):

        with open(filename, "rb") as f:
            data = pickle.load(f)

        self.interp = RegularGridInterpolator(
            (
                data["dV_inclination"],
                data["dV_rdvz"],
                data["dV_boost"]
            ),
            data["mass"]
        )

    def mass(self, dV_inclination, dV_rdvz, dV_boost):

        point = np.array([
            dV_inclination,
            dV_rdvz,
            dV_boost
        ])

        val = self.interp(point)
        return float(np.asarray(val).squeeze())


import random



def _test_mass_database(
        data,
        n_tests=20,
        tolerance=1e-4
):
    """
    Randomly sample grid points and verify:

    1. Stored database value
       == fresh Hestia convergence

    2. Interpolator value
       == fresh Hestia convergence

    Parameters
    ----------
    data : loaded mass database
    n_tests : int
        Number of random points to test
    tolerance : float
        Maximum allowed mass difference [kg]
    """

    interp = MassInterpolator()

    n_incl = len(data["dV_inclination"])
    n_rdvz = len(data["dV_rdvz"])
    n_boost = len(data["dV_boost"])

    failures = []

    for test_no in range(n_tests):

        i = random.randrange(n_incl)
        j = random.randrange(n_rdvz)
        k = random.randrange(n_boost)

        dv_inc = data["dV_inclination"][i]
        dv_rdvz = data["dV_rdvz"][j]
        dv_boost = data["dV_boost"][k]

        stored = data["mass"][i, j, k]

        sc = Hestia(
            dV_inclination=dv_inc,
            dV_rdvz=dv_rdvz,
            dV_boost=dv_boost,
            convergence_tolerance=1e-6
        )

        sc._converge()

        rerun = sc.lower_stage_wet_mass

        interpolated = interp.mass(
            dv_inc,
            dv_rdvz,
            dv_boost
        )

        db_error = abs(stored - rerun)
        interp_error = abs(interpolated - rerun)

        print(
            f"Test {test_no+1:02d} | "
            f"(i,j,k)=({i},{j},{k}) | "
            f"(inc, rdvz, boost)=({round(dv_inc)},{round(dv_rdvz)},{round(dv_boost)}) | "
            f"Mass={rerun:.3f} | "
            f"DB err={db_error:.3e} kg | "
            f"Interp err={interp_error:.3e} kg"
        )

        if (
            db_error > tolerance
            or interp_error > tolerance
        ):
            failures.append(
                {
                    "ijk": (i, j, k),
                    "stored": stored,
                    "rerun": rerun,
                    "interpolated": interpolated,
                    "db_error": db_error,
                    "interp_error": interp_error
                }
            )

    print()
    print("=" * 60)

    if failures:
        print(f"FAILED: {len(failures)} / {n_tests}")

        for f in failures:
            print()
            print(f["ijk"])
            print("stored      :", f["stored"])
            print("rerun       :", f["rerun"])
            print("interpolated:", f["interpolated"])
            print("db_error    :", f["db_error"])
            print("interp_error:", f["interp_error"])

        raise AssertionError(
            f"{len(failures)} tests exceeded tolerance"
        )

    else:
        print(f"PASSED: {n_tests}/{n_tests}")

def _test_interpolator_no_nans(
    data,
    n_tests=100_000,
    seed=42,
):
    rng = np.random.default_rng(seed)

    interp = MassInterpolator()

    inc_min, inc_max = data["dV_inclination"][0], data["dV_inclination"][-1]
    rdv_min, rdv_max = data["dV_rdvz"][0], data["dV_rdvz"][-1]
    boo_min, boo_max = data["dV_boost"][0], data["dV_boost"][-1]

    # First make sure the database itself is clean
    n_grid_nans = np.isnan(data["mass"]).sum()

    print(f"NaNs in source grid: {n_grid_nans}")

    if n_grid_nans:
        raise AssertionError(
            f"Source grid contains {n_grid_nans} NaNs"
        )

    for i in range(n_tests):

        point = np.array([
            rng.uniform(inc_min, inc_max),
            rng.uniform(rdv_min, rdv_max),
            rng.uniform(boo_min, boo_max),
        ])

        val = interp.interp(point)

        if np.isnan(val).any():
            raise AssertionError(
                f"NaN returned at point {point}"
            )

    print(f"PASSED: {n_tests} random in-range points")

def _test_all_grid_points(data):

    interp = MassInterpolator()

    for inc in data["dV_inclination"]:
        for rdvz in data["dV_rdvz"]:
            for boost in data["dV_boost"]:

                val = interp.interp([inc, rdvz, boost])

                if np.isnan(val).any():
                    raise AssertionError(
                        f"NaN at exact grid point "
                        f"({inc}, {rdvz}, {boost})"
                    )

    print("PASSED: all grid points")



def plot_interp_heatmap(
    fixed_axis,
    fixed_value,
    resolution=200,
):
    """
    Plot interpolated mass heatmap while fixing one DV.

    fixed_axis:
        "inclination"
        "rdvz"
        "boost"

    fixed_value:
        value of fixed DV [m/s]
    """

    interp = MassInterpolator()

    data = load_mass_database()

    inc_min = data["dV_inclination"][0]
    inc_max = data["dV_inclination"][-1]

    rdvz_min = data["dV_rdvz"][0]
    rdvz_max = data["dV_rdvz"][-1]

    boost_min = data["dV_boost"][0]
    boost_max = data["dV_boost"][-1]

    if fixed_axis == "inclination":

        x = np.linspace(rdvz_min, rdvz_max, resolution)
        y = np.linspace(boost_min, boost_max, resolution)

        X, Y = np.meshgrid(x, y)

        Z = np.empty_like(X)

        for i in range(resolution):
            for j in range(resolution):

                Z[i, j] = interp.mass(
                    fixed_value,
                    X[i, j],
                    Y[i, j]
                )

        xlabel = "Rendezvous ΔV [m/s]"
        ylabel = "Boost ΔV [m/s]"
        title = f"Inclination ΔV fixed = {fixed_value:.0f} m/s"

    elif fixed_axis == "rdvz":

        x = np.linspace(inc_min, inc_max, resolution)
        y = np.linspace(boost_min, boost_max, resolution)

        X, Y = np.meshgrid(x, y)

        Z = np.empty_like(X)

        for i in range(resolution):
            for j in range(resolution):

                Z[i, j] = interp.mass(
                    X[i, j],
                    fixed_value,
                    Y[i, j]
                )

        xlabel = "Inclination ΔV [m/s]"
        ylabel = "Boost ΔV [m/s]"
        title = f"Rendezvous ΔV fixed = {fixed_value:.0f} m/s"

    elif fixed_axis == "boost":

        x = np.linspace(inc_min, inc_max, resolution)
        y = np.linspace(rdvz_min, rdvz_max, resolution)

        X, Y = np.meshgrid(x, y)

        Z = np.empty_like(X)

        for i in range(resolution):
            for j in range(resolution):

                Z[i, j] = interp.mass(
                    X[i, j],
                    Y[i, j],
                    fixed_value
                )

        xlabel = "Inclination ΔV [m/s]"
        ylabel = "Rendezvous ΔV [m/s]"
        title = f"Boost ΔV fixed = {fixed_value:.0f} m/s"

    else:
        raise ValueError(
            "fixed_axis must be "
            "'inclination', 'rdvz', or 'boost'"
        )

    plt.figure(figsize=(10, 8))

    pcm = plt.pcolormesh(
        X,
        Y,
        Z,
        shading="auto"
    )

    plt.colorbar(pcm, label="Mass [kg]")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    SC = Hestia(
        dV_inclination=3000,
        dV_rdvz=10000,
        dV_boost=5000,
        verbose=True,
        convergence_tolerance=0.001
    )

    SC._converge()

    resolution = 15
    dVs_incl = np.linspace(0, 3500, resolution)
    dVs_rdvz = np.linspace(0, 20000, resolution)
    dVs_boost = np.linspace(0, 7500, resolution)
    # data = generate_mass_database(dVs_incl, dVs_rdvz, dVs_boost)
    data = load_mass_database()
    plot_interp_heatmap("inclination", 3000)
    plot_interp_heatmap("rdvz", 15000)
    plot_interp_heatmap("boost", 7500)
    # plot_mass_database(data)
    # plot_mass_database_2(data)
    #
    # _test_mass_database(data, n_tests=10, tolerance=1e-2)
    # _test_interpolator_no_nans(data)
    # _test_all_grid_points(data)