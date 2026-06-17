from Structures.holistic_mass_solver import Vesta
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
from pathlib import Path


def _single_run(args):
    i, ion_dv = args

    sc = Vesta(
        ion_dv=ion_dv,
        allowed_dv_boost=0,
        convergence_tolerance=0.001,
        verbose=False,
    )

    try:
        sc._converge()

        result = {
            "total_mass": sc.lower_stage_wet_mass,
            "payload_mass": sc.static_mass,
            "reactor_truss": sc.Mass_power_truss,
            "engine_mass": sc.Mass_ion,
            "fuel_mass": sc.Mass_ion_fuel,
        }

    except Exception:
        result = {
            "total_mass": np.nan,
            "reactor_truss": np.nan,
            "fuel_mass": np.nan,
            "engine_mass": np.nan,
            "payload_mass": np.nan,
        }

    return i, result
# from concurrent.futures import ProcessPoolExecutor, as_completed
path = Path(__file__).parent / "mass_database_vesta.pkl"
def generate_mass_database(ion_dvs, path=path):

    jobs = [
        (i, ion_dv)
        for i, ion_dv in enumerate(ion_dvs)
    ]

    masses = np.zeros(len(ion_dvs))
    total_mass = np.zeros(len(ion_dvs))
    reactor_mass = np.zeros(len(ion_dvs))
    fuel_mass = np.zeros(len(ion_dvs))
    engine_mass = np.zeros(len(ion_dvs))
    payload_mass = np.zeros(len(ion_dvs))

    with Pool() as p:
        results = tqdm(
            p.imap_unordered(_single_run, jobs, 5),
            desc="Jobs completed",
            total=len(jobs),
        )

        for i, result in results:
            total_mass[i] = result["total_mass"]
            reactor_mass[i] = result["reactor_truss"]
            fuel_mass[i] = result["fuel_mass"]
            engine_mass[i] = result["engine_mass"]
            payload_mass[i] = result["payload_mass"]

    data = {
        "ion_dv": ion_dvs,
        "total_mass": total_mass,
        "reactor_mass": reactor_mass,
        "fuel_mass": fuel_mass,
        "engine_mass": engine_mass,
        "payload_mass": payload_mass,
    }

    with open(path, "wb") as f:
        pickle.dump(data, f)

    return data


from scipy.interpolate import RegularGridInterpolator
import pickle

path = Path(__file__).parent / "mass_database_vesta.pkl"
def load_mass_database(filename=path):
    """
    Load a precomputed Vesta mass database.

    Returns
    -------
    dict with:
        dV_injection : np.ndarray
        dV_rdvz        : np.ndarray
        mass           : np.ndarray (2D grid)
    """

    with open(filename, "rb") as f:
        data = pickle.load(f)

    # required_keys = ["ion_dv", "mass"]
    #
    # missing = [k for k in required_keys if k not in data]
    # if missing:
    #     raise KeyError(f"Missing keys in database file: {missing}")
    #
    # data["ion_dv"] = np.asarray(data["ion_dv"])
    # data["mass"] = np.asarray(data["mass"])
    #
    # expected_shape = (len(data["ion_dv"]),)
    #
    # if data["mass"].shape != expected_shape:
    #     raise ValueError(
    #         f"Expected {expected_shape}, got {data['mass'].shape}"
    #     )

    return data

from scipy.interpolate import interp1d

class MassInterpolator:

    def __init__(self, filename=path):

        with open(filename, "rb") as f:
            data = pickle.load(f)

        self.interp = interp1d(
            data["ion_dv"],
            data["mass"],
            kind="linear",
            bounds_error=False,
            fill_value=np.nan,
        )

    def mass(self, ion_dv):

        val = self.interp(ion_dv)
        return float(np.asarray(val).squeeze())

import matplotlib.pyplot as plt

def plot_mass_curve(interpolator,
                    ion_dv_range,
                    resolution=500,
                    database=None,
                    show_points=False):

    x = np.linspace(*ion_dv_range, resolution)
    y = interpolator.interp(x)

    plt.figure(figsize=(8, 5))

    plt.plot(x, y, label="Interpolated")

    if show_points and database is not None:
        plt.scatter(
            database["ion_dv"],
            database["mass"],
            s=15,
            label="Database Points"
        )

    plt.xlabel("Ion ΔV [m/s]")
    plt.ylabel("Spacecraft Wet Mass [kg]")
    # plt.title("Mass vs Ion ΔV")

    plt.legend()
    plt.tight_layout()
    plt.show()
def plot_mass_breakdown(database):

    x = database["ion_dv"]

    plt.figure(figsize=(10,6))

    plt.stackplot(
        x,
        database["payload_mass"],
        database["reactor_mass"],
        database["engine_mass"],
        database["fuel_mass"],
        labels=[
            "Fixed Mass (Payload, TT&C, and Bus)",
            "Reactor + Truss",
            "Engine System",
            "Fuel",
        ]
    )

    plt.plot(
        x,
        database["total_mass"],
        "k",
        linewidth=2,
        label="Total"
    )

    # ===== design point =====

    design_dv = 10_000

    design_mass = np.interp(
        design_dv,
        database["ion_dv"],
        database["total_mass"]
    )

    plt.plot(
        [-1000000, design_dv],
        [design_mass, design_mass],
        "k:",
        linewidth=2,
    )
    plt.plot(
        [design_dv, design_dv],
        [-100000, design_mass],
        "k:",
        linewidth=2,
    )
    plt.plot(
        design_dv,
        design_mass,
        "ko",
        markersize=6
    )

    plt.annotate(
        f"{design_mass:.0f} kg",
        (design_dv, design_mass),
        xytext=(10, 10),
        textcoords="offset points"
    )

    plt.xlabel("Ion ΔV [m/s]")
    plt.ylabel("Mass [kg]")
    plt.legend()
    plt.tight_layout()
    plt.show()

import random
def _test_mass_database(
    data,
    n_tests=20,
    tolerance=1e-4
):
    """
    Randomly sample grid points and verify:

    1. Stored database value
       == fresh Vesta convergence

    2. Interpolator value
       == fresh Vesta convergence
    """

    interp = MassInterpolator()

    n_inj = len(data["dV_injection"])
    n_rdvz = len(data["dV_rdvz"])

    failures = []

    for test_no in range(n_tests):

        i = random.randrange(n_inj)
        j = random.randrange(n_rdvz)

        dv_inj = data["dV_injection"][i]
        dv_rdvz = data["dV_rdvz"][j]

        stored = data["mass"][i, j]

        sc = Vesta(
            dV_injection=dv_inj,
            dV_rdvz=dv_rdvz,
            allowed_dv_boost=0,
            convergence_tolerance=1e-6,
            verbose=False,
        )
        try:
            sc._converge()

            rerun = sc.lower_stage_wet_mass
        except:
            rerun = np.inf

        interpolated = interp.mass(
            dv_inj,
            dv_rdvz
        )

        db_error = abs(stored - rerun)
        interp_error = abs(interpolated - rerun)

        print(
            f"Test {test_no+1:02d} | "
            f"(i,j)=({i},{j}) | "
            f"(inj, rdvz)=({round(dv_inj)},{round(dv_rdvz)}) | "
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
                    "ij": (i, j),
                    "stored": stored,
                    "rerun": rerun,
                    "interpolated": interpolated,
                    "db_error": db_error,
                    "interp_error": interp_error,
                }
            )

    print()
    print("=" * 60)

    if failures:
        print(f"FAILED: {len(failures)} / {n_tests}")

        for f in failures:
            print()
            print(f["ij"])
            print("stored      :", f["stored"])
            print("rerun       :", f["rerun"])
            print("interpolated:", f["interpolated"])
            print("db_error    :", f["db_error"])
            print("interp_error:", f["interp_error"])

        raise AssertionError(
            f"{len(failures)} tests exceeded tolerance"
        )

    print(f"PASSED: {n_tests}/{n_tests}")

def _test_interpolator_no_nans(
    data,
    n_tests=100_000,
    seed=42,
):
    rng = np.random.default_rng(seed)

    interp = MassInterpolator()

    inj_min, inj_max = (
        data["dV_injection"][0],
        data["dV_injection"][-1]
    )

    rdv_min, rdv_max = (
        data["dV_rdvz"][0],
        data["dV_rdvz"][-1]
    )

    n_grid_nans = np.isnan(data["mass"]).sum()

    print(f"NaNs in source grid: {n_grid_nans}")

    if n_grid_nans:
        raise AssertionError(
            f"Source grid contains {n_grid_nans} NaNs"
        )

    for _ in range(n_tests):

        point = np.array([
            rng.uniform(inj_min, inj_max),
            rng.uniform(rdv_min, rdv_max),
        ])

        val = interp.interp(point)

        if np.isnan(val).any():
            raise AssertionError(
                f"NaN returned at point {point}"
            )

    print(f"PASSED: {n_tests} random in-range points")

def _test_all_grid_points(data):

    interp = MassInterpolator()

    for inj in data["dV_injection"]:
        for rdvz in data["dV_rdvz"]:

            val = interp.interp([inj, rdvz])

            if np.isnan(val).any():
                raise AssertionError(
                    f"NaN at exact grid point "
                    f"({inj}, {rdvz})"
                )

    print("PASSED: all grid points")

from scipy.interpolate import interp1d
from scipy.optimize import brentq
import numpy as np


def find_crossovers(x, y1, y2):
    """
    Return all x locations where y1 == y2.
    """
    diff = y1 - y2

    roots = []

    for i in range(len(x) - 1):

        if np.isnan(diff[i]) or np.isnan(diff[i + 1]):
            continue

        # exact hit
        if diff[i] == 0:
            roots.append(x[i])

        # sign change
        elif diff[i] * diff[i + 1] < 0:

            f = interp1d(
                x[i:i+2],
                diff[i:i+2],
                kind="linear"
            )

            root = brentq(
                lambda xx: float(f(xx)),
                x[i],
                x[i+1]
            )

            roots.append(root)

    return roots

import matplotlib.pyplot as plt


def plot_mass_breakdown_comparison(
    db1,
    db2,
    label1="Ion engines",
    label2="Hall effect thrusters",
):

    x = db1["ion_dv"]

    fig, ax = plt.subplots(figsize=(10, 6))

    y1 = db1["total_mass"]
    y2 = db2["total_mass"]

    y3 = db1["fuel_mass"]
    y4 = db2["fuel_mass"]

    y5 = db1["reactor_mass"]
    y6 = db2["reactor_mass"]

    ax.plot(
        x,
        y1,
        lw=3,
        label=f"{label1}"
    )

    ax.plot(
        x,
        y2,
        lw=3,
        ls="--",
        label=f"{label2}"
    )

    ax.plot(
        x,
        y3,
        lw=1,
        label=f"{label1} fuel mass"
    )

    ax.plot(
        x,
        y4,
        lw=1,
        ls="--",
        label=f"{label2} fuel mass"
    )

    ax.plot(
        x,
        y5,
        lw=1,
        label=f"{label1} reactor mass"
    )

    ax.plot(
        x,
        y6,
        lw=1,
        ls="--",
        label=f"{label2} reactor mass"
    )

    # ---- total-mass crossover ----

    # roots = find_crossovers(x, y1, y2)
    #
    # for root in roots:
    #
    #     mass = np.interp(root, x, y1)
    #
    #     ax.plot(root, mass, "ko")
    #
    #     ax.axvline(
    #         root,
    #         color="k",
    #         ls=":",
    #         alpha=0.5,
    #     )
    #
    #     ax.annotate(
    #         f"{root:.0f} m/s\n{mass:.0f} kg",
    #         (root, mass),
    #         xytext=(10, 10),
    #         textcoords="offset points",
    #     )
    #
    #     print(
    #         f"Total mass crossover at "
    #         f"{root:.1f} m/s "
    #         f"({mass:.1f} kg)"
    #     )

    ax.set_xlabel("Ion ΔV [m/s]")
    ax.set_ylabel("Total Mass [kg]")
    ax.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    V = Vesta(10000, 0)
    V._converge()
    # print(V)
    # masses = []
    # ion_dv = []
    # allowable_boosts = np.linspace(0, 7000, 100)
    #
    # for boost in allowable_boosts:
    #     V = Vesta(14_730, 4_153, boost)
    #     V._converge()
    #     ion_dv.append(V.ion_extra)
    #     masses.append(V.lower_stage_wet_mass)

    # plt.plot(allowable_boosts, masses)
    # plt.plot(allowable_boosts, ion_dv)
    # plt.show()

    path_hall = Path(__file__).parent / "mass_database_vesta_Ariane64_hall.pkl"
    path_ion = Path(__file__).parent / "mass_database_vesta_Ariane64.pkl"
    path = path_hall
    ion_dvs = np.linspace(0, 25000, 500)

    # data = generate_mass_database(
    #     ion_dvs,
    #     path=path
    # )

    db_ion = load_mass_database(path_ion)
    db_hall = load_mass_database(path_hall)

    plot_mass_breakdown_comparison(db_ion, db_hall)


    # interp = MassInterpolator(path)

    # plot_mass_curve(
    #     interp,
    #     ion_dv_range=(
    #         db["ion_dv"].min(),
    #         db["ion_dv"].max()
    #     ),
    #     database=db,
    #     show_points=True,
    # )

    # plot_mass_breakdown(db)