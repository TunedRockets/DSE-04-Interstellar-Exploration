from Structures.holistic_mass_solver import Vesta
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
from pathlib import Path

def _single_run(args):
    """Worker function (must be top-level for multiprocessing)."""
    i, j, dv_inj, dv_rdvz = args

    sc = Vesta(
        dV_injection=dv_inj,
        dV_rdvz=dv_rdvz,
        allowed_dv_boost=0,
        convergence_tolerance=0.001,
        verbose=False,
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

    return i, j, total_mass

# from concurrent.futures import ProcessPoolExecutor, as_completed
path = Path(__file__).parent / "mass_database_vesta.pkl"
def generate_mass_database(dVs_inj, dVs_rdvz, path=path):

    jobs = [
        (i, j, dv_inj, dv_rdvz)
        for i, dv_inj in enumerate(dVs_inj)
        for j, dv_rdvz in enumerate(dVs_rdvz)
    ]

    masses = np.zeros((len(dVs_inj), len(dVs_rdvz)))

    # pbar = tqdm(total=len(jobs), desc="Mass DB")

    with Pool() as p:
        results = tqdm(p.imap_unordered(_single_run, jobs,5), desc="Jobs completed: ", total=len(jobs))

        for result in results:
            i, j, mass = result
            # print("Result appended!")
            masses[i, j] = mass
            # pbar.update(1)

    print("Run complete!")

    # pbar.close()

    data = {
        "dV_injection": dVs_inj,
        "dV_rdvz": dVs_rdvz,
        "mass": masses,
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

    required_keys = ["dV_injection", "dV_rdvz", "mass"]

    missing = [k for k in required_keys if k not in data]
    if missing:
        raise KeyError(f"Missing keys in database file: {missing}")

    # Ensure numpy arrays (pickle sometimes preserves weird types)
    data["dV_injection"] = np.asarray(data["dV_injection"])
    data["dV_rdvz"] = np.asarray(data["dV_rdvz"])
    data["mass"] = np.asarray(data["mass"])

    # Basic sanity check
    expected_shape = (
        len(data["dV_injection"]),
        len(data["dV_rdvz"])
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
                data["dV_injection"],
                data["dV_rdvz"]
            ),
            data["mass"]
        )

    def mass(self, dV_injection, dV_rdvz):

        point = np.array([
            dV_injection,
            dV_rdvz
        ])

        val = self.interp(point)
        return float(np.asarray(val).squeeze())

import matplotlib.pyplot as plt

def plot_mass_heatmap(interpolator,
                      dV_inj_range=None,
                      dV_rdvz_range=None,
                      resolution=200,
                      show_points=False,
                      database=None):
    """
    Plot a heatmap of interpolated mass values.

    Parameters
    ----------
    interpolator : MassInterpolator
        Instance of MassInterpolator.

    dV_inj_range : tuple(float, float), optional
        (min, max) injection ΔV range.
        If None, inferred from database.

    dV_rdvz_range : tuple(float, float), optional
        (min, max) rendezvous ΔV range.
        If None, inferred from database.

    resolution : int
        Number of points per axis for interpolation.

    show_points : bool
        If True, overlay original database sample points.

    database : dict, optional
        Output from load_mass_database().
        Required if ranges are omitted or if show_points=True.
    """

    if database is None and (
        dV_inj_range is None
        or dV_rdvz_range is None
        or show_points
    ):
        raise ValueError(
            "database must be supplied when ranges are omitted "
            "or show_points=True"
        )

    if dV_inj_range is None:
        dV_inj_range = (
            database["dV_injection"].min(),
            database["dV_injection"].max()
        )

    if dV_rdvz_range is None:
        dV_rdvz_range = (
            database["dV_rdvz"].min(),
            database["dV_rdvz"].max()
        )

    inj = np.linspace(*dV_inj_range, resolution)
    rdvz = np.linspace(*dV_rdvz_range, resolution)

    X, Y = np.meshgrid(inj, rdvz, indexing="ij")

    points = np.column_stack([
        X.ravel(),
        Y.ravel()
    ])

    Z = interpolator.interp(points).reshape(X.shape)

    plt.figure(figsize=(10, 8))

    im = plt.pcolormesh(
        X,
        Y,
        Z,
        shading="auto"
    )

    cbar = plt.colorbar(im)
    cbar.set_label("Lower Stage Wet Mass [kg]")

    if show_points:
        XI, YI = np.meshgrid(
            database["dV_injection"],
            database["dV_rdvz"],
            indexing="ij"
        )

        plt.scatter(
            XI.ravel(),
            YI.ravel(),
            s=10,
            c="k",
            alpha=0.5,
            label="Database Points"
        )
        plt.legend()

    plt.xlabel("Injection ΔV [m/s]")
    plt.ylabel("Rendezvous ΔV [m/s]")
    plt.title("Vesta Mass Interpolation Heatmap")

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


if __name__ == "__main__":
    # V = Vesta(14_730, 4_153, 7000)
    # V._converge()
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

    resolution = 100
    dVs_inj = np.linspace(0, 7500+8500+2000, resolution)
    dVs_rdvz = np.linspace(0, 25000, resolution)
    path = Path(__file__).parent / "mass_database_vesta_FH_Exp.pkl"
    # data = generate_mass_database(dVs_inj, dVs_rdvz,path=path) # TODO MAKE NEW MASS INTERP WITH FH EXP
    db = load_mass_database(filename=path)

    interp = MassInterpolator(filename=path)

    # _test_mass_database(db)
    # _test_interpolator_no_nans(db)
    # _test_all_grid_points(db)

    plot_mass_heatmap(
        interp,
        database=db,
        resolution=300,
        show_points=False
    )