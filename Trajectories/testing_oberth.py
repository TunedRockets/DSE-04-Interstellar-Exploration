

import numpy as np
import matplotlib.pyplot as plt
import pytest
import matplotlib as mpl
mpl.use('TkAgg')

from src2.orbit import (
    Orbit,
    orbit_from_keplerian,
    oberth_transfer_finder,
    oberth_effect_optimzer,
    dt_from_periapsis_point_and_point,
    lambert_vectors,
    plot_orbit
)

# ===========================
# Global constants
# ===========================

MU_EARTH = 398600.0  # km^3/s^2


# ===========================
# Fixtures
# ===========================

@pytest.fixture
def circular_orbits():
    r1 = 12000.0
    r2 = 7000.0

    ob1 = orbit_from_keplerian(r1, 0, 0, 0, 0, 0, MU_EARTH)
    ob2 = orbit_from_keplerian(r2, 0, 0, 0, 0, np.pi/2, MU_EARTH)

    rp, vp_vec = ob1.theta_to_rv(0)
    vp = np.linalg.norm(vp_vec)

    return ob1, ob2, rp, vp


# ===========================
# 1. Basic functionality
# ===========================

def test_oberth_transfer_basic(circular_orbits):
    ob1, ob2, rp, vp = circular_orbits

    orbit, dt = oberth_transfer_finder(
        rp,
        0,
        ob2,
        MU_EARTH,
        min_time=0,
        max_time=20000
    )

    assert orbit is not None
    assert np.isfinite(dt)
    assert dt > 0


# ===========================
# 2. Time consistency
# ===========================

def test_time_consistency(circular_orbits):
    ob1, ob2, rp, vp = circular_orbits

    t_guess = 5000

    int_loc = ob2.time_to_rv(t_guess)[0]
    dt = dt_from_periapsis_point_and_point(rp, int_loc, MU_EARTH)

    assert np.isfinite(dt)


# ===========================
# 3. Optimizer output
# ===========================

def test_oberth_optimizer(circular_orbits):
    ob1, ob2, rp, vp = circular_orbits

    dv_ins, dv_rdv, orbit, t_dep, t_arr = oberth_effect_optimzer(
        target_object=ob2,
        rp=rp,
        vp=vp,
        tp=0,
        min_time=1000,
        max_time=20000,
        optimize_rendezvous=True
    )

    assert np.isfinite(dv_ins)
    assert np.isfinite(dv_rdv)
    assert orbit is not None
    assert t_arr > t_dep


# ===========================
# 4. Compare vs Lambert
# ===========================

def test_vs_lambert(circular_orbits):
    ob1, ob2, rp, vp = circular_orbits

    dv_ins, dv_rdv, orbit, t_dep, t_arr = oberth_effect_optimzer(
        target_object=ob2,
        rp=rp,
        vp=vp,
        tp=0,
        min_time=1000,
        max_time=20000
    )

    r1, v1 = ob1.time_to_rv(t_dep)
    r2, v2 = ob2.time_to_rv(t_arr)

    v_lam1, _ = lambert_vectors(r1, r2, t_arr - t_dep, MU_EARTH)

    dv_lambert = np.linalg.norm(v_lam1 - v1)

    assert dv_ins <= dv_lambert * 2  # loose sanity bound


# ===========================
# 5. Inclined orbit test
# ===========================

def test_inclined_target():
    ob1 = orbit_from_keplerian(7000, 0, 0, 0, 0, 0, MU_EARTH)
    ob2 = orbit_from_keplerian(
        12000,
        0.2,
        np.radians(30),
        np.radians(40),
        np.radians(60),
        np.pi/3,
        MU_EARTH
    )

    rp, vp_vec = ob1.theta_to_rv(0)
    vp = np.linalg.norm(vp_vec)

    dv_ins, dv_rdv, orbit, t_dep, t_arr = oberth_effect_optimzer(
        target_object=ob2,
        rp=rp,
        vp=vp,
        tp=0,
        min_time=1000,
        max_time=20000
    )

    assert np.isfinite(dv_ins)
    assert orbit is not None


# ===========================
# 6. Failure handling
# ===========================

def test_invalid_geometry():
    ob2 = orbit_from_keplerian(12000, 0, 0, 0, 0, 0, MU_EARTH)

    bad_rp = np.array([0, 0, 0])

    orbit, dt = oberth_transfer_finder(
        bad_rp,
        0,
        ob2,
        MU_EARTH,
        1000,
        20000
    )

    assert orbit is None or not np.isfinite(dt)


# ===========================
# 7. Root behavior (diagnostic)
# ===========================

def test_root_behavior(circular_orbits):
    ob1, ob2, rp, vp = circular_orbits

    ts = np.linspace(1000, 20000, 100)
    vals = []

    for t in ts:
        try:
            int_loc = ob2.time_to_rv(t)[0]
            dt = dt_from_periapsis_point_and_point(rp, int_loc, MU_EARTH)
            vals.append(dt - t if np.isfinite(dt) else np.nan)
        except:
            vals.append(np.nan)

    vals = np.array(vals)

    # ensure at least some valid values exist
    assert np.any(np.isfinite(vals))


# ===========================
# Optional visualization
# ===========================

def run_visual_test():
    print("Running visual test...")

    ob1 = orbit_from_keplerian(7000, 0, 0, 0, 0, 0, MU_EARTH)
    ob2 = orbit_from_keplerian(12000, 0, 0, 0, 0, np.pi/2, MU_EARTH)

    rp, vp_vec = ob1.theta_to_rv(0)
    vp = np.linalg.norm(vp_vec)

    dv_ins, dv_rdv, orbit, t_dep, t_arr = oberth_effect_optimzer(
        target_object=ob2,
        rp=rp,
        vp=vp,
        tp=0,
        min_time=1000,
        max_time=20000,
        optimize_rendezvous=True
    )

    print("DV insertion:", dv_ins)
    print("DV rendezvous:", dv_rdv)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    plot_orbit(ax, ob1, color='blue')
    plot_orbit(ax, ob2, color='green')
    plot_orbit(ax, orbit, color='red')

    plt.title("Oberth Transfer Visualization")
    plt.show()


# ===========================
# Run manually
# ===========================

if __name__ == "__main__":
    run_visual_test()
