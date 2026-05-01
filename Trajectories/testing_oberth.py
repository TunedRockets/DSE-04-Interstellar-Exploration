import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('TkAgg')

from src2.orbit import *

# -----------------------------
# Constants (Earth-like system)
# -----------------------------
mu = 398600.0  # km^3/s^2 (Earth)

# -----------------------------
# Target circular orbit
# -----------------------------
r_target = 6000.0  # km
v_target = np.sqrt(mu / r_target)

target_orbit = Orbit(
    p=r_target,
    e=0.0,
    i=0.0,
    RAAN=0.0,
    arg_p=0.0,
    t_p=0.0,
    sgp=mu
)

# -----------------------------
# Starting point
# -----------------------------
rp_loc = np.array([7000.0, 0.0, 0.0])
vp_vec = np.array([0.0, np.sqrt(mu / np.linalg.norm(rp_loc)), 0.0])
tp = 10000.0

# -----------------------------
# Second point
# -----------------------------
int_loc = np.array([0.0, 12000.0, 2000.0])

# =========================================================
# 1. Periapsis + point reconstruction
# =========================================================
try:
    orbit_pp, dt_pp = orbit_from_periapsis_point_and_point(
        rp_loc,
        int_loc,
        mu,
        tp
    )
except Exception as e:
    print("Periapsis-point method failed:", e)
    orbit_pp, dt_pp = None, None

# =========================================================
# Diagnostic plot
# =========================================================
def residual(t):
    try:
        int_loc_test = target_orbit.time_to_rv(t)[0]
        _, transfer_time = orbit_from_periapsis_point_and_point(
            rp_loc,
            int_loc_test,
            mu,
            t
        )
        return transfer_time - t
    except:
        return np.nan

try:
    ts = np.linspace(-10_000, 10000, 400)
    rs = np.array([residual(t) for t in ts])

    plt.figure()
    plt.axhline(0, color='black', linewidth=1)
    plt.plot(ts, rs)
    plt.axis("equal")
    
    plt.xlabel("Transfer time guess [s]")
    plt.ylabel("transfer_time - t")
    plt.title("Root structure of periapsis-point transfer problem")
    plt.grid()
    plt.show()
except Exception as e:
    print("Diagnostic plot failed:", e)

# =========================================================
# 2. Oberth transfer finder
# =========================================================
try:
    orbit_oberth, tof_oberth = oberth_transfer_finder(
        rp_loc,
        tp,
        target_orbit,
        mu,
        min_time=-1000,
        max_time=1000,
    )
except Exception as e:
    print("Oberth finder failed:", e)
    orbit_oberth, tof_oberth = None, None

# =========================================================
# 3. Oberth optimizer
# =========================================================
try:
    dv1, dv2, orbit_opt, t_dep, t_arr = oberth_effect_optimzer(
        target_object=target_orbit,
        rp=rp_loc,
        vp=vp_vec,
        tp=tp,
        min_time=-100,
        max_time=5 * 86400,
        optimize_rendezvous=True
    )
except Exception as e:
    print("Oberth optimizer failed:", e)
    dv1 = dv2 = orbit_opt = t_dep = t_arr = None

# =========================================================
# Helper for plotting
# =========================================================
def setup_ax(title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(title)
    return fig, ax

# =========================================================
# Individual plots (failure-safe)
# =========================================================

# --- Target orbit ---
try:
    fig, ax = setup_ax("Target Orbit")
    plot_orbit(ax, target_orbit, color='blue', label="Target Orbit")
    plt.axis("equal")
    
    ax.legend()
    plt.show()
except Exception as e:
    print("Target orbit plot failed:", e)

# --- Periapsis-point method ---
try:
    if orbit_pp is not None:
        fig, ax = setup_ax("Periapsis-Point Method")
        plot_orbit(ax, target_orbit, color='blue', label="Target Orbit")
        plot_orbit(ax, orbit_pp, color='green', label="Transfer Orbit")
        ax.scatter(*rp_loc, color='black', label="Departure", s=50)
        ax.scatter(*int_loc, color='orange', label="Intermediate", s=50)
        ax.legend()
        plt.axis("equal")
        
        plt.show()
except Exception as e:
    print("Periapsis-point plot failed:", e)

# --- Oberth finder ---
try:
    if orbit_oberth is not None:
        fig, ax = setup_ax("Oberth Transfer Finder")
        plot_orbit(ax, target_orbit, color='blue', label="Target Orbit")
        plot_orbit(ax, orbit_oberth, color='red', label="Transfer Orbit")
        ax.scatter(*rp_loc, color='black', label="Departure", s=50)
        ax.legend()
        plt.axis("equal")
        
        plt.show()
except Exception as e:
    print("Oberth finder plot failed:", e)

# --- Oberth optimizer ---
try:
    if orbit_opt is not None:
        fig, ax = setup_ax("Oberth Optimizer Result")
        plot_orbit(ax, target_orbit, color='blue', label="Target Orbit")
        plot_orbit(ax, orbit_opt, color='purple', label="Optimized Orbit")
        ax.scatter(*rp_loc, color='black', label="Departure", s=50)
        ax.legend()
        plt.axis("equal")
        plt.show()
except Exception as e:
    print("Oberth optimizer plot failed:", e)

# =========================================================
# Diagnostics print
# =========================================================
print("Periapsis-point dt:", dt_pp)
print("Oberth transfer TOF:", tof_oberth)
print("Optimizer dV insertion:", dv1)
print("Optimizer dV rendezvous:", dv2)
print("Departure time:", t_dep)
print("Arrival time:", t_arr)

# =========================================================
# 4. Randomized intercept tests
# =========================================================

n_tests = 20  # number of random trials
t_min = 0
t_max = 2 * 86400  # 2 days

success_count = 0

for i in range(n_tests):
    print(f"\n=== Test {i+1}/{n_tests} ===")

    try:
        # --- pick random time on target orbit ---
        t_rand = np.random.uniform(t_min, t_max)

        # get target position at that time
        r_target_rand, v_target_rand = target_orbit.time_to_rv(t_rand)

        # --- attempt intercept ---
        orbit_test, tof_test = oberth_transfer_finder(
            rp_loc,
            tp,
            target_orbit,
            mu,
            min_time=t_rand - 2000,
            max_time=t_rand + 2000,
        )

        success_count += 1

        # --- plot result ---
        try:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.set_title(f"Intercept Test {i+1}")

            plot_orbit(ax, target_orbit, color='blue', label="Target Orbit")
            plot_orbit(ax, orbit_test, color='red', label="Transfer Orbit")

            ax.scatter(*rp_loc, color='black', label="Departure", s=50)
            ax.scatter(*r_target_rand, color='orange', label="Target Point", s=50)
            plt.axis("equal")
            ax.legend()
            plt.show()

        except Exception as e:
            print(f"Plot failed for test {i+1}:", e)

        print(f"Success | t_target={t_rand:.2f}, TOF={tof_test:.2f}")

    except Exception as e:
        print(f"Intercept failed for test {i+1}:", e)

# =========================================================
# Summary
# =========================================================
print("\n===================================")
print(f"Success rate: {success_count}/{n_tests}")
print("===================================")


# =========================================================
# 4. Random point intercept tests (pure geometry)
# =========================================================

n_tests = 20
success_count = 0

# radius bounds for random points (tune as needed)
r_min = 6000
r_max = 20000

for i in range(n_tests):
    print(f"\n=== Test {i+1}/{n_tests} ===")

    try:
        # --- random point in 3D space ---
        direction = np.random.normal(size=3)
        direction /= np.linalg.norm(direction)

        radius = np.random.uniform(r_min, r_max)
        int_loc_rand = direction * radius

        # --- attempt reconstruction ---
        orbit_test, dt_test = orbit_from_periapsis_point_and_point(
            rp_loc,
            int_loc_rand,
            mu,
            tp
        )

        success_count += 1

        print(f"Success | dt = {dt_test:.2f} s | r = {radius:.1f} km")

        # --- plot ---
        try:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.set_title(f"Random Intercept Test {i+1}")

            plot_orbit(ax, orbit_test, color='green', label="Reconstructed Orbit")

            ax.scatter(*rp_loc, color='black', label="Periapsis", s=50)
            ax.scatter(*int_loc_rand, color='orange', label="Target Point", s=50)
            plt.axis("equal")
            ax.legend()
            plt.show()

        except Exception as e:
            print(f"Plot failed for test {i+1}:", e)

    except Exception as e:
        print(f"Solver failed for test {i+1}:", e)

# =========================================================
# Summary
# =========================================================
print("\n===================================")
print(f"Success rate: {success_count}/{n_tests}")
print("===================================")
