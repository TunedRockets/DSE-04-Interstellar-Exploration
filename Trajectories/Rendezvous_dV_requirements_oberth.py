'''
Script for generating a distributions of dV expected for the ISO intercept


'''
from src2.orbit import Orbit, oberth_effect_optimzer, plot_orbit, orbit_from_lambert, orbit_from_rv, orbit_from_ephemeris
from src2.get_ISO import get_ISO
from src2.examples import Earth, Jupiter
from src2.utilities import AU, YEAR, SGP_SUN
import matplotlib.pyplot as plt
import numpy as np
import math as m
from tqdm import tqdm
from pathlib import Path
import random
import matplotlib as mpl

PLOT= False

if PLOT:
    mpl.use('TkAgg')

rdvz = False



def add_dv_hist(rm, weights, N, PLOT=False, lon_per=None)->None:
    '''Adds to the dv histogram for the different weights'''
    np.seterr(all="ignore")
    if lon_per is None:
        lon_per_str=""
    else:
        lon_per_str=str(round(np.degrees(lon_per)))

    path = Path(__file__).parent.parent / "data_oberth" / (f"dVhist-{weights["w_insertion"]},{weights["w_relv"]},{rm:.2f},"+lon_per_str)
    
    try:
        with open(path, "r") as file:
            lines = file.readlines()
            count = int(lines[0])
            hist = [int(x) for x in lines[1:]]
        

    except:
        hist = [0 for _ in range(100)]
        count = 0


    ISOs = get_ISO() # sample of ISOs
    while len(ISOs) < N:
        ISOs.extend(get_ISO())
    count += len(ISOs)

    for ISO in tqdm(ISOs,desc="studying ISOs"):
        detect_theta = ISO.crosses_altitude(rm*AU)
        if detect_theta is None: continue
        detect_time = ISO.theta_to_time(-detect_theta)
        try:
            # ===============================
            # Oberth optimization at periapsis
            # ===============================
            theta_pe = 0
            # tp = origin.theta_to_time(theta_pe)
            or_period = origin.period
            tp = random.uniform(0.5*or_period, 1.5*or_period)

            rp_vec, vp_vec = origin.theta_to_rv(theta_pe)
            vp_mag = np.linalg.norm(vp_vec)

            min_time = 100

            insert_dv, rdvz_dv, transfer_orbit, st, et = oberth_effect_optimzer(
                ISO,
                rp_vec,
                vp_mag,
                tp,
                min_time,
                max_time,
                optimize_rendezvous=(weights["w_relv"] > 0),
                period=or_period,
                detect_time=detect_time,
                periods=4
            )

            # ===============================
            # Compute required periapsis direction
            # ===============================
            r_req, v_req = transfer_orbit.theta_to_rv(theta_pe)
            v_req_hat = v_req / np.linalg.norm(v_req)

            # current orbit periapsis direction
            r0, v0 = origin.theta_to_rv(theta_pe)
            v0_hat = v0 / np.linalg.norm(v0)

            # ===============================
            # Angle between directions
            # ===============================
            cos_dtheta = np.clip(np.dot(v0_hat, v_req_hat), -1, 1)
            delta = m.acos(cos_dtheta)

            # ===============================
            # Apoapsis velocity
            # ===============================
            r_a = origin.apoapsis
            a = origin.a
            mu = origin.sgp

            v_ap = m.sqrt(mu * (2 / r_a - 1 / a))

            # ===============================
            # Plane change Δv at apoapsis
            # ===============================
            dv_plane = 2 * v_ap * m.sin(delta / 2)

            # ===============================
            # Total insertion Δv
            # ===============================
            insert_dv += dv_plane

        except:
            continue

        insert_dv = round(insert_dv)
        if PLOT:
            # ===============================
            # Reconstruct rotated orbit (after apoapsis burn)
            # ===============================

            # rotation axis from current to required direction
            axis = np.cross(v0_hat, v_req_hat)
            norm = np.linalg.norm(axis)

            if norm < 1e-10:
                axis = np.array([0, 0, 1])  # fallback
            else:
                axis /= norm

            def rotate(vec):
                return (
                        vec * m.cos(delta) +
                        np.cross(axis, vec) * m.sin(delta) +
                        axis * np.dot(axis, vec) * (1 - m.cos(delta))
                )

            # get apoapsis state
            theta_ap = m.pi
            t_ap = origin.theta_to_time(theta_ap)
            r_ap, v_ap_vec = origin.theta_to_rv(theta_ap)

            # rotate state
            r_rot = rotate(r_ap)
            v_rot = rotate(v_ap_vec)

            # rebuild orbit
            try:
                origin_rot = orbit_from_rv(r_rot, v_rot, origin.sgp, t_ap)
                origin_rot.link_time_and_theta(theta_ap, t_ap)
                origin_rot.normalize()
            except:
                continue

            # ==== plotting ====
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            # plot original orbit
            plot_orbit(ax, origin, time=detect_time, ThreeDee=True, label="Original")

            # plot rotated orbit
            plot_orbit(ax, origin_rot, time=detect_time, ThreeDee=True, label="Rotated")

            try:
                plot_orbit(ax, transfer_orbit, time=et, ThreeDee=True, label="Transfer", max_alt=(40*AU))
            except:
                pass  # lambert sometimes fails

            # plot ISO, earth and jupiter orbit for context
            plot_orbit(ax, ISO, time=et, ThreeDee=True, label="ISO", max_alt=(40*AU))
            plot_orbit(ax, Earth, time=detect_time, ThreeDee=True, label="Earth")
            plot_orbit(ax, Jupiter, time=detect_time, ThreeDee=True, label="Jupiter")

            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            plt.axis("scaled")
            textstr = (
                f"ΔV insert: {insert_dv:.2f} km/s\n"
                f"ΔV rendezvous: {rdvz_dv:.2f} km/s\n"
                f"Intercept distance: {np.linalg.norm(ISO.time_to_rv(et)[0])/AU:.2f} AU\n"
                f"Intercept time: {(et-detect_time)/YEAR:.2f} years\n"
            )

            ax.text2D(0.02, 0.98, textstr,
                      transform=ax.transAxes,
                      fontsize=10,
                      verticalalignment='top',
                      bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

            ax.legend()

            plt.show()
        if insert_dv > 99 or insert_dv < 0 or not np.isfinite(insert_dv): continue
        else:
            hist[round(insert_dv)] += 1
    
    # Save
    path = Path(__file__).parent.parent / "data_oberth" / (f"dVhist-{weights["w_insertion"]},{weights["w_relv"]},{rm:.2f},"+lon_per_str)
    with open(path, "w") as file:
        file.write(str(count) + '\n')
        file.writelines([str(x) + '\n' for x in hist])
    return

def get_dv_hist(rm, weights, lon_per=None)->list[float]:
    '''return normalised histogram of the delta v requirements.
    nomalization includes invalid trajectories, so area under curve will be
    less than 1'''
    if lon_per is None:
        lon_per_str = ""
    else:
        lon_per_str = str(round(np.degrees(lon_per)))
    path = Path(__file__).parent.parent / "data_oberth" / (
                f"dVhist-{weights["w_insertion"]},{weights["w_relv"]},{rm:.2f}," + lon_per_str)

    with open(path, "r") as file:
        lines = file.readlines()
        count = int(lines[0])
        hist = [int(x) for x in lines[1:]]
    return [x/count for x in hist]


def mission_success_probability(detection_distance:float, dV_budget:int, N:int, weight:dict, lon_per=None)->float:
    '''
    Generates total mission probability for the given scenario.
    :param detection_distance: distance (in AU) from the sun that ISOs are detected
    :type: float
    :param dV_buget: total mission dV budget
    :type: int
    :param N: Number of ISOs during the mission
    :type: float
    :param weight: optimizer weight, i.e. an intercept or rendezvouz
    :type: dict
    '''
    p_ISO = ISO_probability(detection_distance,dV_budget, weight, lon_per=lon_per)
    p_least_one = 1-(1-p_ISO)**N
    return p_least_one

def ISO_probability(rm:float,dV_budget:int, weight,lon_per)->float:
    '''calculate individual chance of success for given dv budget and detection distance,
    currently only works with integer dV budgets

    :param detection_distance: distance (in AU) from the sun that ISOs are detected
    :type: float
    :param dV_buget: total mission dV budget
    :type: int
    :param weight: optimizer weight, i.e. an intercept or rendezvouz
    :type: dict'''
    hist = get_dv_hist(rm,weight,lon_per=lon_per)
    return np.sum(hist[:m.floor(dV_budget)+1])

def probability_map(rm: float, weight: dict, guesses: bool = True, show: bool = True, lon_per=None):
    '''generate probability map of N over dV,'''

    Ezell_Loeb_avg_per_annum = 5
    Hoover_seligman_payne_per_annum = 14
    Marceta_seligman_per_annum = 35
    years = 10

    EL_N = Ezell_Loeb_avg_per_annum * years
    HSP_N = Hoover_seligman_payne_per_annum * years
    MS_N = Marceta_seligman_per_annum * years

    N_range = np.arange(10, MS_N + 30, 5)
    V_range = np.arange(4, 50)
    NN, VV = np.meshgrid(N_range, V_range)
    PP = np.vectorize(mission_success_probability)(rm, NN, VV, weight,lon_per=lon_per)
    plt.imshow(PP, origin="lower", aspect="auto", extent=(N_range[0], N_range[-1], V_range[0], V_range[-1]))
    plt.colorbar(location="right", label="Probability of mission success")
    CS = plt.contour(PP, levels=[0.9], origin="lower", aspect="auto",
                     extent=(N_range[0], N_range[-1], V_range[0], V_range[-1]))
    plt.clabel(CS, fmt=lambda x: f"{x * 100:.0f}%")
    plt.ylabel('Delta V budget (km/s)')
    plt.xlabel('number of ISOs during mission time')
    if lon_per is None:
        plt.title(
            f"Probability map for {"rendezvous" if weight["w_relv"] else "intercept"} with detection range of {rm} AU\nAnd estimated ISO detections during {years} year mission")
    else:
        plt.title(
            f"Probability map for {"rendezvous" if weight["w_relv"] else "intercept"} with detection range of {rm} AU\nAnd estimated ISO detections during {years} year mission, perihelion at {np.round(np.degrees(lon_per))} degrees")

    if guesses:
        plt.plot([EL_N, EL_N], [5, 48], ls='--', color="gray")
        plt.text(EL_N + 1, 40, "Ezell, Loeb mean", color="gray")
        plt.plot([HSP_N, HSP_N], [5, 48], ls='--', color="gray")
        plt.text(HSP_N + 1, 30, "Hoover, et al. mean /\nMarčeta, Seligman (conservative)", color="gray")
        plt.plot([MS_N, MS_N], [5, 48], ls='--', color="gray")
        plt.text(MS_N - 1, 20, "Marčeta, Seligman mean", ha="right", color="gray")
    if show:
        plt.show()

def distribution_histogram(rm:float, weight:dict,  show:bool=True, lon_per=None):
    '''generate the histogram of the dV requirements'''

    hist = get_dv_hist(rm, weight, lon_per=lon_per)
    print(f"fraction under 10 km/s: {np.sum(hist[:11]):.3f}\nunder 20 km/s: {np.sum(hist[:21]):.3f}\nunder 40 km/s: {np.sum(hist[:41]):.3f}")
    plt.bar(range(100),hist,width=1)
    plt.xlabel("dV requirement")
    plt.ylabel("probability density")
    plt.title(f"Normalized Histogram of the Delta V requirements for ISO {"rendezvous" if weight["w_relv"] else "intercept"}\nwith detection range {rm} AU. (Normalization includes unreachable ISOs)")
    if show:
        plt.show()



if __name__ == "__main__":

    # ==== settings =====
    icpt_weights = {"w_insertion":1, "w_relv": 0, "w_travel_time":0, "w_intercept_distance":0, "w_intercept_time":0}
    rdvz_weights = {"w_insertion":1, "w_relv": 1, "w_travel_time":0, "w_intercept_distance":0, "w_intercept_time":0}
    if rdvz:
        weight = rdvz_weights
    else:
        weight = icpt_weights

    rm = 5
    detect_distance = rm*AU
    max_time = 50*YEAR

    lon_vals = np.linspace(0, 360, 20)
    all_hists = []

    for lon_per in lon_vals:
        origin = orbit_from_ephemeris(
            2.61696589776 * AU,
            0.987573,
            m.radians(1.303),
            m.radians(100.46457166),
            m.radians(lon_per),
            m.radians(100.464),
            SGP_SUN
        )

        add_dv_hist(rm, weight, 10000, PLOT=PLOT, lon_per=np.radians(lon_per))

        hist = get_dv_hist(rm, weight, lon_per=np.radians(lon_per))
        distribution_histogram(2, icpt_weights, True, lon_per=lon_per)
        plt.figure()
        all_hists.append(hist)

        probability_map(rm, weight, lon_per=np.radians(lon_per))

    # Convert degrees → radians for polar plot
    theta = np.radians(lon_vals)

    # Compute all three thresholds
    frac_under_10 = []
    frac_under_20 = []
    frac_under_30 = []

    for hist in all_hists:  # <-- store hist in loop instead of just one value
        frac_under_10.append(np.sum(hist[:11]))  # 0–10
        frac_under_20.append(np.sum(hist[:21]))  # 0–20
        frac_under_30.append(np.sum(hist[:31]))  # 0–30

    # ==== Polar plot ====
    plt.figure()
    ax = plt.subplot(111, projection='polar')

    ax.plot(theta, frac_under_10, marker='o', label="< 10 km/s")
    ax.plot(theta, frac_under_20, marker='o', label="< 20 km/s")
    ax.plot(theta, frac_under_30, marker='o', label="< 30 km/s")

    ax.set_theta_zero_location("E")  # 0° at right (like your input)
    ax.set_theta_direction(1)  # counterclockwise

    ax.set_title("ΔV thresholds vs Longitude of Periapsis")
    ax.set_rlabel_position(135)
    ax.grid(True)
    ax.legend(loc="upper right")

    plt.show()

