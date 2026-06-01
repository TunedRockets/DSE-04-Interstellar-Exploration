import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# Sources:

# https://inldigitallibrary.inl.gov/content/uploads/50/2026/04/Sort_107145.pdf
# https://ntrs.nasa.gov/api/citations/20220002034/downloads/Mass%20Modeling%20of%20NEP%20Power%20Conversion%20Concepts%20for%20Human%20Mars%20Exploration%2020220325.pdf



# =========================================================
# IDEAL GAS
# =========================================================
class IdealGas:
    def __init__(self, R, gamma):
        self.R = R
        self.gamma = gamma
        self.cp = gamma * R / (gamma - 1)
        self.cv = R / (gamma - 1)

# Example gasses

Helium = IdealGas(2077, 5/3)

# Constants

BAR = 100000

# =========================================================
# T-S DIAGRAM
# =========================================================
class TSDiagram:
    def __init__(self, gas: IdealGas):
        self.R = gas.R
        self.gamma = gas.gamma
        self.cp = gas.cp

        self.states = {}
        self.curves = []

    def entropy(self, T, P):
        return self.cp * np.log(T) - self.R * np.log(P)

    def T_isentropic(self, T1, P1, P2):
        return T1 * (P2 / P1) ** ((self.gamma - 1) / self.gamma)

    def add_state(self, name, T, P):
        self.states[name] = {
            "T": T,
            "P": P,
            "s": self.entropy(T, P)
        }

    def isentropic(self, a, b, label="isentropic"):
        A = self.states[a]
        B = self.states[b]

        P_vals = np.linspace(A["P"], B["P"], 100)
        T_vals = A["T"] * (P_vals / A["P"]) ** ((self.gamma - 1) / self.gamma)
        s_vals = self.entropy(T_vals, P_vals)

        self.curves.append({"s": s_vals, "T": T_vals, "label": label})

    def isobar(self, P, a, b, label="isobar"):
        A = self.states[a]
        B = self.states[b]

        T_vals = np.linspace(A["T"], B["T"], 100)
        s_vals = self.entropy(T_vals, P)

        self.curves.append({"s": s_vals, "T": T_vals, "label": label})

    # ---- FIXED: straight lines ----
    def compressor(self, a, b, eta_c, label="compressor (real)"):
        A = self.states[a]
        B = self.states[b]

        T1, P1 = A["T"], A["P"]
        P2 = B["P"]

        T2s = self.T_isentropic(T1, P1, P2)
        T2 = T1 + (T2s - T1) / eta_c

        s1 = self.entropy(T1, P1)
        s2 = self.entropy(T2, P2)

        self.curves.append({
            "s": np.array([s1, s2]),
            "T": np.array([T1, T2]),
            "label": label
        })

    def turbine(self, a, b, eta_t, label="turbine (real)"):
        A = self.states[a]
        B = self.states[b]

        T3, P3 = A["T"], A["P"]
        P4 = B["P"]

        T4s = self.T_isentropic(T3, P3, P4)
        T4 = T3 - eta_t * (T3 - T4s)

        s3 = self.entropy(T3, P3)
        s4 = self.entropy(T4, P4)

        self.curves.append({
            "s": np.array([s3, s4]),
            "T": np.array([T3, T4]),
            "label": label
        })

    def plot(self, title="T-s Diagram"):
        plt.figure(figsize=(8, 6))

        for c in self.curves:
            plt.plot(c["s"], c["T"], label=c["label"])

        for name, st in self.states.items():
            plt.scatter(st["s"], st["T"])
            plt.text(st["s"], st["T"], f" {name}")

        plt.xlabel("Entropy (J/kg·K)")
        plt.ylabel("Temperature (K)")
        plt.title(title)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()



# =========================================================
# BRAYTON CYCLE (kept, but corrected entropy usage)
# =========================================================
class BraytonCycle:
    def __init__(self, gas, eta_c, eta_t_hp, eta_t_lp):
        self.gas = gas
        self.cp = gas.cp
        self.gamma = gas.gamma
        self.R = gas.R

        self.eta_c = eta_c
        self.eta_t_hp = eta_t_hp
        self.eta_t_lp = eta_t_lp

        self.diagram = TSDiagram(gas)

    def T_isentropic(self, T, P1, P2):
        g = self.gamma
        return T * (P2 / P1) ** ((g - 1) / g)

    def entropy(self, T, P):
        return self.cp * np.log(T) - self.R * np.log(P)

    def compressor(self, T1, P1, P2):
        T2s = self.T_isentropic(T1, P1, P2)
        return T1 + (T2s - T1) / self.eta_c

    def turbine(self, T, P_in, P_out, eta):
        Ts = self.T_isentropic(T, P_in, P_out)
        return T - eta * (T - Ts)

    def w(self, T_out, T_in):
        return self.cp * (T_out - T_in)

    def solve_cycle(self, P1, P2, T1, T3):
        # =====================================================
        # 1. Compressor: 1 → 2
        # =====================================================
        T2 = self.compressor(T1, P1, P2)
        wc = self.w(T2, T1)

        # =====================================================
        # 2. Heat addition: 2 → 3
        # =====================================================
        q_in = self.w(T3, T2)

        # =====================================================
        # 3. HP turbine: solve for P4
        # constraint: w_hp = w_c
        # =====================================================

        def HP_residual(P4):
            T4s = self.T_isentropic(T3, P2, P4)
            T4 = T3 - self.eta_t_hp * (T3 - T4s)

            w_hp = self.w(T3, T4)

            return w_hp - wc

        P4 = fsolve(HP_residual, P1)[0]

        # compute final T4
        T4s = self.T_isentropic(T3, P2, P4)
        T4 = T3 - self.eta_t_hp * (T3 - T4s)

        # =====================================================
        # 4. LP turbine: P4 → P5
        # =====================================================
        P5 = P1

        T5 = self.turbine(T4, P4, P5, self.eta_t_lp)
        w_lp = self.w(T4, T5)

        # =====================================================
        # 5. Net work
        # =====================================================
        w_net = w_lp

        # =====================================================
        # Store states
        # =====================================================
        self.diagram.states.clear()

        self.diagram.add_state("1", T1, P1)
        self.diagram.add_state("2", T2, P2)
        self.diagram.add_state("3", T3, P2)
        self.diagram.add_state("4", T4, P4)
        self.diagram.add_state("5", T5, P5)

        return {
            "T1": T1, "T2": T2, "T3": T3,
            "T4": T4, "T5": T5,
            "P1": P1, "P2": P2, "P4": P4, "P5": P5,
            "wc": wc,
            "w_lp": w_lp,
            "net_specific": w_net,
            "q_in": q_in
        }

    def plot_ts(self):
        self.diagram.curves.clear()

        # self.diagram.isentropic("1", "2", "1-2 compressor")
        self.diagram.compressor("1", "2", self.eta_c, "1-2 compressor")
        self.diagram.isobar(self.diagram.states["2"]["P"], "2", "3", "2-3 heat input")
        # self.diagram.isentropic("3", "4", "3-4 turbine")
        self.diagram.turbine("3", "4", self.eta_t_hp, "3-4 HP turbine")
        # self.diagram.isobar(self.diagram.states["4"]["P"], "4", "5", "4-5 exhaust")
        self.diagram.turbine("4", "5", self.eta_t_lp, "4-5 LP turbine")
        self.diagram.isobar(self.diagram.states["5"]["P"], "5", "1", "5-1 closure")

        self.diagram.plot("Brayton Cycle")


# =========================================================
# TEST
# =========================================================
def TSDiagramTest():
    ts = TSDiagram(Helium)

    P1, P2 = 1e5, 8e5
    T1, T3 = 300, 1200

    ts.add_state("1", T1, P1)

    T2s = ts.T_isentropic(T1, P1, P2)
    ts.add_state("2s", T2s, P2)
    ts.add_state("2", T1 + (T2s - T1) / 0.85, P2)

    ts.add_state("3", T3, P2)

    T4s = ts.T_isentropic(T3, P2, P1)
    ts.add_state("4s", T4s, P1)
    ts.add_state("4", T3 - 0.88 * (T3 - T4s), P1)

    ts.isentropic("1", "2s", "1→2 ideal")
    ts.isobar(P2, "2", "3", "2→3 heat")
    ts.isentropic("3", "4s", "3→4 ideal")

    ts.compressor("1", "2", 0.85)
    ts.turbine("3", "4", 0.88)

    ts.plot("Brayton T-s (fixed)")

def efficiency_heatmap(engine, P1, T3):

    T1_vals = np.linspace(250, 350, 30)
    P2_vals = np.linspace(4*BAR, 20*BAR, 30)

    eta_map = np.zeros((len(T1_vals), len(P2_vals)))

    for i, T1 in enumerate(T1_vals):
        for j, P2 in enumerate(P2_vals):

            sol = engine.solve_cycle(P1, P2, T1, T3)

            eta = sol["net_specific"] / sol["q_in"]

            eta_map[i, j] = eta

    plt.figure(figsize=(8, 6))
    plt.imshow(
        eta_map,
        origin="lower",
        aspect="auto",
        extent=[P2_vals[0]/BAR, P2_vals[-1]/BAR,
                T1_vals[0], T1_vals[-1]]
    )

    plt.colorbar(label="Thermal Efficiency")

    plt.xlabel("P2 / P1 (pressure ratio)")
    plt.ylabel("T1 (K)")
    plt.title("Brayton Cycle Efficiency Map")

    plt.tight_layout()
    plt.show()

class BraytonSizing:
    def __init__(
        self,
        engine,
        radiator_areal_density=15.0,
        compressor_Ds=3.0,
        compressor_mass_coeff=2.0e4,
        turbine_specific_power=4000.0
    ):
        self.engine = engine

        # kg/m²
        self.radiator_areal_density = radiator_areal_density

        # Balje compressor parameters
        self.compressor_Ds = compressor_Ds
        self.compressor_mass_coeff = compressor_mass_coeff

        # W/kg
        self.turbine_specific_power = turbine_specific_power

    # =====================================================
    # MASS FLOW
    # =====================================================

    def mass_flow(self, W_elec, sol):
        return W_elec / sol["net_specific"]

    # =====================================================
    # COMPRESSOR MASS
    #
    # Balje diameter estimate:
    #
    # D = Ds * sqrt(Q) / Δh^(1/4)
    #
    # m = kc * D³
    # =====================================================

    def compressor_mass(self, mdot, wc, P1, T1):

        rho1 = P1 / (self.engine.R * T1)

        Q = mdot / rho1

        D = (
            self.compressor_Ds
            * np.sqrt(Q)
            / (wc ** 0.25)
        )

        m_comp = self.compressor_mass_coeff * D**3

        return m_comp

    # =====================================================
    # TURBINE MASS
    #
    # m = Power / Specific Power
    # =====================================================

    def turbine_mass(self, mdot, w_turb):

        Pturb = mdot * abs(w_turb)

        return Pturb / self.turbine_specific_power

    # =====================================================
    # RADIATOR MASS
    # =====================================================

    def radiator_mass(self, Q_rej, T_hot, emissivity=0.85):

        sigma = 5.670374419e-8
        T_space = 3.0

        q_flux = emissivity * sigma * (
            T_hot**4 - T_space**4
        )

        area = Q_rej / q_flux

        return area * self.radiator_areal_density

    # =====================================================
    # TOTAL MASS
    # =====================================================

    def estimate(self, W_elec, sol):

        mdot = self.mass_flow(W_elec, sol)

        m_comp = self.compressor_mass(
            mdot,
            sol["wc"],
            sol["P1"],
            sol["T1"]
        )

        m_turb = self.turbine_mass(
            mdot,
            sol["w_lp"]
        )

        cp = self.engine.cp

        Q_rej = mdot * cp * max(
            sol["T4"] - sol["T1"],
            0.0
        )

        m_rad = self.radiator_mass(
            Q_rej,
            sol["T1"]
        )

        total = m_comp + m_turb + m_rad

        return {
            "mass_flow": mdot,
            "compressor_mass": m_comp,
            "turbine_mass": m_turb,
            "radiator_mass": m_rad,
            "total_mass": total
        }

def evaluate_system(
    engine,
    sizer,
    W_elec,
    P1,
    P2,
    T1,
    T3
):

    sol = engine.solve_cycle(
        P1,
        P2,
        T1,
        T3
    )

    w_net = sol["net_specific"]

    if w_net <= 0:
        return None

    mdot = W_elec / w_net

    # -------------------------
    # Compressor
    # -------------------------

    m_comp = sizer.compressor_mass(
        mdot,
        sol["wc"],
        sol["P1"],
        sol["T1"]
    )

    # -------------------------
    # Turbine
    # -------------------------

    m_turb = sizer.turbine_mass(
        mdot,
        sol["w_lp"]
    )

    m_alternator = w_net/5000

    # -------------------------
    # Heat rejection
    # -------------------------

    cp = engine.cp

    Q_rej = mdot * cp * max(
        sol["T4"] - sol["T1"],
        0.0
    )

    m_rad = sizer.radiator_mass(
        Q_rej,
        sol["T1"]
    )

    total = m_comp + m_turb + m_rad + m_alternator

    return (
        total,
        m_comp,
        m_turb,
        m_rad,
        m_alternator,
        mdot
    )


def mass_heatmap(engine, W_elec, P1, T3,
                 min_T1=400, max_T1=1500,
                 min_P2=None, max_P2=None,
                 res=120,
                 limit=1000):
    if min_P2 is None:
        min_P2 = 1.1*P1

    if max_P2 is None:
        max_P2 = 10*P1

    sizer = BraytonSizing(engine)

    T1_vals = np.linspace(min_T1, max_T1, res)
    P2_vals = np.linspace(min_P2, max_P2, res)

    M = np.full((len(T1_vals), len(P2_vals)), np.nan)

    best = (1e99, None)

    for i, T1 in enumerate(T1_vals):
        for j, P2 in enumerate(P2_vals):

            out = evaluate_system(engine, sizer, W_elec, P1, P2, T1, T3)

            if out is None:
                continue

            total, mc, mt, mr, m_alternator, mdot = out
            if total>limit:
                total=np.nan

            M[i, j] = total

            if total < best[0]:
                best = (total, (T1, P2, mc, mt, mr, m_alternator, mdot))

    # =========================
    # PLOT MASS MAP
    # =========================
    plt.figure(figsize=(9, 6))

    plt.imshow(
        M,
        origin="lower",
        aspect="auto",
        extent=[P2_vals[0]/BAR, P2_vals[-1]/BAR,
                T1_vals[0], T1_vals[-1]]
    )

    plt.colorbar(label="Total Mass (kg)")
    plt.xlabel("HP Pressure P2 (Bar)")
    plt.ylabel("T1 (K)")
    plt.title("Brayton System Mass Map")

    # =========================
    # OPTIMUM
    # =========================
    T1_opt, P2_opt, mc, mt, mr, m_alternator, mdot = best[1]

    plt.scatter(P2_opt/BAR, T1_opt, color="red", label="optimum")
    plt.legend()
    plt.tight_layout()
    plt.show()

    sol = engine.solve_cycle(P1, P2_opt, T1_opt, T3)

    efficiency = sol["net_specific"]/sol["q_in"]
    engine.plot_ts()

    # =========================
    # MASS FLOW PLOT AT OPTIMUM
    # =========================
    print("\n--- OPTIMUM DESIGN ---")
    print(f"T1: {T1_opt:.2f} K")
    print(f"P2: {P2_opt/BAR:.2f} bar")
    print(f"Mass flow: {mdot:.4f} kg/s")
    print(f"Thermal efficiency: {100*efficiency:.2f} %")
    print(f"Compressor mass: {mc:.2f} kg")
    print(f"Turbine mass: {mt:.2f} kg")
    print(f"Alternator mass: {m_alternator:.2f} kg")
    print(f"Radiator mass: {mr:.2f} kg")
    print(f"TOTAL MASS: {best[0]:.2f} kg")

    return best

# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":

    TSDiagramTest()

    # engine = BraytonCycle(Helium, 0.85, 0.88, 0.90)

    # sol = engine.solve_cycle(
    #     P1=1*BAR,
    #     P2=10*BAR,
    #     T1=300,
    #     T3=1500+273.15
    # )
    #
    # engine.plot_ts()
    #
    # print("Net specific work:", sol["net_specific"], "W / (kg/s)")
    # print("Overall efficiency:", sol["net_specific"]/sol["q_in"])
    #
    #
    # efficiency_heatmap(engine, 1*BAR, 1500+273.15)

    engine = BraytonCycle(Helium, 0.85, 0.88, 0.90)
    pressures = np.linspace(10, 40, 10)
    masses = []
    iss_radiator_pressure = 3447 * 1000  # Pa

    for pressure in pressures:
        best = mass_heatmap(
            engine,
            W_elec=30000,
            P1=pressure * BAR,
            T3=2000 + 273.15
        )
        print(f"LP Pressure: {pressure:.2f} bar")
        masses.append(best[0])

    # Convert ISS radiator pressure to bar
    iss_radiator_pressure_bar = iss_radiator_pressure / BAR

    # Plot
    plt.figure()
    plt.plot(pressures, masses, marker='o')
    plt.axvline(
        x=iss_radiator_pressure_bar,
        color='r',
        linestyle='--',
        label=f'ISS radiator ({iss_radiator_pressure_bar:.2f} bar)'
    )

    plt.xlabel("Pressure (bar)")
    plt.ylabel("Mass")
    plt.title("System Mass vs Helium Pressure")
    plt.grid(True)
    plt.legend()
    plt.show()