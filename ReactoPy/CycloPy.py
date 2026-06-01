import numpy as np
import matplotlib.pyplot as plt

class IdealGas:
    def __init__(self, R, gamma):
        self.R = R
        self.gamma = gamma

# Example gasses

Helium = IdealGas(2077, 5/3)

class TSDiagram:
    """
    Ideal gas T-s diagram with:
    - states
    - isentropic + quasi-isentropic processes
    - compressor/turbine efficiency
    """

    def __init__(self, gas: IdealGas):
        self.R = gas.R
        self.gamma = gas.gamma
        self.cp = self.gamma * self.R / (self.gamma - 1)

        self.states = {}
        self.curves = []

    # -------------------------
    # THERMODYNAMICS
    # -------------------------
    def entropy(self, T, P):
        return self.cp * np.log(T) - self.R * np.log(P)

    def T_isentropic(self, T1, P1, P2):
        return T1 * (P2 / P1) ** ((self.gamma - 1) / self.gamma)

    # -------------------------
    # STATE MANAGEMENT
    # -------------------------
    def add_state(self, name, T, P):
        self.states[name] = {
            "T": T,
            "P": P,
            "s": self.entropy(T, P)
        }

    def get(self, name):
        return self.states[name]

    # -------------------------
    # IDEAL PROCESSES
    # -------------------------
    def isentropic(self, a, b, n=100, label="isentropic"):
        A = self.states[a]
        B = self.states[b]

        P_vals = np.linspace(A["P"], B["P"], n)

        T_vals = A["T"] * (P_vals / A["P"]) ** ((self.gamma - 1) / self.gamma)
        s_vals = self.entropy(T_vals, P_vals)

        self.curves.append({
            "s": s_vals,
            "T": T_vals,
            "label": label
        })

    def isobar(self, P, a, b, n=100, label="isobar"):
        A = self.states[a]
        B = self.states[b]

        T_vals = np.linspace(A["T"], B["T"], n)
        s_vals = self.entropy(T_vals, P)

        self.curves.append({
            "s": s_vals,
            "T": T_vals,
            "label": label
        })

    # -------------------------
    # QUASI-ISENTROPIC PROCESSES
    # -------------------------
    def compressor(self, a, b, eta_c, n=100, label="compressor (real)"):
        A = self.states[a]
        B = self.states[b]

        P_vals = np.linspace(A["P"], B["P"], n)

        T1 = A["T"]
        P1 = A["P"]

        T_vals = []
        for P2 in P_vals:
            T2s = self.T_isentropic(T1, P1, P2)
            T2 = T1 + (T2s - T1) / eta_c
            T_vals.append(T2)

        T_vals = np.array(T_vals)
        s_vals = self.entropy(T_vals, P_vals)

        self.curves.append({
            "s": s_vals,
            "T": T_vals,
            "label": label
        })

    def turbine(self, a, b, eta_t, n=100, label="turbine (real)"):
        A = self.states[a]
        B = self.states[b]

        P_vals = np.linspace(A["P"], B["P"], n)

        T3 = A["T"]
        P3 = A["P"]

        T_vals = []
        for P4 in P_vals:
            T4s = self.T_isentropic(T3, P3, P4)
            T4 = T3 - eta_t * (T3 - T4s)
            T_vals.append(T4)

        T_vals = np.array(T_vals)
        s_vals = self.entropy(T_vals, P_vals)

        self.curves.append({
            "s": s_vals,
            "T": T_vals,
            "label": label
        })

    # -------------------------
    # PLOT
    # -------------------------
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

def TSDiagramTest():
    ts = TSDiagram(Helium)

    # -------------------------
    # Conditions
    # -------------------------
    P1, P2 = 1e5, 8e5
    T1, T3 = 300, 1200

    eta_c = 0.85
    eta_t = 0.88

    # -------------------------
    # Ideal reference states
    # -------------------------
    T2s = ts.T_isentropic(T1, P1, P2)
    T4s = ts.T_isentropic(T3, P2, P1)

    ts.add_state("1", T1, P1)
    ts.add_state("2s", T2s, P2)
    ts.add_state("2", T1 + (T2s - T1) / eta_c, P2)

    ts.add_state("3", T3, P2)

    ts.add_state("4s", T4s, P1)
    ts.add_state("4", T3 - eta_t * (T3 - T4s), P1)

    # -------------------------
    # Ideal cycle
    # -------------------------
    ts.isentropic("1", "2s", label="1→2s ideal")
    ts.isobar(P2, "2", "3", label="2→3 heat addition")
    ts.isentropic("3", "4s", label="3→4s ideal")

    # -------------------------
    # REAL (quasi-isentropic)
    # -------------------------
    ts.compressor("1", "2", eta_c, label="1→2 compressor (real)")
    ts.turbine("3", "4", eta_t, label="3→4 turbine (real)")

    ts.plot("Brayton Cycle: Ideal vs Real (Quasi-Isentropic)")

if __name__ == "__main__":
    TSDiagramTest()