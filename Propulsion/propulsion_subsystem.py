"""
Propulsion subsystem preliminary sizing tool for the ISO mission report.

Uses report equations:
  Eq. 3.16: DeltaV = Isp*g0*ln(m0/mf)
  Eq. 3.17: DeltaV_total = sum over sequential burns
  Eq. 3.18: F = mdot*Isp*g0

Extra equations added for useful subsystem sizing:
  mf = m0 / exp(DeltaV/(Isp*g0))       [rearranged Tsiolkovsky]
  propellant_used = m0 - mf
  burn_time = propellant_used / mdot
  fuel_mass = propellant_used/(1 + oxidizer_to_fuel_ratio)  [biprop split]
  oxidizer_mass = propellant_used - fuel_mass

All default numbers are arbitrary placeholders and should be replaced by your
final trajectory, launcher, spacecraft, and propulsion selections.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp
from typing import Iterable, Optional

G0 = 9.80665  # standard gravity [m/s^2]


@dataclass(frozen=True)
class Engine:
    """Basic engine/thruster definition."""

    name: str
    isp_s: float                 # specific impulse [s]
    thrust_n: Optional[float] = None  # thrust [N], optional for burn-time estimate
    mixture_ratio_of: Optional[float] = None  # oxidizer/fuel mass ratio for bipropellant

    @property
    def exhaust_velocity_m_s(self) -> float:
        """ve = Isp * g0."""
        return self.isp_s * G0


@dataclass(frozen=True)
class BurnResult:
    burn_number: int
    delta_v_m_s: float
    initial_mass_kg: float
    final_mass_kg: float
    propellant_used_kg: float
    mass_flow_kg_s: Optional[float]
    burn_time_s: Optional[float]
    fuel_mass_kg: Optional[float]
    oxidizer_mass_kg: Optional[float]


def final_mass_after_burn(initial_mass_kg: float, delta_v_m_s: float, engine: Engine) -> float:
    """Rearranged Tsiolkovsky rocket equation."""
    if initial_mass_kg <= 0:
        raise ValueError("initial_mass_kg must be positive")
    if delta_v_m_s < 0:
        raise ValueError("delta_v_m_s cannot be negative")
    if engine.isp_s <= 0:
        raise ValueError("engine.isp_s must be positive")

    return initial_mass_kg / exp(delta_v_m_s / engine.exhaust_velocity_m_s)


def propellant_for_burn(initial_mass_kg: float, delta_v_m_s: float, engine: Engine) -> float:
    """Propellant needed for one impulsive burn."""
    return initial_mass_kg - final_mass_after_burn(initial_mass_kg, delta_v_m_s, engine)


def mass_flow_rate(engine: Engine) -> Optional[float]:
    """Eq. 3.18 rearranged: mdot = F/(Isp*g0)."""
    if engine.thrust_n is None:
        return None
    if engine.thrust_n <= 0:
        raise ValueError("engine.thrust_n must be positive when provided")
    return engine.thrust_n / engine.exhaust_velocity_m_s


def split_bipropellant(propellant_kg: float, oxidizer_to_fuel_ratio: Optional[float]) -> tuple[Optional[float], Optional[float]]:
    """Split total propellant into fuel and oxidizer for bipropellant systems."""
    if oxidizer_to_fuel_ratio is None:
        return None, None
    if oxidizer_to_fuel_ratio <= 0:
        raise ValueError("oxidizer_to_fuel_ratio must be positive")

    fuel = propellant_kg / (1.0 + oxidizer_to_fuel_ratio)
    oxidizer = propellant_kg - fuel
    return fuel, oxidizer


def make_equal_burns(total_delta_v_m_s: float, number_of_burns: int) -> list[float]:
    """Create an arbitrary equal-burn schedule when only total Delta-V and burn count are known."""
    if number_of_burns <= 0:
        raise ValueError("number_of_burns must be at least 1")
    if total_delta_v_m_s < 0:
        raise ValueError("total_delta_v_m_s cannot be negative")
    return [total_delta_v_m_s / number_of_burns] * number_of_burns


def simulate_burn_sequence(initial_mass_kg: float, burn_delta_vs_m_s: Iterable[float], engine: Engine) -> list[BurnResult]:
    """Apply Eq. 3.17: after every burn, the final mass becomes the next initial mass."""
    results: list[BurnResult] = []
    current_mass = initial_mass_kg
    mdot = mass_flow_rate(engine)

    for i, dv in enumerate(burn_delta_vs_m_s, start=1):
        final_mass = final_mass_after_burn(current_mass, dv, engine)
        propellant = current_mass - final_mass
        burn_time = None if mdot is None else propellant / mdot
        fuel, oxidizer = split_bipropellant(propellant, engine.mixture_ratio_of)

        results.append(
            BurnResult(
                burn_number=i,
                delta_v_m_s=dv,
                initial_mass_kg=current_mass,
                final_mass_kg=final_mass,
                propellant_used_kg=propellant,
                mass_flow_kg_s=mdot,
                burn_time_s=burn_time,
                fuel_mass_kg=fuel,
                oxidizer_mass_kg=oxidizer,
            )
        )
        current_mass = final_mass

    return results


def print_results(system_name: str, engine: Engine, results: list[BurnResult]) -> None:
    """Pretty-print burn sizing results."""
    total_dv = sum(r.delta_v_m_s for r in results)
    total_propellant = sum(r.propellant_used_kg for r in results)
    total_fuel = sum(r.fuel_mass_kg or 0.0 for r in results)
    total_oxidizer = sum(r.oxidizer_mass_kg or 0.0 for r in results)
    total_burn_time = None if any(r.burn_time_s is None for r in results) else sum(r.burn_time_s or 0.0 for r in results)

    print(f"\n{system_name}")
    print("-" * len(system_name))
    print(f"Engine/thruster: {engine.name}")
    print(f"Isp: {engine.isp_s:.1f} s | ve: {engine.exhaust_velocity_m_s:.1f} m/s")
    print(f"Number of burns: {len(results)} | Total Delta-V: {total_dv:.1f} m/s")
    print(f"Initial mass: {results[0].initial_mass_kg:.2f} kg")
    print(f"Final mass:   {results[-1].final_mass_kg:.2f} kg")
    print(f"Total propellant used: {total_propellant:.2f} kg")

    if engine.mixture_ratio_of is not None:
        print(f"  Fuel used:     {total_fuel:.2f} kg")
        print(f"  Oxidizer used: {total_oxidizer:.2f} kg")
    else:
        print("  Fuel/oxidizer split: not applicable or not specified")

    if total_burn_time is not None:
        print(f"Total estimated burn time: {total_burn_time:.1f} s ({total_burn_time / 3600:.2f} h)")

    for r in results:
        print(
            f"  Burn {r.burn_number}: DV={r.delta_v_m_s:.1f} m/s, "
            f"m0={r.initial_mass_kg:.2f} kg, mf={r.final_mass_kg:.2f} kg, "
            f"prop={r.propellant_used_kg:.2f} kg"
        )


def main() -> None:
    # Arbitrary placeholder inputs requested by the user.
    # Replace these with final launcher/spacecraft trajectory values later.
    launcher_total_delta_v = 9_400.0     # m/s, arbitrary launcher insertion value
    launcher_number_of_burns = 2
    launcher_initial_mass = 549_000.0    # kg, arbitrary launcher stack/upper-stage mass

    spacecraft_total_delta_v = 18_000.0  # m/s, arbitrary ISO transfer/intercept value
    spacecraft_number_of_burns = 4
    spacecraft_initial_mass = 3_000.0    # kg, report mentions ~3000 kg preliminary spacecraft mass

    # Engines based on report-listed technology ranges.
    # Launcher: chemical LOX/RP-1 placeholder, Isp within listed 300-350 s range.
    launcher_engine = Engine(
        name="Chemical LOX/RP-1 placeholder",
        isp_s=330.0,
        thrust_n=7_600_000.0,
        mixture_ratio_of=2.56,  # typical O/F; added for fuel/oxidizer split, not from report
    )

    # Spacecraft: NEXT-like ion propulsion placeholder, Isp within listed 4100-4200 s range.
    spacecraft_engine = Engine(
        name="NEXT-like ion thruster placeholder",
        isp_s=4150.0,
        thrust_n=0.236,
        mixture_ratio_of=None,  # xenon/electric propulsion: total propellant only
    )

    launcher_burns = make_equal_burns(launcher_total_delta_v, launcher_number_of_burns)
    spacecraft_burns = make_equal_burns(spacecraft_total_delta_v, spacecraft_number_of_burns)

    launcher_results = simulate_burn_sequence(launcher_initial_mass, launcher_burns, launcher_engine)
    spacecraft_results = simulate_burn_sequence(spacecraft_initial_mass, spacecraft_burns, spacecraft_engine)

    print_results("Launcher propulsion estimate", launcher_engine, launcher_results)
    print_results("Spacecraft propulsion estimate", spacecraft_engine, spacecraft_results)

    combined_propellant = sum(r.propellant_used_kg for r in launcher_results + spacecraft_results)
    print("\nMission total")
    print("-------------")
    print(f"Combined propellant used by launcher + spacecraft: {combined_propellant:.2f} kg")


if __name__ == "__main__":
    main()
