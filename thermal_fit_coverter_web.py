"""
thermal_fit_convert.py

Small helper layer on top of:
  https://github.com/erickschulz/thermal-network

Features:
- Fit Foster (parallel RC) network to Zth(t) data
- Convert Foster <-> Cauer
- Sanity checks (DC Rth, monotonicity, positivity, ordering, round-trip stability)

Install (one-time):
  pip install git+https://github.com/erickschulz/thermal-network

Typical usage:
  from thermal_fit_convert import fit_foster_from_zth, foster_to_cauer_rc, run_all_sanity_checks

  foster, report = fit_foster_from_zth(tp, zth, n_layers=4, trim_ss=True)
  cauer = foster_to_cauer_rc(foster)

  run_all_sanity_checks(tp, zth, foster=foster, cauer=cauer, verbose=True)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union, Any

import numpy as np

from thermal_network.networks import FosterNetwork, CauerNetwork
from thermal_network.conversions import foster_to_cauer, cauer_to_foster
from thermal_network.fitting import (
    fit_foster_network,
    fit_optimal_foster_network,
    trim_steady_state,
    OptimizationConfig,
)
from thermal_network.impedance import (
    foster_impedance_time_domain,
    foster_impedance_freq_domain,
    cauer_impedance_freq_domain,
)

try:
    from flask import Flask, jsonify, request
except Exception:  # pragma: no cover - optional dependency for web usage
    Flask = None  # type: ignore
    jsonify = None  # type: ignore
    request = None  # type: ignore


ArrayLike = Union[np.ndarray, list, tuple]


# ----------------------------
# Helper data structures
# ----------------------------

@dataclass
class FitReport:
    """Human-readable + machine-usable summary of what happened."""
    n_layers: int
    dc_rth_data: float
    dc_rth_model: float
    rmse: float
    max_abs_err: float
    notes: Tuple[str, ...] = ()


# ----------------------------
# Core: Fit Foster model
# ----------------------------

def fit_foster_from_zth(
    tp: ArrayLike,
    zth: ArrayLike,
    n_layers: Optional[int] = None,
    *,
    max_layers: int = 10,
    selection_criterion: str = "bic",
    trim_ss: bool = True,
    tau_floor: Optional[float] = None,
    config: Optional[OptimizationConfig] = None,
    random_seed: int = 0,
) -> Tuple[FosterNetwork, FitReport]:
    """
    Fit Zth(t) to a Foster network.

    Parameters
    ----------
    tp, zth:
      Time points and impedance values (same length). tp must be > 0 and increasing.
    n_layers:
      If provided, fit exactly this order.
      If None, do automatic model selection from 1..max_layers using selection_criterion.
    trim_ss:
      If True, trims steady-state tail per library helper (keeps first steady-state point).
    tau_floor:
      If set, constrains smallest time constant tau_i = R_i*C_i to be >= tau_floor.
    config:
      OptimizationConfig for optimizer / tolerances etc.
    random_seed:
      Seed for initial guess randomization.

    Returns
    -------
    (FosterNetwork, FitReport)
    """
    tp = np.asarray(tp, dtype=float).copy()
    zth = np.asarray(zth, dtype=float).copy()
    _basic_input_checks(tp, zth)

    notes = []

    # Optional: trim steady-state tail (common when datasheet has long flat tail).
    if trim_ss:
        tp2, zth2 = trim_steady_state(tp, zth)
        if len(tp2) < len(tp):
            notes.append(f"Trimmed steady-state tail: {len(tp)} -> {len(tp2)} points.")
        tp, zth = tp2, zth2

    # Fit
    if n_layers is not None:
        if n_layers < 1:
            raise ValueError("n_layers must be >= 1.")
        result = fit_foster_network(
            tp, zth, n_layers, config=config, random_seed=random_seed, tau_floor=tau_floor
        )
        foster = result.network
    else:
        if max_layers < 1:
            raise ValueError("max_layers must be >= 1.")
        if selection_criterion.lower() not in ("aic", "bic"):
            raise ValueError("selection_criterion must be 'aic' or 'bic'.")
        sel = fit_optimal_foster_network(
            tp,
            zth,
            max_layers=max_layers,
            selection_criterion=selection_criterion.lower(),
            config=config,
            random_seed=random_seed,
            tau_floor=tau_floor,
        )
        foster = sel.best_model.network
        notes.append(
            f"Auto-selected {sel.best_model.n_layers} layers by {selection_criterion.upper()}."
        )

    # Build fit report
    z_fit = foster_impedance_time_domain(foster, tp)
    rmse = float(np.sqrt(np.mean((z_fit - zth) ** 2)))
    max_abs_err = float(np.max(np.abs(z_fit - zth)))

    dc_data = float(zth[-1])
    dc_model = float(np.sum(foster.r))  # Foster Zth(t->inf) = sum(R_i)

    report = FitReport(
        n_layers=int(len(foster.r)),
        dc_rth_data=dc_data,
        dc_rth_model=dc_model,
        rmse=rmse,
        max_abs_err=max_abs_err,
        notes=tuple(notes),
    )

    # Run light checks immediately; full checks are in run_all_sanity_checks().
    _check_positive_rc(foster.r, foster.c, name="Foster")
    _check_time_constant_ordering(foster)

    return foster, report


# ----------------------------
# Core: Convert Foster <-> Cauer
# ----------------------------

def foster_to_cauer_rc(foster: FosterNetwork) -> CauerNetwork:
    """Convert Foster -> Cauer using library symbolic continued-fraction expansion."""
    return foster_to_cauer(foster)


def cauer_to_foster_rc(cauer: CauerNetwork) -> FosterNetwork:
    """Convert Cauer -> Foster using library symbolic partial-fraction expansion."""
    return cauer_to_foster(cauer)


# ----------------------------
# Sanity checks (recommended)
# ----------------------------

def run_all_sanity_checks(
    tp: ArrayLike,
    zth: ArrayLike,
    *,
    foster: Optional[FosterNetwork] = None,
    cauer: Optional[CauerNetwork] = None,
    rtol_dc: float = 0.03,
    rtol_roundtrip: float = 1e-6,
    rtol_impedance: float = 1e-3,
    verbose: bool = True,
) -> Dict[str, bool]:
    """
    A practical battery of checks.

    Checks:
    - input data sanity (tp inc, tp>0, Zth finite, mostly nonnegative)
    - Zth monotonic nondecreasing (soft check; allows small noise)
    - Foster/Cauer positivity
    - DC Rth consistency: Zth(t_last) ~ sum(R_foster)  (rtol_dc)
    - Foster tau ordering is canonical
    - Round-trip conversion stability:
        Foster -> Cauer -> Foster (params close)
        Cauer  -> Foster -> Cauer  (params close)
    - Impedance equivalence in frequency domain: |Z_foster(jw)| ~ |Z_cauer(jw)|

    Returns dict of check_name -> pass/fail.
    """
    tp = np.asarray(tp, dtype=float)
    zth = np.asarray(zth, dtype=float)
    _basic_input_checks(tp, zth)

    results: Dict[str, bool] = {}

    results["data_monotonic_soft"] = _check_zth_monotonic_soft(zth, tol=0.01)

    if foster is not None:
        results["foster_positive_rc"] = _check_positive_rc(foster.r, foster.c, name="Foster", ret_bool=True)
        results["foster_tau_ordering"] = _check_time_constant_ordering(foster, ret_bool=True)

        dc_ok = np.isclose(float(zth[-1]), float(np.sum(foster.r)), rtol=rtol_dc, atol=0.0)
        results["dc_rth_match_data_vs_foster"] = bool(dc_ok)

    if cauer is not None:
        results["cauer_positive_rc"] = _check_positive_rc(cauer.r, cauer.c, name="Cauer", ret_bool=True)

    # Round-trip checks (only if we have at least one network)
    if foster is not None:
        c1 = foster_to_cauer_rc(foster)
        f2 = cauer_to_foster_rc(c1)

        # FosterNetwork is canonical-sorted by tau; compare arrays directly.
        results["roundtrip_foster_params"] = bool(
            np.allclose(np.asarray(foster.r), np.asarray(f2.r), rtol=rtol_roundtrip, atol=0.0)
            and np.allclose(np.asarray(foster.c), np.asarray(f2.c), rtol=rtol_roundtrip, atol=0.0)
        )

        # Impedance match check (freq domain) vs converted Cauer
        results["foster_vs_cauer_impedance_freq"] = _check_impedance_equivalence_freq(
            foster=foster, cauer=c1, rtol=rtol_impedance
        )

    if cauer is not None:
        f1 = cauer_to_foster_rc(cauer)
        c2 = foster_to_cauer_rc(f1)
        results["roundtrip_cauer_params"] = bool(
            np.allclose(np.asarray(cauer.r), np.asarray(c2.r), rtol=rtol_roundtrip, atol=0.0)
            and np.allclose(np.asarray(cauer.c), np.asarray(c2.c), rtol=rtol_roundtrip, atol=0.0)
        )

    if verbose:
        _print_check_summary(results)

    return results


# ----------------------------
# Internal checks / utilities
# ----------------------------

def _basic_input_checks(tp: np.ndarray, zth: np.ndarray) -> None:
    if tp.ndim != 1 or zth.ndim != 1:
        raise ValueError("tp and zth must be 1D arrays.")
    if len(tp) != len(zth):
        raise ValueError("tp and zth must have the same length.")
    if len(tp) < 5:
        raise ValueError("Need at least ~5 points for a meaningful fit.")
    if not np.all(np.isfinite(tp)) or not np.all(np.isfinite(zth)):
        raise ValueError("tp and zth must be finite.")
    if np.any(tp <= 0):
        raise ValueError("All tp must be > 0.")
    if np.any(np.diff(tp) <= 0):
        raise ValueError("tp must be strictly increasing.")

    # Zth for a step response should be >= 0; allow slight negative due to digitization noise.
    if np.min(zth) < -0.02 * max(1e-12, float(np.max(np.abs(zth)))):
        raise ValueError("zth has strongly negative values; check sign/unit/digitization.")


def _check_zth_monotonic_soft(zth: np.ndarray, tol: float = 0.01) -> bool:
    """
    Thermal step-response impedance should be nondecreasing.
    Allow small violations due to noise: up to tol * full_scale.
    """
    full_scale = max(1e-12, float(np.max(zth) - np.min(zth)))
    drops = np.diff(zth)
    worst_drop = float(np.min(drops))
    return worst_drop >= -tol * full_scale


def _check_positive_rc(r: ArrayLike, c: ArrayLike, *, name: str, ret_bool: bool = False) -> bool:
    r = np.asarray(r, dtype=float)
    c = np.asarray(c, dtype=float)
    ok = bool(np.all(r > 0) and np.all(c > 0))
    if not ok and not ret_bool:
        raise ValueError(f"{name} has non-positive R or C values.")
    return ok


def _check_time_constant_ordering(foster: FosterNetwork, *, ret_bool: bool = False) -> bool:
    """
    FosterNetwork is supposed to be stored in canonical tau-sorted order (tau increasing).
    (Library claims auto-sorting by tau at init.)
    """
    r = np.asarray(foster.r, dtype=float)
    c = np.asarray(foster.c, dtype=float)
    tau = r * c
    ok = bool(np.all(np.diff(tau) >= 0))
    if not ok and not ret_bool:
        raise ValueError("Foster time constants are not sorted (tau not nondecreasing).")
    return ok


def _check_impedance_equivalence_freq(
    *, foster: FosterNetwork, cauer: CauerNetwork, rtol: float
) -> bool:
    """
    Verify Foster and Cauer frequency-domain impedances are equivalent:
      Z_foster(jw) == Z_cauer(jw)
    We compare magnitudes over a log-spaced frequency sweep.
    """
    # Choose a sweep that covers the Foster taus loosely
    tau = np.asarray(foster.r) * np.asarray(foster.c)
    tau_min = float(np.min(tau))
    tau_max = float(np.max(tau))

    # frequencies around 1/tau; broaden by 2 decades each side
    w_min = 1.0 / max(1e-18, tau_max) / 100.0
    w_max = 1.0 / max(1e-18, tau_min) * 100.0

    w = np.logspace(np.log10(w_min), np.log10(w_max), 200)
    s = 1j * w

    zf = foster_impedance_freq_domain(foster, s)
    zc = cauer_impedance_freq_domain(cauer, s)

    # Compare relative error on magnitude (robust to tiny complex phase differences)
    mag_f = np.abs(zf)
    mag_c = np.abs(zc)
    denom = np.maximum(1e-18, mag_f)
    rel_err = np.max(np.abs(mag_f - mag_c) / denom)
    return bool(rel_err <= rtol)


def _print_check_summary(results: Dict[str, bool]) -> None:
    width = max(len(k) for k in results.keys())
    print("\nSanity check summary")
    print("-" * (width + 10))
    for k, v in results.items():
        print(f"{k:<{width}}  :  {'PASS' if v else 'FAIL'}")
    print("-" * (width + 10))


# ----------------------------
# Optional CLI demo
# ----------------------------

def _parse_array(name: str, value: Any) -> np.ndarray:
    if not isinstance(value, (list, tuple, np.ndarray)):
        raise ValueError(f"{name} must be a list of numbers.")
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D list of numbers.")
    return arr


def _network_to_dict(network: Union[FosterNetwork, CauerNetwork]) -> Dict[str, list]:
    return {
        "r": [float(x) for x in np.asarray(network.r, dtype=float)],
        "c": [float(x) for x in np.asarray(network.c, dtype=float)],
    }


def _report_to_dict(report: FitReport) -> Dict[str, Any]:
    return {
        "n_layers": report.n_layers,
        "dc_rth_data": report.dc_rth_data,
        "dc_rth_model": report.dc_rth_model,
        "rmse": report.rmse,
        "max_abs_err": report.max_abs_err,
        "notes": list(report.notes),
    }


def _example_data() -> Tuple[np.ndarray, np.ndarray]:
    true_f = FosterNetwork(r=[0.2, 0.8, 0.5], c=[15.0, 1.0, 4.0])
    tp = np.logspace(-2, 2, 250)
    z_clean = foster_impedance_time_domain(true_f, tp)
    rng = np.random.default_rng(0)
    z_noisy = z_clean + 0.01 * np.max(z_clean) * rng.standard_normal(z_clean.shape)
    z_noisy = np.maximum(z_noisy, 1e-12)
    return tp, z_noisy


def create_app() -> "Flask":
    if Flask is None:
        raise RuntimeError("Flask is not installed. Run `pip install flask`.")

    app = Flask(__name__)

    @app.get("/")
    def health() -> Any:
        return jsonify({"status": "ok"})

    @app.post("/fit")
    def fit_endpoint() -> Any:
        payload = request.get_json(silent=True) or {}
        try:
            if not payload:
                tp, zth = _example_data()
                foster, report = fit_foster_from_zth(tp, zth, n_layers=3, trim_ss=False)
                return jsonify(
                    {
                        "foster": _network_to_dict(foster),
                        "report": _report_to_dict(report),
                        "example": True,
                    }
                )

            tp = _parse_array("tp", payload.get("tp"))
            zth = _parse_array("zth", payload.get("zth"))

            n_layers = payload.get("n_layers")
            max_layers = int(payload.get("max_layers", 10))
            selection_criterion = str(payload.get("selection_criterion", "bic"))
            trim_ss = bool(payload.get("trim_ss", True))
            tau_floor = payload.get("tau_floor")
            random_seed = int(payload.get("random_seed", 0))

            foster, report = fit_foster_from_zth(
                tp,
                zth,
                n_layers=n_layers,
                max_layers=max_layers,
                selection_criterion=selection_criterion,
                trim_ss=trim_ss,
                tau_floor=tau_floor,
                config=None,
                random_seed=random_seed,
            )

            return jsonify(
                {
                    "foster": _network_to_dict(foster),
                    "report": _report_to_dict(report),
                }
            )
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

    @app.post("/convert")
    def convert_endpoint() -> Any:
        payload = request.get_json(silent=True) or {}
        try:
            if not payload:
                tp, zth = _example_data()
                foster, _ = fit_foster_from_zth(tp, zth, n_layers=3, trim_ss=False)
                cauer = foster_to_cauer_rc(foster)
                return jsonify(
                    {
                        "foster": _network_to_dict(foster),
                        "cauer": _network_to_dict(cauer),
                        "example": True,
                    }
                )

            mode = str(payload.get("mode", "")).strip().lower()
            if mode == "foster_to_cauer":
                if "foster" in payload:
                    fr = payload["foster"] or {}
                    foster = FosterNetwork(
                        r=_parse_array("foster.r", fr.get("r")),
                        c=_parse_array("foster.c", fr.get("c")),
                    )
                else:
                    foster = FosterNetwork(
                        r=_parse_array("r", payload.get("r")),
                        c=_parse_array("c", payload.get("c")),
                    )
                cauer = foster_to_cauer_rc(foster)
                return jsonify({"cauer": _network_to_dict(cauer)})
            if mode == "cauer_to_foster":
                if "cauer" in payload:
                    cr = payload["cauer"] or {}
                    cauer = CauerNetwork(
                        r=_parse_array("cauer.r", cr.get("r")),
                        c=_parse_array("cauer.c", cr.get("c")),
                    )
                else:
                    cauer = CauerNetwork(
                        r=_parse_array("r", payload.get("r")),
                        c=_parse_array("c", payload.get("c")),
                    )
                foster = cauer_to_foster_rc(cauer)
                return jsonify({"foster": _network_to_dict(foster)})

            raise ValueError("mode must be 'foster_to_cauer' or 'cauer_to_foster'.")
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

    @app.post("/checks")
    def checks_endpoint() -> Any:
        payload = request.get_json(silent=True) or {}
        try:
            if not payload:
                tp, zth = _example_data()
                foster, _ = fit_foster_from_zth(tp, zth, n_layers=3, trim_ss=False)
                cauer = foster_to_cauer_rc(foster)
                results = run_all_sanity_checks(
                    tp,
                    zth,
                    foster=foster,
                    cauer=cauer,
                    verbose=False,
                )
                return jsonify(
                    {
                        "results": results,
                        "foster": _network_to_dict(foster),
                        "cauer": _network_to_dict(cauer),
                        "example": True,
                    }
                )

            tp = _parse_array("tp", payload.get("tp"))
            zth = _parse_array("zth", payload.get("zth"))

            foster = None
            if "foster" in payload:
                fr = payload["foster"] or {}
                foster = FosterNetwork(
                    r=_parse_array("foster.r", fr.get("r")),
                    c=_parse_array("foster.c", fr.get("c")),
                )

            cauer = None
            if "cauer" in payload:
                cr = payload["cauer"] or {}
                cauer = CauerNetwork(
                    r=_parse_array("cauer.r", cr.get("r")),
                    c=_parse_array("cauer.c", cr.get("c")),
                )

            results = run_all_sanity_checks(
                tp,
                zth,
                foster=foster,
                cauer=cauer,
                verbose=False,
            )
            return jsonify({"results": results})
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=8000, debug=False)
