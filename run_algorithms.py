# run_algorithms.py
# VS Code–friendly: runs with the CONFIG block (no CLI) or via CLI.
from __future__ import annotations
import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt

from game import Game
from komwu import (
    Komwu,
    KomwuL,
    KomwuEO,
    KomwuEOO,
    KomwuDecay,
    KomwuEOEnd,
    KomwuKD,
    KomwuEOKD,
    KomwuTolShrinkExp,
    KomwuTolShrinkRuleA,
    KomwuTolShrinkRuleBScale,
    KomwuTolShrinkRuleBStop,
    KomwuTolShrinkCutoff,
    KomwuTolShrinkLip,
    KomwuMinLogitsPeriodic,
    KomwuMinLogitsAlways,
)

# ====================== EDIT THIS WHEN RUNNING IN VS CODE ======================
CONFIG = {
    "game": "game_instances/K23.game",
    "algos": "omwu,tolshrink_exp,tolshrink_rulea,tolshrink_ruleb_scale,tolshrink_ruleb_stop,tolshrink_cutoff,tolshrink_lip,minlogits_periodic,minlogits_always",  # defaults: baseline + all shrink variants
    "L": 5,
    "alpha_b": 1.0,
    "alpha_s": 3.0,
    "Kf": 3,
    "rho": 1.0,
    "T": 20000,
    "eta": 1.0,
    "precision": "normal",
    "print_every": 500,
    "plot_every": 100,
    "seed": 0,
    # When True: track and plot max per-player regret over time (no bar chart).
    "plot_total_regret": True,
}
# =============================================================================

# =========================== numerics helpers ===========================

def get_dtype(precision: str):
    if precision not in {"normal", "precise"}:
        raise ValueError("precision must be 'normal' or 'precise'")
    return np.float64 if precision == "normal" else np.longdouble


def utility_gradient_safe(game: Game, p: int, strategies: dict[int, np.ndarray]) -> np.ndarray:
    """
    Wraps game.utility_gradient to ignore *underflow* only.
    Underflow of reach products to 0 is expected off-reach and harmless.
    """
    with np.errstate(under='ignore'):
        return game.utility_gradient(p, strategies)

# ================================ metrics ===============================

def nash_conv_and_ev(
    game: Game,
    x: dict[int, np.ndarray],
    grads: list[np.ndarray] | None = None,
) -> tuple[float, float, float]:
    """
    Returns (nashconv, exploitability, ev0).

    - For 2-player zero-sum games:
        nashconv       = (BR_0 - EV_0) + (BR_1 - EV_1)
        exploitability = nashconv / 2
      so this matches the usual definition.

    - For n_players > 2:
        nashconv       = NaN
        exploitability = mean_p [ BR_p - EV_p ]  (average per-player BR gap)

    ev0 is always the expected value for player 0 under x.
    """
    n_players = game.n_players

    if grads is None:
        grads = [utility_gradient_safe(game, p, x) for p in range(n_players)]

    evs = [float(np.dot(grads[p], x[p])) for p in range(n_players)]
    brs = [float(game.tpxs[p].best_response_value(grads[p])) for p in range(n_players)]
    per_expl = [brs[p] - evs[p] for p in range(n_players)]

    if n_players == 2:
        nconv = per_expl[0] + per_expl[1]
        exploit = 0.5 * nconv
    else:
        nconv = float("nan")
        exploit = float(np.mean(per_expl))

    ev0 = evs[0]
    return nconv, exploit, ev0

# ============================== algo factory ============================

def _parse_bracket_kv(text: str) -> dict:
    if "[" not in text:
        return {}
    _, rest = text.split("[", 1)
    rest = rest.rstrip("]")
    if not rest.strip():
        return {}
    kv = {}
    for piece in rest.split(","):
        k, v = piece.split("=")
        k = k.strip().casefold()
        v = v.strip()
        if k in ("l", "breadth", "optimism"):
            k = "l"
        kv[k] = v
    return kv


def make_algo(spec: str, defaults: dict):
    """
    Parse algo spec like 'omwu_l[L=5]' and return:
        (label, factory)

    factory(tpx, eta, dtype) -> Komwu-like instance
    """
    text = spec.strip()
    base = text.split("[", 1)[0].strip().lower()
    kv = _parse_bracket_kv(text)

    def get_float(name: str, default_key: str):
        if name in kv:
            return float(kv[name])
        return float(defaults[default_key])

    def get_int(name: str, default_key: str):
        if name in kv:
            return int(kv[name])
        return int(defaults[default_key])

    if base == "omwu":
        label = "OMWU"
        return (label, lambda tpx, eta, dt: Komwu(tpx, eta=eta, dtype=dt))

    elif base == "omwu_l":
        L = get_int("l", "L")
        label = f"OMWU-L(L={L})"
        return (label, lambda tpx, eta, dt, L=L: KomwuL(tpx, eta=eta, L=L, dtype=dt))

    elif base == "omwu_eo":
        L = get_int("l", "L")
        label = f"OMWU-EO(L={L})"
        return (label, lambda tpx, eta, dt, L=L: KomwuEO(tpx, eta=eta, L=L, dtype=dt))

    elif base == "omwu_eoo":
        L = get_int("l", "L")
        label = f"OMWU-EOO(L={L})"
        return (label, lambda tpx, eta, dt, L=L: KomwuEOO(tpx, eta=eta, L=L, dtype=dt))

    elif base == "omwu_decay":
        ab = get_float("alpha_b", "alpha_b")
        aS = get_float("alpha_s", "alpha_s")
        Kf = get_int("kf", "Kf")
        rho = get_float("rho", "rho")
        label = f"OMWU-DECAY(αb={ab},αs={aS},Kf={Kf},ρ={rho})"
        return (
            label,
            lambda tpx, eta, dt, ab=ab, aS=aS, Kf=Kf, rho=rho: KomwuDecay(
                tpx, eta=eta, alpha_b=ab, alpha_s=aS, Kf=Kf, rho=rho, dtype=dt
            ),
        )

    elif base == "omwu_eo_end":
        L = get_int("l", "L")
        label = f"OMWU-EO-END(L={L})"
        return (label, lambda tpx, eta, dt, L=L: KomwuEOEnd(tpx, eta=eta, L=L, dtype=dt))

    elif base == "omwu_kd":
        # KD parameters live only inside the bracket, not in defaults.
        # If you omit them, defaults K=10, D=4 are used.
        K_str = kv.get("k", kv.get("kwin", kv.get("window", None)))
        D_str = kv.get("d", kv.get("drop", kv.get("ddrop", None)))
        K = int(K_str) if K_str is not None else 10
        D = int(D_str) if D_str is not None else 4
        label = f"OMWU-KD(K={K},D={D})"
        return (
            label,
            lambda tpx, eta, dt, K=K, D=D: KomwuKD(tpx, eta=eta, K=K, D=D, dtype=dt),
        )

    elif base == "omwu_eo_kd":
        # EO breadth uses L; KD params via brackets (defaults: K=10,D=4)
        L = get_int("l", "L")
        K_str = kv.get("k", kv.get("kwin", kv.get("window", None)))
        D_str = kv.get("d", kv.get("drop", kv.get("ddrop", None)))
        K = int(K_str) if K_str is not None else 10
        D = int(D_str) if D_str is not None else 4
        label = f"OMWU-EO-KD(L={L},K={K},D={D})"
        return (
            label,
            lambda tpx, eta, dt, L=L, K=K, D=D: KomwuEOKD(
                tpx, eta=eta, L=L, K=K, D=D, dtype=dt
            ),
        )

    else:
        raise ValueError(f"Unknown algo spec: '{spec}'")

# ================================ runner ================================

def run_one_game(
    path: str,
    algos: list[str],
    T: int,
    eta: float,
    precision: str,
    print_every: int,
    plot_every: int,
    seed: int,
    default_L: int,
    default_alpha_b: float,
    default_alpha_s: float,
    default_Kf: int,
    default_rho: float,
    plot_total_regret: bool = False,
):
    """
    Run specified KOMWU variants on a .game file.

    - Supports any number of players (n_players >= 2).
    - Exploitability curve:
        * For 2p0s: usual exploitability (NashConv/2).
        * For n>2: mean per-player BR gap.
    - If plot_total_regret=True:
        * Tracks max per-player (cumulative) regret over time for each algorithm.
        * Prints final per-player regrets.
        * Plots max regret vs iteration (log–log).
    """
    DTYPE = get_dtype(precision)
    # Important: underflow is expected; raise on overflow/divide/invalid only.
    np.seterr(over='raise', divide='raise', invalid='raise', under='ignore')
    rng = np.random.default_rng(seed)

    game = Game(path)

    defaults = {
        "L": int(default_L),
        "alpha_b": float(default_alpha_b),
        "alpha_s": float(default_alpha_s),
        "Kf": int(default_Kf),
        "rho": float(default_rho),
    }

    # One agent per player per algorithm (with per-variant shrink metadata)
    algo_bundles: list[dict] = []
    shrink_variants = {
        "tolshrink_exp": (KomwuTolShrinkExp, "tolshrink_exp"),
        "tolshrink_rulea": (KomwuTolShrinkRuleA, "tolshrink_rulea"),
        "tolshrink_ruleb_scale": (KomwuTolShrinkRuleBScale, "tolshrink_ruleb_scale"),
        "tolshrink_ruleb_stop": (KomwuTolShrinkRuleBStop, "tolshrink_ruleb_stop"),
        "tolshrink_cutoff": (KomwuTolShrinkCutoff, "tolshrink_cutoff"),
        "tolshrink_lip": (KomwuTolShrinkLip, "tolshrink_lip"),
        "minlogits_periodic": (KomwuMinLogitsPeriodic, "minlogits_periodic"),
        "minlogits_always": (KomwuMinLogitsAlways, "minlogits_always"),
    }

    for spec in algos:
        base = spec.strip().split("[", 1)[0].strip().lower()

        if base in shrink_variants:
            cls, family = shrink_variants[base]
            num_runs = 25
            for idx in range(num_runs):
                P = int(rng.integers(10, 30))
                eps0 = float(rng.uniform(0.01, 0.4))
                log_gamma = rng.uniform(np.log(1e-7), np.log(1e-1))
                gamma = float(np.exp(log_gamma))

                label = f"{family} #{idx+1} (P={P}, eps0={eps0:.3f}, gamma={gamma:.3e})"
                agents = []
                for p in range(game.n_players):
                    if family == "minlogits_always":
                        ag = cls(game.tpxs[p], eta=eta, dtype=DTYPE)
                    elif family == "minlogits_periodic":
                        ag = cls(game.tpxs[p], eta=eta, P=P, dtype=DTYPE)
                    else:
                        ag = cls(game.tpxs[p], eta=eta, P=P, eps0=eps0, gamma=gamma, dtype=DTYPE)
                    ag.b += np.asarray(rng.normal(0, 1e-12, size=ag.b.shape), dtype=DTYPE)
                    ag._compute_x()
                    agents.append(ag)

                shrink_meta = {
                    "family": family,
                    "mode": family,
                    "P": 1 if family == "minlogits_always" else P,
                    "eps0": eps0,
                    "gamma": gamma,
                    "C": getattr(agents[0], "C", 0.5),
                    "tol": getattr(agents[0], "tol", 0.0),
                    "scale_factor": getattr(agents[0], "scale_factor", 0.5),
                    "gap_cutoff": getattr(agents[0], "gap_cutoff", 1e-4),
                    "gap_ema": None,
                    "gap_alpha": 0.1,
                    "eps_scale": 1.0,
                    "enabled": True,
                    "step": 0,
                }

                algo_bundles.append({"label": label, "agents": agents, "family": family, "shrink": shrink_meta})
            continue

        label, factory = make_algo(spec, defaults)
        agents = []
        for p in range(game.n_players):
            ag = factory(game.tpxs[p], eta, DTYPE)
            ag.b += np.asarray(rng.normal(0, 1e-12, size=ag.b.shape), dtype=DTYPE)
            ag._compute_x()
            agents.append(ag)
        algo_bundles.append({"label": label, "agents": agents, "family": label, "shrink": None})

    # Time-series metrics
    curves = {
        bundle["label"]: {"times": [], "exploit": [], "nashconv": []}
        for bundle in algo_bundles
    }
    regret_curves = (
        {
            bundle["label"]: {"times": [], "max_regret": []}
            for bundle in algo_bundles
        }
        if plot_total_regret
        else {}
    )


    # ============================ main loop ============================
    for t in range(1, T + 1):
        for bundle in algo_bundles:
            label = bundle["label"]
            agents = bundle["agents"]
            shrink_meta = bundle["shrink"]

            strategies_pre = {p: agents[p].next_strategy() for p in range(game.n_players)}
            grads = [utility_gradient_safe(game, p, strategies_pre) for p in range(game.n_players)]

            for p in range(game.n_players):
                agents[p].observe_gradient(grads[p])

            shrink_gap = None
            strategies_post = {p: agents[p].next_strategy() for p in range(game.n_players)}
            if shrink_meta is not None:
                shrink_meta["step"] += 1
                grads_post = [utility_gradient_safe(game, p, strategies_post) for p in range(game.n_players)]
                nc, ex, _ = nash_conv_and_ev(game, strategies_post, grads=grads_post)
                shrink_gap = abs(nc)
                if shrink_meta["gap_ema"] is None:
                    shrink_meta["gap_ema"] = shrink_gap
                else:
                    alpha = shrink_meta["gap_alpha"]
                    shrink_meta["gap_ema"] = (1.0 - alpha) * shrink_meta["gap_ema"] + alpha * shrink_gap

                P = shrink_meta["P"]
                do_shrink = shrink_meta["enabled"] and P > 0 and (shrink_meta["step"] % P == 0)
                if shrink_meta["mode"] == "tolshrink_cutoff" and shrink_meta["gap_ema"] is not None:
                    if shrink_meta["gap_ema"] <= shrink_meta["gap_cutoff"]:
                        shrink_meta["enabled"] = False
                        do_shrink = False

                if shrink_meta["mode"] == "tolshrink_rulea" and shrink_meta["gap_ema"] is not None and do_shrink:
                    eps_check = shrink_meta["eps0"] * np.exp(-shrink_meta["gamma"] * shrink_meta["step"])
                    if eps_check > shrink_meta["C"] * shrink_meta["gap_ema"]:
                        do_shrink = False

                if do_shrink:
                    eps_t = shrink_meta["eps_scale"] * shrink_meta["eps0"] * np.exp(-shrink_meta["gamma"] * shrink_meta["step"])
                    candidates = []
                    for ag in agents:
                        if shrink_meta["mode"].startswith("minlogits"):
                            cand = ag.shrink_candidate_minlogits()
                        elif shrink_meta["mode"] == "tolshrink_lip":
                            cand = ag.shrink_candidate_lip(eps_t)
                        else:
                            cand = ag.shrink_candidate_tv(eps_t)
                        candidates.append(cand)

                    if shrink_meta["mode"] in {"tolshrink_ruleb_scale", "tolshrink_ruleb_stop"}:
                        backups = [(ag.b.copy(), ag.next_strategy().copy()) for ag in agents]
                        for ag, cand in zip(agents, candidates):
                            ag.apply_candidate(cand[0], cand[1])
                        strategies_post = {p: agents[p].next_strategy() for p in range(game.n_players)}
                        grads_after = [utility_gradient_safe(game, p, strategies_post) for p in range(game.n_players)]
                        nc_after, _, _ = nash_conv_and_ev(game, strategies_post, grads=grads_after)
                        gap_after = abs(nc_after)
                        tol = shrink_meta["tol"]
                        if gap_after > (1.0 + tol) * shrink_gap:
                            for ag, (b_old, x_old) in zip(agents, backups):
                                ag.apply_candidate(b_old, x_old)
                            if shrink_meta["mode"] == "tolshrink_ruleb_scale":
                                shrink_meta["eps_scale"] *= shrink_meta["scale_factor"]
                            else:
                                shrink_meta["enabled"] = False
                            strategies_post = {p: agents[p].next_strategy() for p in range(game.n_players)}
                            grads_post = [utility_gradient_safe(game, p, strategies_post) for p in range(game.n_players)]
                            nc, ex, _ = nash_conv_and_ev(game, strategies_post, grads=grads_post)
                            shrink_gap = abs(nc)
                        else:
                            shrink_gap = gap_after
                    else:
                        for ag, cand in zip(agents, candidates):
                            ag.apply_candidate(cand[0], cand[1])
                        strategies_post = {p: agents[p].next_strategy() for p in range(game.n_players)}
                        grads_post = [utility_gradient_safe(game, p, strategies_post) for p in range(game.n_players)]
                        nc, ex, _ = nash_conv_and_ev(game, strategies_post, grads=grads_post)
                        shrink_gap = abs(nc)
                        shrink_meta["gap_ema"] = (1.0 - shrink_meta["gap_alpha"]) * shrink_meta["gap_ema"] + shrink_meta["gap_alpha"] * shrink_gap

            # sample metrics
            if (t % plot_every) == 0 or t == 1:
                grads_sample = [utility_gradient_safe(game, p, strategies_post) for p in range(game.n_players)]
                nc, ex, _ = nash_conv_and_ev(game, strategies_post, grads=grads_sample)
                curves[label]["times"].append(t)
                curves[label]["exploit"].append(ex)
                curves[label]["nashconv"].append(nc)

                if plot_total_regret:
                    regs = [float(ag.regret()) for ag in agents]
                    max_reg = max(regs)
                    regret_curves[label]["times"].append(t)
                    regret_curves[label]["max_regret"].append(max_reg)

            if (t % print_every) == 0 or t == 1:
                ex = curves[label]["exploit"][-1] if curves[label]["exploit"] else float("nan")
                nc = curves[label]["nashconv"][-1] if curves[label]["nashconv"] else float("nan")
                print(
                    f"[{label:32s}] t={t:6d}  exploit={ex:.6e}  "
                    f"nashconv={nc:.6e}  η={eta:.2e}"
                )

        # ensure final sample at T
        if t == T:
            for bundle in algo_bundles:
                label = bundle["label"]
                agents = bundle["agents"]
                strategies_post = {p: agents[p].next_strategy() for p in range(game.n_players)}
                grads_sample = [utility_gradient_safe(game, p, strategies_post) for p in range(game.n_players)]
                nc, ex, _ = nash_conv_and_ev(game, strategies_post, grads=grads_sample)
                if curves[label]["times"] and curves[label]["times"][-1] == T:
                    continue
                curves[label]["times"].append(T)
                curves[label]["exploit"].append(ex)
                curves[label]["nashconv"].append(nc)

    # select best shrink run per family (by final exploitability)
    family_best_label: dict[str, str] = {}
    shrink_families = set(shrink_variants.keys())
    for bundle in algo_bundles:
        label = bundle["label"]
        fam = bundle["family"]
        final_val = curves[label]["exploit"][-1] if curves[label]["exploit"] else float("inf")
        if fam not in shrink_families:
            family_best_label[label] = label
            continue
        best_so_far = family_best_label.get(fam)
        if best_so_far is None:
            family_best_label[fam] = label
        else:
            prev_val = curves[best_so_far]["exploit"][-1] if curves[best_so_far]["exploit"] else float("inf")
            if final_val < prev_val:
                family_best_label[fam] = label

    plot_labels = set(family_best_label.values())

    # ============================= plots ==============================

    # Exploitability curve(s)
    plt.figure(figsize=(8, 5))
    for label in curves:
        if label not in plot_labels:
            continue
        xs = np.array(curves[label]["times"], dtype=float)
        ys = np.array(curves[label]["exploit"], dtype=float)
        plt.plot(xs, ys, label=label)
    plt.yscale("log")
    plt.xlabel("Iterations")
    plt.ylabel("Exploitability")
    plt.title(f"{path}")
    plt.grid(True, which="both", ls=":")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Regret curves (max per-player regret vs iteration)
    if plot_total_regret:
        print("\n=== Final regrets (per algorithm, per player) ===")
        for bundle in algo_bundles:
            label = bundle["label"]
            agents = bundle["agents"]
            if label not in plot_labels:
                continue
            regs = [float(ag.regret()) for ag in agents]
            total = sum(regs)
            max_reg = max(regs)
            per_str = ", ".join(f"p{idx}={r:.3e}" for idx, r in enumerate(regs))
            print(f"{label:32s}  [{per_str}]  total={total:.3e}  max={max_reg:.3e}")

        plt.figure(figsize=(8, 5))
        for label in regret_curves:
            if label not in plot_labels:
                continue
            xs = np.array(regret_curves[label]["times"], dtype=float)
            ys = np.array(regret_curves[label]["max_regret"], dtype=float)
            plt.plot(xs, ys, label=label)
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Iteration")
        plt.ylabel("Max individual regret")
        plt.title(f"Max per-player regret vs iteration on {path}")
        plt.grid(True, which="both", ls=":")
        plt.legend()
        plt.tight_layout()
        plt.show()

# ============================ CLI + VS CODE =============================

def build_arg_parser():
    ap = argparse.ArgumentParser(
        description="Run KOMWU variants on poker .game files (multi-player EFG)."
    )
    ap.add_argument(
        "--game",
        type=str,
        required=True,
        help="Path to .game (e.g., K23.game, L233.game, Leduc.game)",
    )
    ap.add_argument(
        "--algos",
        type=str,
        required=True,
        help=(
            "Comma-separated list, e.g.: "
            "omwu,omwu_l[L=5],omwu_eo[L=6],omwu_eoo[L=6],"
            "omwu_decay[alpha_b=1,alpha_s=1,Kf=3,rho=1],"
            "omwu_eo_end[L=3],omwu_kd[K=10,D=4],omwu_eo_kd[L=5,K=10,D=4]. "
            "Keys inside [] are case-insensitive; you may also use "
            "breadth= or optimism= for L, and kwin/window or drop/ddrop for KD."
        ),
    )
    ap.add_argument("--L", type=int, default=5, help="Global optimism level for *-L/EO/EOO/EO-END.")
    ap.add_argument("--alpha-b", type=float, default=1.0, dest="alpha_b", help="Global α_b for DECAY.")
    ap.add_argument("--alpha-s", type=float, default=1.0, dest="alpha_s", help="Global α_s for DECAY.")
    ap.add_argument("--Kf", type=int, default=1, help="Global K_f for DECAY.")
    ap.add_argument("--rho", type=float, default=1.0, help="Global ρ for DECAY.")
    ap.add_argument("--T", type=int, default=20000, help="Iterations")
    ap.add_argument("--eta", type=float, default=0.1, help="Learning rate")
    ap.add_argument(
        "--precision",
        type=str,
        default="precise",
        choices=["normal", "precise"],
        help="Float64 vs longdouble",
    )
    ap.add_argument("--print-every", type=int, default=500, help="Print frequency")
    ap.add_argument("--plot-every", type=int, default=100, help="Curve sampling frequency")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed")
    # Toggle for tracking/plotting regret over time
    ap.add_argument(
        "--plot-total-regret",
        dest="plot_total_regret",
        action="store_true",
        help="Track and plot max per-player regret vs iteration, and print final regrets.",
    )
    ap.add_argument("--no-plot-total-regret", dest="plot_total_regret", action="store_false")
    ap.set_defaults(plot_total_regret=False)
    return ap


def main_cli():
    parser = build_arg_parser()
    args = parser.parse_args()
    algos = [s for s in args.algos.split(",") if s.strip()]
    run_one_game(
        path=args.game,
        algos=algos,
        T=args.T,
        eta=args.eta,
        precision=args.precision,
        print_every=args.print_every,
        plot_every=args.plot_every,
        seed=args.seed,
        default_L=args.L,
        default_alpha_b=args.alpha_b,
        default_alpha_s=args.alpha_s,
        default_Kf=args.Kf,
        default_rho=args.rho,
        plot_total_regret=args.plot_total_regret,
    )


def main_vscode():
    algos = [s for s in CONFIG["algos"].split(",") if s.strip()]
    run_one_game(
        path=CONFIG["game"],
        algos=algos,
        T=int(CONFIG["T"]),
        eta=float(CONFIG["eta"]),
        precision=str(CONFIG["precision"]),
        print_every=int(CONFIG["print_every"]),
        plot_every=int(CONFIG["plot_every"]),
        seed=int(CONFIG["seed"]),
        default_L=int(CONFIG["L"]),
        default_alpha_b=float(CONFIG["alpha_b"]),
        default_alpha_s=float(CONFIG["alpha_s"]),
        default_Kf=int(CONFIG["Kf"]),
        default_rho=float(CONFIG["rho"]),
        plot_total_regret=bool(CONFIG.get("plot_total_regret", False)),
    )


if __name__ == "__main__":
    # If any CLI-style arg is present, use CLI; otherwise use CONFIG (VS Code).
    if len(sys.argv) > 1 and any(arg.startswith("-") for arg in sys.argv[1:]):
        main_cli()
    else:
        print(
            "[run_algorithms.py] No CLI args detected → using CONFIG block at top of file.\n"
            f"  game={CONFIG['game']}  algos={CONFIG['algos']}  L={CONFIG['L']}  "
            f"αb={CONFIG['alpha_b']}  αs={CONFIG['alpha_s']}  Kf={CONFIG['Kf']}  ρ={CONFIG['rho']}  "
            f"plot_total_regret={CONFIG.get('plot_total_regret', False)}\n"
        )
        main_vscode()
