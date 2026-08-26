"""CMA-ES: gradient-free direct policy search, implemented from scratch.

Why a gradient-free method is in this benchmark
-----------------------------------------------
Several of these missions have exactly the structure evolutionary strategies
are good at and policy gradients are bad at: a low-dimensional control problem
whose real signal -- did the spacecraft arrive, with how much propellant, with
how much thruster life left -- arrives once, at the end, after tens of
thousands of steps. A policy gradient has to propagate that through a value
function; CMA-ES just reads the episode return. It is a genuinely different
point in the design space and it deserves an honest entry rather than a
strawman, so this is the real algorithm: the standard ``mu/mu_w, lambda``
variant with weighted intermediate recombination, rank-one *and* rank-mu
covariance updates, and cumulative step-size adaptation.

The implementation is deliberately split in two. :class:`CMAES` is a plain
optimiser over ``R^n`` with an ``ask``/``tell`` interface and no knowledge of
reinforcement learning, which is what makes it verifiable against the standard
test functions (sphere, Rosenbrock, Rastrigin) instead of only against
"the agent seemed to improve". :class:`CMAESAgent` wraps it around the flat
weight vector of a small deterministic MLP and drives it from
``on_episode_end``.

Cost model, stated plainly for the comparison table: one generation costs
``popsize`` episodes (``4 + floor(3 ln n)``, so 23 episodes at n = 677 for the
default 36 -> 16 -> 5 policy), and nothing is learned in between. Against PPO's
per-step updates that is a lot of environment interaction per unit of progress,
and the comparison should say so.

Note on ``cma``: the PyPI package is not a dependency of this project and is
not installed. This is a first-party implementation, cross-checked against
Hansen's published pseudocode and against the reference optima of the test
functions in the self-test.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from ..core.registry import AGENT
from .base import Agent, TrainStats, Transition
from .networks import RunningMeanStd

logger = logging.getLogger(__name__)

__all__ = ["CMAES", "CMAESAgent", "FlatMLPPolicy", "rastrigin", "rosenbrock", "sphere"]


# --- standard test functions (used by the self-test and by tests/) -----------
def sphere(x: np.ndarray) -> float:
    """Separable, perfectly conditioned. Minimum 0 at the origin."""
    return float(np.sum(np.asarray(x, dtype=np.float64) ** 2))


def rosenbrock(x: np.ndarray) -> float:
    """Non-separable banana valley. Minimum 0 at ``(1, ..., 1)``.

    The one that actually tests the covariance adaptation: a diagonal-only
    method has to crawl along the valley floor, a full-covariance one learns
    the valley's orientation and walks it.
    """
    a = np.asarray(x, dtype=np.float64)
    return float(np.sum(100.0 * (a[1:] - a[:-1] ** 2) ** 2 + (1.0 - a[:-1]) ** 2))


def rastrigin(x: np.ndarray) -> float:
    """Highly multimodal, ~``10^n`` local minima. Minimum 0 at the origin.

    CMA-ES with the default population size is *expected* to fail this from a
    distant start -- it is a local method with a global-ish step size. It is
    included because a method that claims to solve it at default settings is
    reporting a bug, and because it is solvable with a large population, which
    is a real and reportable property.
    """
    a = np.asarray(x, dtype=np.float64)
    return float(10.0 * a.size + np.sum(a**2 - 10.0 * np.cos(2.0 * math.pi * a)))


class CMAES:
    """Covariance Matrix Adaptation Evolution Strategy, ``mu/mu_w, lambda``.

    Minimises. Use ``ask()`` to draw a generation, evaluate it however you
    like, and ``tell()`` the fitnesses back in the same order.

    Parameters
    ----------
    x0:
        Initial distribution mean; its length fixes ``n``.
    sigma0:
        Initial step size. Should be roughly a third of the search interval
        you expect the optimum to lie in, per Hansen's guidance.
    popsize:
        ``lambda``. Defaults to ``4 + floor(3 ln n)``. Larger populations buy
        robustness on multimodal problems at a linear cost in evaluations.
    mu:
        Parents. Defaults to ``popsize // 2``, which with the default
        logarithmic weights is the recommended setting.
    bounds:
        Optional ``(low, high)`` box. Candidates are clipped on the way out and
        the clipped values are what ``tell`` sees, so the distribution learns
        the boundary rather than repeatedly proposing across it.
    max_condition:
        Guard on ``max(D^2)/min(D^2)``. Above it the covariance is nudged back
        towards isotropy; an unbounded condition number is how CMA-ES fails on
        a degenerate objective, and failing loudly is better than producing
        NaNs twenty generations later.
    """

    def __init__(
        self,
        x0: Sequence[float] | np.ndarray,
        sigma0: float = 0.3,
        *,
        popsize: int | None = None,
        mu: int | None = None,
        seed: int = 0,
        bounds: tuple[float, float] | None = None,
        tol_x: float = 1e-12,
        tol_fun: float = 1e-12,
        max_condition: float = 1e14,
    ) -> None:
        self.mean = np.asarray(x0, dtype=np.float64).reshape(-1).copy()
        self.n = int(self.mean.size)
        if self.n < 1:
            raise ValueError("CMAES needs at least one dimension")
        if sigma0 <= 0.0:
            raise ValueError("sigma0 must be positive")
        self.sigma = float(sigma0)
        self.sigma0 = float(sigma0)
        self.bounds = bounds
        self.tol_x = float(tol_x)
        self.tol_fun = float(tol_fun)
        self.max_condition = float(max_condition)

        n = self.n
        self.popsize = int(popsize) if popsize else 4 + int(math.floor(3 * math.log(n)))
        self.popsize = max(4, self.popsize)
        self.mu = int(mu) if mu else self.popsize // 2
        self.mu = max(1, min(self.mu, self.popsize))

        # Logarithmically decreasing recombination weights (Hansen's default).
        w = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = w / w.sum()
        self.mueff = float(1.0 / np.sum(self.weights**2))

        # Adaptation rates. These are not free parameters -- they are the
        # published defaults and changing them changes the algorithm.
        self.cc = (4.0 + self.mueff / n) / (n + 4.0 + 2.0 * self.mueff / n)
        self.cs = (self.mueff + 2.0) / (n + self.mueff + 5.0)
        self.c1 = 2.0 / ((n + 1.3) ** 2 + self.mueff)
        self.cmu = min(
            1.0 - self.c1,
            2.0 * (self.mueff - 2.0 + 1.0 / self.mueff) / ((n + 2.0) ** 2 + self.mueff),
        )
        self.damps = (
            1.0
            + 2.0 * max(0.0, math.sqrt((self.mueff - 1.0) / (n + 1.0)) - 1.0)
            + self.cs
        )
        #: Expectation of ``||N(0, I)||``, the CSA reference length.
        self.chiN = math.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n * n))

        self.pc = np.zeros(n)
        self.ps = np.zeros(n)
        self.B = np.eye(n)
        self.D = np.ones(n)
        self.C = np.eye(n)
        self.invsqrtC = np.eye(n)
        self.eigeneval = 0
        self.counteval = 0
        self.generation = 0

        self.best_x = self.mean.copy()
        self.best_f = math.inf
        self.last_best_f = math.inf
        self.median_f = math.inf
        self._recent_best: list[float] = []
        self._pending: np.ndarray | None = None
        self._rng = np.random.default_rng(seed)
        self.seed = int(seed)

    # --- the interface -------------------------------------------------------
    def ask(self) -> np.ndarray:
        """Draw one generation: ``(popsize, n)``."""
        z = self._rng.standard_normal((self.popsize, self.n))
        y = (z * self.D) @ self.B.T
        x = self.mean + self.sigma * y
        if self.bounds is not None:
            x = np.clip(x, self.bounds[0], self.bounds[1])
        self._pending = x
        return x

    def tell(
        self, solutions: np.ndarray, fitnesses: Sequence[float] | np.ndarray
    ) -> None:
        """Feed back the evaluated generation and advance the distribution."""
        x = np.asarray(solutions, dtype=np.float64).reshape(-1, self.n)
        f = np.asarray(fitnesses, dtype=np.float64).reshape(-1)
        if x.shape[0] != f.size:
            raise ValueError(
                f"got {x.shape[0]} solutions and {f.size} fitnesses"
            )
        if not np.all(np.isfinite(f)):
            # An infeasible or crashed episode should not be allowed to
            # dominate the ranking; push it to the back instead of poisoning
            # the weighted mean with an inf.
            worst = np.max(f[np.isfinite(f)]) if np.any(np.isfinite(f)) else 0.0
            f = np.where(np.isfinite(f), f, worst + abs(worst) + 1.0)

        order = np.argsort(f, kind="stable")
        x_sorted = x[order]
        f_sorted = f[order]
        self.counteval += x.shape[0]
        self.generation += 1

        if f_sorted[0] < self.best_f:
            self.best_f = float(f_sorted[0])
            self.best_x = x_sorted[0].copy()
        self.last_best_f = float(f_sorted[0])
        self.median_f = float(np.median(f_sorted))
        self._recent_best.append(self.last_best_f)
        if len(self._recent_best) > 10 + int(30 * self.n / self.popsize):
            self._recent_best.pop(0)

        xold = self.mean.copy()
        parents = x_sorted[: self.mu]
        self.mean = self.weights @ parents

        y = (self.mean - xold) / self.sigma
        self.ps = (1.0 - self.cs) * self.ps + math.sqrt(
            self.cs * (2.0 - self.cs) * self.mueff
        ) * (self.invsqrtC @ y)

        # hsig stalls the rank-one update when the evolution path is long
        # because sigma is growing fast, not because the search is directed.
        denom = math.sqrt(
            max(1e-300, 1.0 - (1.0 - self.cs) ** (2.0 * self.counteval / self.popsize))
        )
        hsig = float(np.linalg.norm(self.ps)) / denom / self.chiN < (
            1.4 + 2.0 / (self.n + 1.0)
        )
        self.pc = (1.0 - self.cc) * self.pc
        if hsig:
            self.pc += math.sqrt(self.cc * (2.0 - self.cc) * self.mueff) * y

        artmp = (parents - xold) / self.sigma
        rank_one = np.outer(self.pc, self.pc)
        if not hsig:
            rank_one = rank_one + self.cc * (2.0 - self.cc) * self.C
        rank_mu = (artmp.T * self.weights) @ artmp
        self.C = (
            (1.0 - self.c1 - self.cmu) * self.C
            + self.c1 * rank_one
            + self.cmu * rank_mu
        )

        self.sigma *= math.exp(
            (self.cs / self.damps)
            * (float(np.linalg.norm(self.ps)) / self.chiN - 1.0)
        )
        if not math.isfinite(self.sigma) or self.sigma <= 0.0:
            logger.warning("CMAES: sigma became %s; resetting to sigma0", self.sigma)
            self.sigma = self.sigma0

        self._maybe_eigendecompose()

    def _maybe_eigendecompose(self) -> None:
        """Refresh ``B``, ``D`` and ``C^{-1/2}``, amortised over generations.

        The eigendecomposition is the only ``O(n^3)`` step. Doing it every
        generation would dominate the runtime for any interesting ``n``, and it
        is unnecessary: the covariance changes by at most ``c1 + cmu`` per
        generation, so refreshing on that schedule keeps the sampling
        distribution accurate to the same order as the update itself.
        """
        interval = self.popsize / max(self.c1 + self.cmu, 1e-12) / self.n / 10.0
        if self.counteval - self.eigeneval <= interval:
            return
        self.eigeneval = self.counteval
        self.C = np.triu(self.C) + np.triu(self.C, 1).T   # enforce symmetry
        try:
            d2, B = np.linalg.eigh(self.C)
        except np.linalg.LinAlgError:  # pragma: no cover - numerically extreme
            logger.warning("CMAES: eigendecomposition failed; resetting covariance")
            self.C = np.eye(self.n)
            self.B, self.D = np.eye(self.n), np.ones(self.n)
            self.invsqrtC = np.eye(self.n)
            return

        d2 = np.maximum(d2, 0.0)
        largest = float(d2.max()) if d2.size else 1.0
        floor = largest / self.max_condition if largest > 0.0 else 1e-300
        if np.any(d2 < floor):
            d2 = np.maximum(d2, floor)
            self.C = (B * d2) @ B.T
            logger.debug("CMAES: covariance condition capped at %.1e", self.max_condition)
        self.B = B
        self.D = np.sqrt(d2)
        self.invsqrtC = (B / self.D) @ B.T

    # --- diagnostics ---------------------------------------------------------
    @property
    def condition_number(self) -> float:
        d2 = self.D**2
        lo = float(d2.min())
        return float(d2.max() / lo) if lo > 0.0 else math.inf

    @property
    def axis_ratio(self) -> float:
        return float(self.D.max() / self.D.min()) if self.D.min() > 0 else math.inf

    def stop(self) -> dict[str, float]:
        """Triggered termination conditions, empty while the run is healthy."""
        out: dict[str, float] = {}
        spread = self.sigma * float(np.max(self.D))
        if spread < self.tol_x:
            out["tolx"] = spread
        if len(self._recent_best) >= 10:
            span = max(self._recent_best) - min(self._recent_best)
            if span < self.tol_fun:
                out["tolfun"] = span
        if self.condition_number > self.max_condition:
            out["conditioncov"] = self.condition_number
        return out

    def optimize(
        self,
        fn: Callable[[np.ndarray], float],
        max_evals: int = 10_000,
        ftarget: float | None = None,
        verbose: bool = False,
    ) -> dict[str, Any]:
        """Convenience loop for offline benchmarking of the optimiser itself."""
        while self.counteval < max_evals:
            pop = self.ask()
            fitness = [float(fn(row)) for row in pop]
            self.tell(pop, fitness)
            if verbose and self.generation % 10 == 0:
                logger.info(
                    "gen %4d evals %6d best %.6e sigma %.3e axis %.2e",
                    self.generation, self.counteval, self.best_f,
                    self.sigma, self.axis_ratio,
                )
            if ftarget is not None and self.best_f <= ftarget:
                return self._result("ftarget")
            stop = self.stop()
            if stop:
                return self._result(next(iter(stop)))
        return self._result("maxevals")

    def _result(self, reason: str) -> dict[str, Any]:
        return {
            "x": self.best_x.copy(),
            "f": self.best_f,
            "evaluations": self.counteval,
            "generations": self.generation,
            "sigma": self.sigma,
            "stop": reason,
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "mean": self.mean.copy(),
            "sigma": self.sigma,
            "C": self.C.copy(),
            "B": self.B.copy(),
            "D": self.D.copy(),
            "invsqrtC": self.invsqrtC.copy(),
            "pc": self.pc.copy(),
            "ps": self.ps.copy(),
            "best_x": self.best_x.copy(),
            "best_f": self.best_f,
            "counteval": self.counteval,
            "eigeneval": self.eigeneval,
            "generation": self.generation,
            "rng": self._rng.bit_generator.state,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.mean = np.asarray(state["mean"], dtype=np.float64).copy()
        self.sigma = float(state["sigma"])
        self.C = np.asarray(state["C"], dtype=np.float64).copy()
        self.B = np.asarray(state["B"], dtype=np.float64).copy()
        self.D = np.asarray(state["D"], dtype=np.float64).copy()
        self.invsqrtC = np.asarray(state["invsqrtC"], dtype=np.float64).copy()
        self.pc = np.asarray(state["pc"], dtype=np.float64).copy()
        self.ps = np.asarray(state["ps"], dtype=np.float64).copy()
        self.best_x = np.asarray(state["best_x"], dtype=np.float64).copy()
        self.best_f = float(state["best_f"])
        self.counteval = int(state["counteval"])
        self.eigeneval = int(state["eigeneval"])
        self.generation = int(state["generation"])
        if state.get("rng") is not None:
            self._rng.bit_generator.state = state["rng"]

    def __repr__(self) -> str:
        return (
            f"<CMAES n={self.n} popsize={self.popsize} mu={self.mu} "
            f"gen={self.generation} sigma={self.sigma:.3e} best={self.best_f:.6e}>"
        )


class FlatMLPPolicy:
    """Deterministic tanh MLP whose weights live in one flat vector.

    Numpy rather than torch on purpose. There is no gradient to take here, the
    networks are tiny, and a numpy forward pass avoids a per-step torch graph
    setup that would otherwise dominate the rollout cost of the one agent in
    this benchmark that needs the most rollouts.
    """

    def __init__(
        self, obs_dim: int, action_dim: int, hidden: Sequence[int] = (16,)
    ) -> None:
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.hidden = tuple(int(h) for h in hidden)
        dims = [self.obs_dim, *self.hidden, self.action_dim]
        self.shapes: list[tuple[int, ...]] = []
        for a, b in zip(dims[:-1], dims[1:]):
            self.shapes.append((a, b))
            self.shapes.append((b,))
        self.size = int(sum(int(np.prod(s)) for s in self.shapes))
        self._params: list[np.ndarray] = []
        self.set_flat(np.zeros(self.size))

    def set_flat(self, flat: np.ndarray) -> None:
        v = np.asarray(flat, dtype=np.float64).reshape(-1)
        if v.size != self.size:
            raise ValueError(f"expected {self.size} parameters, got {v.size}")
        out: list[np.ndarray] = []
        i = 0
        for shape in self.shapes:
            k = int(np.prod(shape))
            out.append(v[i : i + k].reshape(shape))
            i += k
        self._params = out

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        x = np.asarray(obs, dtype=np.float64).reshape(-1)
        n_layers = len(self._params) // 2
        for layer in range(n_layers):
            w = self._params[2 * layer]
            b = self._params[2 * layer + 1]
            x = x @ w + b
            x = np.tanh(x)      # tanh on the head too: the action box is [-1, 1]
        return x

    def initial_flat(self, rng: np.random.Generator, scale: float) -> np.ndarray:
        """Zeros at ``scale = 0`` (the neutral command), else a scaled init.

        Zeros are a defensible start here rather than a degenerate one: a zero
        action decodes to half throttle, mid operating point, pure prograde --
        a sane prior. CMA-ES breaks the symmetry on its first generation
        because it perturbs, not because it differentiates.
        """
        if scale <= 0.0:
            return np.zeros(self.size)
        flat = np.zeros(self.size)
        i = 0
        for shape in self.shapes:
            k = int(np.prod(shape))
            if len(shape) == 2:
                gain = scale / math.sqrt(shape[0])
                flat[i : i + k] = rng.normal(0.0, gain, size=k)
            i += k
        return flat


@AGENT.register("cmaes", kind="evolution", learns=True)
class CMAESAgent(Agent):
    """Direct policy search over a small deterministic MLP with CMA-ES.

    Episodic by construction: :meth:`act` runs whichever candidate is currently
    under evaluation, and :meth:`on_episode_end` is what actually advances the
    search. :meth:`update` is a no-op that reports the latest generation's
    statistics, per the :class:`Agent` contract for episodic learners.

    Parameters
    ----------
    hidden:
        Policy hidden widths. Kept small on purpose: CMA-ES maintains a full
        ``n x n`` covariance, so the default ``(16,)`` (677 parameters for the
        canonical 36/5 interface) is already at the sensible end. ``()`` gives
        a 185-parameter linear policy, which is a fast and surprisingly strong
        setting on these tasks. Anything with a 256-wide layer is a misuse of
        the method, and is warned about.
    sigma0:
        Initial step size in weight space.
    episodes_per_candidate:
        Average this many episodes per candidate before scoring it. Above 1 it
        costs proportionally more episodes and buys a less noisy ranking, which
        matters on the missions with randomised initial states.
    eval_policy:
        What ``deterministic=True`` returns -- ``"best"`` (best candidate seen,
        the honest "what you would fly") or ``"mean"`` (the distribution mean,
        Hansen's recommended final solution). Both are reasonable; ``"best"``
        is the default because these fitness evaluations are noisy and the mean
        of a noisy population is not always a policy anyone evaluated.
    normalize_obs:
        Whiten observations with running statistics. On by default -- this
        observation vector mixes metres from the Sun with a wear fraction in
        [0, 1], and a tanh MLP fed the raw version saturates.
    """

    name = "cmaes"
    learns = True
    uses_constraints = False

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        *,
        hidden: Sequence[int] = (16,),
        sigma0: float = 0.2,
        popsize: int | None = None,
        mu: int | None = None,
        seed: int = 0,
        episodes_per_candidate: int = 1,
        eval_policy: str = "best",
        normalize_obs: bool = True,
        obs_clip: float = 10.0,
        init_scale: float = 0.0,
        weight_bound: float | None = 10.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(obs_dim, action_dim, **kwargs)
        if eval_policy not in ("best", "mean"):
            raise ValueError("eval_policy must be 'best' or 'mean'")
        self.seed = int(seed)
        self.eval_policy = eval_policy
        self.episodes_per_candidate = max(1, int(episodes_per_candidate))
        self.normalize_obs = bool(normalize_obs)
        self.obs_clip = float(obs_clip)

        self._rng = np.random.default_rng(self.seed)
        self.policy = FlatMLPPolicy(obs_dim, action_dim, hidden=hidden)
        if self.policy.size > 5_000:
            logger.warning(
                "CMAESAgent: %d policy parameters is far past where a full "
                "covariance matrix is sensible (%.1e entries); consider "
                "hidden=() or a narrower layer",
                self.policy.size,
                float(self.policy.size) ** 2,
            )

        x0 = self.policy.initial_flat(self._rng, float(init_scale))
        bounds = (
            None if weight_bound is None else (-float(weight_bound), float(weight_bound))
        )
        self.es = CMAES(
            x0, sigma0, popsize=popsize, mu=mu, seed=self.seed, bounds=bounds
        )
        self.obs_rms = RunningMeanStd(obs_dim) if self.normalize_obs else None

        self.config.update(
            hidden=self.policy.hidden,
            sigma0=float(sigma0),
            popsize=self.es.popsize,
            mu=self.es.mu,
            seed=self.seed,
            episodes_per_candidate=self.episodes_per_candidate,
            eval_policy=self.eval_policy,
            normalize_obs=self.normalize_obs,
            n_parameters=self.policy.size,
        )

        self._population = self.es.ask()
        self._candidate = 0
        self._fitness = np.full(self.es.popsize, np.nan)
        self._repeat_returns: list[float] = []
        self._episodes = 0
        self._last_stats = TrainStats()
        self._generation_returns: list[float] = []
        self._activate(self._population[0])

    # --- policy plumbing -----------------------------------------------------
    def _activate(self, flat: np.ndarray) -> None:
        self._active = np.asarray(flat, dtype=np.float64).copy()
        self.policy.set_flat(self._active)

    def _eval_params(self) -> np.ndarray:
        if self.eval_policy == "mean":
            return self.es.mean
        return self.es.best_x if math.isfinite(self.es.best_f) else self.es.mean

    def _prepare(self, obs: np.ndarray) -> np.ndarray:
        arr = np.asarray(obs, dtype=np.float64).reshape(-1)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        if self.obs_rms is None:
            return arr
        return self.obs_rms.normalize(arr, clip=self.obs_clip).astype(np.float64)

    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        x = self._prepare(obs)
        if deterministic:
            # Swap in the evaluation weights for this call only, so an eval
            # episode interleaved with training cannot disturb the candidate
            # currently being scored.
            saved = self._active
            self.policy.set_flat(self._eval_params())
            action = self.policy(x)
            self.policy.set_flat(saved)
        else:
            action = self.policy(x)
        return np.clip(action, -1.0, 1.0).astype(np.float32)

    def observe_transition(self, tr: Transition) -> None:
        if self.obs_rms is not None:
            self.obs_rms.update(np.asarray(tr.obs, dtype=np.float64).reshape(-1))

    # --- the search ----------------------------------------------------------
    def on_episode_end(self, episode_return: float, info: dict[str, Any]) -> None:
        """Score the current candidate; advance the generation when it is full."""
        self._episodes += 1
        value = float(episode_return)
        if not math.isfinite(value):
            logger.warning(
                "CMAESAgent: non-finite episode return (%s); scoring it as the "
                "worst outcome rather than propagating it into the mean",
                value,
            )
            value = -math.inf
        self._repeat_returns.append(value)
        self._generation_returns.append(value)
        if len(self._repeat_returns) < self.episodes_per_candidate:
            return

        mean_return = float(np.mean(self._repeat_returns))
        self._repeat_returns.clear()
        # CMA-ES minimises; RL maximises.
        self._fitness[self._candidate] = -mean_return
        self._candidate += 1

        if self._candidate < self.es.popsize:
            self._activate(self._population[self._candidate])
            return

        self.es.tell(self._population, self._fitness)
        returns = np.asarray(self._generation_returns, dtype=np.float64)
        finite = returns[np.isfinite(returns)]
        self._last_stats = TrainStats().update(
            generation=float(self.es.generation),
            evaluations=float(self.es.counteval),
            episodes=float(self._episodes),
            sigma=float(self.es.sigma),
            best_return=float(-self.es.best_f),
            generation_best_return=float(-self.es.last_best_f),
            generation_mean_return=float(finite.mean()) if finite.size else float("nan"),
            generation_std_return=float(finite.std()) if finite.size else float("nan"),
            axis_ratio=float(self.es.axis_ratio),
            condition_number=float(self.es.condition_number),
            n_parameters=float(self.policy.size),
        )
        stop = self.es.stop()
        if stop:
            logger.info(
                "CMAESAgent: convergence criterion %s reached at generation %d; "
                "the search will keep sampling but has stopped making progress",
                sorted(stop),
                self.es.generation,
            )
        self._generation_returns.clear()
        self._population = self.es.ask()
        self._fitness = np.full(self.es.popsize, np.nan)
        self._candidate = 0
        self._activate(self._population[0])

    def update(self) -> TrainStats:
        """No-op between generations; reports the last completed one."""
        return self._last_stats

    def reset(self) -> None:
        """Nothing per-episode to clear: the policy is a pure function.

        The candidate index deliberately advances in
        :meth:`on_episode_end`, not here, so a runner that resets an episode
        without finishing it re-evaluates the same candidate instead of
        silently skipping one.
        """

    # --- bookkeeping ---------------------------------------------------------
    def set_seed(self, seed: int) -> None:
        self.seed = int(seed)
        self.config["seed"] = self.seed
        self._rng = np.random.default_rng(self.seed)
        self.es._rng = np.random.default_rng(self.seed)

    @property
    def num_parameters(self) -> int:
        return self.policy.size

    @staticmethod
    def _checkpoint_path(path: str | Path) -> Path:
        """The single definition of where a CMA-ES checkpoint lives.

        Both :meth:`save` and :meth:`load` go through this. They used to
        normalise independently -- ``np.savez`` *appends* ``.npz`` to a name
        that lacks it while ``Path.with_suffix`` *replaces* the extension --
        so ``save("run/cmaes.pt")`` wrote ``run/cmaes.pt.npz`` and
        ``load("run/cmaes.pt")`` went looking for ``run/cmaes.npz``. The
        checkpoint never came back, and since the runner saves a trained policy
        and reloads it for the held-out evaluation, every CMA-ES score in a
        sweep would have belonged to a different policy than the one trained.
        """
        p = Path(path)
        return p if p.suffix == ".npz" else p.with_suffix(".npz")

    def save(self, path: str | Path) -> None:
        p = self._checkpoint_path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        state = self.es.state_dict()
        np.savez(
            p,
            active=self._active,
            candidate=self._candidate,
            population=self._population,
            fitness=self._fitness,
            obs_mean=self.obs_rms.mean if self.obs_rms else np.zeros(1),
            obs_var=self.obs_rms.var if self.obs_rms else np.ones(1),
            obs_count=np.array([self.obs_rms.count if self.obs_rms else 0.0]),
            **{k: v for k, v in state.items() if k != "rng"},
        )

    def load(self, path: str | Path) -> None:
        p = self._checkpoint_path(path)
        with np.load(p, allow_pickle=False) as data:
            self.es.load_state_dict(
                {k: data[k] for k in data.files if k in
                 ("mean", "sigma", "C", "B", "D", "invsqrtC", "pc", "ps",
                  "best_x", "best_f", "counteval", "eigeneval", "generation")}
            )
            self._population = data["population"]
            self._fitness = data["fitness"]
            self._candidate = int(data["candidate"])
            if self.obs_rms is not None and data["obs_count"][0] > 0.0:
                self.obs_rms.mean = data["obs_mean"]
                self.obs_rms.var = data["obs_var"]
                self.obs_rms.count = float(data["obs_count"][0])
            self._activate(data["active"])

    def __repr__(self) -> str:
        return (
            f"<CMAESAgent n={self.policy.size} popsize={self.es.popsize} "
            f"gen={self.es.generation} candidate={self._candidate} "
            f"sigma={self.es.sigma:.3e}>"
        )
