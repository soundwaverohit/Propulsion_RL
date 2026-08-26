"""Model-predictive control by the cross-entropy method.

Plan a horizon of actions, execute the first one, throw the rest away, replan.
The planner is CEM: sample action sequences from a diagonal Gaussian, score
them under a model, refit the Gaussian to the best few, repeat. It is
embarrassingly simple, has no learned policy at all, and is often very hard to
beat when the model is good.

Two models, and the difference between them is the point
--------------------------------------------------------
``model="oracle"`` plans by ``deepcopy``-ing the environment and rolling it
forward. That is cheating, deliberately: it is the answer to "what is
achievable here with perfect dynamics knowledge and a finite planning budget",
which is the single most useful reference point in the whole study. Without it,
a mediocre score from every RL method is ambiguous -- the algorithms might be
failing, or the mission might simply be hard. With it, that ambiguity is
resolved by one number. Oracle CEM does not learn (``learns = False``); there
is nothing to train.

``model="learned"`` is PETS: a probabilistic ensemble of dynamics models
trained on real transitions, with trajectories propagated through the ensemble
so that model disagreement -- epistemic uncertainty -- shows up as variance in
the return estimate rather than as false confidence. This one learns.

Honesty about compute
---------------------
CEM is expensive in a way that per-step-cost tables usually hide: it spends
``population * horizon * iterations`` model evaluations *per environment step*,
and for the oracle that is real environment steps. A comparison that reports
only sample efficiency would make this method look free. Every instance
therefore tracks its own planning wall time and reports
``plan_time_ms_per_step`` in :meth:`update`, and the budget knobs (``horizon``,
``population``, ``elites``, ``iterations``) are all configurable so the
compute/quality trade can be swept rather than asserted.
"""

from __future__ import annotations

import copy
import hashlib
import logging
import math
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn

from ..core.registry import AGENT
from ..core.types import CanonicalCommand
from .base import Agent, TrainStats, Transition
from .buffers import ReplayBuffer
from .networks import count_parameters, resolve_device

logger = logging.getLogger(__name__)

__all__ = ["CEMMPCAgent", "EnsembleDynamics"]

#: What to command when no model is usable. Decodes to full throttle, mid
#: operating point, pure prograde -- the same thing :class:`ProgradeAgent`
#: does, chosen because it is the sane default rather than because it is good.
_FALLBACK_ACTION = CanonicalCommand(1.0, 0.5, 0.0, 0.0, 0.5).to_array()


class _EnsembleLinear(nn.Module):
    """``ensemble_size`` independent linear layers evaluated as one bmm."""

    def __init__(self, ensemble_size: int, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.ensemble_size = int(ensemble_size)
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        # Truncated-normal-ish init at 1/sqrt(2 * in_dim), the PETS default.
        std = 1.0 / (2.0 * math.sqrt(in_dim))
        self.weight = nn.Parameter(
            torch.randn(self.ensemble_size, in_dim, out_dim) * std
        )
        self.bias = nn.Parameter(torch.zeros(self.ensemble_size, 1, out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x`` is ``(E, B, in_dim)`` -> ``(E, B, out_dim)``."""
        return torch.baddbmm(self.bias, x, self.weight)

    def extra_repr(self) -> str:
        return f"E={self.ensemble_size}, in={self.in_dim}, out={self.out_dim}"


class EnsembleDynamics(nn.Module):
    """Probabilistic ensemble predicting ``(delta_obs, reward, cost)``.

    Three choices that matter and are easy to get wrong:

    * It predicts the **delta** of the observation, not the observation. Over a
      36-wide vector where most channels barely move per step, regressing the
      absolute next state means the model spends all its capacity learning the
      identity function and its errors are large in exactly the channels the
      planner cares about.
    * It predicts **reward and cost too**. A planner needs a reward function,
      and the alternative -- requiring every mission to expose a differentiable
      reward over observations -- would couple the planner to mission internals
      it has no business knowing.
    * Each member has its own **heteroscedastic variance head** with softly
      bounded log-variance, trained by Gaussian negative log-likelihood. That
      splits "the dynamics are noisy here" (aleatoric, the variance head) from
      "the members disagree here" (epistemic, the spread across members), and
      only the second should make a planner cautious.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        *,
        ensemble_size: int = 5,
        hidden: Sequence[int] = (200, 200),
        probabilistic: bool = True,
    ) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.ensemble_size = int(ensemble_size)
        self.probabilistic = bool(probabilistic)
        self.out_dim = self.obs_dim + 2          # delta_obs, reward, cost

        in_dim = self.obs_dim + self.action_dim
        widths = [in_dim, *[int(h) for h in hidden]]
        self.layers = nn.ModuleList(
            _EnsembleLinear(self.ensemble_size, a, b)
            for a, b in zip(widths[:-1], widths[1:])
        )
        self.head = _EnsembleLinear(self.ensemble_size, widths[-1], 2 * self.out_dim)

        # Soft log-variance bounds, learned but penalised (Chua et al. 2018).
        self.max_logvar = nn.Parameter(torch.full((self.out_dim,), 0.5))
        self.min_logvar = nn.Parameter(torch.full((self.out_dim,), -10.0))

        # Normalisation statistics; buffers so they ride along in state_dict.
        self.register_buffer("in_mean", torch.zeros(in_dim))
        self.register_buffer("in_std", torch.ones(in_dim))
        self.register_buffer("out_mean", torch.zeros(self.out_dim))
        self.register_buffer("out_std", torch.ones(self.out_dim))
        self.register_buffer("fitted", torch.zeros(1))

    def fit_normalizers(self, inputs: np.ndarray, targets: np.ndarray) -> None:
        """Per-dimension whitening statistics, refit before each training run."""
        dev = self.in_mean.device
        in_mean = torch.as_tensor(inputs.mean(0), dtype=torch.float32, device=dev)
        in_std = torch.as_tensor(inputs.std(0), dtype=torch.float32, device=dev)
        out_mean = torch.as_tensor(targets.mean(0), dtype=torch.float32, device=dev)
        out_std = torch.as_tensor(targets.std(0), dtype=torch.float32, device=dev)
        # A constant channel (a zero-padded observation slot, say) has zero
        # spread; leaving it at zero would divide by ~0 and hand the network an
        # enormous input for a channel that carries no information at all.
        self.in_mean.copy_(in_mean)
        self.in_std.copy_(torch.clamp(in_std, min=1e-6))
        self.out_mean.copy_(out_mean)
        self.out_std.copy_(torch.clamp(out_std, min=1e-6))
        self.fitted.fill_(1.0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalised ``(E, B, obs+act)`` in, normalised ``(mean, logvar)`` out."""
        h = x
        for layer in self.layers:
            h = torch.nn.functional.silu(layer(h))
        out = self.head(h)
        mean, logvar = out[..., : self.out_dim], out[..., self.out_dim :]
        logvar = self.max_logvar - torch.nn.functional.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + torch.nn.functional.softplus(logvar - self.min_logvar)
        return mean, logvar

    def _normalize_in(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, action], dim=-1)
        return (x - self.in_mean) / self.in_std

    def loss(
        self, obs: torch.Tensor, action: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gaussian NLL (or MSE when deterministic) plus the log-var penalty."""
        mean, logvar = self(self._normalize_in(obs, action))
        target_n = (target - self.out_mean) / self.out_std
        mse = (mean - target_n) ** 2
        if self.probabilistic:
            inv_var = torch.exp(-logvar)
            nll = (mse * inv_var + logvar).mean(dim=(1, 2)).sum()
            nll = nll + 0.01 * (self.max_logvar.sum() - self.min_logvar.sum())
        else:
            nll = mse.mean(dim=(1, 2)).sum()
        return nll, mse.mean().detach()

    @torch.no_grad()
    def predict(
        self, obs: torch.Tensor, action: torch.Tensor, sample: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One step for every ensemble member: ``(next_obs, reward, cost)``.

        ``obs`` and ``action`` are ``(E, B, ...)``; the returned tensors keep
        that layout, so a whole population is propagated through the whole
        ensemble in one call.
        """
        mean, logvar = self(self._normalize_in(obs, action))
        if sample and self.probabilistic:
            mean = mean + torch.randn_like(mean) * torch.exp(0.5 * logvar)
        raw = mean * self.out_std + self.out_mean
        delta, reward, cost = raw[..., : self.obs_dim], raw[..., -2], raw[..., -1]
        return obs + delta, reward, cost


@AGENT.register("cem_mpc", kind="planning", learns=True)
class CEMMPCAgent(Agent):
    """Cross-entropy-method MPC over an oracle or a learned dynamics model.

    Parameters
    ----------
    model:
        ``"oracle"`` (deepcopy the real environment; does not learn) or
        ``"learned"`` (PETS ensemble; does).
    env:
        The live environment, for the oracle model. May also be supplied later
        with :meth:`set_env`. It must be the *same object* the runner is
        stepping, because the planner rolls forward from its current internal
        state -- the observation alone does not determine it.
    horizon, population, elites, iterations:
        The planning budget. Defaults are modest on purpose; the oracle model
        spends ``population * horizon * iterations`` real environment steps per
        action, so the defaults here are a starting point for a sweep, not a
        recommendation.
    alpha:
        Fraction of the previous CEM mean retained per refit. Smoothing stops a
        single lucky elite set from throwing the plan away.
    gamma:
        Discount inside the planning horizon.
    cost_penalty:
        Weight on ``info["constraint_cost"]`` in the planning objective. CEM
        has no Lagrangian machinery; this is its (blunt) constraint handling.
    risk_aversion:
        Learned model only. Scores a sequence as ``mean - risk_aversion * std``
        across ensemble members, so plans that rely on a part of the state
        space the members disagree about are penalised. ``0`` is risk-neutral.
    """

    name = "cem_mpc"
    uses_constraints = False

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        *,
        model: str = "oracle",
        env: Any = None,
        horizon: int = 12,
        population: int = 64,
        elites: int = 8,
        iterations: int = 4,
        alpha: float = 0.1,
        gamma: float = 0.99,
        init_std: float = 0.5,
        min_std: float = 0.05,
        cost_penalty: float = 10.0,
        risk_aversion: float = 0.0,
        seed: int = 0,
        device: str = "cpu",
        # --- learned model only ---
        ensemble_size: int = 5,
        model_hidden: Sequence[int] = (200, 200),
        model_lr: float = 1e-3,
        model_batch: int = 256,
        model_epochs: int = 20,
        buffer_size: int = 200_000,
        warmup_steps: int = 1_000,
        train_every: int = 1_000,
        probabilistic: bool = True,
        weight_decay: float = 1e-4,
        **kwargs: Any,
    ) -> None:
        super().__init__(obs_dim, action_dim, **kwargs)
        if model not in ("oracle", "learned"):
            raise ValueError("model must be 'oracle' or 'learned'")
        if elites > population:
            raise ValueError("elites cannot exceed population")
        self.model = model
        #: Oracle CEM has nothing to train, and telling the runner otherwise
        #: would spend a training budget on it for no reason.
        self.learns = model == "learned"

        self.horizon = int(horizon)
        self.population = int(population)
        self.elites = int(elites)
        self.iterations = int(iterations)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.init_std = float(init_std)
        self.min_std = float(min_std)
        self.cost_penalty = float(cost_penalty)
        self.risk_aversion = float(risk_aversion)
        self.seed = int(seed)
        self.device = resolve_device(device)
        self._rng = np.random.default_rng(self.seed)

        self.config.update(
            model=self.model,
            horizon=self.horizon,
            population=self.population,
            elites=self.elites,
            iterations=self.iterations,
            gamma=self.gamma,
            cost_penalty=self.cost_penalty,
            risk_aversion=self.risk_aversion,
            seed=self.seed,
        )

        self._mean = np.zeros((self.horizon, self.action_dim), dtype=np.float64)
        self._std = np.full(
            (self.horizon, self.action_dim), self.init_std, dtype=np.float64
        )

        # --- planning cost accounting ---
        self._plan_time_s = 0.0
        self._plan_calls = 0
        self._model_evals = 0

        # --- oracle ---
        self._env: Any = None
        self._oracle_ok = False
        if env is not None:
            self.set_env(env)

        # --- learned ---
        self.dynamics: EnsembleDynamics | None = None
        self.buffer: ReplayBuffer | None = None
        self._model_opt: torch.optim.Optimizer | None = None
        self.model_batch = int(model_batch)
        self.model_epochs = int(model_epochs)
        self.warmup_steps = int(warmup_steps)
        self.train_every = int(train_every)
        self._steps_seen = 0
        self._steps_since_train = 0
        self._model_trained = False
        self._last_model_loss = float("nan")
        if self.model == "learned":
            torch.manual_seed(self.seed)
            self.dynamics = EnsembleDynamics(
                obs_dim,
                action_dim,
                ensemble_size=int(ensemble_size),
                hidden=model_hidden,
                probabilistic=bool(probabilistic),
            ).to(self.device)
            self._model_opt = torch.optim.Adam(
                self.dynamics.parameters(),
                lr=float(model_lr),
                weight_decay=float(weight_decay),
            )
            self.buffer = ReplayBuffer(
                int(buffer_size), obs_dim, action_dim, seed=self.seed
            )
            self.config.update(
                ensemble_size=int(ensemble_size),
                model_hidden=tuple(int(h) for h in model_hidden),
                warmup_steps=self.warmup_steps,
                train_every=self.train_every,
            )
        self._fallback_warned = False

    # --- the oracle model ----------------------------------------------------
    def set_env(self, env: Any) -> bool:
        """Attach the live environment for oracle planning.

        Returns whether oracle planning is actually available. An environment
        that cannot be deep-copied -- one holding a file handle, a socket, a
        compiled simulator with a C pointer in it -- is not a bug to raise on;
        it just means this reference point is unavailable for that
        configuration, and the agent says so once and falls back.
        """
        self._env = env
        self._oracle_ok = False
        if env is None:
            return False
        if not (hasattr(env, "step") and hasattr(env, "reset")):
            logger.warning(
                "CEMMPCAgent: object passed as env has no step/reset; "
                "oracle planning disabled"
            )
            return False
        try:
            probe = copy.deepcopy(env)
        except Exception as exc:  # noqa: BLE001 - any failure means no oracle
            logger.warning(
                "CEMMPCAgent: environment is not deep-copyable (%s: %s); "
                "oracle planning disabled, falling back to %s",
                type(exc).__name__,
                exc,
                "the learned model" if self.model == "learned" else "a fixed action",
            )
            return False
        del probe
        self._oracle_ok = True
        return True

    def _score_oracle(self, sequences: np.ndarray) -> np.ndarray:
        """Discounted return of each action sequence in the real environment."""
        scores = np.empty(sequences.shape[0], dtype=np.float64)
        for i, seq in enumerate(sequences):
            try:
                sim = copy.deepcopy(self._env)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "CEMMPCAgent: deepcopy failed mid-plan (%s); disabling oracle",
                    type(exc).__name__,
                )
                self._oracle_ok = False
                return scores[:i] if i else np.zeros(sequences.shape[0])
            total = 0.0
            discount = 1.0
            for t in range(self.horizon):
                obs, reward, terminated, truncated, info = sim.step(
                    seq[t].astype(np.float32)
                )
                cost = float(info.get("constraint_cost", 0.0)) if info else 0.0
                total += discount * (float(reward) - self.cost_penalty * cost)
                discount *= self.gamma
                self._model_evals += 1
                if terminated or truncated:
                    break
            scores[i] = total
        return scores

    # --- the learned model ---------------------------------------------------
    def _score_learned(self, obs: np.ndarray, sequences: np.ndarray) -> np.ndarray:
        """Discounted return under the ensemble, aggregated across members."""
        assert self.dynamics is not None
        e = self.dynamics.ensemble_size
        n = sequences.shape[0]
        dev = self.device
        state = torch.as_tensor(
            np.broadcast_to(obs.astype(np.float32), (e, n, self.obs_dim)).copy(),
            device=dev,
        )
        acts = torch.as_tensor(sequences.astype(np.float32), device=dev)
        total = torch.zeros(e, n, device=dev)
        discount = 1.0
        for t in range(self.horizon):
            a = acts[:, t, :].unsqueeze(0).expand(e, n, self.action_dim)
            state, reward, cost = self.dynamics.predict(state, a, sample=True)
            total = total + discount * (
                reward - self.cost_penalty * torch.clamp(cost, min=0.0)
            )
            discount *= self.gamma
            self._model_evals += e * n
        score = total.mean(dim=0)
        if self.risk_aversion != 0.0 and e > 1:
            score = score - self.risk_aversion * total.std(dim=0)
        return score.cpu().numpy().astype(np.float64)

    def _train_model(self) -> float:
        """Refit the ensemble on everything in the buffer. Returns the loss."""
        assert self.dynamics is not None and self.buffer is not None
        assert self._model_opt is not None
        size = len(self.buffer)
        if size < max(self.model_batch, 2):
            return float("nan")

        obs = self.buffer.obs[:size]
        actions = self.buffer.actions[:size]
        next_obs = self.buffer.next_obs[:size]
        rewards = self.buffer.rewards[:size]
        costs = self.buffer.costs[:size]
        inputs = np.concatenate([obs, actions], axis=1)
        targets = np.concatenate(
            [next_obs - obs, rewards[:, None], costs[:, None]], axis=1
        )
        self.dynamics.fit_normalizers(inputs, targets)

        e = self.dynamics.ensemble_size
        batches = max(1, size // self.model_batch)
        losses: list[float] = []
        self.dynamics.train()
        for _ in range(self.model_epochs):
            for _ in range(batches):
                # One independent draw per member: bootstrap resampling is what
                # makes the members disagree for the right reason.
                idx = self._rng.integers(0, size, size=(e, self.model_batch))
                o = torch.as_tensor(obs[idx], device=self.device)
                a = torch.as_tensor(actions[idx], device=self.device)
                y = torch.as_tensor(targets[idx], device=self.device)
                loss, _ = self.dynamics.loss(o, a, y)
                self._model_opt.zero_grad(set_to_none=True)
                loss.backward()
                self._model_opt.step()
                losses.append(float(loss.item()))
        self.dynamics.eval()
        self._model_trained = True
        return float(np.mean(losses)) if losses else float("nan")

    # --- the planner ---------------------------------------------------------
    def _plan_rng(self, obs: np.ndarray, deterministic: bool) -> np.random.Generator:
        """Reproducible stream in eval, the agent's own stream in training.

        Determinism for a sampling planner has to be constructed, not assumed:
        seeding from the observation makes ``act`` a pure function of state
        given the warm-start cache, which ``reset`` clears.
        """
        if not deterministic:
            return self._rng
        digest = hashlib.blake2b(
            np.asarray(obs, dtype=np.float64).tobytes()
            + self.seed.to_bytes(8, "little", signed=True),
            digest_size=8,
        ).digest()
        return np.random.default_rng(int.from_bytes(digest, "little"))

    def _usable_model(self) -> str | None:
        if self.model == "oracle" and self._oracle_ok and self._env is not None:
            return "oracle"
        if self.model == "learned" and self._model_trained:
            return "learned"
        # A learned agent whose model is not trained yet can still fall back to
        # the oracle if someone handed it an environment, and vice versa.
        if self._oracle_ok and self._env is not None:
            return "oracle"
        if self.dynamics is not None and self._model_trained:
            return "learned"
        return None

    def _fallback(self) -> np.ndarray:
        if not self._fallback_warned:
            self._fallback_warned = True
            logger.warning(
                "CEMMPCAgent: no usable model (model=%r, oracle=%s, trained=%s); "
                "emitting the fixed prograde fallback action",
                self.model,
                self._oracle_ok,
                self._model_trained,
            )
        # The planner supports reduced-actuator ablations, so its emergency
        # action must honour the width it was constructed with as faithfully
        # as a planned action does.  Keep the canonical prograde values where
        # they exist and use neutral commands for any extra dimensions.
        fallback = np.zeros(self.action_dim, dtype=_FALLBACK_ACTION.dtype)
        width = min(self.action_dim, _FALLBACK_ACTION.size)
        fallback[:width] = _FALLBACK_ACTION[:width]
        return fallback

    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        which = self._usable_model()
        if which is None:
            return self._fallback()

        obs_arr = np.asarray(obs, dtype=np.float64).reshape(-1)
        rng = self._plan_rng(obs_arr, deterministic)
        mean = self._mean.copy()
        std = self._std.copy()
        started = time.perf_counter()

        elites = np.empty((0, self.horizon, self.action_dim))
        for _ in range(self.iterations):
            samples = mean + std * rng.standard_normal(
                (self.population, self.horizon, self.action_dim)
            )
            np.clip(samples, -1.0, 1.0, out=samples)
            if which == "oracle":
                scores = self._score_oracle(samples)
                if not self._oracle_ok:      # disabled mid-plan
                    self._plan_time_s += time.perf_counter() - started
                    self._plan_calls += 1
                    which = self._usable_model()
                    if which is None:
                        return self._fallback()
                    return self.act(obs, deterministic)
            else:
                scores = self._score_learned(obs_arr, samples)

            order = np.argsort(-scores, kind="stable")[: self.elites]
            elites = samples[order]
            new_mean = elites.mean(axis=0)
            new_std = elites.std(axis=0)
            mean = self.alpha * mean + (1.0 - self.alpha) * new_mean
            std = np.maximum(self.alpha * std + (1.0 - self.alpha) * new_std,
                             self.min_std)

        self._plan_time_s += time.perf_counter() - started
        self._plan_calls += 1

        action = np.clip(mean[0], -1.0, 1.0).astype(np.float32)
        # Warm start: shift the plan one step and re-open the last slot. The
        # single biggest quality win available to a receding-horizon planner,
        # and per-episode state, so reset() clears it.
        self._mean = np.roll(mean, -1, axis=0)
        self._mean[-1] = 0.0
        self._std = np.full(
            (self.horizon, self.action_dim), self.init_std, dtype=np.float64
        )
        return action

    # --- Agent interface -----------------------------------------------------
    def observe_transition(self, tr: Transition) -> None:
        if self.buffer is None:
            return
        self.buffer.add_transition(tr)
        self._steps_seen += 1
        self._steps_since_train += 1

    def update(self) -> TrainStats:
        stats = TrainStats().update(
            plan_time_ms_per_step=float(self.plan_time_ms_per_step),
            plan_calls=float(self._plan_calls),
            model_evals_per_step=float(
                self._model_evals / self._plan_calls if self._plan_calls else 0.0
            ),
            planning_budget=float(self.population * self.horizon * self.iterations),
        )
        if not self.learns or self.buffer is None:
            return stats
        ready = (
            len(self.buffer) >= self.warmup_steps
            and self._steps_since_train >= self.train_every
        )
        if ready:
            self._last_model_loss = self._train_model()
            self._steps_since_train = 0
        return stats.update(
            model_loss=self._last_model_loss,
            model_trained=float(self._model_trained),
            buffer_size=float(len(self.buffer)),
            total_steps=float(self._steps_seen),
        )

    def reset(self) -> None:
        """Clear the warm-started plan; the model and buffer survive."""
        self._mean = np.zeros((self.horizon, self.action_dim), dtype=np.float64)
        self._std = np.full(
            (self.horizon, self.action_dim), self.init_std, dtype=np.float64
        )

    def set_seed(self, seed: int) -> None:
        self.seed = int(seed)
        self.config["seed"] = self.seed
        self._rng = np.random.default_rng(self.seed)
        if self.buffer is not None:
            self.buffer.set_seed(self.seed)

    # --- diagnostics ---------------------------------------------------------
    @property
    def plan_time_ms_per_step(self) -> float:
        """Mean planning wall time per environment step, milliseconds."""
        if not self._plan_calls:
            return 0.0
        return 1000.0 * self._plan_time_s / self._plan_calls

    @property
    def num_parameters(self) -> int:
        if self.dynamics is None:
            return 0     # oracle CEM has no parameters at all, and should say so
        return count_parameters([self.dynamics])

    def save(self, path: str | Path) -> None:
        if self.dynamics is None:
            logger.info("CEMMPCAgent(oracle) has no parameters to save")
            return
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "agent": type(self).__name__,
                "config": dict(self.config),
                "dynamics": self.dynamics.state_dict(),
                "optimizer": None
                if self._model_opt is None
                else self._model_opt.state_dict(),
                "model_trained": self._model_trained,
            },
            p,
        )

    def load(self, path: str | Path) -> None:
        if self.dynamics is None:
            logger.info("CEMMPCAgent(oracle) has no parameters to load")
            return
        try:
            state = torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:  # pragma: no cover - torch < 1.13
            state = torch.load(path, map_location=self.device)
        self.dynamics.load_state_dict(state["dynamics"])
        if state.get("optimizer") and self._model_opt is not None:
            self._model_opt.load_state_dict(state["optimizer"])
        self._model_trained = bool(state.get("model_trained", True))

    def __repr__(self) -> str:
        return (
            f"<CEMMPCAgent model={self.model!r} H={self.horizon} "
            f"pop={self.population} iters={self.iterations} "
            f"plan={self.plan_time_ms_per_step:.1f}ms/step>"
        )
