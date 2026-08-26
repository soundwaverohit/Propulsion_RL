"""Constrained PPO via a Lagrangian multiplier on the constraint cost.

Why this agent exists
---------------------
It carries one of the headline hypotheses of the benchmark. Nuclear propulsion
has hard safety constraints -- fuel temperature, prompt-criticality margin,
thermal stress -- where a violation is not a bad reward, it is a destroyed
reactor. Folding those into the reward with a fixed penalty weight means
hand-tuning that weight per mission and per propulsion system, and the tuned
value trades safety against performance at a rate nobody chose deliberately.
A Lagrangian method instead *learns* the weight from the constraint itself, so
the prediction is: constrained RL should beat unconstrained RL on the nuclear
systems, and be pure overhead on solar-electric ones where nothing binds.

For that to be a testable prediction rather than a slogan, the multiplier and
the constraint it tracks have to be visible. Every :meth:`update` reports
``lagrange_lambda``, ``cost_jc``, ``cost_limit`` and
``constraint_violation_rate``, so "the multiplier never moved" and "the
multiplier moved and the policy ignored it" are distinguishable in the logs.

The method
----------
Two critics, one policy. The cost critic is trained exactly like the reward
critic -- its own GAE, its own discount, since :class:`RolloutBuffer` already
carries the parallel channel -- and the surrogate is optimised on the effective
advantage::

    A = (A_r - lambda * A_c) / (1 + lambda)

The ``1 / (1 + lambda)`` is not cosmetic. Without it the effective advantage
grows without bound as the multiplier climbs, and against a fixed clip range
that is a silently growing step size exactly when the policy is in trouble. It
also means the combined advantage is deliberately **not** re-standardised --
doing so would divide the shrink straight back out -- so the reward and cost
advantages are standardised separately and combined afterwards. That is the one
reason :meth:`_minibatch_loss` restates the base class's loss rather than
delegating to it.

The multiplier is ``lambda = softplus(nu)`` with dual gradient ascent on
``nu``::

    nu <- nu + lambda_lr * (J_C - d)

Softplus rather than a clamp, for two reasons. A clamped multiplier resting at
zero sits on a boundary where the projected gradient is identically zero in one
direction, so it re-enters the feasible region as a jump rather than a slope,
and that jump is a step change in the effective advantage that PPO's trust
region was never asked to absorb. Softplus is smooth everywhere and strictly
positive, so the same trajectory is a continuous ramp. The dual step is taken
on ``nu`` directly rather than through the softplus Jacobian: applying the
Jacobian scales the effective learning rate by ``sigmoid(nu)``, which vanishes
as ``nu`` goes negative, so a comfortably-feasible run would anneal away its
own ability to ever respond again.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
import torch
from torch import nn

from ..core.registry import AGENT
from .base import TrainStats, Transition
from .buffers import RolloutBuffer
from .networks import ValueNetwork, count_parameters
from .ppo import PPOAgent

logger = logging.getLogger(__name__)

__all__ = ["LagrangianPPOAgent"]

#: ``softplus`` is the identity long before this; the cap exists so a
#: pathological run cannot overflow ``nu``, not to bound the constraint.
_NU_MAX = 60.0


def _softplus(x: float) -> float:
    """Numerically safe scalar softplus."""
    return x if x > 30.0 else math.log1p(math.exp(x))


def _inverse_softplus(y: float) -> float:
    """``nu`` such that ``softplus(nu) == y``, for a requested initial lambda."""
    if y <= 0.0:
        return -10.0          # lambda ~ 4.5e-5: effectively off, not stuck
    if y > 30.0:
        return y
    return math.log(math.expm1(y))


def _standardize(x: torch.Tensor) -> torch.Tensor:
    return (x - x.mean()) / (x.std(unbiased=False) + 1e-8)


@AGENT.register("lagrangian_ppo", kind="on_policy", learns=True, constrained=True)
class LagrangianPPOAgent(PPOAgent):
    """PPO with a learned Lagrangian multiplier on the constraint cost.

    Parameters
    ----------
    cost_limit:
        ``d`` in ``J_C <= d``: the tolerated **undiscounted episodic**
        constraint cost. ``0.0`` demands strict feasibility, which is the right
        setting for a hard safety constraint and a slow one for a soft budget.
    lambda_lr:
        Dual ascent step on ``nu``. Deliberately far larger than the policy
        learning rate: the dual variable takes one step per rollout, not one
        per minibatch, so it needs to move meaningfully in a single update.
    lambda_init:
        Starting multiplier. Small and non-zero, so the first constrained
        rollout already carries some pressure.
    cost_gamma, cost_gae_lambda:
        Discount and trace for the cost critic. ``cost_gamma`` defaults to
        ``gamma``; a value nearer 1 is often better, because a constraint
        violation 500 steps away is still a destroyed reactor.
    cost_vf_coef:
        Weight on the cost-critic regression loss.
    cost_limit_decay:
        Optional geometric annealing of the limit towards ``cost_limit_final``,
        applied once per update. For curriculum-style tightening; off by
        default because an annealed limit makes runs harder to compare.

    Everything else is :class:`PPOAgent`'s.
    """

    name = "lagrangian_ppo"
    learns = True
    uses_constraints = True

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        *,
        cost_limit: float = 25.0,
        lambda_lr: float = 0.035,
        lambda_init: float = 0.05,
        cost_gamma: float | None = None,
        cost_gae_lambda: float | None = None,
        cost_vf_coef: float = 0.5,
        cost_limit_final: float | None = None,
        cost_limit_decay: float = 1.0,
        **kwargs: Any,
    ) -> None:
        # Set before super().__init__: the base constructor calls
        # _make_buffer() while building itself, and that reads these.
        self.cost_limit = float(cost_limit)
        self.cost_limit_final = (
            self.cost_limit if cost_limit_final is None else float(cost_limit_final)
        )
        self.cost_limit_decay = float(cost_limit_decay)
        self.lambda_lr = float(lambda_lr)
        self.cost_vf_coef = float(cost_vf_coef)
        self._cost_gamma_arg = cost_gamma
        self._cost_lambda_arg = cost_gae_lambda
        self._nu = _inverse_softplus(float(lambda_init))

        super().__init__(obs_dim, action_dim, **kwargs)

        self.cost_gamma = float(
            self.gamma if cost_gamma is None else cost_gamma
        )
        self.cost_gae_lambda = float(
            self.gae_lambda if cost_gae_lambda is None else cost_gae_lambda
        )
        self.cost_value_net = ValueNetwork(
            obs_dim, hidden=self.hidden, activation=self.activation
        ).to(self.device)
        # The base built its optimiser before this net existed (see
        # _learnable), so the cost critic joins as an extra param group rather
        # than by rebuilding and losing the configured defaults.
        self.optimizer.add_param_group(
            {"params": list(self.cost_value_net.parameters())}
        )

        self.config.update(
            cost_limit=self.cost_limit,
            lambda_lr=self.lambda_lr,
            lambda_init=float(lambda_init),
            cost_gamma=self.cost_gamma,
            cost_gae_lambda=self.cost_gae_lambda,
            cost_vf_coef=self.cost_vf_coef,
        )

        # Episodic cost bookkeeping. J_C is the *undiscounted* episodic cost,
        # which is what the limit is stated in.
        self._episode_cost = 0.0
        self._episode_len = 0
        self._recent_ep_costs: list[float] = []
        self._recent_ep_lens: list[float] = []
        self._episode_len_estimate = float(self.rollout_steps)
        self._cost_v_losses: list[float] = []
        self._last_jc = 0.0

    # --- construction hooks --------------------------------------------------
    def _make_buffer(self) -> RolloutBuffer:
        return RolloutBuffer(
            self.rollout_steps,
            self.obs_dim,
            self.action_dim,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            cost_gamma=self._cost_gamma_arg,
            cost_lambda=self._cost_lambda_arg,
            track_cost=True,
        )

    def _learnable(self) -> list[nn.Parameter]:
        params = super()._learnable()
        cost_net = getattr(self, "cost_value_net", None)
        if cost_net is not None:
            params = params + list(cost_net.parameters())
        return params

    # --- the multiplier ------------------------------------------------------
    @property
    def lagrange_multiplier(self) -> float:
        """Current ``lambda >= 0``."""
        return _softplus(self._nu)

    @property
    def nu(self) -> float:
        """The unconstrained dual parameter behind the softplus."""
        return float(self._nu)

    def update_lambda(self, mean_episodic_cost: float) -> float:
        """One dual gradient-ascent step; returns the new ``lambda``.

        Public and side-effect-local on purpose: the dual dynamics are the
        thing this agent is being asked to demonstrate, so they are testable
        without collecting a rollout.
        """
        jc = float(mean_episodic_cost)
        if not math.isfinite(jc):
            logger.warning("lagrangian_ppo: non-finite J_C (%s); skipping dual step", jc)
            return self.lagrange_multiplier
        self._nu = float(np.clip(self._nu + self.lambda_lr * (jc - self.cost_limit),
                                 -_NU_MAX, _NU_MAX))
        return self.lagrange_multiplier

    # --- rollout plumbing ----------------------------------------------------
    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        action = super().act(obs, deterministic)
        if self._cached is not None:
            with torch.no_grad():
                t = self._to_tensor(self._cached["obs_n"])
                self._cached["cost_value"] = float(self.cost_value_net(t).item())
        return action

    def _cost_value_of(self, obs_n: np.ndarray) -> float:
        with torch.no_grad():
            return float(self.cost_value_net(self._to_tensor(obs_n)).item())

    def _extra_add_kwargs(
        self, tr: Transition, next_obs_n: np.ndarray
    ) -> dict[str, float]:
        cached = self._cached
        if cached is not None and "cost_value" in cached:
            cost_value = float(cached["cost_value"])
        else:
            cost_value = self._cost_value_of(self._normalize_obs(tr.obs))
        # Mirrors the base's next_value rule: only a truncation needs the
        # bootstrap, every other step reads values[t+1] out of the buffer.
        next_cost_value = 0.0
        if tr.truncated and not tr.terminated:
            next_cost_value = self._cost_value_of(next_obs_n)
        return {"cost_value": cost_value, "next_cost_value": next_cost_value}

    def _last_cost_value(self) -> tuple[float, ...]:
        if self._last_next_obs_n is None or self._last_terminated:
            return (0.0,)
        return (self._cost_value_of(self._last_next_obs_n),)

    def observe_transition(self, tr: Transition) -> None:
        super().observe_transition(tr)
        if not self.learns:
            return
        self._episode_cost += float(tr.cost)
        self._episode_len += 1
        if tr.terminated or tr.truncated:
            self._recent_ep_costs.append(self._episode_cost)
            self._recent_ep_lens.append(float(self._episode_len))
            self._episode_cost = 0.0
            self._episode_len = 0

    def reset(self) -> None:
        super().reset()
        # A runner that resets without ever delivering a terminal transition
        # would otherwise carry one episode's cost into the next.
        self._episode_cost = 0.0
        self._episode_len = 0

    # --- constraint accounting ----------------------------------------------
    def _rollout_cost_stats(self) -> tuple[float, float]:
        """``(J_C, violation_rate)`` for the rollout about to be trained on.

        ``J_C`` is the mean undiscounted cost of the episodes that *finished*
        during the rollout. When none finished -- long missions with a rollout
        shorter than an episode, which is the normal case here -- it is
        estimated as the mean per-step cost times the running estimate of
        episode length, so the quantity stays on the same scale the limit is
        expressed in instead of silently becoming a per-step cost.
        """
        n = len(self.buffer)
        costs = self.buffer.costs[:n]
        violation_rate = float(np.mean(costs > 0.0)) if n else 0.0

        if self._recent_ep_lens:
            self._episode_len_estimate = float(np.mean(self._recent_ep_lens))
        self._episodes_completed = len(self._recent_ep_costs)
        if self._recent_ep_costs:
            jc = float(np.mean(self._recent_ep_costs))
        elif n:
            jc = float(np.mean(costs)) * self._episode_len_estimate
        else:
            jc = 0.0
        self._recent_ep_costs.clear()
        self._recent_ep_lens.clear()
        self._last_jc = jc
        return jc, violation_rate

    def _anneal_cost_limit(self) -> None:
        if self.cost_limit_decay >= 1.0:
            return
        target = self.cost_limit_final
        self.cost_limit = target + (self.cost_limit - target) * self.cost_limit_decay
        self.config["cost_limit"] = self.cost_limit

    # --- the update ----------------------------------------------------------
    def _train_epochs(self) -> TrainStats:
        jc, violation_rate = self._rollout_cost_stats()
        # Dual step first, so the policy epochs below optimise against the
        # multiplier that reflects the rollout they were collected under.
        lam = self.update_lambda(jc)
        self._cost_v_losses = []

        stats = super()._train_epochs()
        self._anneal_cost_limit()

        n = len(self.buffer)
        cost_returns = self.buffer.cost_returns[:n]
        return stats.update(
            lagrange_lambda=float(lam),
            lagrange_nu=float(self._nu),
            cost_jc=float(jc),
            cost_limit=float(self.cost_limit),
            cost_surplus=float(jc - self.cost_limit),
            constraint_violation_rate=float(violation_rate),
            cost_rate=float(np.mean(self.buffer.costs[:n])) if n else 0.0,
            cost_return_mean=float(np.mean(cost_returns)) if n else 0.0,
            cost_value_loss=(
                float(np.mean(self._cost_v_losses))
                if self._cost_v_losses
                else float("nan")
            ),
            episodes_completed=float(len(self._recent_ep_costs)),
        )

    def _minibatch_loss(
        self, mb: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, float]] | None:
        """Clipped surrogate on the Lagrangian advantage, plus both critics.

        This restates rather than extends :meth:`PPOAgent._minibatch_loss`
        because the base standardises the advantage it is handed, and doing
        that *after* the ``1 / (1 + lambda)`` shrink would cancel it exactly.
        """
        new_logp, entropy = self.policy.evaluate(mb["obs"], mb["actions"])
        log_ratio = new_logp - mb["logprobs"]
        ratio = log_ratio.exp()

        lam = self.lagrange_multiplier
        adv_r = _standardize(mb["advantages"])
        adv_c = _standardize(mb["cost_advantages"])
        adv = (adv_r - lam * adv_c) / (1.0 + lam)

        pg1 = -adv * ratio
        pg2 = -adv * torch.clamp(ratio, 1.0 - self.clip, 1.0 + self.clip)
        policy_loss = torch.max(pg1, pg2).mean()

        new_value = self.value_net(mb["obs"])
        new_cost_value = self.cost_value_net(mb["obs"])
        if self.clip_vloss:
            clipped_v = mb["values"] + torch.clamp(
                new_value - mb["values"], -self.clip, self.clip
            )
            value_loss = 0.5 * torch.max(
                (new_value - mb["returns"]) ** 2, (clipped_v - mb["returns"]) ** 2
            ).mean()
            clipped_cv = mb["cost_values"] + torch.clamp(
                new_cost_value - mb["cost_values"], -self.clip, self.clip
            )
            cost_value_loss = 0.5 * torch.max(
                (new_cost_value - mb["cost_returns"]) ** 2,
                (clipped_cv - mb["cost_returns"]) ** 2,
            ).mean()
        else:
            value_loss = 0.5 * ((new_value - mb["returns"]) ** 2).mean()
            cost_value_loss = 0.5 * ((new_cost_value - mb["cost_returns"]) ** 2).mean()

        entropy_mean = entropy.mean()
        loss = (
            policy_loss
            + self.vf_coef * value_loss
            + self.cost_vf_coef * cost_value_loss
            - self.ent_coef * entropy_mean
        )

        if not torch.isfinite(loss):
            logger.warning(
                "lagrangian_ppo: non-finite loss (%s); skipping minibatch",
                loss.item(),
            )
            return None

        self._cost_v_losses.append(float(cost_value_loss.item()))
        with torch.no_grad():
            approx_kl = ((ratio - 1.0) - log_ratio).mean()
            clip_fraction = ((ratio - 1.0).abs() > self.clip).float().mean()

        return loss, {
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": float(entropy_mean.item()),
            "approx_kl": float(approx_kl.item()),
            "clip_fraction": float(clip_fraction.item()),
        }

    # --- bookkeeping ---------------------------------------------------------
    @property
    def num_parameters(self) -> int:
        return count_parameters([self.policy, self.value_net, self.cost_value_net])

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        state.update(
            cost_value_net=self.cost_value_net.state_dict(),
            nu=self._nu,
            cost_limit=self.cost_limit,
            episode_len_estimate=self._episode_len_estimate,
        )
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        super().load_state_dict(state)
        if state.get("cost_value_net") is not None:
            self.cost_value_net.load_state_dict(state["cost_value_net"])
        self._nu = float(state.get("nu", self._nu))
        self.cost_limit = float(state.get("cost_limit", self.cost_limit))
        self._episode_len_estimate = float(
            state.get("episode_len_estimate", self._episode_len_estimate)
        )

    def __repr__(self) -> str:
        return (
            f"<LagrangianPPOAgent lambda={self.lagrange_multiplier:.4f} "
            f"cost_limit={self.cost_limit:.3f} J_C={self._last_jc:.3f}>"
        )
