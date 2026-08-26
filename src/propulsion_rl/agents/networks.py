"""Neural network building blocks shared by the learned agents.

Everything here is deliberately small and explicit. The benchmark compares
algorithms, so the *networks* must not be a confounding variable: PPO, SAC and
TD3 all draw their torsos from the same :class:`MLP` with the same
initialisation conventions, and any architectural difference between them is a
difference the algorithm actually requires.

Two conventions are worth stating up front because getting them wrong is
silent rather than loud:

Orthogonal initialisation gains
    ``sqrt(2)`` on hidden layers (the standard gain for ReLU/tanh torsos),
    ``0.01`` on a policy mean head so the initial policy is near-deterministic
    at the centre of the action box instead of saturating tanh immediately, and
    ``1.0`` on value/Q heads so early value predictions are not artificially
    shrunk toward zero. These are the numbers every reference PPO uses; they
    matter most in the first few thousand steps.

Where the log-std lives
    :class:`GaussianPolicy` (PPO) uses a **state-independent** log-std: a bare
    ``nn.Parameter`` vector. On-policy methods estimate the policy gradient
    from a single batch of on-policy samples, so a state-conditioned std adds
    variance to that estimate for very little benefit, and a global std anneals
    smoothly as a scalar exploration schedule.
    :class:`SquashedGaussianPolicy` (SAC) uses a **state-dependent** log-std
    head, because SAC's objective is entropy-regularised: the policy is
    *supposed* to be near-deterministic where the Q-function is sharp and broad
    where it is flat, and the temperature controller can only trade entropy
    against return if the policy can vary its entropy per state.

Randomness
    Sampling never touches the global torch RNG. Every stochastic module owns a
    :class:`NoiseSource` with its own ``torch.Generator``, so a run stays
    reproducible even when other code in the same process reseeds torch.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch import nn

logger = logging.getLogger(__name__)

__all__ = [
    "DeterministicPolicy",
    "GaussianPolicy",
    "MLP",
    "NoiseSource",
    "QNetwork",
    "RunningMeanStd",
    "SquashedGaussianPolicy",
    "ValueNetwork",
    "count_parameters",
    "gaussian_entropy",
    "gaussian_log_prob",
    "hard_update",
    "orthogonal_init",
    "resolve_activation",
    "resolve_device",
    "soft_update",
    "squashed_gaussian_log_prob",
    "to_numpy",
    "to_tensor",
]

SQRT2 = math.sqrt(2.0)

#: Bounds on the state-dependent log-std of :class:`SquashedGaussianPolicy`.
LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0

#: Guard inside the tanh log-det-Jacobian correction. Doubles as the cap on the
#: correction once ``tanh(u)**2`` rounds to exactly 1.0 in float32 (|u| >~ 9).
TANH_EPS = 1e-6

_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "elu": nn.ELU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "leaky_relu": nn.LeakyReLU,
    "identity": nn.Identity,
}


# --- numpy <-> torch bridge --------------------------------------------------
# torch's fast zero-copy numpy bridge is compiled against a specific numpy ABI.
# When the installed torch predates numpy 2.x, ``torch.as_tensor(ndarray)`` and
# ``Tensor.numpy()`` both raise "Numpy is not available" -- which is exactly the
# situation in this project's default interpreter (torch 2.0.1 + numpy 2.4.6).
# The buffer protocol and DLPack are ABI-independent and still work, so these
# two helpers probe once and then take the fast path when it exists and the
# portable path when it does not. Every numpy/torch conversion in the agents
# goes through them; the correct long-term fix is to align the two versions.
_NP_FOR_TORCH: dict[torch.dtype, Any] = {
    torch.float32: np.float32,
    torch.float64: np.float64,
    torch.int64: np.int64,
    torch.int32: np.int32,
    torch.bool: np.bool_,
}
_BRIDGE_OK: bool | None = None


def _bridge_ok() -> bool:
    global _BRIDGE_OK
    if _BRIDGE_OK is None:
        try:
            torch.as_tensor(np.zeros(1, dtype=np.float32)).numpy()
            _BRIDGE_OK = True
        except Exception:  # noqa: BLE001 - any failure means "use the fallback"
            _BRIDGE_OK = False
            logger.warning(
                "torch's numpy bridge is unavailable (torch %s vs numpy %s); "
                "falling back to the buffer protocol / DLPack. Aligning the two "
                "versions will be faster.",
                torch.__version__,
                np.__version__,
            )
    return _BRIDGE_OK


def to_tensor(
    x: Any,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Convert a numpy array (or array-like) to a torch tensor. Always a copy."""
    np_dtype = _NP_FOR_TORCH.get(dtype, np.float32)
    arr = np.ascontiguousarray(np.asarray(x, dtype=np_dtype))
    if _bridge_ok():
        return torch.as_tensor(arr, dtype=dtype).to(device)
    if arr.size == 0:
        # torch.frombuffer rejects a zero-length buffer.
        return torch.empty(arr.shape, dtype=dtype, device=device)
    # ``bytearray`` gives torch a writable buffer it exclusively owns, so the
    # tensor never aliases numpy memory that may be freed or mutated later.
    flat = torch.frombuffer(bytearray(arr.tobytes()), dtype=dtype)
    return flat.reshape(arr.shape).to(device)


def to_numpy(t: torch.Tensor, dtype: Any = np.float32) -> np.ndarray:
    """Convert a torch tensor to a fresh numpy array. Always a copy."""
    t = t.detach().cpu().contiguous()
    if _bridge_ok():
        return np.array(t.numpy(), dtype=dtype)
    try:
        return np.array(np.from_dlpack(t), dtype=dtype)
    except Exception:  # noqa: BLE001 - last resort, correct if slow
        return np.array(t.tolist(), dtype=dtype)


# --- small helpers -----------------------------------------------------------
def resolve_activation(activation: str | type[nn.Module]) -> type[nn.Module]:
    """Accept either an activation name or an ``nn.Module`` subclass."""
    if isinstance(activation, str):
        key = activation.lower()
        if key not in _ACTIVATIONS:
            raise ValueError(
                f"unknown activation '{activation}'. Known: {sorted(_ACTIVATIONS)}"
            )
        return _ACTIVATIONS[key]
    if isinstance(activation, type) and issubclass(activation, nn.Module):
        return activation
    raise TypeError(f"activation must be a name or nn.Module subclass, got {activation!r}")


def resolve_device(device: str | torch.device) -> torch.device:
    """Resolve a device string, falling back to CPU with a warning.

    The default everywhere in this package is ``"cpu"`` and that is a
    performance decision, not a limitation: the largest network here is roughly
    a 36 -> 256 -> 256 -> 5 MLP with batches of 64-256, which is far too small
    to amortise kernel-launch and host/device transfer overhead. CPU is
    measurably faster than MPS or CUDA at this size, and it keeps runs bitwise
    reproducible. ``"cuda"`` / ``"mps"`` are honoured when explicitly asked for.
    """
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available; falling back to CPU")
        return torch.device("cpu")
    if dev.type == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        logger.warning("MPS requested but not available; falling back to CPU")
        return torch.device("cpu")
    return dev


def orthogonal_init(
    module: nn.Module, gain: float = SQRT2, bias_const: float = 0.0
) -> nn.Module:
    """Orthogonally initialise a ``nn.Linear`` in place and return it.

    Safe to hand to :meth:`torch.nn.Module.apply`; non-linear layers are left
    untouched.
    """
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        nn.init.constant_(module.bias, bias_const)
    return module


def soft_update(source: nn.Module, target: nn.Module, tau: float) -> None:
    """Polyak update ``target <- tau * source + (1 - tau) * target``."""
    with torch.no_grad():
        for p, tp in zip(source.parameters(), target.parameters()):
            tp.mul_(1.0 - tau).add_(p.detach(), alpha=tau)


def hard_update(source: nn.Module, target: nn.Module) -> None:
    """Copy ``source`` parameters and buffers into ``target``."""
    target.load_state_dict(source.state_dict())


class NoiseSource:
    """Reproducible standard-normal noise from a private ``torch.Generator``.

    ``torch.distributions`` samples from the *global* torch RNG, which makes a
    run's reproducibility hostage to any other code in the process that calls
    ``torch.manual_seed``. Since this benchmark's whole value rests on seeded,
    comparable runs, every stochastic policy draws its noise here instead.
    """

    def __init__(self, seed: int = 0) -> None:
        self.seed = int(seed)
        self._gen: torch.Generator | None = None
        self._gen_device: torch.device | None = None
        self._for_device: torch.device | None = None

    def seed_(self, seed: int) -> None:
        """Reseed; the generator is rebuilt lazily on the next draw."""
        self.seed = int(seed)
        self._gen = None
        self._gen_device = None
        self._for_device = None

    def _ensure(self, device: torch.device) -> None:
        if self._gen is not None and self._for_device == device:
            return
        try:
            gen = torch.Generator(device=device)
            gen_device = device
        except (RuntimeError, TypeError):
            # e.g. MPS on older torch builds: generate on CPU and copy across.
            gen = torch.Generator()
            gen_device = torch.device("cpu")
        gen.manual_seed(self.seed)
        self._gen = gen
        self._gen_device = gen_device
        self._for_device = device

    def randn(
        self,
        shape: Sequence[int],
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        self._ensure(device)
        x = torch.randn(
            tuple(shape), generator=self._gen, device=self._gen_device, dtype=dtype
        )
        return x if self._gen_device == device else x.to(device)

    def get_state(self) -> dict[str, Any]:
        state = None if self._gen is None else self._gen.get_state()
        return {"seed": self.seed, "state": state}

    def set_state(self, d: dict[str, Any]) -> None:
        self.seed = int(d.get("seed", 0))
        self._gen = None
        self._gen_device = None
        self._for_device = None
        saved = d.get("state")
        if saved is not None:
            self._ensure(torch.device("cpu"))
            assert self._gen is not None
            try:
                self._gen.set_state(saved)
            except RuntimeError:  # state came from a different device
                logger.warning("could not restore generator state; reseeding instead")
                self._gen.manual_seed(self.seed)


# --- core modules ------------------------------------------------------------
class MLP(nn.Module):
    """Plain feed-forward torso with orthogonal initialisation.

    Parameters
    ----------
    in_dim, out_dim:
        Input and output widths.
    hidden:
        Hidden layer widths, e.g. ``(256, 256)``. May be empty for a linear map.
    activation:
        Name (``"relu"``, ``"tanh"``, ...) or ``nn.Module`` subclass.
    output_activation:
        Applied after the final linear layer; ``None`` means a linear head.
    hidden_gain, output_gain, output_bias:
        Orthogonal-init gains. See the module docstring for the conventions --
        ``0.01`` for policy mean heads, ``1.0`` for value/Q heads.
    layer_norm:
        Insert ``LayerNorm`` after each hidden activation. Off by default; the
        reference results in this benchmark are run without it.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden: Sequence[int] = (256, 256),
        activation: str | type[nn.Module] = "relu",
        output_activation: str | type[nn.Module] | None = None,
        hidden_gain: float = SQRT2,
        output_gain: float = 1.0,
        output_bias: float = 0.0,
        layer_norm: bool = False,
    ) -> None:
        super().__init__()
        act_cls = resolve_activation(activation)
        layers: list[nn.Module] = []
        prev = int(in_dim)
        for width in hidden:
            linear = nn.Linear(prev, int(width))
            orthogonal_init(linear, gain=hidden_gain)
            layers.append(linear)
            if layer_norm:
                layers.append(nn.LayerNorm(int(width)))
            layers.append(act_cls())
            prev = int(width)
        head = nn.Linear(prev, int(out_dim))
        orthogonal_init(head, gain=output_gain, bias_const=output_bias)
        layers.append(head)
        if output_activation is not None:
            layers.append(resolve_activation(output_activation)())
        self.net = nn.Sequential(*layers)
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GaussianPolicy(nn.Module):
    """Diagonal Gaussian policy with a **state-independent** log-std (PPO).

    The mean head is a plain MLP; the log-std is a free parameter vector shared
    across states. Actions are *not* squashed -- PPO's importance ratio needs
    the density of the raw Gaussian sample, and the environment clips to the
    action box anyway. :meth:`sample` therefore returns the unclipped sample;
    the caller clips before stepping the environment and keeps the raw value
    for the log-prob.

    ``log_std`` is clamped to a wide band on use. The clamp is a blow-up guard,
    not a schedule: PPO essentially never reaches it, but a diverged update that
    drives std to 0 would otherwise produce infinite log-probs and poison the
    whole run.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden: Sequence[int] = (256, 256),
        activation: str | type[nn.Module] = "tanh",
        log_std_init: float = 0.0,
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.mu = MLP(
            obs_dim,
            action_dim,
            hidden=hidden,
            activation=activation,
            hidden_gain=SQRT2,
            output_gain=0.01,
        )
        self.log_std = nn.Parameter(torch.full((int(action_dim),), float(log_std_init)))
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        self.noise = NoiseSource(seed)

    def set_seed(self, seed: int) -> None:
        self.noise.seed_(seed)

    def _mu_std(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu = self.mu(obs)
        log_std = self.log_std.clamp(self.log_std_min, self.log_std_max)
        return mu, log_std.exp().expand_as(mu)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self._mu_std(obs)

    def sample(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(action, log_prob)``; the mean when ``deterministic``."""
        mu, std = self._mu_std(obs)
        if deterministic:
            return mu, gaussian_log_prob(mu, mu, std)
        eps = self.noise.randn(mu.shape, mu.device, mu.dtype)
        action = mu + std * eps
        return action, gaussian_log_prob(action, mu, std)

    def evaluate(
        self, obs: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(log_prob, entropy)`` of ``action`` under the current policy."""
        mu, std = self._mu_std(obs)
        log_prob = gaussian_log_prob(action, mu, std)
        entropy = gaussian_entropy(std)
        return log_prob, entropy


class SquashedGaussianPolicy(nn.Module):
    """tanh-squashed diagonal Gaussian with a **state-dependent** log-std (SAC).

    Sample ``u ~ N(mu(s), std(s))``, emit ``a = tanh(u)`` so actions live in
    ``(-1, 1)^d`` by construction, and correct the density for the squashing:

    .. math::
        \\log \\pi(a|s) = \\log N(u; \\mu, \\sigma)
                          - \\sum_i \\log\\left(1 - \\tanh(u_i)^2 + \\epsilon\\right)

    The subtracted term is the log-determinant of the tanh Jacobian. Omitting it
    -- or getting its sign wrong -- does **not** crash anything: the policy still
    trains, it just optimises the wrong entropy. Automatic temperature tuning
    then chases a target entropy measured in the wrong units and quietly settles
    at the wrong exploration level, which is exactly the kind of bug that would
    invalidate a benchmark rather than break it. ``tests`` check this against a
    numerical Jacobian.

    ``epsilon = 1e-6`` matches the reference SAC implementations and, usefully,
    caps the correction at ``log(1e-6)`` per dimension once ``tanh(u)**2``
    rounds to exactly 1 in float32 (around ``|u| > 9``), where the exact
    expression would return ``-inf``.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden: Sequence[int] = (256, 256),
        activation: str | type[nn.Module] = "relu",
        log_std_min: float = LOG_STD_MIN,
        log_std_max: float = LOG_STD_MAX,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        hidden = tuple(int(h) for h in hidden)
        if not hidden:
            raise ValueError("SquashedGaussianPolicy needs at least one hidden layer")
        act_cls = resolve_activation(activation)
        torso: list[nn.Module] = []
        prev = self.obs_dim
        for width in hidden:
            linear = nn.Linear(prev, width)
            orthogonal_init(linear, gain=SQRT2)
            torso.extend((linear, act_cls()))
            prev = width
        self.torso = nn.Sequential(*torso)
        self.mu_head = orthogonal_init(nn.Linear(prev, self.action_dim), gain=0.01)
        self.log_std_head = orthogonal_init(nn.Linear(prev, self.action_dim), gain=0.01)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        self.noise = NoiseSource(seed)

    def set_seed(self, seed: int) -> None:
        self.noise.seed_(seed)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(mu, log_std)`` before squashing."""
        h = self.torso(obs)
        mu = self.mu_head(h)
        raw = self.log_std_head(h)
        # Smooth squashing into the band rather than torch.clamp: a hard clamp
        # zeroes the gradient at the boundary and can pin log_std there forever.
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (
            torch.tanh(raw) + 1.0
        )
        return mu, log_std

    def sample(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
        with_logprob: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """Reparameterised sample.

        Returns ``(action, log_prob, mean_action)`` where ``action`` and
        ``mean_action`` are both already squashed into ``(-1, 1)``. The sample
        is reparameterised (``mu + std * eps``), so gradients flow through it --
        that is what makes SAC's actor loss a pathwise derivative rather than a
        score-function estimator.
        """
        mu, log_std = self(obs)
        std = log_std.exp()
        mean_action = torch.tanh(mu)
        if deterministic:
            return mean_action, None, mean_action
        eps = self.noise.randn(mu.shape, mu.device, mu.dtype)
        u = mu + std * eps
        action = torch.tanh(u)
        log_prob = squashed_gaussian_log_prob(u, mu, std) if with_logprob else None
        return action, log_prob, mean_action

    def log_prob(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Log-density of an already-squashed ``action`` in ``(-1, 1)``."""
        mu, log_std = self(obs)
        a = action.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        u = torch.atanh(a)
        return squashed_gaussian_log_prob(u, mu, log_std.exp())


class DeterministicPolicy(nn.Module):
    """tanh-bounded deterministic actor for TD3.

    No distribution at all: TD3's actor is a point map and all exploration is
    injected externally as additive Gaussian noise, which is why the agent (not
    this module) owns the exploration schedule.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden: Sequence[int] = (256, 256),
        activation: str | type[nn.Module] = "relu",
    ) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.net = MLP(
            obs_dim,
            action_dim,
            hidden=hidden,
            activation=activation,
            output_activation="tanh",
            hidden_gain=SQRT2,
            output_gain=0.01,
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class QNetwork(nn.Module):
    """State-action value ``Q(s, a)``, returned with the trailing dim squeezed."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden: Sequence[int] = (256, 256),
        activation: str | type[nn.Module] = "relu",
    ) -> None:
        super().__init__()
        self.net = MLP(
            int(obs_dim) + int(action_dim),
            1,
            hidden=hidden,
            activation=activation,
            hidden_gain=SQRT2,
            output_gain=1.0,
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, action], dim=-1)).squeeze(-1)


class ValueNetwork(nn.Module):
    """State value ``V(s)``, returned with the trailing dim squeezed."""

    def __init__(
        self,
        obs_dim: int,
        hidden: Sequence[int] = (256, 256),
        activation: str | type[nn.Module] = "tanh",
    ) -> None:
        super().__init__()
        self.net = MLP(
            int(obs_dim),
            1,
            hidden=hidden,
            activation=activation,
            hidden_gain=SQRT2,
            output_gain=1.0,
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


# --- densities ---------------------------------------------------------------
_LOG_2PI = math.log(2.0 * math.pi)


def gaussian_log_prob(
    x: torch.Tensor, mu: torch.Tensor, std: torch.Tensor
) -> torch.Tensor:
    """Diagonal-Gaussian log-density, summed over the action dimension."""
    log_std = std.log()
    per_dim = -0.5 * ((x - mu) / std) ** 2 - log_std - 0.5 * _LOG_2PI
    return per_dim.sum(-1)


def gaussian_entropy(std: torch.Tensor) -> torch.Tensor:
    """Differential entropy of a diagonal Gaussian, summed over dimensions."""
    return (std.log() + 0.5 * (_LOG_2PI + 1.0)).sum(-1)


def squashed_gaussian_log_prob(
    u: torch.Tensor, mu: torch.Tensor, std: torch.Tensor
) -> torch.Tensor:
    """Log-density of ``a = tanh(u)`` where ``u ~ N(mu, std)``.

    The change-of-variables term is ``-sum_i log|d a_i / d u_i|`` with
    ``d tanh(u)/du = 1 - tanh(u)^2``. See :class:`SquashedGaussianPolicy`.
    """
    log_prob = gaussian_log_prob(u, mu, std)
    correction = torch.log(1.0 - torch.tanh(u) ** 2 + TANH_EPS).sum(-1)
    return log_prob - correction


# --- observation normalisation ----------------------------------------------
class RunningMeanStd:
    """Streaming mean/variance via Chan et al.'s parallel (Welford) update.

    Observations in this project span many orders of magnitude -- metres from
    the Sun next to a wear fraction in [0, 1] -- so unnormalised inputs make the
    first layer's conditioning terrible and the comparison between algorithms a
    comparison of how well each tolerates bad scaling. Normalising is a
    correctness requirement here, not a tweak.

    The parallel form updates from a whole batch in one shot with the same
    numerics as the sequential Welford recurrence, so it is both cheap in the
    rollout loop and stable over the millions of steps a mission run takes.
    """

    def __init__(self, shape: int | tuple[int, ...], epsilon: float = 1e-4) -> None:
        self.shape = (shape,) if isinstance(shape, int) else tuple(shape)
        self.mean = np.zeros(self.shape, dtype=np.float64)
        self.var = np.ones(self.shape, dtype=np.float64)
        self.count = float(epsilon)

    def update(self, x: np.ndarray) -> None:
        """Update from a batch ``(n, *shape)`` or a single sample ``(*shape,)``."""
        arr = np.asarray(x, dtype=np.float64)
        if arr.shape == self.shape:
            arr = arr.reshape((1, *self.shape))
        if arr.ndim != len(self.shape) + 1:
            raise ValueError(f"expected batch of {self.shape}, got {arr.shape}")
        if arr.shape[0] == 0:
            return
        self.update_from_moments(
            arr.mean(axis=0), arr.var(axis=0), float(arr.shape[0])
        )

    def update_from_moments(
        self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: float
    ) -> None:
        delta = batch_mean - self.mean
        total = self.count + batch_count
        new_mean = self.mean + delta * (batch_count / total)
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * (self.count * batch_count / total)
        self.mean = new_mean
        self.var = m2 / total
        self.count = total

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(self.var)

    def normalize(
        self, x: np.ndarray, clip: float | None = 10.0, epsilon: float = 1e-8
    ) -> np.ndarray:
        """Whiten ``x`` with the current statistics (float32 out)."""
        out = (np.asarray(x, dtype=np.float64) - self.mean) / np.sqrt(
            self.var + epsilon
        )
        if clip is not None:
            out = np.clip(out, -clip, clip)
        return out.astype(np.float32)

    def denormalize(self, x: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
        return (
            np.asarray(x, dtype=np.float64) * np.sqrt(self.var + epsilon) + self.mean
        ).astype(np.float32)

    # --- persistence ---------------------------------------------------------
    def state_dict(self) -> dict[str, Any]:
        return {
            "shape": self.shape,
            "mean": self.mean.copy(),
            "var": self.var.copy(),
            "count": self.count,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.shape = tuple(state["shape"])
        self.mean = np.asarray(state["mean"], dtype=np.float64).reshape(self.shape)
        self.var = np.asarray(state["var"], dtype=np.float64).reshape(self.shape)
        self.count = float(state["count"])

    def __repr__(self) -> str:
        return f"RunningMeanStd(shape={self.shape}, count={self.count:.1f})"


def count_parameters(modules: Iterable[nn.Module | nn.Parameter]) -> int:
    """Learnable (``requires_grad``) parameter count over modules/parameters.

    Target networks are expected to have been frozen with
    ``requires_grad_(False)``, which is both correct (they are never optimised)
    and what keeps this count honest in the comparison table.
    """
    total = 0
    seen: set[int] = set()
    for m in modules:
        params = [m] if isinstance(m, nn.Parameter) else list(m.parameters())
        for p in params:
            if p.requires_grad and id(p) not in seen:
                seen.add(id(p))
                total += p.numel()
    return total
