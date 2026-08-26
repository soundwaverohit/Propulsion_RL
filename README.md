# Propulsion_RL

A simulation benchmark for **pairing reinforcement-learning methods with
spacecraft propulsion systems**, and finding which pairing delivers the most
economically efficient transport.

The unit of study is a pairing — `(AI method × propulsion system × mission)`.
The benchmark sweeps the full cross product, measures technical outcomes first
and economics second, and reports a ranked answer with confidence intervals and
a Pareto front.

## The question

Different propulsion technologies pose *qualitatively different control
problems*. Electric propulsion is a months-long, power-limited, wear-limited
optimisation. Nuclear propulsion is a short, hot, hard-safety-constrained one
where waiting costs you propellant to boiloff. So the best controller for one
should not be the best controller for the other — and the size of that
interaction effect is the thing worth measuring.

## What is modelled

**Electric propulsion** — Hall effect thrusters (SPT-100, NASA HERMeS/AEPS) and
gridded ion engines (NSTAR, NEXT-C). Voltage-dependent efficiency, Child–Langmuir
space-charge limits, channel-wall and grid erosion, cathode aging, thermal
state, xenon throughput life.

**Nuclear propulsion** — nuclear thermal (Pewee/NERVA class and a modern HALEU
concept) and nuclear electric (Brayton conversion, Kilopower class). Six-group
point kinetics, Doppler feedback, prompt-criticality margin, Xe-135 poisoning
with its ~9-hour post-shutdown peak, decay heat, thermal-stress-limited ramps,
LH₂ boiloff.

**Missions** — LEO→GEO transfer, Earth–Mars cargo, fast crewed Mars, GEO
station-keeping. Chosen so that no single propulsion family wins them all.

**AI methods** — PPO, SAC, TD3, Lagrangian-constrained PPO, CEM-MPC (with both
an oracle model and a learned PETS-style ensemble), CMA-ES direct policy search,
and a set of properly-tuned scripted baselines including Edelbaum optimal
steering and a life-aware controller.

**Economics** — `$/kg delivered`, levelized cost of transport, NPV over trip
time, with three bracketing price books and a sensitivity analysis that says
which conclusions survive the uncertainty in reactor and xenon prices.

## Install

Core needs only numpy, scipy, torch, pandas and pyyaml — the Gymnasium API is
implemented natively and the RL algorithms are written directly in PyTorch, so
nothing else is required to run the benchmark.

```bash
pip install -e ".[dev,viz]"
```

Optional cross-validation against the standard ecosystem:

```bash
pip install -e ".[sb3]"     # gymnasium + stable-baselines3
```

## Use

```bash
propulsion-rl list                              # registered propulsion, missions, agents
propulsion-rl demo                              # end-to-end smoke run, under a minute
propulsion-rl run --config configs/smoke.yaml
propulsion-rl run --config configs/electric_vs_nuclear.yaml --workers 8
propulsion-rl analyze --sweep electric_vs_nuclear
propulsion-rl plot --sweep electric_vs_nuclear
```

Or from Python:

```python
from propulsion_rl.envs.propulsion_env import make_env
from propulsion_rl.core.registry import AGENT

env = make_env("hall_hermes", "earth_mars_cargo", seed=0)
agent = AGENT.make("sac", obs_dim=36, action_dim=5)

obs, info = env.reset(seed=0)
done = False
while not done:
    action = agent.act(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

print(info["mission_result"], info["economics"])
```

## Testing

```bash
pytest                    # fast suite
pytest -m "not slow"      # skip long-running physics validation
pytest --cov              # coverage report
```

The conformance suites are parametrised over the registries, so adding a new
thruster or mission automatically brings the full invariant battery with it —
mass/energy consistency, power respect, determinism, monotone wear, and purity
of the read-only accessors.

## Documentation

* [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — layering, the canonical
  action/observation contracts, the step loop, and the fair-comparison protocol.
