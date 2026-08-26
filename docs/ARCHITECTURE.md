# Propulsion_RL — Architecture

## What this is

A simulation benchmark that answers one question:

> **Which AI/RL control method best complements which propulsion technology, and
> which pairing delivers the most economically efficient transport?**

The unit of study is a **pairing**: `(AI method × propulsion system × mission)`.
The benchmark sweeps the full cross product, measures technical outcomes first
and economic outcomes second, and reports a ranked, statistically defensible
answer with a Pareto front.

The interesting hypothesis is not "RL beats a PID controller." It is that
**there is an interaction term** — that the best control method is *different*
for electric than for nuclear propulsion, because the two have qualitatively
different control problems:

| | Electric propulsion | Nuclear propulsion |
|---|---|---|
| Horizon | Months–years of continuous low thrust | Minutes of high thrust, separated by long coasts |
| Dominant trade | Isp vs. thruster erosion life | Isp vs. fuel damage and safety margin |
| Binding limit | Available power (falls as 1/r²) | Fuel temperature, prompt-criticality margin |
| Failure mode | Wear-out, gradual | Excursion, abrupt |
| Time cost | Ops cost, payload delay | LH₂ boiloff — propellant literally evaporates |
| Expected fit | Long-horizon credit assignment: model-based, SAC | Hard safety constraints: constrained/Lagrangian RL |

If that table is right, the benchmark should reproduce it. If it is wrong, the
benchmark should say so — which is the point of building it rather than
asserting it.

## Layering

Strict one-way dependencies. Nothing below reaches upward.

```
                    experiments/          sweep, fair-comparison runner, stats, CLI
                         │
                    ┌────┴────┐
                 agents/    envs/         policies          Gym-compatible Env
                    │         │
                    │    ┌────┴─────┬──────────┐
                    │  missions/  spacecraft/  economics/
                    │     │          │            │
                    └─────┴──────────┴────────────┘
                                 │
                          propulsion/               physics of the hardware
                                 │
                              core/                 types, constants, registry
```

* **`core/`** — the contracts. `types.py` defines every structure that crosses a
  module boundary; `constants.py` is the single source of truth for physics
  constants; `registry.py` maps names to factories so a sweep is configured by
  strings in YAML, not by imports.
* **`propulsion/`** — stateful hardware models. Own everything from the power
  terminal to the exhaust plane: efficiency, thermal state, wear, failure. Own
  nothing outside it — no orbital mechanics, no mission logic, no dollars.
* **`spacecraft/`** — trajectory propagation, orbital elements, the power bus.
* **`missions/`** — the task: gravity field, initial state, progress, reward,
  termination.
* **`envs/`** — assembles the above into a Gymnasium-API environment.
* **`economics/`** — a pure post-processor over finished missions.
* **`agents/`** — scripted controllers and RL algorithms, interchangeable.
* **`experiments/`** — the sweep and the statistics that turn runs into an answer.

## The two interface decisions that make the comparison possible

### 1. A canonical action space, shared by every propulsion system

All systems accept the same 5-dimensional action in `[-1, 1]`:

| # | Component | Meaning | Electric | Nuclear |
|---|---|---|---|---|
| 0 | `throttle` | Fraction of the *allowable* envelope | anode mass flow | propellant flow / pump speed |
| 1 | `operating_point` | The Isp↔thrust trade knob | discharge voltage | chamber temperature |
| 2 | `thrust_yaw` | In-plane steering, from the velocity vector | — | — |
| 3 | `thrust_pitch` | Out-of-plane steering | — | — |
| 4 | `thermal_margin` | Protect hardware vs. extract performance | cathode/cooling flow | coolant flow, turbine bypass |

Each system implements `decode_action()` to map these onto native actuators.
A zero action is a 50 %-throttle prograde burn — a sane default for an
untrained policy, so early training is not dominated by the policy accidentally
pointing backwards.

### 2. A fixed-width observation, zero-padded

`36 = 12 (mission) + 8 (vehicle) + 16 (propulsion)`, all normalised to ≈`[-1, 1]`.

Fixed width means a policy trained on a Hall thruster can be *evaluated
zero-shot* on a nuclear thermal stage. Transfer between propulsion systems
becomes measurable rather than an architectural impossibility.

## The step loop

Ordering matters; bugs here silently poison every downstream number.

1. Decode `action → CanonicalCommand`
2. Build `StepContext` — mass, available bus power (minus propulsion
   housekeeping), heliocentric radius, eclipse, RNG. **Skipped for
   `self_powered` systems**, which carry a reactor.
3. `propulsion.step(command, ctx)` → thrust, ṁ, Isp, power, heat, events
4. `power_bus.step(...)` with the actual draw
5. `dynamics.propagate(...)` — RK4 with substeps, **mass depleting within the
   step**, thrust direction fixed in the *rotating* RTN frame
6. Read `constraints()` and `health()`
7. `mission.reward(...)` → decomposed `RewardTerms`
8. `mission.terminated(...)`, plus the env's own step-limit truncation
9. Assemble observation, `Telemetry`, and `info`

`info["constraint_cost"]` is always present, so a constrained-RL agent can read
the safety signal while knowing nothing else about the environment's internals.

## Fair-comparison protocol

Most "which RL algorithm is best" studies are wrong for a small number of
recurring reasons. The runner is built to avoid them:

* **Equal environment-step budget**, not equal wall-clock and not equal
  gradient updates. Wall-clock and planning compute are recorded separately and
  reported — an agent that wins only by spending 100× compute per step has not
  won in a way that matters.
* **Multiple seeds, paired across cells**, with bootstrap confidence intervals
  on every headline number. Never a single seed.
* **Held-out evaluation episodes**, disjoint from training seeds, run with
  `deterministic=True` and with observation-normalisation statistics frozen.
* **Scripted baselines get the same evaluation and no training budget.** An RL
  method that cannot beat a well-tuned Edelbaum steering law has not earned its
  complexity, and the benchmark must be able to say so.
* **Multiple-comparison correction** (Holm–Bonferroni) across the matrix, and
  an explicit "not significant" verdict when that is the truth.

## Economics: deliberately the smaller half

Economics never touches the physics and, by default, never enters the reward.
It consumes a finished `MissionResult` plus the propulsion system's
`BillOfMaterials` — propulsion reports *what it is*, never dollars — so cost
assumptions can be swept independently of the physics.

The headline metric is **`$/kg delivered`**, which folds performance, trip time
and hardware cost into one number comparable across a 30 kW solar-electric tug
and a 500 MW nuclear stage. Three cost models (`reference`, `reusable`,
`conservative`) bracket the answer rather than pretending one price book is
true, and a tornado analysis states plainly which conclusions are robust and
which are artefacts of an assumed xenon or reactor price.

The `reusable` model is where propulsion life becomes economically
load-bearing: an agent that preserves thruster life by running at lower
discharge voltage can deliver a cheaper `$/kg` even while flying slower. That
is a real trade, and it is exactly the kind of thing an RL policy can find and a
fixed operating point cannot.

## Dependencies

Core requires only **numpy, scipy, torch, pandas, pyyaml**. The Gymnasium API
surface is implemented natively and the RL algorithms are written directly in
PyTorch, so the benchmark runs with nothing else installed. `gymnasium` and
`stable-baselines3` are optional extras used only for cross-validation.
