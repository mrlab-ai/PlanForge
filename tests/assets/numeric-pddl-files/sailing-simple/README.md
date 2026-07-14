# sailing-simple

A hand-computable simplification of the sailing domain used to reason about and
test **numeric domain abstractions** and **cost partitioning**. It is a
benchmark, not a target domain — the mechanisms under test must stay
domain-agnostic.

## Domain (`domain.pddl`)

- State: boat coordinates `x(b)`, `y(b)`; per-person target `tx(p)`, `ty(p)`
  (set in `:init`, never modified → constants, like `d(t)` in real sailing).
- Moves (all unit cost): axis moves change one coordinate by `±1`
  (`go_east/west/north/south`); diagonal moves change both by `±0.5`
  (`go_north_east`, …).
- `save_person(b, t)`: precondition `x(b) = tx(t) ∧ y(b) = ty(t)` (**exact
  position**), effect `saved(t)`. **Unit cost like every move** — all h\*
  below count moves *plus* saves.
- Goal: all persons `saved`.

## Instances and hand-derived optima

| File | Boats | Persons (target) | h\* | What it is for |
|---|---|---|---|---|
| `prob_1b1p_x.pddl` | b0(0,0) | p0(10,0) | 11 (10+1) | 1-D workhorse: cost-stealing `alpha1` vs `alpha2`; regression build of `alpha2` |
| `prob_1b1p_diag.pddl` | b0(0,0) | p0(5,5) | 11 (10+1) | operator changes both x and y (0.5 diagonals); "one interval is infinite" footprint case; `u=x+y` root |
| `prob_2b1p.pddl` | b0(0,0), b1(100,0) | p0(10,0) | 11 (10+1) | multi-boat teleport / untracked-achiever admissibility limit |
| `prob_2b2p_x.pddl` | b0(0,0), b1(30,0) | p0(10,0), p1(40,0) | 22 ((10+1)×2) | additive multi-boat fixture: disjoint near routes should sum under CP |
| `prob_2b2p_assign.pddl` | b0(0,0), b1(100,0) | p0(5,0), p1(−5,0) | 17 (5+1+10+1) | assignment gap fixture: per-person minima underestimate one boat doing both trips |
| `prob_1b2p_x.pddl` | b0(0,0) | p0(10,0), p1(15,0) | 17 (15+2) | nested/prefix targets, **overlapping** abstractions; region CP recovers 16–17 depending on order (naive sum = 27) |
| `prob_1b2p_diag.pddl` | b0(0,0) | p0(5,5), p1(10,10) | 22 (20+2) | additive CP across persons via residual segments (chain route) |
| `prob_1b4p_axes.pddl` | b0(0,0) | pe(10,0), pw(−10,0), pn(0,10), ps(0,−10) | 74 (70+4) | the "8 abstractions" target; strict-dominance-over-LMc demonstrator (LMc-style counting ≈ 44, chain-abstraction CP ≥ 64) |
| `prob_1b1p_far.pddl` | b0(0,0) | p0(100,0) | 101 (100+1) | sailing-ipc scale proxy: deep 1-D chain; >64-overlap / hull-fallback / build-budget stress |

**h\* values are hand-derived and must be machine-verified with `astar(blind())`
(or an exact search) before any heuristic test is pinned against them.** The
state spaces are finite enough under duplicate detection (0.5-lattice within
the cost radius) that blind A\* is feasible on all instances.

## Intended abstractions (ground truth for tests)

- **`prob_1b1p_x`.** `alpha1` = fine partitions on `x∈[0,5]`, coarse tail
  `[5,∞)` (built by progression). `alpha2` = fine on `[5,10]`, coarse head
  `(−∞,5]` (built by regression/target-centered). Alone: `alpha1` = 6
  (5 moves + optimistic save in the `[5,∞)` tail where `x=10` is *unknown*),
  `alpha2` = 6 (1 teleport-cross from the head + 4 fine moves + save).
  - Label/operator CP: `alpha1` saturates the single `go_east` scalar and the
    save → `alpha2` residual 0 → **h = 6** (the cost-stealing bug).
  - Region/transition CP: `alpha1` charges `go_east` only on source
    `x∈[0,5]` pieces; `alpha2` keeps residual on `[5,10]` (the boundary
    crossing piece `[4,5]` is charged once) → **h = 10** in either order.
    h\* = 11.
- **`prob_1b2p_x`.** `alpha1` fine `[0,10]` (p0, alone h=11), `alpha2` fine
  `[0,15]` (p1, alone h=16), overlapping on `[0,10]`. Region CP →
  **16** (`alpha1` first: 11, then `alpha2` keeps 5 moves on `[10,15]` and
  its own save_p1 label = 5) or **17 = h\*** (`alpha2` first: 16, then
  `alpha1` keeps only its distinct save_p0 label = 1).
  Order-sensitivity fixture; naive independent sum = 27 (inadmissible).
- **`prob_1b1p_diag` / `prob_1b2p_diag`.** Useful root is `u = x + y`
  (`transform_linear_task`); each diagonal move is `u += 1`. A single 1-D
  abstraction on `u` counts the moves exactly. Also the cross-dimension
  sharing fixture: a diagonal op charged in an x-abstraction must reduce its
  residual for a y-abstraction (one concrete NE move cannot pay both
  dimensions) — `A_x + A_y` under CP ≈ 11, not 20.
- **`prob_1b4p_axes`.** Chain abstractions `x × saved(pe) × saved(pw)` and
  `y × saved(pn) × saved(ps)` (≈ 84 abstract states each) count the round
  trips exactly (32 each); region CP sums them → ≥ 64. LMc-style counting
  cannot see return legs → ≈ 44. h\* = 74 (the gap to 74 is axis-interleaving,
  invisible to both).
- **`prob_2b1p`.** An abstraction with `saved(p0)` must keep the x-root of
  *every* boat that can save (b0 **and** b1) or exclude `saved(p0)`; otherwise
  b1 teleports and h collapses to ~1. Correct abstraction → **11**.
- **`prob_2b2p_x`.** Each goal has a different nearest boat and each boat's
  move labels are distinct ground operators. Complementary per-goal
  abstractions can therefore cost-partition the two near routes additively:
  10 moves + save for p0, and 10 moves + save for p1 → **22**.
- **`prob_2b2p_assign`.** Both persons' nearest achiever is b0. Per-person
  abstractions see two independent minima of 5 moves + save and sum to about
  **12**, but the real plan must route one boat from one person to the other,
  adding the 10-move transfer: h\* = **17**.

## Verifying

```bash
target/release/planforge --max-time 60s \
  tests/assets/numeric-pddl-files/sailing-simple/domain.pddl \
  tests/assets/numeric-pddl-files/sailing-simple/prob_1b1p_x.pddl \
  --search 'astar(blind())'
```
