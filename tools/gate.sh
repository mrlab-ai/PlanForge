#!/usr/bin/env bash
# Tiered verification gate.
#
# Everything used to run at every commit: full test suite, a full-LTO release
# build, and the 21-domain benchmark. That is 6-10 minutes per commit, and most
# of it is wasted on changes that provably cannot alter behaviour.
#
# The tiers below exist because the checks answer different questions:
#
#   fast     -- does the crate I touched still work?
#   semantic -- did plan costs or expansion counts move?
#   timing   -- did it get slower?
#
# Measured cost of the full suite: one crate dominates it completely. The
# `tests` crate group is 116.7s of a ~122s run; the next largest group is 5.1s.
# So `scripts/gate.sh fast <crate>` on the crate you touched is seconds, and
# running the whole suite is almost entirely the cost of that one group.
#
# Measured, so that nobody repeats the mistake this script was first written
# on: the `experiment` profile (lto = false, codegen-units = 16) is NOT faster
# to build than full release here -- 52s vs 53s. LTO is not the bottleneck;
# compiling ~100k lines at opt-level 3 is. So there is no cheap optimized
# build, and the savings have to come from not building an optimized binary at
# all, and from running fewer instances when you do.
#
# What that leaves:
#   fast     -- no optimized build. This is the one to run constantly.
#   semantic -- one optimized build, then the 18 instances that take
#               milliseconds; skips the three that dominate a full run.
#   full     -- all 21, for a change that could plausibly affect only a big one.
#   timing   -- the only tier that needs release + repeated interleaved runs.
#
# WARNING about `timing`, found the hard way: it runs the DEFAULT search spec,
# `astar(blind())`. Blind search touches no heuristic code at all, so if your
# change is in a heuristic, an abstraction, lm-cut or the numeric bounds, this
# tier exercises NONE of it and its wobble is pure machine noise. Pass a spec
# that reaches your change:
#   scripts/gate.sh timing "astar(lmcutnumeric())"
#   scripts/gate.sh timing "astar(domain_abstraction())"
# A heuristic change measured under `blind()` is not evidence of anything.
#
# Plan costs and expansion counts are semantics and cannot be changed by an
# optimisation level, so `semantic` and `full` deliberately ignore the timing
# columns; only `timing` looks at the clock.
#
# Usage:
#   scripts/gate.sh fast [crate...]   before every commit
#   scripts/gate.sh semantic          before any commit that could change behaviour
#   scripts/gate.sh timing            when you claim something is faster or slower
#   scripts/gate.sh all               before pushing
set -euo pipefail

JOBS=${CARGO_BUILD_JOBS:-6}
export CARGO_BUILD_JOBS=$JOBS
ASSETS=tests/assets/numeric-pddl-files
BASELINE=${BASELINE:-/tmp/planforge-baseline}

# The 21 benchmark domains split by cost. `quick` is every instance that solves
# in well under a tenth of a second, which is most of them; `slow` is the three
# that dominate a full run.
SLOW='minecraft-pogo-advanced rover-unit satellite'

die() { echo "GATE FAILED: $*" >&2; exit 1; }

fast() {
  local crates=("$@")
  if [ ${#crates[@]} -eq 0 ]; then
    cargo test --workspace --exclude planforge-cplex
  else
    local args=()
    for c in "${crates[@]}"; do args+=(-p "$c"); done
    cargo test "${args[@]}"
  fi
  cargo clippy --workspace --exclude planforge-cplex --all-targets -- -D warnings
  cargo fmt --all --check
}

# There is no cheaper optimized build than release (measured above), so this is
# just the release build. It is fine for a cost/expansion comparison and wrong
# for a stopwatch, which is why `timing` builds its own and repeats runs.
build_semantic() {
  cargo build --release --bin planforge >/dev/null 2>&1 \
    || die "release build failed"
  cp target/release/planforge /tmp/planforge-gate
}

# Compare costs and expansions only. Timing columns are deliberately ignored.
compare() {
  local bin=$1 subset=$2 diffs=0 checked=0
  for dir in "$ASSETS"/*/; do
    local name dom prob
    name=$(basename "$dir")
    [ "$name" = sailing-simple ] && continue
    dom="$dir/domain.pddl"; [ -f "$dom" ] || continue
    if [ "$subset" = quick ] && grep -qw "$name" <<<"$SLOW"; then continue; fi
    prob=$(find "$dir" -maxdepth 1 -name '*.pddl' ! -name domain.pddl | head -1)
    [ -n "$prob" ] || continue
    local a b
    a=$("$BASELINE" "$dom" "$prob" 2>&1 | grep -oP 'Plan cost: \K[0-9.]+|Expanded \K[0-9]+' | paste -sd'|')
    b=$("$bin"      "$dom" "$prob" 2>&1 | grep -oP 'Plan cost: \K[0-9.]+|Expanded \K[0-9]+' | paste -sd'|')
    checked=$((checked + 1))
    if [ "$a" != "$b" ]; then echo "  DIFF $name: $a -> $b"; diffs=$((diffs + 1)); fi
  done
  echo "  checked $checked, differing $diffs"
  [ "$diffs" -eq 0 ] || die "$diffs domain(s) changed cost or expansions"
}

case "${1:-all}" in
  fast)     shift; fast "$@" ;;
  semantic) build_semantic; compare /tmp/planforge-gate quick ;;
  full)     build_semantic; compare /tmp/planforge-gate all ;;
  timing)
    SPEC=${2:-astar(blind())}
    echo "timing with --search '$SPEC'"
    case "$SPEC" in
      *blind*) echo "  NOTE: blind search touches no heuristic code -- pass a spec that reaches your change" ;;
    esac
    cargo build --release --bin planforge >/dev/null 2>&1 || die "release build failed"
    cp target/release/planforge /tmp/planforge-timing
    # ABBA so machine drift cancels; A-then-B exaggerated an 8% delta to 13%.
    for inst in minecraft-pogo-advanced satellite rover-unit; do
      local_dir="$ASSETS/$inst"
      p=$(find "$local_dir" -maxdepth 1 -name '*.pddl' ! -name domain.pddl | head -1)
      echo "$inst:"
      for _ in 1 2 3; do
        for bin in "$BASELINE" /tmp/planforge-timing /tmp/planforge-timing "$BASELINE"; do
          printf ' %s' "$("$bin" --search "$SPEC" "$local_dir/domain.pddl" "$p" 2>&1 | grep -oP 'Search time: \K[0-9.]+')"
        done
      done
      echo
    done
    echo "columns are baseline,new,new,baseline per round -- compare medians, not single runs"
    ;;
  all)      fast; build_semantic; compare /tmp/planforge-gate all ;;
  *)        die "unknown tier '${1}'; use fast | semantic | full | timing | all" ;;
esac
echo "gate '${1:-all}' passed"
