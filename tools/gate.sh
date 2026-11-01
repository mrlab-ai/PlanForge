#!/usr/bin/env bash
# Tiered verification gate.
#
# The tiers answer different questions:
#
#   fast     -- does the crate I touched still work? (tests, clippy, rustfmt)
#   semantic -- did plan costs or search effort move? The `search_snapshot`
#               test runs the main heuristic families on the benchmark corpus
#               and compares cost, expanded and generated states with
#               tests/snapshots/search.txt. After an intended behaviour change,
#               rewrite the snapshot with
#                 PLANFORGE_BLESS=1 cargo test -p tests --release -- --ignored search_snapshot
#               and commit the diff with the change.
#   timing   -- did it get slower? The only tier that compares run times.
#
# Plan costs and expansion counts cannot be changed by an optimisation level,
# so only `timing` looks at the clock. A profile without LTO is not faster to
# build here (52s vs 53s): compiling at opt-level 3 is the cost, not LTO.
#
# WARNING about `timing`: it runs the DEFAULT search spec, `astar(blind())`.
# Blind search touches no heuristic code, so for a change in a heuristic, an
# abstraction, LM-cut or the numeric bounds, pass a spec that reaches it:
#   tools/gate.sh timing "astar(lmcutnumeric())"
#
# Usage:
#   tools/gate.sh fast [crate...]   before every commit
#   tools/gate.sh semantic          before any commit that could change behaviour
#   tools/gate.sh timing [spec]     when you claim something is faster or slower
#                                   (compares with BASELINE, a planforge binary)
#   tools/gate.sh all               before pushing
set -euo pipefail

JOBS=${CARGO_BUILD_JOBS:-6}
export CARGO_BUILD_JOBS=$JOBS
ASSETS=tests/assets/numeric-pddl-files
BASELINE=${BASELINE:-/tmp/planforge-baseline}

die() { echo "GATE FAILED: $*" >&2; exit 1; }

# The crates every gate covers. `cargo --workspace` would also pull in
# planforge-cplex, which needs CPLEX headers, so the list is explicit.
CRATES=(planforge-sas planforge-translate planforge-search planforge-heuristics planforge planforge-config-derive tests)

fast() {
  local crates=("$@")
  [ ${#crates[@]} -eq 0 ] && crates=("${CRATES[@]}")
  local args=()
  for c in "${crates[@]}"; do args+=(-p "$c"); done
  cargo test "${args[@]}"
  local all=()
  for c in "${CRATES[@]}"; do all+=(-p "$c"); done
  cargo clippy "${all[@]}" --all-targets -- -D warnings
  cargo fmt --all --check
}

semantic() {
  cargo test -p tests --release -- --ignored search_snapshot \
    || die "search behaviour differs from tests/snapshots/search.txt"
}

timing() {
  local spec=$1
  [ -x "$BASELINE" ] || die "BASELINE=$BASELINE is not a planforge binary"
  echo "timing with --search '$spec'"
  case "$spec" in
    *blind*) echo "  NOTE: blind search touches no heuristic code -- pass a spec that reaches your change" ;;
  esac
  cargo build --release --bin planforge >/dev/null 2>&1 || die "release build failed"
  cp target/release/planforge /tmp/planforge-timing
  # ABBA so machine drift cancels; A-then-B exaggerated an 8% delta to 13%.
  for inst in minecraft-pogo-advanced satellite rover-unit; do
    local dir="$ASSETS/$inst" problem
    problem=$(find "$dir" -maxdepth 1 -name '*.pddl' ! -name domain.pddl | sort | head -1)
    echo "$inst:"
    for _ in 1 2 3; do
      for bin in "$BASELINE" /tmp/planforge-timing /tmp/planforge-timing "$BASELINE"; do
        printf ' %s' "$("$bin" --search "$spec" "$dir/domain.pddl" "$problem" 2>&1 | grep -oP 'Search time: \K[0-9.]+')"
      done
    done
    echo
  done
  echo "columns are baseline,new,new,baseline per round -- compare medians, not single runs"
}

case "${1:-all}" in
  fast)     shift; fast "$@" ;;
  semantic) semantic ;;
  timing)   timing "${2:-astar(blind())}" ;;
  all)      fast; semantic ;;
  *)        die "unknown tier '${1}'; use fast | semantic | timing | all" ;;
esac
echo "gate '${1:-all}' passed"
