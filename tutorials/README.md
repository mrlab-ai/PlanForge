# Tutorials

- [`python/prototyping_search.py`](python/prototyping_search.py) builds `h_max`
  and an eager A* loop in Python; start here when experimenting with a new
  search or heuristic on small tasks.
- [`python/state_space.py`](python/state_space.py) asks Rust to enumerate a
  complete reachable graph, reads its CSR transitions and exact `h_star`, and
  tabulates goal-count error; use it for heuristic analysis or supervised
  targets.
- [`python/goal_count.ipynb`](python/goal_count.ipynb) is an interactive first
  heuristic that compares a Python callback with Rust-backed search.
- [`rust-goal-count/`](rust-goal-count/) implements a native `Heuristic`; it is
  for extensions evaluated at every search node.
- [`rust-custom-search/`](rust-custom-search/) registers a new priority policy
  outside the search crate while keeping the expansion loop monomorphized.

Use Rust for anything on the per-node hot path. Use Python for prototyping when
the FFI cost per call is acceptable; bulk state-space enumeration still runs
once in Rust and returns arrays to Python.

## Install the Python module

After a PyPI release, install it with `pip install planforge`. From a checkout,
build and install the same abi3 wheel without a global maturin installation:

```console
uv venv .venv --python 3.13
uvx maturin@1.7 build --release --locked \
  --manifest-path planforge-py/Cargo.toml --out /tmp/planforge-wheels
uv pip install --python .venv/bin/python /tmp/planforge-wheels/planforge-*.whl
```

Run the scripts from the repository root with `.venv/bin/python`, or open the
notebook with that environment as its kernel. Build the Rust tutorials
explicitly because they are not default workspace members:

```console
cargo build -p tutorial-goal-count -p tutorial-custom-search
```
