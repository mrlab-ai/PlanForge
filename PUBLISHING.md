# Publishing PlanForge

The crates.io release consists of these packages:

- `planforge-config-derive`
- `planforge-sas`
- `planforge-translate`
- `planforge-cplex`
- `planforge-search`
- `planforge-sgd`
- `planforge-searcher`
- `planforge`

Publish them in this order, waiting until each version is available from the
crates.io index before continuing:

1. `planforge-config-derive`
2. `planforge-sas`
3. `planforge-translate`
4. `planforge-cplex`
5. `planforge-search`
6. `planforge-sgd`
7. `planforge-searcher`
8. `planforge`

The primary pipeline order is `planforge-config-derive` → `planforge-sas` →
`planforge-translate` → `planforge-search` → `planforge-searcher` →
`planforge`. The two additional placements are dependencies too:
`planforge-search` optionally depends on `planforge-cplex`, and
`planforge-searcher` optionally depends on `planforge-sgd`.

`planforge-cplex` is safe to package and document without a local CPLEX
installation because its native API is disabled by default. Enabling its
`cplex` feature checks for the CPLEX headers and static library and fails with
an installation-specific error if they are absent. The `planforge-search`
`cplex` feature enables both the dependency and its native API.

The `planforge-py` package is distributed as a Python wheel and source archive
through PyPI, not crates.io. The `tests` package and every package under
`tutorials/` are workspace-only. All of them set `publish = false`.

Before a release, run the repository gates and package each crates.io package
in the order above. `cargo package` is the dry run; uploading with
`cargo publish` is a separate maintainer action and requires a crates.io token.
