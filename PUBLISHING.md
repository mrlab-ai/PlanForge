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

## Pre-release package verification

On 2026-08-18, the three packages without unpublished PlanForge dependencies
passed both archive creation and packaged-source compilation:

| Package | `cargo package --no-verify` | `cargo package` |
| --- | --- | --- |
| `planforge-config-derive` | pass | pass |
| `planforge-sas` | pass | pass |
| `planforge-cplex` | pass | pass |

The other five commands stop during registry resolution because this is the
first release and their required PlanForge `0.1.0` packages are not in the
crates.io index yet:

| Package | First missing registry dependency |
| --- | --- |
| `planforge-translate` | `planforge-sas` |
| `planforge-search` | `planforge-config-derive` |
| `planforge-sgd` | `planforge-sas` |
| `planforge-searcher` | `planforge-sas` |
| `planforge` | `planforge-sas` |

This is why publishing and index propagation must follow the order above. As a
separate packaged-source check, all five archives compiled successfully when
the preceding crates.io dependencies were patched to the same `0.1.0`
workspace sources. Those patches model an indexed dependency; they do not
replace the plain `cargo package` release check after each preceding upload.

Exact-name searches of the public crates.io registry on 2026-08-18 found no
existing entries for any of the eight names above. This is a point-in-time
check, not a reservation; check again immediately before the first upload.
