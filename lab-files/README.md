# Lab support files

New IPC benchmark experiments must filter
`project.SUITE_NUMERIC_OTHERS_NO_0_COVERAGE` through
`exclude_superseded_others_domains` from `benchmark_suites.py`.

The old `others/petri-net` suite and `ipc2026/petri-net` encode the same 20
instances. The IPC 2026 encoding corrects the domain and replaces the old
suite. Historical experiment scripts retain both suites for reproducibility;
new experiments keep only `petri-net-ipc26`.

With the benchmark selection used by experiments 0053--0057, this changes the
executed suite from 642 tasks in 30 domains to 622 tasks in 29 domains.
