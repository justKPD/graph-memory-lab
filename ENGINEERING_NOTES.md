# Engineering decisions and negative results

24 October 2026. Preserve these boundaries in interviews and outreach.

- The standard save_on_cpu hook failed on a saved sparse CSR tensor in the library
  path. The lab offloads dense floating saved activations and keeps sparse/integer
  structure resident. Tests cover every backend/policy after this change.
- Sparse CSR invariants disallow raw duplicate column entries. The library path
  coalesces duplicates into summed weights; the custom multigraph kernel preserves
  duplicate index entries. Both implement the same sum, with floating tolerances.
- The native kernel gathers from dual CSR rather than materializing an E-by-F
  message array. Backward uses transpose CSR; conversion and retained structure
  incur costs, which must be included when interpreting a crossover.
- Current CUDA implementation is float32/float64 and first-order only. It is not a
  generic operator library, not optimized for all graph distributions, and not
  benchmarked against every available GNN implementation.
- The model is a repeated residual graph processor with degree normalization.
  It tests a systems mechanism. It does not establish accuracy on a scientific task.
- The council is deterministic and produces auditable specialist opinions. A
  simple greedy controller receives equivalent validated information as a strong
  baseline. Never attribute an improvement to 'multiple agents' without that test.
- Local ridge selection was evaluated on held-out profiles and retained as a
  negative experiment when it tied the simple training-selected rule.
- Controller replay timings are descriptive: only one scripted replay per strategy,
  in fixed strategy order. Completed-step counts and numerical checks are the
  relevant functional observations, not proof of a throughput or SLO improvement.
- A bounded same-process reservation changes real device memory pressure. Logical
  budget traces are separate constraints; neither is a measured game workload.
- Hash-chain verification is tamper-evident against a retained root, not an
  authenticated or tamper-proof ledger. A writer can regenerate a whole chain.
- Public deployment and mail sending require the user's approval under handbook
  Section 0. Build and test the local report and static deployment bundle first.

## Command chat and hosting update

The user authorized Vercel/Railway deployment on 24 October. Vercel account access
is unavailable, Railway Google connection failed, and an anonymous temporary
deployment returns HTTP 403; public accessibility is NOT established. See
results/deployment_status.json. No further permission is needed to deploy once
the actual existing account is signed in.

The new command chat has a separate exact-Origin loopback control endpoint,
fixed worker argv, UUID output paths, one-job admission, cancellation and timeout.
Its real GPU run/cancel checks passed. Public chat is saved-evidence only.

## Production deployment completed

The user completed account and CLI authorization. The dedicated drona3/graph-memory-lab
project is deployed at https://graph-memory-lab.vercel.app . Unauthenticated HTTP
checks match the local release, and public chat/browser tests passed. Earlier
authentication errors are historical; no further deployment permission is pending.
Public hosting serves evidence and command suggestions; the GPU worker stays local.
