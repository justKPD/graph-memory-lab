# Lab command chat

Public/static commands: scope; results; help; status; compare uniform_train budget
100; explain uniform_train/cuda_csr/resident; propose ring_holdout budget 120;
replay council; replay council step 1; verify.

Local server only (http://127.0.0.1:8765): run replay; job status; cancel replay.
The runner executes the existing four-controller uniform_train GPU workload in a
new UUID output directory, with a 180-second timeout and at most one active job.
It never interpolates chat text into shell code or overwrites the original study.
Cancellation targets the single actual worker process; partial output is not a
completed result. Poll job status manually after launch. Refresh does not cancel.
A server restart loses in-memory job status; this is a local research MVP, not a
persistent production scheduler. Keep the server running until its job stops.

The public website cannot control the GPU on your laptop. This is deterministic
command routing and suggestions, not an LLM or unrestricted natural-language agent.
Unsupported commands, evidence edits, unknown fields, concurrent work and remote
control attempts are refused. Public hosting has no GPU/API-key requirement.

Validated: 12 router checks; 9 local API checks, including an actual completed GPU
replay with zero observed cap violations, and an actually cancelled second job.
Results: results/chat_router_tests.json and results/control_api_tests.json.
