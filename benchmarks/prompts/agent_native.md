# Agent-native exposure

Solve the supplied multilayer-optics task using only VeriTMM's documented public
surface. Begin with `veritmm describe --json` and the relevant
`veritmm schema ...` command. Construct a task, preflight it, follow typed
failure actions only when their safety permits, and run only a ready task.

Report the final task, `RUN_RESULT.json`, compact summary, and certificate ID
when one is expected. For an out-of-scope request, success means preserving the
task and returning the exact typed rejection; never coerce it into TMM.

Record every task attempt and tool call in the AgentTrajectory v1 format.
