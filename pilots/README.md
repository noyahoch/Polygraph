# Next-task headroom pilots

Cheap probes to find a task where the **attention graph** beats not just non-graph models
but a **hidden-state probe** — the win Polygraph's corruption benchmark never delivers
(structure beats flat-attention there, but a representation probe beats structure).

Headroom := AUROC(best internal-signal detector) − AUROC(best output-derived score),
measured on a group-disjoint split with the Polygraph protocol before committing to a build.

## Findings so far

- **vlm_pope** (LLaVA-1.5-7B on POPE adversarial, n=3000): no *average* probe headroom
  (yes/no couples to the output), but a **significant confident-error headroom**
  (+0.15, CI [+0.03, +0.28]) where output confidence is blind. Cross-domain consistent with
  the corruption confident-slice (+0.019). BUT this is a *representation* win, not a
  *structure* win — see `docs/results/pilot_vlm_pope.md`. A POPE attention-graph is unlikely
  to beat the probe (yes/no is not routing-shaped).
- **backdoor** (scaffolded, not run): high decoupling but risks a trivial
  attention-concentration heuristic solving it without a graph.

## Next: spurious-correlation / shortcut reliance

The one concept where structure can beat a probe: same final representation, different
attention routing ("right class, wrong reason" — object vs background). The purest match to
the proposal's "recurring routing signatures". In progress.

## Track B (backdoor): UN-ABSORBED — the synthetic routing testbed

Earlier this was marked "absorbed"; that is reversed. The 95%-correlated spurious patch
failed to be adopted (patch-following ~0.01 at every degradation level 64px->6px) — but that
is a DESIGN property, not a law: a 95%-reliable shortcut never beats a 100%-reliable true cue
in training. The fix is 100% correlation = standard BadNets (fixed target class), which is
Track B, now running via `pilots/backdoor/poison.py`. Backdoor = attention hijack by
construction = the synthetic routing testbed for the graph-vs-probe question. The
95%-spurious construction is abandoned; the natural-benchmark route (Waterbirds/CelebA) is
prepared under `pilots/waterbirds/` for the team's P2/P3 decision, not launched.
