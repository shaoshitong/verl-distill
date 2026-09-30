# Fresh-noise DMD rollout

`method.params.rollout_initial_noise: gaussian` uses `torch.randn_like(clean)`
for the initial Generator rollout state in both DMD phases (G and Fake score).
The FP32 clean latent supplies only shape/device/dtype; its values do not enter
the fresh noise. Score sigma/noise keep their existing independent sampling.
The default `dataset` retains the paired initial noise without extra RNG draws.
REFLOW always retains the paired dataset noise. Exit mode is independent: omitting
`dmd_rollout_loss_mode` retains the existing random exit/debug-last behavior.

For the specifically authorized paired-to-fresh resume with checkpoint interval
100, set `runtime.resume_from` and `runtime.resume_initial_noise_migration` to a
mapping containing the source checkpoint `state.json` SHA256 under
`source_state_sha256`. Set `runtime.save_every_fake_updates: 100` and remove
`init_reflow_from`. The narrow migration rejects any other numerical contract
change, including optimizer, dataset, GA or all-exit-mode changes. It restores
both G/Fake weights AND Adam states, data cursor and RNG; update counters continue.
This migration flag is for the transition checkpoint only. Subsequent ordinary
resumes of a fresh-noise checkpoint should omit the migration flag.
