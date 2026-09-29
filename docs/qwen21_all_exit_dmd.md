# Qwen DMD: supervise every rollout exit

Set the following in the existing recipe (all other settings can remain unchanged):

```yaml
method:
  params:
    dmd_rollout_loss_mode: all_exits
    generator_input: rollout_dataset_noise
```

Omitting `dmd_rollout_loss_mode` retains `random_exit`, including its random exit,
forced-debug behavior, RNG draws and loss scaling. The option applies only to
DMD Generator and Fake phases; REFLOW is unchanged. No defaults are inserted
into the saved params, so existing configurations retain their checkpoint contract.
Changing the option changes the checkpoint contract; it is not a silent exact resume.

In `all_exits`, each data sample runs one six-forward Generator trajectory.
Each forward predicts x0 and computes the next detached Euler state using the
same velocity. Each exit independently draws score sigma from the existing
schedule and fresh score Gaussian noise. Independent draws may happen to select
the same discrete sigma; no rejection/uniqueness constraint changes its marginal.
The score noise is not the stored initial rollout noise.

For G, every exit backpropagates the existing DMD surrogate. For Fake, G runs
without gradients and every exit backpropagates the existing Fake score loss.
Each loss is divided by `6 * gradient_accumulation_steps`. Parameters stay fixed
through the trajectory; gradient clipping, optimizer step, and update counters
advance once after accumulation. The Fake:G update ratio is unchanged. Gradients
are reduced per backward (no new FSDP no_sync/full-gradient memory requirement).

Fake-phase Generator shards stay resident until the last rollout forward to
avoid six offload/reload pairs. This uses additional shard memory during earlier
Fake backwards compared with random_exit; longest-sample memory needs a GPU trial.
G-phase score offload retains its existing per-backward behavior.

Logs contain six exit records per sample with their score sigma, individual loss,
exit index, mode, and accumulation divisor. Global loss is the average over exits
and data microbatches. Dataset batch size/GA are unchanged; six correlated exits
are not six independent data samples. Debug stores only the final 6NFE exit; it
does not suppress training of earlier exits or force their sampling.

CPU regression tests compare shared-trajectory outputs and accumulated G/Fake
gradients to six independent detached-prefix reference runs (6 vs 21 G forwards).
No distributed GPU throughput/convergence claim is made by these tests.
