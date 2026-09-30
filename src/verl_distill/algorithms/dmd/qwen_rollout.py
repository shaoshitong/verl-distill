"""Dataset-noise rollout with a detached Euler prefix and one x0 exit call."""
import torch


def detached_prefix_rollout(initial_noise, levels, exit_index, predict, *, train_exit):
    """predict(x, sigma) is called exit_index+1 times; only the last may build a graph.

    Callers synchronize exit_index across FSDP ranks. The six-point grid itself
    can depend on each sample's spatial shape, exactly as in normal inference.
    """
    if not 0 <= exit_index < len(levels) - 1:
        raise ValueError("Exit index must select a nonzero rollout timestep")
    x = initial_noise.detach().float()
    with torch.no_grad():
        for index in range(exit_index):
            sigma = levels[index:index + 1]
            velocity = predict(x, sigma)
            x = (x + (levels[index + 1] - levels[index]) * velocity.float()).detach()
    exit_input = x.detach()
    sigma = levels[exit_index:exit_index + 1]
    with torch.set_grad_enabled(train_exit):
        velocity = predict(exit_input, sigma)
        generated = exit_input - sigma * velocity.float()
    return generated, exit_input


def detached_rollout_step(x, levels, step_index, predict, *, train_exit):
    """One shared trajectory step; next state never retains the exit graph."""
    if not 0 <= step_index < len(levels) - 1:
        raise ValueError("Step must select a nonzero rollout timestep")
    exit_input = x.detach().float()
    sigma = levels[step_index:step_index + 1]
    with torch.set_grad_enabled(train_exit):
        velocity = predict(exit_input, sigma)
        generated = exit_input - sigma * velocity.float()
    with torch.no_grad():
        next_input = (exit_input + (levels[step_index + 1] - levels[step_index])
                      * velocity.float()).detach()
    return generated, exit_input, next_input


def initial_rollout_noise(clean, dataset_noise, *, phase, rollout_input, source="dataset"):
    """Fresh per-rank DMD rollout noise; REFLOW always retains paired data noise."""
    if source not in ("dataset", "gaussian"):
        raise ValueError("Unknown rollout_initial_noise")
    if phase == "reflow" or (rollout_input and source == "dataset"):
        return dataset_noise.to(clean.device)
    return torch.randn_like(clean)
