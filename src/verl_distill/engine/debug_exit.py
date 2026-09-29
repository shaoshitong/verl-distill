"""Debug snapshots reuse the actual update, optionally at the final exit."""
def capture_update(phase, fake_updates, interval, *, benchmark=False):
    if interval <= 0:
        raise ValueError('Debug interval must be positive')
    update = fake_updates + 1 if phase == 'fake_score' else fake_updates
    return not benchmark and phase != 'reflow' and update % interval == 0

def force_debug_exit(capture, enabled, rollout_input):
    return bool(capture and enabled and rollout_input)
