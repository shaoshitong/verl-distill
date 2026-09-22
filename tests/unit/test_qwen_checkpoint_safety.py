"""Failure injection for checkpoint publication and the real post-update boundary."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from verl_distill.algorithms.dmd.qwen_image21 import UpdateState
from verl_distill.engine import qwen_checkpoint as cp
from verl_distill.trainers.qwen_image21 import post_update_actions


@pytest.fixture
def local_checkpoint(monkeypatch, tmp_path):
    monkeypatch.setattr(cp.dist, 'get_world_size', lambda: 1)
    monkeypatch.setattr(cp.dist, 'all_gather_object', lambda out, value: out.__setitem__(0, value))
    monkeypatch.setattr(cp, 'capture_rng_state', lambda: {'test_rng': 123})
    def payload(path, *args, **kwargs):
        Path(path).mkdir(parents=True)
        (Path(path) / '.metadata').write_bytes(b'metadata')
        (Path(path) / '__0_0.distcp').write_bytes(b'checkpoint-payload')
    monkeypatch.setattr(cp, 'save_distributed_training_state', payload)
    state = UpdateState(reflow_updates=1000)
    def save(name):
        return cp.save_checkpoint(tmp_path / name, {'generator': object()},
                                  {'generator': object()}, state,
                                  SimpleNamespace(state_dict=lambda: {'position': 8}),
                                  {'test': True}, {}, rank=0)
    return state, save


@pytest.mark.parametrize('phase', ['reflow', 'generator'])
def test_debug_failure_leaves_committed_checkpoint(local_checkpoint, tmp_path, phase):
    state, save = local_checkpoint
    if phase == 'generator':
        state.fake_updates, state.generator_updates, state.dmd_initialized = 600, 120, True
    def fail_debug():
        raise RuntimeError('injected VAE/debug failure')
    runtime = {'debug_after_reflow': True, 'save_every_fake_updates': 600,
               'debug_every_fake_updates': 20}
    with pytest.raises(RuntimeError, match='injected VAE'):
        post_update_actions(state, phase, 1000, 3000, runtime, save, fail_debug)
    path = tmp_path / ('reflow_step_001000' if phase == 'reflow' else 'fake_step_000600')
    assert cp.inspect_checkpoint(path, {'test': True})['updates'] == state.state_dict()


def test_rank_sidecar_failure_never_commits_or_enters_debug(local_checkpoint, monkeypatch, tmp_path):
    state, save = local_checkpoint
    calls = []
    def fail(*args, **kwargs):
        raise OSError('injected disk full')
    monkeypatch.setattr(cp.torch, 'save', fail)
    with pytest.raises(RuntimeError, match='checkpoint sidecars.*disk full'):
        post_update_actions(state, 'reflow', 1000, 3000, {'debug_after_reflow': True},
                            save, lambda: calls.append('debug'))
    assert not calls
    assert not (tmp_path / 'reflow_step_001000').exists()
    assert not (tmp_path / 'reflow_step_001000.incomplete/COMPLETE').exists()
    with pytest.raises(ValueError, match='incomplete'):
        cp.inspect_checkpoint(tmp_path / 'reflow_step_001000.incomplete', {'test': True})


def test_truncated_payload_rejected_before_loading(local_checkpoint):
    _, save = local_checkpoint
    path = save('saved')
    (path / 'generator/__0_0.distcp').write_bytes(b'x')
    with pytest.raises(ValueError, match='truncated'):
        cp.inspect_checkpoint(path, {'test': True})


def test_changed_metadata_rejected(local_checkpoint):
    _, save = local_checkpoint
    path = save('saved')
    p = path / 'state.json'
    p.write_text(p.read_text() + '\n')
    with pytest.raises(ValueError, match='digest mismatch'):
        cp.inspect_checkpoint(path, {'test': True})


def test_reflow_checkpoint_and_rollout_cadence():
    runtime = {'reflow_checkpoint_steps': [1,500,1000], 'debug_every_reflow_updates': 20,
               'debug_after_reflow': True, 'save_every_fake_updates':600,
               'debug_every_fake_updates':20}
    state = UpdateState()
    events = []
    for step in range(1,1001):
        state.reflow_updates = step
        post_update_actions(state,'reflow',1000,3000,runtime,
                            lambda name:events.append((step,'save',name)),
                            lambda:events.append((step,'debug',None)))
    assert [e[0] for e in events if e[1]=='save'] == [1,500,1000]
    assert [e[0] for e in events if e[1]=='debug'] == list(range(20,1001,20))
    for step in [500,1000]:
        assert [e[1] for e in events if e[0]==step] == ['save','debug']
    state.dmd_initialized = True
    events.clear()
    for fake in range(5,3001,5):
        state.fake_updates, state.generator_updates = fake, fake//5
        post_update_actions(state,'generator',1000,3000,runtime,
                            lambda name:events.append((fake,'save',name)),
                            lambda:events.append((fake,'debug',None)))
    assert [e[0] for e in events if e[1]=='save'] == [600,1200,1800,2400,3000]
    assert [e[0] for e in events if e[1]=='debug'] == list(range(20,3001,20))
    for step in [600,1200,1800,2400,3000]:
        assert [e[1] for e in events if e[0]==step] == ['save','debug']


def test_debug_paths_distinguish_reflow_steps_from_dmd():
    from verl_distill.trainers.qwen_image21_debug import debug_event_name
    assert debug_event_name(UpdateState(reflow_updates=20)) == 'reflow_step_000020'
    assert debug_event_name(UpdateState(reflow_updates=40)) == 'reflow_step_000040'
    assert debug_event_name(UpdateState(1000,20,4,True)) == 'fake_step_000020'


def test_early_dmd_checkpoint_commits_before_debug():
    runtime = {'early_dmd_checkpoint_steps': [5, 40, 45],
               'save_every_fake_updates': 600, 'debug_every_fake_updates': 20}
    state = UpdateState(reflow_updates=510, fake_updates=40,
                        generator_updates=8, dmd_initialized=True)
    events = []
    post_update_actions(state, 'generator', 510, 3000, runtime,
                        lambda name: events.append(('save', name)),
                        lambda: events.append(('debug', None)))
    assert events == [('save', 'fake_step_000040'), ('debug', None)]


def test_first_refinement_debug_after_checkpoint():
    state = UpdateState(reflow_updates=1)
    runtime = {'reflow_checkpoint_steps':[1,20,100], 'debug_every_reflow_updates':20,
               'debug_reflow_steps':[1]}
    events=[]
    post_update_actions(state,'reflow',100,3000,runtime,
                        lambda name:events.append('save'),lambda:events.append('debug'))
    assert events==['save','debug']
