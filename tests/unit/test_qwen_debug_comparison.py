import torch
from verl_distill.trainers import qwen_image21_debug as debug


def test_comparison_same_noise_and_no_intermediate_tensors(monkeypatch):
    class Schedule:
        def levels(self, height, width, steps=6, *, generator=True, device='cpu'):
            calls.append((steps, generator))
            return torch.linspace(1, 0, steps + 1, device=device)
    calls = []
    forwards = []
    def velocity(model, x, sigma, condition):
        forwards.append(sigma.item())
        return torch.ones_like(x)
    monkeypatch.setattr(debug, 'predict_velocity', velocity)
    row = dict(height=32, width=48, seed=42)
    six = debug.rollout_sample(None, None, row, Schedule(), 'cpu')
    comparison = debug.rollout_sample(None, None, row, Schedule(), 'cpu',
                                     steps=25, official_schedule=True, capture_trajectory=False)
    assert calls == [(6, True), (25, False)]
    assert len(forwards) == 31
    assert torch.equal(six[0], comparison[0])
    torch.testing.assert_close(comparison[1], comparison[0] - 1)
    assert len(comparison[2]) == 25
    assert all(set(step) == {'index', 'sigma', 'next_sigma'} for step in comparison[2])
    assert all('predicted_x0' in step for step in six[2])


def test_first_eight_paired_outputs(tmp_path, monkeypatch):
    import json
    from PIL import Image
    from verl_distill.algorithms.dmd.qwen_image21 import UpdateState
    class Schedule:
        config = {'shift_terminal': .02}
        def levels(self, height, width, steps=6, *, generator=True, device='cpu'):
            return torch.linspace(1, 0, steps + 1)
    class Store:
        def get(self, row, device): return None
    class Decoder:
        def decode(self, packed, height, width):
            return Image.new('RGBA', (width, height)), None
        def offload(self): pass
    rows = [dict(id=f'row-{i}', height=32, width=32, seed=i) for i in range(64)]
    csv = tmp_path / 'prompts.csv'
    csv.write_text('test')
    monkeypatch.setattr(debug, 'predict_velocity', lambda model, x, sigma, cond: torch.ones_like(x))
    monkeypatch.setattr(debug, 'collective_call', lambda name, fn: fn())
    monkeypatch.setattr(debug, 'capture_rng_state', lambda: None)
    monkeypatch.setattr(debug, 'restore_rng_state', lambda state: None)
    monkeypatch.setattr(debug.torch.cuda, 'max_memory_allocated', lambda device: 0)
    monkeypatch.setattr(debug.dist, 'all_gather_object', lambda counts, count: counts.__setitem__(0, count))
    debug.debug_event(tmp_path, UpdateState(reflow_updates=20), torch.nn.Identity(), {},
                      {'records': rows}, Store(), Schedule(), Decoder(), 'cpu', 0, 1,
                      tmp_path, csv, {}, prompt_count=8, comparison_steps=25)
    event = tmp_path / 'debug/reflow_step_000020'
    assert (event / 'COMPLETE').exists()
    assert {p.name for p in (event / 'rollout').iterdir()} == {r['id'] for r in rows[:8]}
    assert len(list((event / 'rollout_25step').glob('*/image.png'))) == 8
    assert not list((event / 'rollout_25step').glob('*/steps'))
    for row in rows[:8]:
        six, other = event/'rollout'/row['id'], event/'rollout_25step'/row['id']
        assert torch.equal(torch.load(six/'initial_noise.pt'), torch.load(other/'initial_noise.pt'))
        meta = json.loads((other/'metadata.json').read_text())
        assert meta['nfe'] == 25 and meta['cfg'] == 'conditional_only'
