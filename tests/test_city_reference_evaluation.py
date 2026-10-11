from types import SimpleNamespace

from tools.evaluate_city_references import evaluate_reference


def test_dta_updates_once_only_on_training_seed_and_evaluation_keeps_fixed_policy():
    calls = []
    method = SimpleNamespace(iteration=0, select=lambda *a, **kw: None,
                             begin_episode=lambda *a, **kw: None)
    def update(rows, windows, *, episode):
        assert rows == ['real'] and windows == ['windows'] and episode == 1
        method.iteration += 1
        return {'dta_iteration': method.iteration}
    method.on_episode = update
    def run(select, **kwargs):
        calls.append((kwargs['seed'], kwargs.get('record_windows', False), method.iteration))
        return ['real'], ['windows'], dict(seed=kwargs['seed'], completion_rate=1., global_reward_sum=-2.)
    training, evaluations = evaluate_reference(SimpleNamespace(run_episode=run), method, [10042, 10043, 10044], 43)
    assert calls == [(43, True, 0), (10042, False, 1), (10043, False, 1), (10044, False, 1)]
    assert training['update']['dta_iteration'] == 1
    assert [row['seed'] for row in evaluations] == [10042, 10043, 10044]


def test_shortest_has_one_evaluation_round_and_no_training_episode():
    calls = []
    def run(select, **kwargs):
        assert select == 'shortest'
        calls.append(kwargs['seed'])
        return [], [], dict(seed=kwargs['seed'], completion_rate=1., global_reward_sum=-2.)
    training, evaluations = evaluate_reference(SimpleNamespace(run_episode=run), None, [10042, 10043, 10044], 43)
    assert training is None and len(evaluations) == 3
    assert calls == [10042, 10043, 10044]
