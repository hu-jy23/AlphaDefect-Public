import numpy as np
import torch

from sts2rl.group_objectives import (
    action_weights, clipped_terms, overlong_penalty, reference_kl,
    relative_advantages, sequence_coefficients, sequence_loss,
)


def test_group_normalization_and_zero_variance():
    np.testing.assert_array_equal(relative_advantages([3, 3, 3, 3]), np.zeros(4))
    adv = relative_advantages([0, 1, 2, 3])
    assert abs(adv.mean()) < 1e-7
    assert abs(adv.std() - 1) < 1e-7


def test_sequence_and_action_weighting_differ_on_unequal_lengths():
    np.testing.assert_allclose(action_weights([1, 3], "grpo"), [.5, 1/6, 1/6, 1/6])
    np.testing.assert_allclose(action_weights([1, 3], "dapo"), [.25]*4)


def test_dapo_clip_higher_retains_positive_gradient():
    value = torch.tensor([np.log(1.25)], dtype=torch.float64, requires_grad=True)
    low = clipped_terms(value, torch.ones(1), .2, .2).sum()
    high = clipped_terms(value, torch.ones(1), .2, .28).sum()
    assert torch.autograd.grad(low, value, retain_graph=True)[0].item() == 0
    assert torch.autograd.grad(high, value)[0].item() > 0


def test_dspo_chunked_gradient_matches_full_sequence_autograd():
    # Unequal lengths and both clipping directions exercise sequence ownership.
    owners = torch.tensor([0, 0, 1, 1, 1, 2, 3])
    lengths = torch.bincount(owners)
    values = torch.tensor([.1, -.1, .15, .2, .1, -.4, .4],
                          dtype=torch.float64, requires_grad=True)
    advantages = torch.tensor([1., -1., -1., 1.], dtype=torch.float64)
    means = torch.zeros(4, dtype=torch.float64).index_add(0, owners, values) / lengths
    exact, = torch.autograd.grad(sequence_loss(means, advantages, .2, .2), values)
    _, coeff = sequence_coefficients(means, advantages, .2, .2)
    chunked = torch.zeros_like(values)
    for begin, end in [(0, 3), (3, 5), (5, 7)]:
        part = values.detach()[begin:end].requires_grad_(True)
        surrogate = (part * coeff[owners[begin:end]] / lengths[owners[begin:end]]).sum()
        chunked[begin:end], = torch.autograd.grad(surrogate, part)
    torch.testing.assert_close(exact, chunked)


def test_reference_kl_and_overlong_boundaries():
    torch.testing.assert_close(reference_kl(torch.zeros(3), torch.zeros(3)), torch.zeros(3))
    assert (reference_kl(torch.tensor([-.5, .5]), torch.zeros(2)) > 0).all()
    np.testing.assert_allclose(overlong_penalty([7, 8, 9, 10, 11], 10, 2), [0, 0, -.5, -1, -1])


def test_run_milestones_use_boss_count_column_24():
    from types import SimpleNamespace
    from sts2rl.group_train import milestone_reward
    def obs(act, floor, bosses=0, victory=False):
        header = np.zeros(34, dtype=np.int32)
        header[24] = bosses
        return SimpleNamespace(act=act, floor=floor, header=header, done=victory, victory=victory)
    rewards = dict(floor=.02, enter_act_1=.1, enter_act_2=.5,
                   act3_first_boss=1.5, act3_second_boss=2., victory=2.)
    np.testing.assert_allclose(milestone_reward(obs(0, 16), obs(1, 17), rewards), .12)
    assert milestone_reward(obs(2, 48), obs(2, 48, 1), rewards) == 1.5
    assert milestone_reward(obs(2, 48, 1), obs(2, 48, 1), rewards) == 0
    assert milestone_reward(obs(2, 48, 1), obs(2, 48, 2, True), rewards) == 4
