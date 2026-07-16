import pytest

from admet_platform.training.task_sampler import RoundRobinTaskSampler


TASKS = ("bbb_martins", "herg_karim", "ames")


def test_round_robin_sequence_is_deterministic_and_balanced() -> None:
    sampler = RoundRobinTaskSampler(TASKS)

    sequence = [sampler.next_task() for _ in range(8)]

    assert sequence == ["bbb_martins", "herg_karim", "ames", "bbb_martins", "herg_karim", "ames", "bbb_martins", "herg_karim"]
    assert sampler.logical_pass == 2


def test_sampler_save_and_resume_preserves_next_task_and_counts() -> None:
    sampler = RoundRobinTaskSampler(TASKS)
    first = sampler.next_task()
    sampler.record_batch(first, 4)
    sampler.next_task()
    state = sampler.state_dict()

    resumed = RoundRobinTaskSampler(TASKS)
    resumed.load_state_dict(state)

    assert resumed.next_task() == "ames"
    assert resumed.batch_counts == {"bbb_martins": 1, "herg_karim": 0, "ames": 0}
    assert resumed.example_counts["bbb_martins"] == 4


def test_sampler_rejects_incompatible_task_order() -> None:
    state = RoundRobinTaskSampler(TASKS).state_dict()

    with pytest.raises(ValueError, match="incompatible"):
        RoundRobinTaskSampler(reversed(TASKS)).load_state_dict(state)
