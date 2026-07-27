import pytest

from admet_platform.training.task_sampler import (
    ProbabilisticTaskSampler,
    RoundRobinTaskSampler,
    build_task_sampler,
)


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


def test_temperature_sampling_probabilities_and_sequence_are_deterministic() -> None:
    rows = {"bbb_martins": 100, "herg_karim": 400, "ames": 900}
    first = build_task_sampler("temperature", TASKS, rows, seed=42, alpha=0.5)
    second = build_task_sampler("temperature", TASKS, rows, seed=42, alpha=0.5)

    assert isinstance(first, ProbabilisticTaskSampler)
    assert first.probabilities == pytest.approx(
        {"bbb_martins": 1 / 6, "herg_karim": 2 / 6, "ames": 3 / 6}
    )
    first_sequence = [first.next_task() for _ in range(20)]
    second_sequence = [second.next_task() for _ in range(20)]
    assert first_sequence == second_sequence


@pytest.mark.parametrize(
    ("strategy", "expected"),
    [
        ("uniform", {"bbb_martins": 1 / 3, "herg_karim": 1 / 3, "ames": 1 / 3}),
        ("proportional", {"bbb_martins": 0.1, "herg_karim": 0.3, "ames": 0.6}),
    ],
)
def test_probability_strategies(strategy: str, expected: dict[str, float]) -> None:
    sampler = build_task_sampler(
        strategy,
        TASKS,
        {"bbb_martins": 10, "herg_karim": 30, "ames": 60},
        seed=7,
    )
    assert isinstance(sampler, ProbabilisticTaskSampler)
    assert sampler.probabilities == pytest.approx(expected)


def test_probabilistic_sampler_resume_is_exact() -> None:
    rows = {task: index + 1 for index, task in enumerate(TASKS)}
    sampler = build_task_sampler("temperature", TASKS, rows, seed=9, alpha=0.5)
    for _ in range(11):
        task = sampler.next_task()
        sampler.record_batch(task, 2)
    state = sampler.state_dict()

    resumed = build_task_sampler("temperature", TASKS, rows, seed=9, alpha=0.5)
    resumed.load_state_dict(state)

    assert [sampler.next_task() for _ in range(20)] == [
        resumed.next_task() for _ in range(20)
    ]
