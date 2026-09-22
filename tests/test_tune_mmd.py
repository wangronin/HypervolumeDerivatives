import json
from types import SimpleNamespace

import pytest

from scripts import benchmark_MMD, tune_MMD


class DeterministicTrial:
    def __init__(self, kernel="rational_quadratic"):
        self.kernel = kernel
        self.requested = []

    def suggest_categorical(self, name, values):
        self.requested.append(name)
        return self.kernel if name == "kernel" else values[0]

    def suggest_float(self, name, low, high, log=False):
        self.requested.append(name)
        assert log
        return (low * high) ** 0.5


def tuning_args(**overrides):
    values = {
        "kernels": list(tune_MMD.KERNELS),
        "beta_min": 1e-4,
        "beta_max": 1.0,
        "theta_min": 1e-3,
        "theta_max": 1e4,
        "regularization_choices": [False, True],
        "boundary_constraints": True,
        "algorithm": "NSGA-III",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_tuning_space_uses_matching_beta_and_smooth_kernels() -> None:
    trial = DeterministicTrial()
    config = tune_MMD.sample_configuration(trial, tuning_args())

    assert set(tune_MMD.KERNELS) == {"rbf", "rational_quadratic", "linear"}
    assert config == {
        "kernel": "rational_quadratic",
        "beta": pytest.approx(0.01),
        "regularization": False,
        "theta_multiplier": pytest.approx(10**0.5),
    }
    assert set(trial.requested) == {"kernel", "beta", "regularization", "theta_multiplier"}
    assert tune_MMD.study_name("IDTLZ1", tuning_args()) == (
        "MMDNewton-MMDMatching-IDTLZ1-NSGA-III-bounds"
    )


def test_linear_kernel_does_not_request_a_length_scale() -> None:
    trial = DeterministicTrial(kernel="linear")

    config = tune_MMD.sample_configuration(trial, tuning_args())

    assert "theta_multiplier" not in config
    assert "theta_multiplier" not in trial.requested


def test_tuning_parser_has_no_indicator_or_line_search_switch() -> None:
    parser = tune_MMD.build_parser()
    destinations = {action.dest for action in parser._actions}
    kernel_action = next(action for action in parser._actions if action.dest == "kernels")

    assert "matching" not in destinations
    assert "line_searches" not in destinations
    assert "laplace" not in kernel_action.choices
    assert bool(tune_MMD.jax_config.jax_enable_x64)


def test_benchmark_discovers_and_loads_tuner_output(tmp_path) -> None:
    path = tmp_path / "MMDNewton-MMDMatching-IDTLZ1-NSGA-III-bounds-best.json"
    result = {
        "problem": "IDTLZ1",
        "algorithm": "NSGA-III",
        "indicator": "MMDMatching",
        "boundary_constraints": True,
        "best_config": {
            "kernel": "rbf",
            "beta": 0.2,
            "regularization": True,
            "theta_multiplier": 3.0,
        },
    }
    path.write_text(json.dumps(result))
    args = SimpleNamespace(
        best_config=None,
        tuning_dir=tmp_path,
        problem="IDTLZ1",
        algorithm="NSGA-III",
        boundary_constraints=True,
    )

    selected_path, loaded = benchmark_MMD.find_best_result(args)

    assert selected_path == path
    assert loaded == result


def test_benchmark_uses_thirty_parallel_workers_by_default() -> None:
    args = benchmark_MMD.build_parser().parse_args(["IDTLZ1"])

    assert args.n_jobs == 30
    assert benchmark_MMD.build_parser().parse_args(["IDTLZ1", "--n-jobs", "1"]).n_jobs == 1
