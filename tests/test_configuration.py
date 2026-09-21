import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hvd.problems import IDTLZ1, IDTLZ2, IDTLZ3, IDTLZ4


def test_direct_submodule_import_enables_jax_x64() -> None:
    environment = os.environ.copy()
    environment.pop("JAX_ENABLE_X64", None)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import hvd.mmd; import jax; "
            "print(jax.config.jax_enable_x64, jax.numpy.ones(1).dtype)",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.stdout.strip() == "True float64"


def test_jax_x64_configuration_is_centralized() -> None:
    package = Path(__file__).parents[1] / "hvd"
    occurrences = [
        path.relative_to(package)
        for path in package.rglob("*.py")
        if "JAX_ENABLE_X64" in path.read_text()
    ]

    assert occurrences == [Path("__init__.py")]


@pytest.mark.parametrize("problem_type", [IDTLZ1, IDTLZ2, IDTLZ3, IDTLZ4])
def test_mmd_reference_points_dominate_idtlz_pareto_front(problem_type) -> None:
    config = pd.read_csv(
        Path(__file__).parents[1] / "scripts" / "ref_point.csv",
        index_col="problem",
    )
    reference_point = config.loc[problem_type.__name__].dropna().to_numpy(dtype=float)
    pareto_front = problem_type().get_pareto_front()

    assert reference_point.shape == (pareto_front.shape[1],)
    assert np.all(reference_point > np.max(pareto_front, axis=0))
