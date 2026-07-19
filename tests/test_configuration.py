import os
import subprocess
import sys
from pathlib import Path


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
