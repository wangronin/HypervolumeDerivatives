import numpy as np
import pytest

from hvd.utils import regularize_hessian_block


def test_block_regularization_damps_points_independently() -> None:
    coupling = np.array([[0.1, -0.2], [0.3, 0.05]])
    hessian = np.block(
        [
            [np.diag([-2.0, 1.0]), coupling],
            [coupling.T, np.diag([1e-12, 2.0])],
        ]
    )

    regularized = regularize_hessian_block(
        hessian,
        block_size=2,
        min_eigenvalue=1e-6,
        max_condition_number=100.0,
    )

    assert np.allclose(regularized[:2, 2:], coupling)
    assert np.allclose(regularized[2:, :2], coupling.T)
    first_shift = regularized[0, 0] - hessian[0, 0]
    second_shift = regularized[2, 2] - hessian[2, 2]
    assert first_shift > second_shift > 0
    for start in (0, 2):
        eigenvalues = np.linalg.eigvalsh(regularized[start : start + 2, start : start + 2])
        assert eigenvalues[0] >= 1e-6 - 1e-12
        assert eigenvalues[-1] / eigenvalues[0] <= 100.0 + 1e-10


def test_block_regularization_leaves_well_conditioned_blocks_unchanged() -> None:
    hessian = np.diag([1.0, 2.0, 3.0, 4.0])

    regularized = regularize_hessian_block(hessian, block_size=2)

    assert np.array_equal(regularized, hessian)


@pytest.mark.parametrize(
    "hessian,block_size",
    [
        (np.ones((2, 3)), 1),
        (np.eye(3), 2),
        (np.eye(2), 0),
    ],
)
def test_block_regularization_rejects_invalid_shapes(hessian, block_size) -> None:
    with pytest.raises(ValueError):
        regularize_hessian_block(hessian, block_size)
