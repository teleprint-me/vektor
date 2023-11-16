"""
tests/core/test_vektor.py
"""
import pytest

from vektor.core.vektor import Vektor


@pytest.fixture
def sample_vektor():
    return Vektor(matrix=[[1, 2], [3, 4]])


def test_init_with_matrix():
    matrix = [[1.0, 2.0], [3.0, 4.0]]
    v = Vektor(matrix=matrix)
    assert v.matrix == matrix


def test_init_with_shape():
    shape = (2, 3)
    v = Vektor(shape=shape)
    expected_matrix = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    assert v.matrix == expected_matrix


def test_type_enforcement():
    with pytest.raises(TypeError):
        Vektor(matrix=[[1, 2], [3, "4"]])


@pytest.mark.parametrize(
    "input_matrix,expected_output",
    [([[1, 2], [3, 4]], [[1.0, 2.0], [3.0, 4.0]]), ([[5, 6]], [[5.0, 6.0]])],
)
def test_matrix_initialization(input_matrix, expected_output):
    v = Vektor(matrix=input_matrix)
    assert v.matrix == expected_output
