import numpy as np
import pytest

from pyclustifyr.similarity import cosine, kl_divergence, vector_similarity


def test_cosine_identical_vectors_is_one():
    v = np.array([1.0, 2.0, 3.0])
    assert cosine(v, v) == pytest.approx(1.0)


def test_cosine_orthogonal_vectors_is_zero():
    assert cosine(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(0.0)


def test_kl_divergence_matches_r_reference():
    # cross-checked against clustifyr:::kl_divergence(v1, v2)
    v1 = np.array([1, 3, 2, 1, 4, 4, 0, 3, 2, 2, 2, 1, 3, 1, 1, 3, 5, 1, 2, 0], dtype=float)
    v2 = np.array(
        [21, 24, 18, 15, 15, 23, 20, 24, 21, 29, 14, 23, 18, 23, 10, 17, 30, 22, 25, 17], dtype=float
    )
    assert kl_divergence(v1, v2) == pytest.approx(0.609890964641179, abs=1e-12)


def test_kl_divergence_matches_r_reference_second_case():
    v1 = np.array([4, 4, 5, 8, 3, 8, 9, 6, 6, 2, 3, 3, 6, 4, 7, 5, 6, 11, 4, 7], dtype=float)
    v2 = np.array([11, 5, 8, 4, 5, 6, 2, 6, 10, 6, 7, 8, 7, 5, 9, 8, 9, 4, 8, 6], dtype=float)
    assert kl_divergence(v1, v2) == pytest.approx(0.703515453712026, abs=1e-12)


def test_vector_similarity_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        vector_similarity(np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0]), "cosine")


def test_vector_similarity_rejects_unknown_method():
    with pytest.raises(ValueError):
        vector_similarity(np.array([1.0]), np.array([1.0]), "pearson")
