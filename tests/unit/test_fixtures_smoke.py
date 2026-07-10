def test_pbmc_matrix_small_shape(pbmc_matrix_small):
    assert pbmc_matrix_small.shape == (2000, 2638)
    assert pbmc_matrix_small.index[0] == "PPBP"


def test_cbmc_ref_shape(cbmc_ref):
    assert cbmc_ref.shape == (2000, 13)


def test_pbmc_meta_shape(pbmc_meta):
    assert pbmc_meta.shape == (2638, 9)


def test_pbmc_vargenes_len(pbmc_vargenes):
    assert len(pbmc_vargenes) == 2000
