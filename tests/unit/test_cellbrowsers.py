import gzip
import io
from unittest.mock import patch

import pytest

from pyclustifyr.cellbrowsers import get_ucsc_reference

META_TSV = "Cell\tcluster\nc1\tA\nc2\tA\nc3\tB\nc4\tB\n"
MATRIX_TSV = "gene\tc1\tc2\tc3\tc4\ng1\t1.0\t2.0\t3.0\t4.0\ng2\t0.0\t0.0\t5.0\t5.0\n"


def _fake_response(status_code, text=None, content=None):
    resp = type("Resp", (), {})()
    resp.status_code = status_code
    if text is not None:
        resp.text = text
    if content is not None:
        resp.content = content
    return resp


def _gzip_bytes(text: str) -> bytes:
    buf = io.BytesIO()
    with gzip.open(buf, "wt") as fh:
        fh.write(text)
    return buf.getvalue()


def _mock_get(url, timeout=None):
    if url.endswith("meta.tsv"):
        return _fake_response(200, text=META_TSV)
    if url.endswith("exprMatrix.tsv.gz"):
        return _fake_response(200, content=_gzip_bytes(MATRIX_TSV))
    return _fake_response(404)


def test_get_ucsc_reference_builds_correct_urls_and_averages():
    with patch("pyclustifyr.cellbrowsers.requests.get", side_effect=_mock_get) as mock_get:
        ref = get_ucsc_reference(
            "https://cells.ucsc.edu/?ds=some-dataset", cluster_col="cluster", if_log=False, output_log=False
        )

    called_urls = [call.args[0] for call in mock_get.call_args_list]
    assert "https://cells.ucsc.edu/some-dataset/meta.tsv" in called_urls
    assert "https://cells.ucsc.edu/some-dataset/exprMatrix.tsv.gz" in called_urls

    assert list(ref.columns) == ["A", "B"]
    assert ref.loc["g1", "A"] == pytest.approx(1.5)
    assert ref.loc["g1", "B"] == pytest.approx(3.5)
    assert ref.loc["g2", "A"] == pytest.approx(0.0)


def test_get_ucsc_reference_joins_subdataset_path_with_slashes():
    with patch("pyclustifyr.cellbrowsers.requests.get", side_effect=_mock_get) as mock_get:
        get_ucsc_reference("https://cells.ucsc.edu/?ds=a+b+c", cluster_col="cluster", if_log=False)

    called_urls = [call.args[0] for call in mock_get.call_args_list]
    assert "https://cells.ucsc.edu/a/b/c/meta.tsv" in called_urls
    assert "https://cells.ucsc.edu/a/b/c/exprMatrix.tsv.gz" in called_urls


def test_get_ucsc_reference_requires_ds_query_param():
    with pytest.raises(ValueError, match="ds="):
        get_ucsc_reference("https://cells.ucsc.edu/", cluster_col="cluster")


def test_get_ucsc_reference_raises_on_missing_metadata():
    def missing_meta(url, timeout=None):
        return _fake_response(404)

    with patch("pyclustifyr.cellbrowsers.requests.get", side_effect=missing_meta):
        with pytest.raises(ValueError, match="unable to find metadata"):
            get_ucsc_reference("https://cells.ucsc.edu/?ds=nope", cluster_col="cluster")


def test_get_ucsc_reference_warns_on_unlogged_data():
    big_matrix = "gene\tc1\tc2\tc3\tc4\ng1\t100.0\t200.0\t300.0\t400.0\n"

    def mock_get(url, timeout=None):
        if url.endswith("meta.tsv"):
            return _fake_response(200, text=META_TSV)
        return _fake_response(200, content=_gzip_bytes(big_matrix))

    with patch("pyclustifyr.cellbrowsers.requests.get", side_effect=mock_get):
        with pytest.warns(UserWarning, match="likely not log transformed"):
            get_ucsc_reference("https://cells.ucsc.edu/?ds=nope", cluster_col="cluster")
