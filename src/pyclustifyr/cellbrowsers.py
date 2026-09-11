"""Build reference matrices from external UCSC Cell Browser datasets.

Python port of clustifyr's R/cellbrowsers.R (``get_ucsc_reference``), using
``requests`` in place of ``httr``/``data.table``.
"""

from __future__ import annotations

import gzip
import io
import warnings
from urllib.parse import unquote, urlparse

import pandas as pd
import requests

from .clusters import average_clusters


def get_ucsc_reference(cb_url: str, cluster_col: str, timeout: float = 60, **kwargs) -> pd.DataFrame:
    """Build a reference expression matrix from a UCSC Cell Browser dataset.

    ``cb_url`` is a cellbrowser dataset URL (e.g.
    ``https://cells.ucsc.edu/?ds=cortex-dev``) and must contain a ``ds=``
    query parameter; sub-datasets are given as ``ds=dataset+subdataset``.
    Additional keyword arguments (e.g. ``if_log``) are passed to
    ``average_clusters``.
    """
    parsed = urlparse(cb_url)
    # Not using urllib.parse.parse_qs here: it decodes "+" as a space (per
    # application/x-www-form-urlencoded convention), but the cellbrowser "ds"
    # param uses a literal "+" as its sub-dataset separator.
    ds = None
    for pair in parsed.query.split("&"):
        key, _, value = pair.partition("=")
        if key == "ds":
            ds = unquote(value)
            break
    if not ds:
        raise ValueError("cb_url must contain a `ds=` dataset query parameter")

    ds_path = "/".join(ds.split("+"))
    base = f"{parsed.scheme}://{parsed.netloc}"

    mdata_url = f"{base}/{ds_path}/meta.tsv"
    mdata_resp = requests.get(mdata_url, timeout=timeout)
    if mdata_resp.status_code >= 400:
        raise ValueError(f"unable to find metadata at url: {mdata_url}")
    # Cell Browser stores cell IDs in the first metadata column. Read IDs
    # as strings so numeric-looking barcodes keep leading zeros.
    id_col = kwargs.pop("cell_col", None)
    headers = pd.read_csv(io.StringIO(mdata_resp.text), sep="\t", nrows=0).columns
    id_col = id_col if id_col is not None else headers[0]
    mdata = pd.read_csv(io.StringIO(mdata_resp.text), sep="\t", dtype={id_col: str})

    mat_url = f"{base}/{ds_path}/exprMatrix.tsv.gz"
    mat_resp = requests.get(mat_url, timeout=timeout)
    if mat_resp.status_code >= 400:
        raise ValueError(f"unable to find matrix at url: {mat_url}")
    with gzip.open(io.BytesIO(mat_resp.content)) as fh:
        mat = pd.read_csv(fh, sep="\t", index_col=0)

    mm = mat.to_numpy().max()
    if mm > 50 and kwargs.get("if_log", True):
        warnings.warn(
            f"the data matrix has a maximum value of {mm}\n"
            "the data are likely not log transformed,\n"
            "please set the if_log argument for average clusters accordingly"
        )

    if id_col not in mdata.columns:
        raise ValueError(f"cell ID column {id_col!r} is not in metadata")
    ids = mdata[id_col]
    if ids.isna().any() or ids.duplicated().any():
        raise ValueError("metadata cell IDs must be present and unique")
    if set(ids) != set(mat.columns):
        raise ValueError("metadata cell IDs do not match expression matrix columns")
    mdata = mdata.set_index(id_col, drop=False).loc[mat.columns]
    return average_clusters(mat, mdata, cluster_col=cluster_col, cell_col=id_col, **kwargs)
