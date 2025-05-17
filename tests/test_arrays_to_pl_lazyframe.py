import pytest
import pandas as pd
import numpy as np
import polars as pl

from your_module import arrays_to_pl_lazyframe  # adjust import path as needed


def test_pandas_dataframe_input():
    # Create a simple pandas DataFrame
    pdf = pd.DataFrame({
        "col1": [1, 2, 3],
        "col2": [4.0, 5.0, 6.0]
    })
    # Convert to a Polars LazyFrame
    lz = arrays_to_pl_lazyframe(pdf)
    assert isinstance(lz, pl.LazyFrame)
    # Collect to a Polars DataFrame and compare
    df = lz.collect()
    assert df.shape == (3, 2)
    # Compare column names and values
    assert df.columns == ["col1", "col2"]
    # Convert back to pandas for deep equality check
    assert df.to_pandas().equals(pdf)


def test_numpy_ndarray_input():
    # Create a 2x3 numpy ndarray
    arr = np.array([[1, 2, 3], [7, 8, 9]])
    # Convert to a Polars LazyFrame
    lz = arrays_to_pl_lazyframe(arr)
    assert isinstance(lz, pl.LazyFrame)
    # Collect to a Polars DataFrame
    df = lz.collect()
    # Expect 3 rows and 2 columns
    assert df.shape == (3, 2)
    # Check default column names '0' and '1'
    assert df.columns == ["0", "1"]
    # Check values
    expected = pd.DataFrame({
        "0": [1, 2, 3],
        "1": [7, 8, 9]
    })
    assert df.to_pandas().equals(expected)


def test_list_of_lists_input():
    # Create a list of lists
    data = [[10, 20, 30, 40], [100, 200, 300, 400]]
    # Convert to a Polars LazyFrame
    lz = arrays_to_pl_lazyframe(data)
    assert isinstance(lz, pl.LazyFrame)
    # Collect to a Polars DataFrame
    df = lz.collect()
    # Expect 4 rows and 2 columns
    assert df.shape == (4, 2)
    # Check default column names '0' and '1'
    assert df.columns == ["0", "1"]
    # Check values
    expected = pd.DataFrame({
        "0": [10, 20, 30, 40],
        "1": [100, 200, 300, 400]
    })
    assert df.to_pandas().equals(expected)
