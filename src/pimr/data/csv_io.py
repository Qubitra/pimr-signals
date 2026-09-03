"""CSV persistence for dict-of-arrays data (L1 books, bar tables).

Extracted from tests/csv_manager.py.
"""

import csv

import numpy as np


def save_csv(data, path="synthetic_l1_book.csv"):
    """
    Save a data to a CSV file, with one header row per event, naming the columns.
    Parameters
    ----------
    data : dict of numpy.ndarray
        Price data (L1 books).
    path : str
        Destination path of the CSV file.
    """
    columns = list(data.keys())
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(zip(*(data[c] for c in columns)))


def load_csv(path="l1_book.csv"):
    """
    Read a CSV data into a dict of numpy arrays.
    Parameters
    ----------
    path : str
        Path of the CSV file to read. The first row must be a header naming the columns.
    Returns
    -------
    dict of numpy.ndarray
        One float64 array per column; ``bid_size`` and ``ask_size`` are cast back to integers when present.
    Raises
    ------
    FileNotFoundError
        If there is no file at ``path``.
    """
    try:
        with open(path, newline="") as f:
            columns = f.readline().strip().split(",")
    except FileNotFoundError:
        raise FileNotFoundError(
            f"{path} not found"
        ) from None

    data = np.loadtxt(path, delimiter=",", skiprows=1, unpack=True, ndmin=2)
    book = dict(zip(columns, data))

    for key in ("bid_size", "ask_size"):
        if key in book:
            book[key] = book[key].astype(np.int64)
    return book
