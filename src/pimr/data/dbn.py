"""Loader for Databento DBN files (.dbn.zst) via the databento-dbn package.

databento-dbn is only a decoder: it yields one Metadata object plus one record
object per book event (MBP1Msg for the mbp-1 schema). It has no OHLC/candle
aggregation or plotting helpers -- Databento serves ready-made candles as a
separate schema (e.g. 'ohlcv-1h') at download time. Candles are therefore
built locally with pimr.data.bars.build_ohlc.

Requires the optional `databento-dbn` package (not needed by the rest of
pimr.data). Extracted from tests/dbn_data.py.
"""

import numpy as np
from databento_dbn import Compression, DBNDecoder, Metadata, MBP1Msg, FIXED_PRICE_SCALE, UNDEF_PRICE

PRICE_SCALE = 1.0 / FIXED_PRICE_SCALE  # fixed-point int -> dollars
CHUNK_SIZE = 1 << 22                   # 4 MiB of compressed bytes per decoder write


def load_dbn(path, max_rows=None):
    """
    Load one MBP-1 .dbn.zst file into numpy arrays.

    The file is streamed through DBNDecoder in CHUNK_SIZE pieces, so the whole
    (much larger) decompressed stream is never held in memory at once. The
    output layout matches pimr.data.glbx_csv.load_GLBX, plus the instrument id
    of every event and the id -> raw-symbol map from the file metadata (needed
    because parent symbology like 'GC.FUT' mixes many contracts in one file).

    Args:
        path (str): Path to a .dbn.zst file with the mbp-1 schema.
        max_rows (int or None): Stop after this many events (None = all).

    Returns:
        dict:
            - "ts" (int64): event timestamps (ts_event) in ns since the epoch.
            - "action" (str): event type per row ('T' = trade, 'A' = add, ...).
            - "price" (float64): event price in dollars, NaN if undefined.
            - "size" (float64): event size in contracts.
            - "mid" (float64): top-of-book mid price in dollars, NaN where
              either side of the book is empty.
            - "instrument_id" (int64): instrument id per event.
            - "symbols" (dict): instrument_id -> raw symbol (e.g. 'GCZ6').
        All arrays have equal length and are sorted by time.
    """
    decoder = DBNDecoder(compression=Compression.ZSTD)

    ts_list, action_list, price_list = [], [], []
    size_list, mid_list, iid_list = [], [], []
    symbols = {}
    done = False

    with open(path, "rb") as f:
        while not done:
            chunk = f.read(CHUNK_SIZE)
            if not chunk:
                break
            for rec in decoder.write_and_decode(chunk):
                if isinstance(rec, MBP1Msg):
                    ts_list.append(rec.ts_event)
                    action_list.append(rec.action)
                    px = rec.price
                    price_list.append(px * PRICE_SCALE if px != UNDEF_PRICE else np.nan)
                    size_list.append(rec.size)
                    bid, ask = rec.bid_px_00, rec.ask_px_00
                    mid_list.append((bid + ask) / 2 * PRICE_SCALE
                                    if bid != UNDEF_PRICE and ask != UNDEF_PRICE
                                    else np.nan)
                    iid_list.append(rec.instrument_id)
                    if max_rows is not None and len(ts_list) >= max_rows:
                        done = True
                        break
                elif isinstance(rec, Metadata):
                    # mappings: raw symbol -> [{start_date, end_date, symbol}, ...]
                    # where 'symbol' is the instrument id as a string
                    for raw_symbol, intervals in rec.mappings.items():
                        for interval in intervals:
                            if interval["symbol"].isdigit():
                                symbols[int(interval["symbol"])] = raw_symbol

    ts = np.array(ts_list, dtype=np.int64)
    order = np.argsort(ts, kind="stable")
    return {
        "ts": ts[order],
        "action": np.array(action_list, dtype=str)[order],
        "price": np.array(price_list, dtype=np.float64)[order],
        "size": np.array(size_list, dtype=np.float64)[order],
        "mid": np.array(mid_list, dtype=np.float64)[order],
        "instrument_id": np.array(iid_list, dtype=np.int64)[order],
        "symbols": symbols,
    }


def load_dbn_many(paths, trades_only=False):
    """
    Load several DBN files and concatenate them in time.

    Args:
        paths (list of str): Paths to .dbn.zst files with the mbp-1 schema.
        trades_only (bool): If True, keep only trade events (action == 'T')
            of each file before concatenating, which saves memory when only
            candles are needed. The "action" key is dropped in that case.

    Returns:
        dict: Same layout as load_dbn, with the "symbols" maps of all files
        merged. Arrays are the per-file arrays concatenated in file order
        (pass paths sorted by date to keep the result chronological).
    """
    keys = ("ts", "action", "price", "size", "mid", "instrument_id")
    parts, symbols = [], {}
    for p in paths:
        part = load_dbn(p)
        symbols.update(part["symbols"])
        if trades_only:
            is_trade = part["action"] == "T"
            part = {key: part[key][is_trade] for key in keys if key != "action"}
        parts.append(part)

    out_keys = tuple(k for k in keys if not (trades_only and k == "action"))
    data = {key: np.concatenate([part[key] for part in parts]) for key in out_keys}
    data["symbols"] = symbols
    return data


def most_traded_instrument(instrument_id, symbols=None):
    """
    Find the instrument with the most events (the front-month contract).

    Parent symbology ('GC.FUT') mixes every outright and calendar spread in
    one file; plots and per-contract statistics normally want only the most
    traded one. Pass trade events only to select by trade count.

    Args:
        instrument_id (numpy.ndarray): Instrument id per event (int64).
        symbols (dict or None): instrument_id -> raw symbol map, as returned
            by load_dbn. Used only for the returned symbol name.

    Returns:
        tuple: (front_id, front_symbol, count) — the winning instrument id,
        its raw symbol (str(front_id) if unknown), and its event count.
    """
    ids, counts = np.unique(instrument_id, return_counts=True)
    front_id = int(ids[np.argmax(counts)])
    front_symbol = (symbols or {}).get(front_id, str(front_id))
    return front_id, front_symbol, int(counts.max())
