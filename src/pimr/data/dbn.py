"""
Loader for Databento DBN files (.dbn.zst) via the databento-dbn package.

databento-dbn is only a decoder: it yields trades schema Metadata object which contains 
carries one TradeMsg per executed trade and nothing else (no book updates). It has no 
OHLC/candle aggregation or plotting helpers. For now there is no supprot for mbp-1 files 
because of efficiency.

Requires the optional `databento-dbn` package (not needed by the rest of
pimr.data).
"""


import numpy as np
from databento_dbn import Compression, DBNDecoder, Metadata, TradeMsg, FIXED_PRICE_SCALE, UNDEF_PRICE

PRICE_SCALE = 1.0 / FIXED_PRICE_SCALE  # fixed-point int -> dollars
CHUNK_SIZE = 1 << 22                   # 4 MiB of compressed bytes per decoder write
SCHEMA = "trades"

ARRAY_KEYS = ("ts", "price", "size", "side", "instrument_id")


def load_dbn(path, max_rows=None, most_traded=True):
    """
    Load one trades-schema .dbn.zst file into numpy arrays.
    The file is streamed through DBNDecoder in CHUNK_SIZE pieces, so the
    decompressed stream is never held in memory at once. The file metadata supplies the instrument id -> raw symbol map (needed because parent symbology like 'GC.FUT' mixes every outright and calendar spread in one file. Spread prices can be negative).

    Args:
        path (str): Path to a .dbn.zst file with the trades schema.
        max_rows (int or None): Stop after this many trades (None = all).

    Returns:
        dict:
            - "ts" (int64): trade timestamps (ts_event) in ns since the epoch.
            - "price" (float64): trade price in dollars, NaN if undefined.
            - "size" (float64): trade size in contracts.
            - "side" (str): aggressor side per trade as reported by Databento:'B' = buy aggressor (bid side initiated), 'A' = sell aggressor (ask side initiated), 'N' = not specified.
            - "instrument_id" (int64): instrument id per trade.
            - "symbols" (dict): instrument_id -> raw symbol (e.g. 'GCZ6').
        All arrays have equal length and are sorted by time.

    Raises:
        ValueError: If the file metadata reports a schema other than 'trades' e.g. an mbp-1 metadata file.
    """
    decoder = DBNDecoder(compression=Compression.ZSTD)

    ts_list, price_list, iid_list = [], [], []
    size_list, side_list = [], []
    symbols = {}
    done = False
    with open(path, "rb") as f:
        while not done:
            chunk = f.read(CHUNK_SIZE)
            if not chunk:
                break
            for rec in decoder.write_and_decode(chunk):
                if isinstance(rec, TradeMsg):
                    ts_list.append(rec.ts_event)
                    px = rec.price
                    price_list.append(px * PRICE_SCALE if px != UNDEF_PRICE else np.nan)
                    size_list.append(rec.size)
                    side_list.append(rec.side)
                    iid_list.append(rec.instrument_id)
                    if max_rows is not None and len(ts_list) >= max_rows:
                        done = True
                        break
                elif isinstance(rec, Metadata): # mappings: raw symbol -> [{start_date, end_date, symbol}, ...] where 'symbol' is the instrument id as a string
                    for raw_symbol, intervals in rec.mappings.items():
                        for interval in intervals:
                            instrument = interval["symbol"]
                            if instrument.isdigit():
                                symbols[int(instrument)] = raw_symbol
    
    
    ts = np.array(ts_list, dtype=np.int64)
    price = np.array(price_list, dtype=np.float64)
    instrument_id = np.array(iid_list, dtype=np.int64)
    sizes = np.array(size_list, dtype=np.float64)
    side = np.array(side_list, dtype=str)

    if most_traded and instrument_id.size:
        trade_ids, counts = np.unique(instrument_id, return_counts=True)
        main_id = trade_ids[np.argmax(counts)]
        keep = instrument_id == main_id
        
        ts = ts[keep]
        price = price[keep]
        instrument_id = instrument_id[keep]
        sizes = sizes[keep]
        side = side[keep]

    order = np.argsort(ts, kind="stable")

    return {
        "ts": ts[order],
        "price": price[order],
        "side": side[order],
        "size": sizes[order],
        "instrument_id": instrument_id[order],
        "symbols": symbols,
    }



def load_dbn_batch(paths):
    """
    Load several trades-schema DBN files and concatenate them in time.

    Args:
        paths (list of str): Paths to .dbn.zst files with the trades schema.

    Returns:
        dict: Same layout as load_trades, with the "symbols" maps of all files
        merged. Arrays are the per-file arrays concatenated in file order
        (pass paths sorted by date to keep the result chronological).

    Raises:
        ValueError: If paths is empty.
    """
    
    paths = list(paths)
    if not paths:
        raise ValueError("paths is empty: no .dbn.zst files to load")

    parts, symbols = [], {}
    for p in paths:
        part = load_dbn(p)
        symbols.update(part["symbols"])
        parts.append(part)

    data = {key: np.concatenate([part[key] for part in parts]) for key in ARRAY_KEYS}
    data["symbols"] = symbols
    return data


## Warning[2026-09-29]: most_traded_instrument() will be deprecated and removed in a future updates. The load_dbn() has a most_traded option. 
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
