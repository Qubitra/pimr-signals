"""NumPy-based spectral analysis and frequency-based forecasting."""

import numpy as np

def spectrogram(x, window_size, hop_size, sample_rate=1.0):
    if window_size < 8:
        raise ValueError("window size must be bigger equal to 8")
    window = np.hanning(window_size)
    # window = np.ones(window_size)
    starts = np.arange(0, len(x) - window_size + 1, hop_size)

    power_frames = []
    for start in starts:
        frame = x[start : start + window_size]
        frame = frame - frame.mean()
        spectrum = np.fft.rfft(frame * window)
        power_frames.append(np.abs(spectrum) ** 2)

    power = np.array(power_frames).T
    frequencies = np.fft.rfftfreq(window_size, d=1 / sample_rate)
    times = (starts + (window_size - 1) / 2) / sample_rate

    return frequencies, times, power



def spectral_forecast(window, horizon=1, n_peaks=1, sample_rate=1.0):
    window = np.asarray(window, dtype=float)
    n = len(window)
    if n < 8:
        raise ValueError("window must contain at least 8 samples")

    # Find candidate frequencies from the window's spectrum.
    tapered = (window - window.mean()) * np.hanning(n)
    power = np.abs(np.fft.rfft(tapered)) ** 2
    frequencies = np.fft.rfftfreq(n, d=1.0 / sample_rate)

    # Ignore DC. Choose strong peaks while avoiding adjacent bins
    # that likely belong to the same spectral peak.
    ranked_bins = np.argsort(power[1:])[::-1] + 1
    selected_bins = []
    for bin_index in ranked_bins:
        if all(abs(bin_index - chosen) >= 2 for chosen in selected_bins):
            selected_bins.append(bin_index)
        if len(selected_bins) == n_peaks:
            break

    selected_freqs = frequencies[selected_bins]

    # Fit an offset plus sine/cosine terms on the observed window.
    sample_times = np.arange(n) / sample_rate
    columns = [np.ones(n)]
    for frequency in selected_freqs:
        phase = 2 * np.pi * frequency * sample_times
        columns.extend((np.cos(phase), np.sin(phase)))

    design = np.column_stack(columns)
    beta = np.linalg.lstsq(design, window, rcond=None)[0]

    # Evaluate the same fitted waves at future sample times.
    future_times = (n - 1 + np.arange(1, horizon + 1)) / sample_rate
    future_design = [np.ones(horizon)]
    for frequency in selected_freqs:
        phase = 2 * np.pi * frequency * future_times
        future_design.extend((np.cos(phase), np.sin(phase)))

    future_design = np.column_stack(future_design)
    prediction = future_design @ beta
    return prediction, selected_freqs

