#!/usr/bin/env python3
"""
DSGBR test prototype - quick parameter testing tool
Test file: data/sphere_5/340/sphere_dgx_340_periodogram.npz (relative to project root)
WARNING: Never save to the original file!
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from DSGBR import dsgbr_detector
from inspection import _psd_limits_within_band

# Test data file path (relative to project root)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_FILE = os.path.join(_PROJECT_ROOT, "data/sphere_5/340/sphere_dgx_340_periodogram.npz")


def load_data(filepath=TEST_FILE):
    """Load NPZ file and extract frequencies and PSD."""
    print(f"Loading data from: {filepath}")

    data = np.load(filepath, allow_pickle=True)
    frequencies = np.asarray(data["frequencies"])
    psd = np.asarray(data["psd"])

    print(f"  Frequency range: {frequencies[0]:.6f} to {frequencies[-1]:.6f}")
    print(f"  Number of points: {len(frequencies)}")
    print(f"  PSD range: {psd.min():.2e} to {psd.max():.2e}")

    # Load existing peaks if available
    existing_peaks = None
    if "peak_frequencies" in data:
        existing_peaks = np.asarray(data["peak_frequencies"])
        print(f"  Existing peaks detected: {len(existing_peaks)}")

    return frequencies, psd, existing_peaks


def test_dsgbr(frequencies, psd, **params):
    """Run DSGBR detector with specified parameters."""
    print("\nRunning DSGBR with parameters:")
    for key, value in params.items():
        print(f"  {key}: {value}")

    # Create case_info dict with parameters
    case_info = dict(params)

    # Run detector and get support series
    result = dsgbr_detector(frequencies, psd, case_info=case_info, return_support=True)
    if len(result) == 3:
        peak_freqs, peak_heights, support = result
    else:
        peak_freqs, peak_heights = result
        support = {}

    print(f"  Peaks detected: {len(peak_freqs)}")
    if len(peak_freqs) > 0:
        print(f"  First 5 peaks: {peak_freqs[:5]}")

    return peak_freqs, peak_heights, support


def plot_results(frequencies, psd, peak_freqs, peak_heights, support, title_suffix=""):
    """Create inspection-style plot showing PSD + support series + detected peaks."""

    fig, ax = plt.subplots(figsize=(14, 6))

    # Set x-limits
    x_min = 2e-4
    x_max = 1.0

    # Main PSD plot - THICKER line
    ax.loglog(frequencies, psd, color="#1f77b4", linewidth=2.5, alpha=0.9, label="Periodogram")
    ax.set_xlim(x_min, x_max)

    # Set y-limits based on PSD data BEFORE adding overlays
    y_min, y_max = _psd_limits_within_band(frequencies, psd, x_min=x_min, x_max=x_max)
    ax.set_ylim(y_min, y_max)

    # Overlay support series
    search_series = support.get("search_series", np.array([]))
    baseline_series = support.get("baseline_series", np.array([]))
    ratio_series = support.get("ratio_series", np.array([]))

    if search_series.size > 0:
        ax.loglog(
            frequencies,
            np.maximum(search_series, 1e-300),
            color="#ff8c00",
            alpha=0.7,
            linewidth=0.8,
            linestyle="--",
            label="SEARCH",
            zorder=3,
        )

    if baseline_series.size > 0:
        ax.loglog(
            frequencies,
            np.maximum(baseline_series, 1e-300),
            color="0.25",
            alpha=0.55,
            linewidth=1.0,
            linestyle=":",
            label="BASELINE",
            zorder=3,
        )

    # Add ratio on secondary axis
    ratio_ax = None
    if ratio_series.size > 0:
        ratio_ax = ax.twinx()
        ratio_ax.set_xscale("log")
        ratio_ax.set_xlim(x_min, x_max)

        # Get ratio threshold from detector config
        detector_cfg = support.get("detector_config", {})
        ratio_threshold = detector_cfg.get("ratio_threshold", 1.3)

        # Normalize ratio by threshold
        normalized_ratio = ratio_series / ratio_threshold

        ratio_ax.plot(
            frequencies,
            normalized_ratio,
            color="#b22222",
            linewidth=1.2,
            alpha=0.85,
            label="RATIO / threshold",
            zorder=4,
        )

        # Add threshold line at 1.0
        ratio_ax.axhline(1.0, color="#b22222", linewidth=1.0, alpha=0.55, linestyle=(0, (4, 2)), label="= 1", zorder=3)

        # Set ratio axis limits
        positive_ratio = normalized_ratio[normalized_ratio > 0]
        if positive_ratio.size > 0:
            ratio_min = max(positive_ratio.min() * 0.8, 1e-4)
            ratio_max = max(positive_ratio.max() * 1.2, ratio_min * 1.2)
            ratio_ax.set_yscale("log")
            ratio_ax.set_ylim(ratio_min, ratio_max)

        ratio_ax.set_ylabel("(SEARCH / BASELINE) / threshold", color="#b22222")
        ratio_ax.tick_params(axis="y", colors="#b22222")

    # Mark detected peaks
    if len(peak_freqs) > 0:
        ax.scatter(
            peak_freqs,
            peak_heights,
            facecolors="none",
            edgecolors="r",
            s=50,
            marker="o",
            linewidths=1.1,
            alpha=0.9,
            zorder=6,
        )

        # Add frequency labels for first 10 peaks
        for freq, height in zip(peak_freqs[:10], peak_heights[:10]):
            ax.text(freq, height * 1.6, f"{freq:.5f}", rotation=90, fontsize=6, va="bottom", ha="center", color="0.2")

    # Formatting
    ax.set_xlabel("Strouhal Number [St]")
    ax.set_ylabel("PSD")
    ax.set_title(f"DSGBR Test Results{title_suffix}")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()

    # Save figure instead of showing
    filename = f"test_DSGBR{title_suffix.replace(' - ', '_').replace(' ', '_')}.png"
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {filename}")


def plot_comparison(
    frequencies, psd, peak_freqs1, peak_heights1, support1, label1, peak_freqs2, peak_heights2, support2, label2
):
    """Create comparison plot with both parameter sets in the same figure."""

    # Create 2 subplots: main PSD plot and detection threshold panel
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 8), gridspec_kw={"height_ratios": [4, 1]}, sharex=True)

    # Set x-limits
    x_min = 5e-4
    x_max = 1.0

    # TOP PANEL: Main PSD plot - THICKER line
    ax1.loglog(frequencies, psd, color="black", linewidth=2.0, alpha=0.4, label="Periodogram", zorder=1)
    ax1.set_xlim(x_min, x_max)

    # Set y-limits based on PSD data BEFORE adding overlays
    y_min, y_max = _psd_limits_within_band(frequencies, psd, x_min=x_min, x_max=x_max)
    ax1.set_ylim(y_min, y_max)

    # Colors for the two parameter sets
    colors1 = {"search": "#ff4d00", "baseline": "#ff4d00", "peaks": "#ff4d00"}
    colors2 = {"search": "#0066cc", "baseline": "#0066cc", "peaks": "#0066cc"}

    # Plot support series for first parameter set - THINNER SEARCH lines
    search1 = support1.get("search_series", np.array([]))
    baseline1 = support1.get("baseline_series", np.array([]))

    if search1.size > 0:
        ax1.loglog(
            frequencies,
            np.maximum(search1, 1e-300),
            color=colors1["search"],
            alpha=0.5,
            linewidth=1.0,
            linestyle="--",
            label=f"SEARCH ({label1})",
            zorder=2,
        )

    if baseline1.size > 0:
        ax1.loglog(
            frequencies,
            np.maximum(baseline1, 1e-300),
            color=colors1["baseline"],
            alpha=1.0,
            linewidth=1.5,
            linestyle="-",
            label=f"BASELINE ({label1})",
            zorder=2,
        )

    # Plot support series for second parameter set - THINNER SEARCH lines
    search2 = support2.get("search_series", np.array([]))
    baseline2 = support2.get("baseline_series", np.array([]))

    if search2.size > 0:
        ax1.loglog(
            frequencies,
            np.maximum(search2, 1e-300),
            color=colors2["search"],
            alpha=0.5,
            linewidth=1.0,
            linestyle="--",
            label=f"SEARCH ({label2})",
            zorder=2,
        )

    if baseline2.size > 0:
        ax1.loglog(
            frequencies,
            np.maximum(baseline2, 1e-300),
            color=colors2["baseline"],
            alpha=1.0,
            linewidth=1.5,
            linestyle="-",
            label=f"BASELINE ({label2})",
            zorder=2,
        )

    # Mark detected peaks for both parameter sets
    if len(peak_freqs1) > 0:
        ax1.scatter(
            peak_freqs1,
            peak_heights1,
            facecolors="none",
            edgecolors=colors1["peaks"],
            s=60,
            marker="o",
            linewidths=1.5,
            alpha=0.8,
            zorder=5,
            label=f"Peaks ({label1}): {len(peak_freqs1)}",
        )

    if len(peak_freqs2) > 0:
        ax1.scatter(
            peak_freqs2,
            peak_heights2,
            facecolors="none",
            edgecolors=colors2["peaks"],
            s=30,
            marker="s",
            linewidths=1.2,
            alpha=0.8,
            zorder=5,
            label=f"Peaks ({label2}): {len(peak_freqs2)}",
        )

    # Top panel formatting
    ax1.set_ylabel("PSD")
    ax1.set_title("DSGBR Parameter Comparison")
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="upper right", fontsize=9, ncol=2)

    # BOTTOM PANEL: Detection threshold visualization
    # Get ratio series for both parameter sets
    ratio1 = support1.get("ratio_series", np.array([]))
    ratio2 = support2.get("ratio_series", np.array([]))

    # Get threshold values
    cfg1 = support1.get("detector_config", {})
    cfg2 = support2.get("detector_config", {})
    threshold1 = cfg1.get("ratio_threshold", 1.3)
    threshold2 = cfg2.get("ratio_threshold", 1.3)

    # Plot normalized ratios (SEARCH/BASELINE)/threshold
    if ratio1.size > 0:
        normalized_ratio1 = ratio1 / threshold1
        ax2.semilogx(
            frequencies, normalized_ratio1, color=colors1["search"], linewidth=1.5, alpha=0.8, label=f"Ratio ({label1})"
        )

    if ratio2.size > 0:
        normalized_ratio2 = ratio2 / threshold2
        ax2.semilogx(
            frequencies, normalized_ratio2, color=colors2["search"], linewidth=1.5, alpha=0.8, label=f"Ratio ({label2})"
        )

    # Add detection threshold line at y=1 with clearer labeling
    ax2.axhline(y=1.0, color="red", linestyle="-", linewidth=2.0, alpha=0.8, label="Detection Threshold")

    # Apply symlog scale to x-axis only to space out dense St > 0.1 region
    ax2.set_xscale("symlog", linthresh=0.04, linscale=0.3)

    # - Linear threshold: This is the transition point between linear and logarithmic scaling
    # - For St < 0.02: Uses linear scale (maintains good resolution for low frequencies)
    # - For St > 0.02: Uses logarithmic scale (spreads out the  dense peak region)
    # - You chose 0.02 which means the linear region goes up to St=0.02, then log scale takes over
    # - Linear scale factor: Controls how abruptly the transition happens between linear and log scales
    # - Higher values = more gradual/softer transition - Lower values = sharper transition
    # - Range is typically 0.1 to 10, with 1 being a moderate transition
    #   How they work together:
    # - St < 0.02: Linear spacing (equal distance between equal St  values)
    # - St > 0.02: Logarithmic spacing (more space between higher St values)
    # - Around St = 0.02: Smooth transition between the two  scaling modes

    # Add background shading to show detection zones
    ax2.axhspan(1.0, 10, color="lightgreen", alpha=0.15, label="DETECTED (≥1)")
    ax2.axhspan(0.1, 1.0, color="lightcoral", alpha=0.15, label="REJECTED (<1)")

    # Fill areas above threshold for both parameter sets
    if ratio1.size > 0:
        normalized_ratio1 = ratio1 / threshold1
        ax2.fill_between(
            frequencies,
            normalized_ratio1,
            1,
            where=(normalized_ratio1 >= 1),
            color=colors1["search"],
            alpha=0.3,
            interpolate=True,
            label=f"Detected ({label1})",
        )

    if ratio2.size > 0:
        normalized_ratio2 = ratio2 / threshold2
        ax2.fill_between(
            frequencies,
            normalized_ratio2,
            1,
            where=(normalized_ratio2 >= 1),
            color=colors2["search"],
            alpha=0.3,
            interpolate=True,
            label=f"Detected ({label2})",
        )

    # Bottom panel formatting
    ax2.set_xlim(x_min, x_max)
    ax2.set_ylim(0.1, 10)
    ax2.set_yscale("log")
    ax2.set_xlabel("St")
    ax2.set_ylabel("Detection Ratio\n(SEARCH/BASELINE)/threshold")
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="upper right", fontsize=8)

    plt.tight_layout()

    # Save comparison plot
    filename = "DSGBR_test.png"
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {filename}")


def run_dsgbr_single(frequencies, psd, params):
    """Run DSGBR detector with given parameters and return peak info."""
    case_info = dict(params)

    result = dsgbr_detector(frequencies, psd, case_info=case_info, return_support=True)
    if len(result) == 3:
        peak_freqs, peak_heights, support = result
    else:
        peak_freqs, peak_heights = result
        support = {}

    return {"peak_freqs": peak_freqs, "peak_heights": peak_heights, "support": support, "n_peaks": len(peak_freqs)}


def test_parameter_sensitivity(frequencies, psd, param_name, param_values, base_params):
    """Test sensitivity of one parameter across a range of values."""
    print(f"\n{'=' * 60}")
    print(f"TESTING {param_name.upper()} SENSITIVITY")
    print(f"{'=' * 60}")
    print(f"Testing {len(param_values)} values: {param_values[0]:.3f} to {param_values[-1]:.3f}")

    results = []

    for i, value in enumerate(param_values):
        test_params = base_params.copy()
        test_params[param_name] = value

        # Handle special case for smooth_window (must be odd and >= 3)
        if param_name == "smooth_window":
            value = int(value)
            if value % 2 == 0:
                value += 1  # Make odd
            value = max(3, value)  # Minimum value
            test_params[param_name] = value

        result = run_dsgbr_single(frequencies, psd, test_params)
        results.append(
            {
                "param_value": value,
                "n_peaks": result["n_peaks"],
                "peak_freqs": result["peak_freqs"],
                "peak_heights": result["peak_heights"],
                "support": result["support"],
            }
        )

        print(f"  {param_name}={value:.3f}: {result['n_peaks']} peaks")

    return results


def create_unified_sensitivity_plot_27(frequencies, psd, param_ranges, base_params):
    """Create 27-variation sensitivity plot with y-shifted overlays in each subplot."""

    # Create figure with 3 rows x 3 columns
    fig = plt.figure(figsize=(16, 12))

    # Set x-limits for all PSD plots
    x_min, x_max = 5e-4, 0.06
    y_min, y_max = _psd_limits_within_band(frequencies, psd, x_min=x_min, x_max=x_max)

    # Extract parameter values
    sw_values = param_ranges["smooth_window"]
    bwf_values = param_ranges["baseline_window_frac"]
    rt_values = param_ranges["ratio_threshold"]

    # Colors for ratio_threshold variations
    rt_colors = ["#2E8B57", "#FF6B35", "#1E90FF"]
    rt_labels = [f"RT={rt:.2f}" for rt in rt_values]

    # Layout: rows=smooth_window, columns=baseline_window_frac
    for row, sw in enumerate(sw_values):
        for col, bwf in enumerate(bwf_values):
            ax = plt.subplot(3, 3, row * 3 + col + 1)

            # Plot original PSD for reference
            ax.loglog(frequencies, psd, color="black", linewidth=1.0, alpha=0.2, label="PSD")
            ax.set_xlim(x_min, x_max)

            peak_counts = []

            # Overlay 3 ratio_threshold variations with y-shifts
            for i, rt in enumerate(rt_values):
                # Set up parameters for this combination
                params = base_params.copy()
                params["smooth_window"] = int(sw)
                params["baseline_window_frac"] = bwf
                params["ratio_threshold"] = rt

                # Run DSGBR with these parameters
                result = run_dsgbr_single(frequencies, psd, params)
                peak_counts.append(result["n_peaks"])

                # Apply y-shift for visibility
                shift_factor = 10 ** (i * 3.5)  # Larger shifts: 1x, ~3162x, ~10^7x
                shifted_psd = psd * shift_factor

                # Plot shifted PSD
                ax.loglog(frequencies, shifted_psd, color=rt_colors[i], alpha=0.3, linewidth=1.0, label=rt_labels[i])

                # Plot support series with same y-shift
                support = result["support"]
                search_series = support.get("search_series", np.array([]))
                baseline_series = support.get("baseline_series", np.array([]))

                if search_series.size > 0:
                    shifted_search = np.maximum(search_series, 1e-300) * shift_factor
                    ax.loglog(
                        frequencies,
                        shifted_search,
                        color=rt_colors[i],
                        alpha=0.6,
                        linewidth=0.8,
                        linestyle="--",
                    )

                if baseline_series.size > 0:
                    shifted_baseline = np.maximum(baseline_series, 1e-300) * shift_factor
                    ax.loglog(
                        frequencies,
                        shifted_baseline,
                        color=rt_colors[i],
                        alpha=0.8,
                        linewidth=1.2,
                        linestyle=":",
                    )

                # Plot detected peaks with same y-shift
                if len(result["peak_freqs"]) > 0:
                    shifted_peaks = result["peak_heights"] * shift_factor
                    ax.scatter(
                        result["peak_freqs"],
                        shifted_peaks,
                        facecolors="none",
                        edgecolors=rt_colors[i],
                        s=20,
                        marker="o",
                        linewidths=1.0,
                        alpha=0.9,
                    )

            # Update y-limits to accommodate larger shifts
            ax.set_ylim(y_min, y_max * 10000000)

            # Formatting
            ax.grid(True, alpha=0.2)

            # Title showing parameter values and peak counts in single line
            title = f"SW={int(sw)}, BWF={bwf:.3f} | Peaks: {'/'.join(map(str, peak_counts))}"
            ax.set_title(title, fontsize=9)

            # Add labels
            if row == 2:  # Bottom row
                ax.set_xlabel("St", fontsize=10)
            if col == 0:  # Left column
                ax.set_ylabel("PSD (shifted)", fontsize=10)

            # Remove tick markers from internal plots to save space
            if row != 2:  # Not bottom row
                ax.set_xticklabels([])
            if col != 0:  # Not left column
                ax.set_yticklabels([])

            # Add legend to each subplot showing parameter combinations
            # Reorder legend to match visual plot order: blue, orange, green
            legend_order = [2, 1, 0]  # Reorder to match visual appearance
            legend_labels = []
            for idx in legend_order:
                rt = rt_values[idx]
                legend_labels.append(f"SW={int(sw)}, BWF={bwf:.3f}, RT={rt:.2f}")

            # Create custom legend with color coding
            from matplotlib.lines import Line2D

            legend_elements = [
                Line2D([0], [0], color=rt_colors[legend_order[i]], lw=2, label=legend_labels[i])
                for i in range(len(rt_values))
            ]
            ax.legend(
                handles=legend_elements, loc="upper left", fontsize=7, framealpha=0.8, bbox_to_anchor=(0.02, 0.98)
            )

    # Overall title
    fig.suptitle(
        "DSGBR 27-Variation Sensitivity Analysis\n(3 Ratio Thresholds per subplot with y-shifts)", fontsize=14, y=0.95
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.93])

    # Save figure
    filename = "DSGBR_sensitivity_27.png"
    plt.savefig(filename, dpi=500, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {filename}")

    return filename


if __name__ == "__main__":
    # Load test data
    frequencies, psd, existing_peaks = load_data()

    print("\n" + "=" * 60)
    print("TESTING DEFAULT PARAMETERS")

    default_params = {
        "smooth_window": 3,  # SW # Smaller window - higher resolution
        "baseline_window_frac": 0.001,  # BWF # Faster baseline adaptation
        "ratio_threshold": 1.8,  # RT # Lower threshold - more sensitive
        "max_peaks": 5000,  # Allow more peaks
    }

    peak_freqs, peak_heights, support = test_dsgbr(frequencies, psd, **default_params)

    print("\n" + "=" * 60)
    print("TESTING CUSTOM PARAMETERS")

    custom_params = {
        "smooth_window": 5,  # SW # Smaller window - higher resolution
        "baseline_window_frac": 0.002,  # BWF # Faster baseline adaptation
        "ratio_threshold": 1.8,  # RT # Lower threshold - more sensitive
        "max_peaks": 5000,  # Allow more peaks
    }

    peak_freqs_custom, peak_heights_custom, support_custom = test_dsgbr(frequencies, psd, **custom_params)

    # Compare with existing peaks
    if existing_peaks is not None:
        print(f"  Default DSGBR: {len(peak_freqs)}")
        print(f"  Custom DSGBR: {len(peak_freqs_custom)}")

    # Create comparison plot
    plot_comparison(
        frequencies,
        psd,
        peak_freqs,
        peak_heights,
        support,
        "Default",
        peak_freqs_custom,
        peak_heights_custom,
        support_custom,
        "Custom",
    )

    # # Now run sensitivity analysis
    # print("\n" + "=" * 60)
    # print("SENSITIVITY ANALYSIS")
    # print("=" * 60)

    # # Base configuration for sensitivity testing
    # base_params = {
    #     "smooth_window": 9,
    #     "baseline_window_frac": 0.01,
    #     "ratio_threshold": 1.3,
    #     "max_peaks": 5000,
    # }

    # print(f"\nBase parameters: {base_params}")

    # # Define parameter ranges for sensitivity testing - 3 values each
    # param_ranges = {
    #     "smooth_window": [3, 4, 5],
    #     "baseline_window_frac": [0.001, 0.01, 0.1],
    #     "ratio_threshold": [1.0, 1.8, 2.5],
    # }

    # # Run sensitivity analysis for each parameter and collect results
    # all_results = {}
    # for param_name, param_values in param_ranges.items():
    #     results = test_parameter_sensitivity(frequencies, psd, param_name, param_values, base_params)
    #     all_results[param_name] = results

    # # Create 27-variation sensitivity plot
    # create_unified_sensitivity_plot_27(frequencies, psd, param_ranges, base_params)
