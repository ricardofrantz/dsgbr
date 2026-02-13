# Dual Savitzky–Golay Baseline Ratio (DSGBR) Peak Detection Method

## Abstract

The Dual Savitzky–Golay Baseline Ratio (DSGBR) method is a novel spectral peak detection algorithm designed for robust identification of spectral peaks in power spectral density (PSD) data, particularly for quasi-periodic dynamical systems. DSGBR employs a two-stage Savitzky–Golay filtering approach: first applying compact smoothing to preserve peak characteristics while reducing noise (SEARCH series), then applying broader smoothing to estimate a local baseline (BASELINE series). Peaks are identified where the ratio SEARCH/BASELINE exceeds a configurable threshold, enabling robust detection in dense spectral combs and noisy environments while preserving peak height, width, and position accuracy better than conventional moving-average or prominence-based methods.

Key advantages include: (1) superior peak shape preservation through polynomial-based smoothing, (2) adaptive baseline estimation that follows slow spectral variations, (3) robust performance in dense frequency combs typical of quasi-periodic systems, (4) configurable parameters for different signal characteristics, and (5) integrated guardrails for ultra-low-frequency artifacts and frequency-aware spacing control.

## Theoretical Foundation

### Mathematical Formulation

Given a power spectral density P(f) at frequencies f, DSGBR constructs two filtered series:

1. **SEARCH Series Construction:**
   ```
   S(f) = SG[log₁₀(P(f) + ε), w₁, p₁]
   S(f) = 10^S(f)  (if filtering on log scale)
   ```
   where SG[·, w, p] denotes Savitzky–Golay filtering with window size w and polynomial order p, and ε is a small constant (typically 10⁻³⁰⁰) to handle zero values.

2. **BASELINE Series Construction:**
   ```
   B(f) = SG[log₁₀(S(f) + ε), w₂, p₂]
   B(f) = 10^B(f)  (if filtering on log scale)
   ```
   where w₂ >> w₁ to capture slow baseline variations.

3. **Peak Acceptance Criterion:**
   ```
   Peak at f* is accepted if:
   - S(f*) is a local maximum
   - R(f*) = S(f*)/B(f*) ≥ r_threshold
   - Spacing constraints are satisfied
   ```

### Theoretical Justification

The dual-filtering approach addresses fundamental challenges in spectral peak detection:

**Peak Shape Preservation:** Savitzky–Golay filters fit local polynomials via least squares, preserving moments of the underlying signal better than moving averages. Operating on log₁₀(PSD) linearizes exponential decay patterns common in spectral data, making polynomial fits more accurate.

**Adaptive Baseline Estimation:** The two-scale approach separates concerns: the SEARCH series maintains peak fidelity while the BASELINE series tracks slowly varying spectral structure (noise floor variations, broad resonances). This is particularly important for quasi-periodic systems where spectral energy varies significantly across frequency ranges.

**Ratio-Based Detection:** The ratio R(f) = S(f)/B(f) provides a local signal-to-noise estimate that adapts to varying baseline levels. Unlike fixed thresholds on absolute amplitude, this ratio criterion maintains consistent detection sensitivity across different spectral regions.

**Dense Comb Resolution:** Quasi-periodic systems often exhibit dense frequency combs (fundamental frequencies and their linear combinations). DSGBR's compact SEARCH filtering preserves individual peak resolution while the adaptive baseline prevents threshold drift in crowded spectral regions.

## Algorithm Description

### Core Algorithm

```pseudocode
ALGORITHM: DSGBR Peak Detection
INPUT: frequencies[], psd[], config
OUTPUT: peak_frequencies[], peak_heights[]

1. SEARCH SERIES CONSTRUCTION:
   IF config.smooth_on_log:
       search_data = log10(psd + 1e-300)
   ELSE:
       search_data = psd

   search_series = savgol_filter(search_data,
                                 window=config.smooth_window,
                                 polyorder=config.smooth_polyorder)

   IF config.smooth_on_log:
       search_series = 10^search_series

2. BASELINE SERIES CONSTRUCTION:
   IF config.baseline_on_log:
       baseline_data = log10(search_series + 1e-300)
   ELSE:
       baseline_data = search_series

   baseline_window = determine_baseline_window(search_series, config)
   baseline_series = savgol_filter(baseline_data,
                                   window=baseline_window,
                                   polyorder=2)

   IF config.baseline_on_log:
       baseline_series = 10^baseline_series

3. RATIO COMPUTATION AND CANDIDATE DETECTION:
   ratio_series = search_series / max(baseline_series, 1e-300)
   candidates = find_local_maxima(search_series)

4. RATIO THRESHOLD FILTERING:
   accepted = []
   FOR each candidate c in candidates:
       IF ratio_series[c] >= config.ratio_threshold:
           accepted.append(c)

5. SPACING ENFORCEMENT:
   final_peaks = apply_spacing_rules(accepted, frequencies, config)

6. ULTRA-LOW-FREQUENCY GUARDRAIL:
   final_peaks = apply_ulf_guardrail(final_peaks, frequencies,
                                     search_series, config)

7. BAND-BALANCED SELECTION:
   IF len(final_peaks) > config.max_peaks:
       final_peaks = select_by_frequency_bands(final_peaks, config)

8. RETURN peak arrays sorted by amplitude
```

### Inspection Overlays

QA inspection figures now expose the three series that drive DSGBR decisions:

- **SEARCH** — the smoothed detection signal scanned for local maxima (log-log axes).
- **BASELINE** — the broader floor estimate that adjusts to slow spectral trends.
- **RATIO / threshold** — the SEARCH/BASELINE curve normalised by `ratio_threshold`, plotted on a dedicated log-scaled right axis with a reference line at 1 so passes/fails are obvious.

Together these overlays make the acceptance rule `SEARCH / BASELINE ≥ ratio_threshold` explicit: peaks appear where the RATIO curve rises above the horizontal ratio threshold while SEARCH still aligns with the PSD apex.

### Parameter Determination

**Baseline Window Selection:**
```
IF config.baseline_window is specified:
    window = config.baseline_window
ELSE IF config.baseline_window_frac is specified:
    window = max(7, round(len(psd) * config.baseline_window_frac))
ELSE:
    window = max(15, (len(psd) // 200) * 2 + 1)
```

**Frequency-Aware Spacing:**
```
FOR each peak at frequency f:
    IF f >= config.switch_frequency:
        min_distance = config.distance_high
    ELSE:
        min_distance = config.distance_low
```

## Comparative Analysis

### Advantages Over Existing Methods

**vs. Simple Prominence Detection:**
- DSGBR's adaptive baseline follows spectral structure, while fixed prominence assumes constant noise floor
- Better handling of varying signal-to-noise ratios across frequency ranges
- Superior performance in dense frequency combs

**vs. Wavelet-Based Methods:**
- More computationally efficient for real-time applications
- Fewer parameters requiring expert tuning
- Direct interpretability of threshold parameters

**vs. Moving Average Smoothing:**
- Savitzky–Golay preserves peak width and height more accurately
- Better frequency resolution in dense spectral regions
- Reduced phase distortion compared to IIR-based approaches

**vs. Derivative-Based Methods:**
- More robust to noise (operates on smoothed data)
- No requirement for secondary derivative calculations
- Better handling of asymmetric peaks

### Performance Characteristics

**Quasi-Periodic System Optimization:**
- Designed specifically for systems with multiple incommensurate fundamental frequencies
- Handles linear combination frequencies (f₁ ± f₂, 2f₁ ± f₂, etc.)
- Maintains resolution in dense spectral combs around fundamental frequencies

**Computational Complexity:**
- O(N log N) for Savitzky-Golay filtering operations
- O(N) for peak detection and post-processing
- Total: O(N log N) where N is spectrum length

## Parameter Guide

### Core Parameters

| Parameter | Physical Interpretation | Typical Range | Selection Guidelines |
|-----------|------------------------|---------------|---------------------|
| `smooth_window` | Peak resolution scale | 5-15 | Smaller for higher frequency resolution, larger for noise reduction |
| `baseline_window_frac` | Baseline tracking scale | 0.005-0.02 | Smaller for faster baseline adaptation, larger for stability |
| `ratio_threshold` | Detection sensitivity | 1.1-1.8 | Lower for high sensitivity, higher for specificity |
| `switch_frequency` | Dense/sparse transition | 0.01-0.05 | Based on fundamental frequency locations |
| `distance_low/high` | Spacing constraints | 1-5 bins | Based on expected peak density |

### Default Parameter Configuration

The DSGBR test framework (`DSGBR_test.py`) uses optimized default parameters that have been validated for quasi-periodic dynamical systems:

```python
default_params = {
    "smooth_window": 3,        # SW: Smaller window - higher resolution
    "baseline_window_frac": 0.001,  # BWF: Faster baseline adaptation
    "ratio_threshold": 1.8,    # RT: Lower threshold - more sensitive
    "max_peaks": 5000,         # Allow more peaks for dense spectral analysis
}
```

**Parameter Abbreviations (for easy reference):**
- **SW** = `smooth_window`: Controls peak resolution vs. noise tradeoff
- **BWF** = `baseline_window_frac`: Controls baseline adaptation speed
- **RT** = `ratio_threshold`: Controls detection sensitivity vs. specificity

**Rationale for Default Values:**
- **SW = 3**: Very compact window for maximum frequency resolution in dense spectral combs
- **BWF = 0.001**: Extremely responsive baseline that adapts quickly to local spectral variations
- **RT = 1.8**: Moderately sensitive threshold that captures most significant peaks while avoiding excessive noise

These parameters were determined through sensitivity analysis on quasi-periodic flow data and can be fine-tuned by users for specific applications. The sensitivity testing framework in `DSGBR_test.py` allows systematic exploration of parameter space to optimize detection for particular spectral characteristics.

### Ultra-Low-Frequency Guardrail

| Parameter | Purpose | Typical Values |
|-----------|---------|----------------|
| `ulf_fmax` | ULF band definition | 1e-3 to 1e-2 |
| `ulf_min_q` | Quality factor threshold | 5-15 |
| `ulf_max_points` | Maximum ULF peaks | 3-10 |

**Quality Factor Definition:**
Q = f_center / FWHM, where FWHM is full-width at half-maximum of the peak in the SEARCH series.

### Parameter Tuning Guidelines

**For Dense Spectral Combs:**
```
smooth_window: 5-7 (high resolution)
baseline_window_frac: 0.006-0.01 (responsive baseline)
ratio_threshold: 1.2-1.3 (moderate sensitivity)
distance_low: 1 (minimal spacing constraint)
```

**For High-SNR Sparse Spectra:**
```
smooth_window: 9-15 (more smoothing acceptable)
baseline_window_frac: 0.01-0.02 (stable baseline)
ratio_threshold: 1.4-1.6 (higher specificity)
distance_low: 2-3 (avoid close doubles)
```

**For Low-SNR Noisy Data:**
```
smooth_window: 11-15 (more noise reduction)
baseline_window_frac: 0.015-0.025 (stable baseline)
ratio_threshold: 1.1-1.3 (maintain sensitivity)
ulf_min_q: 10-15 (stricter ULF filtering)
```

## Implementation Notes

### Best Practices

1. **Window Size Selection:**
   - Ensure all windows are odd numbers (required by scipy.signal.savgol_filter)
   - Validate window < data_length before filtering
   - Consider frequency resolution requirements when setting smooth_window

2. **Numerical Stability:**
   - Add small epsilon (1e-300) before logarithm operations
   - Use maximum(baseline, epsilon) in ratio calculations to avoid division by zero
   - Handle edge cases where filtering windows exceed data length

3. **Memory Optimization:**
   - Pre-allocate arrays for intermediate results
   - Use memory-mapped arrays for large datasets
   - Consider chunked processing for extremely long time series

4. **Parameter Validation:**
   - Validate smooth_polyorder < smooth_window
   - Ensure ratio_threshold > 1.0
   - Check frequency-dependent parameters are positive

### Common Pitfalls

1. **Over-smoothing:** Large smooth_window values can merge nearby peaks
2. **Under-smoothing:** Small baseline windows cause threshold fluctuations
3. **Inappropriate scaling:** Linear-scale filtering may miss weak peaks in strong-peak presence
4. **Edge effects:** Savitzky–Golay filters have reduced accuracy near array boundaries

### Performance Optimizations

1. **Caching:** Pre-compute filter coefficients for repeated operations
2. **Vectorization:** Use numpy operations instead of Python loops
3. **Early termination:** Stop processing when max_peaks limit reached
4. **Sparse representation:** Store only peak indices/values for large spectra

## Validation and Testing

### Performance Metrics

**Peak Detection Accuracy:**
- True Positive Rate (TPR): fraction of real peaks detected
- False Positive Rate (FPR): fraction of detections that are false peaks
- Peak Position Error: RMS deviation between detected and true peak frequencies
- Peak Height Error: relative error in detected vs. true peak amplitudes

**Spectral Resolution:**
- Minimum Resolvable Separation: closest frequency spacing reliably detected
- Dynamic Range: ratio of strongest to weakest detectable peaks
- Baseline Stability: variation in detection threshold across frequency

### Test Cases

**Synthetic Validation:**
1. **Known Frequency Combs:** Generate synthetic quasi-periodic signals with known fundamental frequencies
2. **Noise Robustness:** Add controlled noise levels and measure degradation
3. **Parameter Sensitivity:** Sweep key parameters and measure stability

**Benchmark Comparisons:**
1. **Standard Datasets:** Apply to established peak detection benchmarks
2. **Cross-Validation:** Compare against expert manual peak identification
3. **Computational Performance:** Timing comparisons with alternative methods

### Robustness Analysis

**Parameter Perturbation:**
Test detection stability under ±10% parameter variations:
- ratio_threshold: Should maintain >90% peak overlap
- baseline_window_frac: Should preserve main peaks
- smooth_window: Should not introduce/remove major peaks

**Signal Conditions:**
- SNR range: 1:1 to 100:1
- Spectral density: sparse (few peaks) to dense (100+ peaks)
- Frequency range: validate across decades of frequency

## Applications and Extensions

### Primary Applications

**Quasi-Periodic Dynamical Systems:**
- Fluid dynamics: wake frequencies, vortex shedding patterns
- Mechanical vibrations: multi-mode oscillations, bearing diagnostics
- Biological rhythms: circadian patterns, neural oscillations
- Economic time series: multi-timescale cycles

**Dense Spectral Analysis:**
- Frequency combs in nonlinear optics
- Harmonic analysis in power systems
- Multi-tone communication signals
- Crystallographic peak identification

### Extensions and Variations

**Multi-Resolution DSGBR:**
Apply DSGBR at multiple scales simultaneously and combine results:
```
peaks_fine = DSGBR(psd, config_high_res)
peaks_coarse = DSGBR(psd, config_low_res)
combined_peaks = merge_multi_scale(peaks_fine, peaks_coarse)
```

**Adaptive Parameter Selection:**
Automatically tune parameters based on spectral characteristics:
- Estimate spectral density and adjust spacing parameters
- Measure noise floor and adapt ratio_threshold
- Detect fundamental frequency range and set switch_frequency

### Experimental Methodologies

Several experimental threshold computation methods have been implemented to explore alternative adaptive detection strategies. These methods preserve the core DSGBR SEARCH series while experimenting with different baseline estimation approaches.

#### Broadband Savitzky-Golay Envelope

**Concept**: Apply additional wide Savitzky-Golay smoothing (5-10× the SEARCH window) to estimate a more stable baseline floor.

**Implementation**:
```python
# In core/experimental/broadband_savgol.py
broadband_window = search_window * broadband_window_multiplier  # 7x default
broadband_baseline = savgol_filter(search_series, broadband_window, polyorder=2)
threshold = broadband_baseline  # More stable baseline estimate
```

**Advantages**:
- Maintains existing DSGBR tooling and infrastructure
- Simple one-parameter tuning (window multiplier)
- Robust to high-frequency noise while preserving peak shapes
- Computationally efficient (single additional smoothing pass)

**Typical Parameters**: `broadband_window_multiplier=7.0`, `broadband_polyorder=2`

#### Log-domain Moving Quantile

**Concept**: Work in log₁₀(PSD) domain with sliding window quantile estimation for robust local floor determination.

**Implementation**:
```python
# In core/experimental/log_quantile.py
log_search = np.log10(np.maximum(search_series, 1e-300))
quantile_floor = moving_quantile(log_search, window, quantile=0.85)
threshold = 10^(safety_factor * quantile_floor)  # Exponentiate back to linear
```

**Advantages**:
- Quantiles more robust than means for sparse spike distributions
- Log-domain processing prevents low-frequency bins from dominating
- Adaptive to local spectral characteristics
- Less sensitive to outlier peaks in baseline estimation

**Typical Parameters**: `target_quantile=0.85`, `safety_factor=1.1`, `quantile_window_frac=0.05`

#### Two-stage EMA Cascade

**Concept**: Apply fast exponential moving average for responsiveness, slow EMA for envelope tracking.

**Implementation**:
```python
# In core/experimental/ema_cascade.py
fast_ema = ema_filter(search_series, alpha_fast=0.1)    # Responsive
slow_ema = ema_filter(search_series, alpha_slow=0.01)   # Envelope
ratio = fast_ema / slow_ema  # Adaptive threshold multiplier
threshold = slow_ema * clip(ratio, min_ratio, max_ratio)
```

**Advantages**:
- Extremely low computational cost (O(N) complexity)
- Continuous adaptation to changing spectral conditions
- Tunable responsiveness through decay factors
- No window size selection required

**Typical Parameters**: `fast_ema_alpha=0.1`, `slow_ema_alpha=0.01`, `min_ema_ratio=1.0`, `max_ema_ratio=10.0`

#### Testing and Validation

All experimental methods are implemented in `core/experimental/` package with:
- Common base class for consistent interface
- Automatic fallback to original DSGBR on failure
- Extended testing framework (`DSGBR_test_experimental.py`)
- 4-panel comparison plots showing baseline vs. experimental methods

**Key Findings** (sphere_5/340 dataset):
- Baseline DSGBR: 653 peaks (potentially over-sensitive)
- Experimental methods: ~25 peaks each (more conservative)
- All methods show different trade-offs between sensitivity and specificity
- Experimental approaches provide more robust peak detection in dense spectral regions

#### Usage Example

```python
from core.experimental import BroadbandSavgolDetector, LogQuantileDetector, EMACascadeDetector

# Test broadband Savitzky-Golay
detector = BroadbandSavgolDetector()
peaks_freq, peaks_height, support = detector.detect_peaks(frequencies, psd)

# Test log-domain quantile
detector = LogQuantileDetector(target_quantile=0.9, safety_factor=1.2)
peaks_freq, peaks_height, support = detector.detect_peaks(frequencies, psd)

# Test EMA cascade
detector = EMACascadeDetector(fast_ema_alpha=0.15, slow_ema_alpha=0.02)
peaks_freq, peaks_height, support = detector.detect_peaks(frequencies, psd)
```

**Time-Varying DSGBR:**
For non-stationary signals, apply DSGBR to sliding windows:
```
FOR each time window:
    peaks_t = DSGBR(psd_window, config)
    track_peak_evolution(peaks_t)
```

**Multi-Dimensional Extension:**
Extend to 2D spectrograms or higher-dimensional spectral data:
- Apply DSGBR along frequency axis
- Use 2D Savitzky-Golay filtering for time-frequency analysis

### Integration with Downstream Analysis

**Combination Classification:**
DSGBR output integrates directly with algorithms that identify linear combinations of fundamental frequencies:
```
peaks = DSGBR(frequencies, psd)
fundamentals = identify_fundamentals(peaks)
combinations = classify_combinations(peaks, fundamentals)
```

**Frequency Tracking:**
For time-series analysis, track peak evolution:
```
FOR each time step:
    peaks_t = DSGBR(psd_t, config)
    trajectories = match_peaks_to_trajectories(peaks_t, trajectories)
```

## Implementation Reference

### Configuration Class Structure

```python
@dataclass
class DSGBRConfig:
    # SEARCH series parameters
    smooth: str = "savgol"
    smooth_window: int = 9
    smooth_polyorder: int = 2
    smooth_on_log: bool = True

    # BASELINE series parameters
    baseline_window: Optional[int] = None
    baseline_window_frac: float = 0.01
    baseline_on_log: bool = True

    # Detection parameters
    ratio_threshold: float = 1.3

    # Spacing parameters
    switch_frequency: float = 2e-2
    distance_low: int = 2
    distance_high: int = 1

    # ULF guardrail
    ulf_fmax: float = 1e-3
    ulf_min_q: float = 9.0
    ulf_max_points: int = 5

    # Selection parameters
    max_peaks: int = 25
    band_strategy: str = "proportional"
    n_bands: int = 10
```

### Core Function Signatures

```python
def dsgbr_detector(
    frequencies: np.ndarray,
    psd: np.ndarray,
    *,
    case_info: Optional[Dict] = None,
    return_support: bool = False,
) -> Union[Tuple[np.ndarray, np.ndarray],
           Tuple[np.ndarray, np.ndarray, Dict]]:
    """Main DSGBR detection function."""

def compute_support_series(
    frequencies: np.ndarray,
    psd: np.ndarray,
    case_info: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Return SEARCH, BASELINE, ratio series for visualization."""
```

## Citation and Attribution

### How to Cite

If you use the DSGBR method in your research, please cite:

```
[Author], "Dual Savitzky–Golay Baseline Ratio (DSGBR) Peak Detection for
Quasi-Periodic Dynamical Systems," [Journal/Conference], [Year].
```

### License and Usage

This method is provided under [LICENSE_TYPE]. You are free to use, modify, and distribute the implementation with appropriate attribution.

### Contact Information

For questions, suggestions, or collaboration opportunities regarding the DSGBR method:
- Email: [CONTACT_EMAIL]
- Repository: [REPOSITORY_URL]
- Issues: [ISSUES_URL]

### Acknowledgments

Development of the DSGBR method was motivated by challenges in analyzing quasi-periodic flow dynamics and the need for robust peak detection in dense spectral combs typical of nonlinear dynamical systems.

## Version History

- v1.0: Initial implementation with core DSGBR algorithm
- v1.1: Added ultra-low-frequency guardrail and band-balanced selection
- v1.2: Enhanced parameter handling and legacy compatibility
- v1.3: Performance optimizations and improved documentation

---

*This document serves as the definitive reference for the DSGBR peak detection method. For implementation details, see the accompanying source code in `core/DSGBR.py`.*
