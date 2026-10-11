# Signal Lab: Radar Echo Detection Using Matched Filtering

## Overview

Signal Lab is a Python simulation exploring how matched filtering and repeated-observation averaging can help detect a weak echo in noise.

The project investigates the relationship between detection probability, false-alarm probability, and threshold calibration.

## Objectives

- Simulate a known pulse received in Gaussian noise.
- Use correlation-based matched filtering to locate a potential echo.
- Estimate a detection threshold from noise-only recordings.
- Compare a fixed threshold with a threshold recalibrated for each observation count.
- Study how averaging independent observations affects detection performance.

## Experimental Setup

The simulation uses a known five-sample pulse, a fixed echo location, a configurable echo amplitude, and independent Gaussian noise. Multiple observations of the same echo are averaged before detection.

The detector declares a successful detection when the correlation score exceeds the threshold and the maximum correlation occurs at the true echo location.

## Threshold Comparison

**Fixed threshold:** The threshold is calibrated using single-observation noise-only recordings and reused for larger observation counts.

**Recalibrated threshold:** A separate threshold is estimated for each observation count using noise-only recordings.

The experiments illustrate that a threshold calibrated for one score distribution may become unsuitable when averaging changes that distribution.

## Results

The recalibrated-threshold experiment showed a strong increase in measured detection probability as the number of observations increased, while the measured false-alarm rate fluctuated around the 1% target.

In the fixed-threshold experiment, detection probability approached zero for larger observation counts under the tested weak-echo conditions.

These are simulation results, not measurements from real radar hardware.

## Limitations

- Noise is modeled as independent Gaussian noise.
- The echo location is fixed and known for evaluation.
- The same echo is present in each averaged observation.
- Thresholds are estimated from finite Monte Carlo samples.
- Real radar systems may include clutter, interference, target motion, and other effects not represented here.

## Tools

Python, NumPy, Matplotlib.

## Future Work

- Test different echo amplitudes and noise levels.
- Add uncertainty estimates to detection and false-alarm measurements.
- Explore unknown target locations and more realistic noise.
- Investigate software-defined radio experiments.

## Reproducibility

The Python source code and result figures should be included in this repository. Exact performance figures should be reported alongside the simulation settings and random-seed configuration used to produce them.