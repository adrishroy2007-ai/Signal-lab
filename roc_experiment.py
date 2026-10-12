
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ----------------------------
# Experiment configuration
# ----------------------------

rng = np.random.default_rng(2026)

pulse = np.array([0.1, 0.4, 0.8, 0.4, 0.1])
noise_std = 0.15
echo_amplitude = 0.2
echo_start = 45
signal_length = 100

calibration_trials = 10_000
evaluation_trials = 20_000
target_false_alarm = 0.01

results_dir = Path("results")
results_dir.mkdir(exist_ok=True)


# ----------------------------
# Generate observations
# ----------------------------

def generate_observation(echo_present: bool) -> np.ndarray:
    """Generate one noisy observation, optionally containing an echo."""
    received = rng.normal(0, noise_std, signal_length)

    if echo_present:
        received[
            echo_start : echo_start + len(pulse)
        ] += echo_amplitude * pulse

    return received


def maximum_correlation(received: np.ndarray) -> tuple[float, int]:
    """Return the largest matched-filter score and its starting index."""
    scores = np.correlate(received, pulse, mode="valid")
    location = int(np.argmax(scores))
    return float(scores[location]), location


def collect_scores(
    trials: int, echo_present: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Collect maximum scores and peak locations."""
    scores = np.empty(trials)
    locations = np.empty(trials, dtype=int)

    for i in range(trials):
        received = generate_observation(echo_present)
        scores[i], locations[i] = maximum_correlation(received)

    return scores, locations


# ----------------------------
# Calibrate threshold
# ----------------------------

calibration_noise_scores, _ = collect_scores(
    calibration_trials, echo_present=False
)

threshold = np.quantile(
    calibration_noise_scores, 1 - target_false_alarm
)

print("ROC experiment")
print(f"Calibration trials: {calibration_trials}")
print(f"Evaluation trials:  {evaluation_trials}")
print(f"Target false-alarm probability: {target_false_alarm:.3f}")
print(f"Calibrated threshold: {threshold:.4f}")


# ----------------------------
# Independent evaluation
# ----------------------------

noise_scores, _ = collect_scores(
    evaluation_trials, echo_present=False
)

echo_scores, echo_locations = collect_scores(
    evaluation_trials, echo_present=True
)

# Sort once so we can efficiently evaluate every threshold.
all_scores = np.unique(np.concatenate([noise_scores, echo_scores]))
thresholds = np.concatenate(([-np.inf], all_scores, [np.inf]))

p_false_alarm = np.array([
    np.mean(noise_scores > t) for t in thresholds
])

p_detection = np.array([
    np.mean(echo_scores > t) for t in thresholds
])

p_correct_localization = np.array([
    np.mean(
        (echo_scores > t) & (echo_locations == echo_start)
    )
    for t in thresholds
])


# ----------------------------
# Save ROC data
# ----------------------------

csv_path = results_dir / "roc_data.csv"

with csv_path.open("w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow([
        "threshold",
        "false_alarm_probability",
        "detection_probability",
        "correct_localization_probability",
    ])

    for row in zip(
        thresholds,
        p_false_alarm,
        p_detection,
        p_correct_localization,
    ):
        writer.writerow(row)


# ----------------------------
# Plot ROC curve
# ----------------------------

plt.figure(figsize=(8, 6))

plt.plot(
    p_false_alarm,
    p_detection,
    label="Matched-filter detector",
)

plt.plot(
    [0, 1],
    [0, 1],
    "--",
    label="Random/no-discrimination reference",
)

plt.xlabel("Probability of False Alarm")
plt.ylabel("Probability of Detection")
plt.title("Signal Lab: Receiver Operating Characteristic")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(results_dir / "roc_curve.png", dpi=200)
plt.close()


# ----------------------------
# Plot localization performance
# ----------------------------

plt.figure(figsize=(8, 6))

plt.plot(
    p_false_alarm,
    p_correct_localization,
    color="green",
    label="Correct localization probability",
)

plt.xlabel("Probability of False Alarm")
plt.ylabel("Probability of Correct Localization")
plt.title("Signal Lab: Detection and Localization")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(results_dir / "localization_curve.png", dpi=200)
plt.close()


# ----------------------------
# Report calibrated operating point
# ----------------------------

evaluation_pfa = np.mean(noise_scores > threshold)
evaluation_pd = np.mean(echo_scores > threshold)

evaluation_localization = np.mean(
    (echo_scores > threshold)
    & (echo_locations == echo_start)
)

print("\nIndependent evaluation at calibrated threshold")
print(f"Threshold:                       {threshold:.4f}")
print(f"Measured false-alarm probability: {evaluation_pfa:.4f}")
print(f"Detection probability:            {evaluation_pd:.4f}")
print(f"Correct localization probability:  {evaluation_localization:.4f}")

print("\nSaved:")
print(f"- {csv_path}")
print(f"- {results_dir / 'roc_curve.png'}")
print(f"- {results_dir / 'localization_curve.png'}")