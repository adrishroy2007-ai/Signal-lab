
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

observation_counts = [1, 2, 5, 10, 20]
trials = 10_000

results_dir = Path("results")
results_dir.mkdir(exist_ok=True)


# ----------------------------
# Run experiment
# ----------------------------

results = []

for n_observations in observation_counts:
    exact_count = 0
    near_count = 0

    for _ in range(trials):
        # Generate independent noisy observations of the same scene.
        averaged_signal = np.zeros(signal_length)

        for _ in range(n_observations):
            received = rng.normal(
                0, noise_std, signal_length
            )

            # The echo has the same amplitude and location
            # in every observation.
            received[
                echo_start : echo_start + len(pulse)
            ] += echo_amplitude * pulse

            averaged_signal += received

        averaged_signal /= n_observations

        # Search for the strongest matched-filter response.
        scores = np.correlate(
            averaged_signal, pulse, mode="valid"
        )
        detected_location = int(np.argmax(scores))

        exact_count += detected_location == echo_start
        near_count += abs(detected_location - echo_start) <= 2

    exact_probability = exact_count / trials
    near_probability = near_count / trials

    results.append(
        (
            n_observations,
            exact_probability,
            near_probability,
        )
    )

    print(
        f"N={n_observations:2d} | "
        f"Exact={exact_probability:.4f} | "
        f"Within ±2={near_probability:.4f}"
    )


# ----------------------------
# Save results
# ----------------------------

csv_path = results_dir / "repeated_localization.csv"

with csv_path.open("w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(
        [
            "observations_averaged",
            "exact_localization_probability",
            "localization_within_2_samples",
        ]
    )
    writer.writerows(results)


# ----------------------------
# Plot results
# ----------------------------

n_values = [row[0] for row in results]
exact_values = [row[1] for row in results]
near_values = [row[2] for row in results]

plt.figure(figsize=(8, 6))

plt.plot(
    n_values,
    exact_values,
    marker="o",
    label="Exact localization",
)

plt.plot(
    n_values,
    near_values,
    marker="s",
    label="Localization within ±2 samples",
)

plt.xlabel("Number of observations averaged (N)")
plt.ylabel("Localization probability")
plt.title("Effect of Averaging on Echo Localization")
plt.ylim(0, 1.05)
plt.xticks(n_values)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plot_path = results_dir / "repeated_localization.png"
plt.savefig(plot_path, dpi=200)
plt.close()

print(f"\nSaved data: {csv_path}")
print(f"Saved plot: {plot_path}")
