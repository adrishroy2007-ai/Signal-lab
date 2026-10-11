import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1. Simulation settings
# -----------------------------
rng = np.random.default_rng(2026)

pulse = np.array([0.1, 0.4, 0.8, 0.4, 0.1])
noise_std = 0.15
echo_amplitude = 0.2
echo_start = 45
signal_length = 100

calibration_trials = 50_000
test_trials = 10_000

observation_counts = range(1, 51)

# -----------------------------
# 2. Calibrate ONCE at N=1
# -----------------------------
noise_maxima = []

for _ in range(calibration_trials):
    noise = rng.normal(0, noise_std, signal_length)

    scores = np.correlate(
        noise, pulse, mode="valid"
    )

    noise_maxima.append(np.max(scores))

threshold = np.percentile(noise_maxima, 99)

print(f"Fixed N=1 threshold: {threshold:.6f}")
print("This threshold will NOT change with N.\n")

# -----------------------------
# 3. Test N=1 through N=50
# -----------------------------
detection_rates = []
false_alarm_rates = []

for n_obs in observation_counts:

    detections = 0
    false_alarms = 0

    for _ in range(test_trials):

        # Average N independent signal-present observations
        signal_scores = []

        for _ in range(n_obs):
            received = rng.normal(
                0, noise_std, signal_length
            )

            received[
                echo_start:echo_start + len(pulse)
            ] += echo_amplitude * pulse

            scores = np.correlate(
                received, pulse, mode="valid"
            )
            signal_scores.append(scores)

        combined_signal = np.mean(
            signal_scores, axis=0
        )

        # Detection requires threshold crossing AND
        # correct localization of the echo
        if (
            np.max(combined_signal) > threshold
            and np.argmax(combined_signal) == echo_start
        ):
            detections += 1

        # Average N independent noise-only observations
        noise_scores = []

        for _ in range(n_obs):
            noise = rng.normal(
                0, noise_std, signal_length
            )

            scores = np.correlate(
                noise, pulse, mode="valid"
            )
            noise_scores.append(scores)

        combined_noise = np.mean(
            noise_scores, axis=0
        )

        if np.max(combined_noise) > threshold:
            false_alarms += 1

    detection_rate = 100 * detections / test_trials
    false_alarm_rate = 100 * false_alarms / test_trials

    detection_rates.append(detection_rate)
    false_alarm_rates.append(false_alarm_rate)

    print(
        f"N={n_obs:2d} | "
        f"Detection={detection_rate:6.2f}% | "
        f"False alarms={false_alarm_rate:6.2f}%"
    )

# -----------------------------
# 4. Plot the results
# -----------------------------
plt.figure(figsize=(10, 5))

plt.plot(
    list(observation_counts),
    detection_rates,
    marker="o",
    markersize=3,
    label="Detection probability"
)

plt.plot(
    list(observation_counts),
    false_alarm_rates,
    marker="s",
    markersize=3,
    label="False-alarm probability"
)

plt.axhline(
    1, linestyle="--",
    label="1% reference"
)

plt.xlabel("Number of observations averaged (N)")
plt.ylabel("Probability (%)")
plt.title("Fixed Threshold Calibrated at N=1")
plt.xticks(range(1, 51, 2))
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
