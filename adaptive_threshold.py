import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(2026)

pulse = np.array([0.1, 0.4, 0.8, 0.4, 0.1])
noise_std = 0.15
echo_amplitude = 0.2
echo_start = 45

trials = 3000
calibration_trials = 5000
observation_counts = list(range(1, 51))

detection_rates = []
false_alarm_rates = []

for n_obs in observation_counts:

    # Calibrate threshold using noise-only recordings
    noise_maxima = []

    for _ in range(calibration_trials):
        scores = []

        for _ in range(n_obs):
            noise = rng.normal(0, noise_std, 100)
            scores.append(
                np.correlate(noise, pulse, mode="valid")
            )

        combined = np.mean(scores, axis=0)
        noise_maxima.append(np.max(combined))

    threshold = np.percentile(noise_maxima, 99)

    detections = 0
    false_alarms = 0

    for _ in range(trials):
        scores = []
        noise_scores = []

        for _ in range(n_obs):
            # Signal-present recording
            received = rng.normal(0, noise_std, 100)
            received[echo_start:echo_start + len(pulse)] += (
                echo_amplitude * pulse
            )

            scores.append(
                np.correlate(received, pulse, mode="valid")
            )

            # Independent noise-only recording
            noise = rng.normal(0, noise_std, 100)
            noise_scores.append(
                np.correlate(noise, pulse, mode="valid")
            )

        combined = np.mean(scores, axis=0)
        noise_only = np.mean(noise_scores, axis=0)

        # Detection requires threshold crossing AND correct location
        if (
            np.max(combined) > threshold
            and np.argmax(combined) == echo_start
        ):
            detections += 1

        if np.max(noise_only) > threshold:
            false_alarms += 1

    detection_rates.append(detections / trials)
    false_alarm_rates.append(false_alarms / trials)

    print(
        f"N={n_obs:2d} | "
        f"Detection={100*detections/trials:6.2f}% | "
        f"False alarms={100*false_alarms/trials:5.2f}%"
    )

plt.plot(
    observation_counts,
    np.array(detection_rates) * 100,
    marker="o",
    label="Detection probability"
)

plt.plot(
    observation_counts,
    np.array(false_alarm_rates) * 100,
    marker="s",
    label="False-alarm probability"
)

plt.xlabel("Number of observations averaged")
plt.ylabel("Probability (%)")
plt.title("Signal Lab: Detection vs. Observation Count")
plt.grid(True)
plt.legend()
plt.show()
