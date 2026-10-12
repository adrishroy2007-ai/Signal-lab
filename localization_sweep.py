
import numpy as np

rng = np.random.default_rng(2026)

pulse = np.array([0.1, 0.4, 0.8, 0.4, 0.1])
noise_std = 0.15
echo_start = 45
signal_length = 100
trials = 10_000

print("Echo amplitude | Exact localization | Within ±2 samples")
print("-" * 59)

for echo_amplitude in [0.2, 0.3, 0.5, 0.75, 1.0]:
    exact = 0
    near = 0

    for _ in range(trials):
        received = rng.normal(0, noise_std, signal_length)

        received[
            echo_start : echo_start + len(pulse)
        ] += echo_amplitude * pulse

        scores = np.correlate(
            received, pulse, mode="valid"
        )
        location = np.argmax(scores)

        exact += location == echo_start
        near += abs(location - echo_start) <= 2

    print(
        f"{echo_amplitude:14.2f} | "
        f"{exact / trials:18.4f} | "
        f"{near / trials:18.4f}"
    )
