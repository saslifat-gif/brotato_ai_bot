# Control rate follow-up

The expanded C++ controller was profiled in live gameplay. At 24 Hz requested bridge reporting, mean state wait was 32.36 ms, movement decision 1.59 ms, normalization 1.80 ms, and vectorization 1.94 ms in the captured early-wave sample.

A 60 Hz reporting trial produced approximately 30 actual control updates/sec over 479 intervals in wave 6. Its initial profile measured 21.81 ms mean state wait, 3.01 ms model inference, 1.95 ms movement decisions, 2.00 ms normalization, and 2.48 ms vectorization. This is not a same-wave controlled comparison and does not establish a win-rate gain or sustained late-game performance.

The local automatic runner now requests 60 Hz by default. This is a reporting request, not a guarantee of 60 control decisions/sec. Existing learning checkpoints and training configuration remain unchanged. v4.run_rate now separately profiles model inference through the unwrapped environment profiler.
