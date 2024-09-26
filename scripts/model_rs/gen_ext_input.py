import os
import numpy as np
from dotenv import load_dotenv

load_dotenv()
DESNITIES = {"4k": 3619, "32k": 29696}  # Number of cortical vertices

nruns = 50
den = "4k"
nverts = DESNITIES[den]

dt = 0.09
tmax = 50 + 1199 * 0.72
t = np.arange(0, tmax + dt, dt)
n_timepoints = len(t)

for i in range(nruns):
    print(f"Generating external input {i}. Shape: {nverts} x {n_timepoints}")
    file = f"{os.getenv('PROJ_DIR')}/data/resting_state/extInput_den-{den}_randseed-{i}.npy"

    np.random.seed(i)
    ext_input = np.random.randn(nverts, n_timepoints)
    np.save(file, ext_input)
