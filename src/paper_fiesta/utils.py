"""
TODO: Store some common utils here, such as plotting etc
"""
import os

# Thibeau
BASE_DIR_FLUX_MODELS_THIBEAU = "/home/twouters2/projects/fiesta/fiestaEM/flux_models/"
BASE_DIR_LC_MODELS_THIBEAU = "/home/twouters2/projects/fiesta/fiestaEM/lightcurve_models/"

# Hauke TODO: add dirs
BASE_DIR_FLUX_MODELS_HAUKE = "./"
BASE_DIR_LC_MODELS_HAUKE = "./"

if not os.path.exists(BASE_DIR_FLUX_MODELS_THIBEAU):
    if not os.path.exists(BASE_DIR_FLUX_MODELS_HAUKE):
        raise ValueError("No valid flux models directory found. Please set the correct path in `utils.py`")
    else:
        BASE_DIR_FLUX_MODELS = BASE_DIR_FLUX_MODELS_HAUKE
        BASE_DIR_LC_MODELS = BASE_DIR_LC_MODELS_HAUKE
        print(f"Using model directory: {BASE_DIR_FLUX_MODELS}")
else:
    BASE_DIR_FLUX_MODELS = BASE_DIR_FLUX_MODELS_THIBEAU
    BASE_DIR_LC_MODELS = BASE_DIR_LC_MODELS_THIBEAU
    print(f"Using model directory: {BASE_DIR_FLUX_MODELS}")