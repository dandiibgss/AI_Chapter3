# Example 3.28e Plotting LazyRegressor R-Squared Results
import importlib.util
import os
import matplotlib.pyplot as plt

# Import data models dari percobaan3.28d.py
if 'models' not in globals():
    try:
        from percobaan3_28d import models
    except ImportError:
        _spec = importlib.util.spec_from_file_location(
            "percobaan3_28d",
            os.path.join(os.path.dirname(__file__), "percobaan3.28d.py")
        )
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        models = _mod.models

# Ambil nama kolom R-Squared yang tersedia
r2_col = 'R-Squared' if 'R-Squared' in models.columns else 'Adjusted R-Squared'

# Plot Visualisasi R-Squared Model Regresi
plt.figure(figsize=(12, 6))
plt.plot(models.index, models[r2_col], '-s', color='r', linewidth=2)
plt.xticks(rotation=90)
plt.title(f"Perbandingan {r2_col} Model Regresi (LazyRegressor)")
plt.xlabel("Nama Model")
plt.ylabel(r2_col)
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()