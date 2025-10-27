import json
import numpy as np
import matplotlib.pyplot as plt

# carica i risultati
with open(r"C:\Users\daria\OneDrive\Desktop\lista_N4\bound_results_S3_cached_totale.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# il file è una lista, non un dizionario
results = data

# estrai valori
c_vals = [r["c"] for r in results]
delta_vals = [r["bound"] for r in results]

# crea una curva smooth interpolando
c_smooth = np.linspace(min(c_vals), max(c_vals), 200)
delta_smooth = np.interp(c_smooth, c_vals, delta_vals)

# plot
plt.figure(figsize=(7,5))
plt.plot(c_vals, delta_vals, "o", label="Punti calcolati")
plt.plot(c_smooth, delta_smooth, "-", alpha=0.7, label="Curva interpolata")
plt.xlabel("c")
plt.ylabel("Δ*(c)")
plt.title("Bound Δ*(c) dal bootstrap")
plt.legend()
plt.grid(True)
plt.show()
