import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import glob
import os

# -----------------------------
# 0. USER INPUT
# -----------------------------
folder = input("Enter the folder path: ").strip()

if not os.path.exists(folder):
    raise ValueError("The specified folder does not exist!")

files = sorted(glob.glob(os.path.join(folder, "*.txt")))

if len(files) == 0:
    raise ValueError("No .txt files found in the folder!")

c_AA = float(input("Enter the ascorbic acid concentration (g dm⁻³): "))
c_AA *= 1000  # convert to mg/dm3

print("\nDetected files:")
for f in files:
    print(f)

# -----------------------------
# 1. FILE LOADING
# -----------------------------
def load_file(file):
    try:
        with open(file, 'r', encoding='utf-8-sig') as f:
            lines = f.readlines()

        if len(lines) < 3:
            raise ValueError("File too short")

        V_added = float(lines[0].strip())

        df = pd.read_csv(
            file,
            sep="\t",
            skiprows=2,
            encoding='utf-8-sig'
        )

        if df.shape[1] < 3:
            raise ValueError("Not enough columns")

        df = df.iloc[:, :3]
        df.columns = ["Time", "Potential", "Current"]

        return V_added, df

    except Exception as e:
        print(f"⚠️ Error in file: {file}")
        print(f"   -> {e}")
        return None, None


# -----------------------------
# 2. FORWARD SCAN ONLY
# -----------------------------
def trim_forward_scan(df):
    potentials = df["Potential"].values
    diffs = np.diff(potentials)

    idx = np.where(diffs < 0)[0]

    if len(idx) > 0:
        return df.iloc[:idx[0] + 1]

    return df


# -----------------------------
# 3. DATA COLLECTION
# -----------------------------
data = []

for file in files:
    V_added, df = load_file(file)

    if df is None:
        continue

    df = trim_forward_scan(df)

    data.append({
        "V_added": V_added,
        "df": df
    })

if len(data) == 0:
    raise ValueError("No valid data found!")

# -----------------------------
# 4. VOLTAMMOGRAMS
# -----------------------------
plt.figure(figsize=(8, 6))

colors = plt.cm.viridis(np.linspace(0, 1, len(data)))

for i, d in enumerate(data):
    df = d["df"]

    E_mV = df["Potential"] * 1000
    I_uA = df["Current"] * 1e6

    d["E"] = E_mV
    d["I"] = I_uA

    plt.plot(E_mV, I_uA, color=colors[i])

    plt.annotate(f"{d['V_added']} µL",
                 xy=(E_mV.iloc[-1], I_uA.iloc[-1]),
                 xytext=(5, 0),
                 textcoords='offset points')

plt.xlabel(r"$\it{E}$ / mV")
plt.ylabel(r"$\it{I}$ / µA")
plt.title("Measured Voltammograms")
plt.grid(False)
plt.tight_layout()

plt.savefig(os.path.join(folder, "Measured Voltammograms.pdf"))
plt.show()

# -----------------------------
# 5. PEAK DETECTION
# -----------------------------
ref = max(data, key=lambda x: x["V_added"])
ref_df = ref["df"]

ref_peak_idx = ref_df["Current"].idxmax()
ref_E = ref_df.loc[ref_peak_idx, "Potential"]

window = 0.01

results = []

for d in data:
    df = d["df"]

    mask = (df["Potential"] > ref_E - window) & (df["Potential"] < ref_E + window)
    sub = df[mask] if len(df[mask]) > 0 else df

    peak_idx = sub["Current"].idxmax()

    peak_I = sub.loc[peak_idx, "Current"]
    peak_E = sub.loc[peak_idx, "Potential"]

    peak_I_uA = peak_I * 1e6
    peak_E_mV = peak_E * 1000

    V_added_dm3 = d["V_added"] * 1e-6
    c_added = (c_AA * V_added_dm3) / (V_added_dm3 + 0.01)

    results.append({
        "V_added": d["V_added"],
        "c_added": c_added,
        "E_peak": peak_E_mV,
        "I_peak": peak_I_uA
    })

df_res = pd.DataFrame(results).sort_values("V_added")

print("\n--- PROCESSED DATA ---")
print(df_res)

# -----------------------------
# 6. TABLE EXPORT
# -----------------------------
fig, ax = plt.subplots(figsize=(8, 4))
ax.axis('off')

table_data = df_res.copy()
table_data.columns = [
    r"$\it{V}_{\mathrm{added}}$ / µL",
    r"$\it{c}_{\mathrm{added}}$ / (mg dm$^{-3}$)",
    r"$\it{E}$ / mV",
    r"$\it{I}_{\mathrm{peak}}$ / µA"
]

table = ax.table(cellText=np.round(table_data.values, 4),
                 colLabels=table_data.columns,
                 loc='center',
                 cellLoc='center'
                 )

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.5)

plt.savefig(os.path.join(folder, "Peak Currents.pdf"), bbox_inches='tight')
plt.close()

# -----------------------------
# 7. REGRESSION (MANUAL)
# -----------------------------
x = df_res["c_added"].values
y = df_res["I_peak"].values

if len(x) < 3:
    raise ValueError("Not enough data points for regression!")

n = len(x)

x_mean = np.mean(x)
y_mean = np.mean(y)

Sxx = np.sum((x - x_mean)**2)
Sxy = np.sum((x - x_mean)*(y - y_mean))

slope = Sxy / Sxx
intercept = y_mean - slope * x_mean

y_fit = slope * x + intercept
residuals = y - y_fit

Sr = np.sqrt(np.sum(residuals**2) / (n - 2))
Sa = Sr / np.sqrt(Sxx)
Sb = Sr * np.sqrt(np.sum(x**2) / (n * Sxx))

ss_reg = np.sum((y_fit - y_mean)**2)
ss_res = np.sum(residuals**2)

F = (ss_reg / 1) / (ss_res / (n - 2))

t_val = stats.t.ppf(0.975, n - 2)

slope_ci = t_val * Sa
intercept_ci = t_val * Sb

x0 = -intercept / slope if slope != 0 else np.nan

Sx = Sr / abs(slope) * np.sqrt(
    (1/n) +
    (y_mean**2 / (slope**2 * Sxx))
) if slope != 0 else np.nan

x0_err = t_val * Sx if not np.isnan(Sx) else np.nan

r_value = np.sqrt(ss_reg / (ss_reg + ss_res))

# -----------------------------
# 8. CALIBRATION CURVE
# -----------------------------
plt.figure(figsize=(8, 6))

for i, row in df_res.iterrows():
    plt.scatter(row["c_added"], row["I_peak"])

x_min = min(min(x), x0) if not np.isnan(x0) else min(x)
x_max = max(x)

x_fit = np.linspace(x_min, x_max, 200)
plt.plot(x_fit, slope * x_fit + intercept)

text = (
    f"y = ({slope:.2e} ± {slope_ci:.2e})x + ({intercept:.2e} ± {intercept_ci:.2e})\n"
    f"R² = {r_value**2:.4f}\n"
    f"F = {F:.2f}\n"
    f"|x₀| = {abs(x0):.2e} ± {x0_err:.2e}"
)

plt.text(0.05, 0.95, text, transform=plt.gca().transAxes,
         verticalalignment='top')

plt.xlabel(r"$\it{c}_{\mathrm{added}}$ / (mg dm$^{-3}$)")
plt.ylabel(r"$\it{I}_{\mathrm{peak}}$ / µA")
plt.title("Calibration Curve")
plt.grid(False)
plt.tight_layout()

plt.savefig(os.path.join(folder, "Calibration Curve.pdf"))
plt.show()

print("\nDone! All files have been saved.")

