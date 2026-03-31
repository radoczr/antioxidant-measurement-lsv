import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import glob
import os

# -----------------------------
# 0. FELHASZNÁLÓI BEMENET
# -----------------------------
folder = input("Add meg a mappa elérési útját: ").strip()

if not os.path.exists(folder):
    raise ValueError("A megadott mappa nem létezik!")

files = sorted(glob.glob(os.path.join(folder, "*.txt")))

if len(files) == 0:
    raise ValueError("Nincs .txt fájl a mappában!")

c_AA = float(input("Add meg az aszkorbinsav koncentrációját (g dm⁻³): "))
c_AA *= 1000  # átváltás mg/dm3-re

print("\nTalált fájlok:")
for f in files:
    print(f)

# -----------------------------
# 1. FÁJL BEOLVASÁS
# -----------------------------
def load_file(file):
    try:
        with open(file, 'r', encoding='utf-8-sig') as f:
            lines = f.readlines()

        if len(lines) < 3:
            raise ValueError("Túl rövid fájl")

        V_added = float(lines[0].strip())

        df = pd.read_csv(
            file,
            sep="\t",
            skiprows=2,
            encoding='utf-8-sig'
        )

        if df.shape[1] < 3:
            raise ValueError("Kevés oszlop")

        df = df.iloc[:, :3]
        df.columns = ["Time", "Potential", "Current"]

        return V_added, df

    except Exception as e:
        print(f"⚠️ Hiba a fájlban: {file}")
        print(f"   -> {e}")
        return None, None


# -----------------------------
# 2. CSAK ELŐREMENŐ SZAKASZ
# -----------------------------
def trim_forward_scan(df):
    potentials = df["Potential"].values
    diffs = np.diff(potentials)

    idx = np.where(diffs < 0)[0]

    if len(idx) > 0:
        return df.iloc[:idx[0] + 1]

    return df


# -----------------------------
# 3. ADATOK ÖSSZEGYŰJTÉSE
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
    raise ValueError("Nincs feldolgozható adat!")

# -----------------------------
# 4. VOLTAMMOGRAMOK ÁBRÁZOLÁSA
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
plt.title("Mért voltammogramok")
plt.grid(False)
plt.tight_layout()

plt.savefig(os.path.join(folder, "Voltammogramok.pdf"))
plt.show()

# -----------------------------
# 5. CSÚCSKERESÉS
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

    # koncentráció számítás (standard addíció)
    V_added_dm3 = d["V_added"] * 1e-6
    c_added = (c_AA * V_added_dm3) / (V_added_dm3 + 0.01)

    results.append({
        "V_added": d["V_added"],
        "c_added": c_added,
        "E_peak": peak_E_mV,
        "I_peak": peak_I_uA
    })

df_res = pd.DataFrame(results).sort_values("V_added")

print("\n--- KIÉRTÉKELT ADATOK ---")
print(df_res)

# -----------------------------
# 6. TÁBLÁZAT EXPORT (PDF)
# -----------------------------
fig, ax = plt.subplots(figsize=(8, 4))
ax.axis('off')

table_data = df_res.copy()
table_data.columns = [
    r"$V_{added}$ / µL",
    r"$c_{added}$ / (mg dm$^{-3}$)",
    r"$E$ / mV",
    r"$I_{peak}$ / µA"
]

table = ax.table(cellText=np.round(table_data.values, 4),
                 colLabels=table_data.columns,
                 loc='center',
                 cellLoc='center')

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.5)

plt.savefig(os.path.join(folder, "Csucsaromok.pdf"), bbox_inches='tight')
plt.close()

# -----------------------------
# 7. REGRESSZIÓ (KÉZI)
# -----------------------------
x = df_res["c_added"].values
y = df_res["I_peak"].values

if len(x) < 3:
    raise ValueError("Túl kevés adat regresszióhoz!")

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
# 8. KALIBRÁCIÓS GÖRBE
# -----------------------------
plt.figure(figsize=(8, 6))

for i, row in df_res.iterrows():
    plt.scatter(row["c_added"], row["I_peak"])

x_min = min(min(x), x0) if not np.isnan(x0) else min(x)
x_max = max(x)

x_fit = np.linspace(x_min, x_max, 200)
plt.plot(x_fit, slope * x_fit + intercept)

text = (
    f"y = ({slope:.2f} ± {slope_ci:.2f})x + ({intercept:.2f} ± {intercept_ci:.2f})\n"
    f"R² = {r_value**2:.2f}\n"
    f"F = {F:.2f}\n"
    f"|x₀| = {abs(x0):.2f} ± {x0_err:.2f}"
)

plt.text(0.05, 0.95, text,
         transform=plt.gca().transAxes,
         verticalalignment='top')

plt.xlabel(r"$c_{added}$ / (mg dm$^{-3}$)")
plt.ylabel(r"$I_{peak}$ / µA")
plt.title("Kalibrációs görbe")
plt.grid(False)
plt.tight_layout()

plt.savefig(os.path.join(folder, "Kalibracios_gorbe.pdf"))
plt.show()

print("\nKész! Minden fájl elmentve.")
