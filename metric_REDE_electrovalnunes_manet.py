import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.patches import Patch


output_dir = Path("electrosvalnunes")
output_dir.mkdir(exist_ok=True)

print("Armazenar os arquivos em:", output_dir.resolve())




# 1. CONFIGURAÇÃO INICIAL
# ============================================================

dataset_path = "Dataset_electrosvalnunes_manet.csv"   # coloque aqui o nome do seu arquivo CSV

df = pd.read_csv(dataset_path)
df.columns = [c.strip() for c in df.columns]

print("Colunas encontradas no dataset:")
print(df.columns.tolist())


# 2. AJUSTE DOS NOMES DAS COLUNAS
#    Altere aqui se no seu CSV os nomes forem diferentes
# ============================================================

def find_col(possible_names):
    for name in possible_names:
        for col in df.columns:
            if name.lower() in col.lower():
                return col
    return None

nodes_col = find_col(["nodes", "topology", "node", "num_nodes", "n_nodes"])
scenario_col = find_col(["scenario", "attack", "class", "label"])
pdr_col = find_col(["pdr"])
throughput_col = find_col(["throughput", "thup"])
delay_col = find_col(["delay"])
energy_col = find_col(["energy"])

print("Coluna de nós:", nodes_col)
print("Coluna de cenário:", scenario_col)
print("Coluna PDR:", pdr_col)
print("Coluna Throughput:", throughput_col)
print("Coluna Delay:", delay_col)
print("Coluna Energy:", energy_col)


# 3. FILTRAR TOPOLOGIAS E CENÁRIOS
# ============================================================

df[nodes_col] = pd.to_numeric(df[nodes_col], errors="coerce")

df = df[df[nodes_col].isin([30, 50, 100])]
df[scenario_col] = df[scenario_col].astype(str)

scenario_order = ["Normal", "Flooding"]
node_order = [30, 50, 100]


# 4. CALCULAR MÉDIA E DESVIO-PADRÃO
# ============================================================

stats = (
    df.groupby([nodes_col, scenario_col])
    .agg(
        PDR_mean=(pdr_col, "mean"),
        PDR_std=(pdr_col, "std"),
        Throughput_mean=(throughput_col, "mean"),
        Throughput_std=(throughput_col, "std"),
        Delay_mean=(delay_col, "mean"),
        Delay_std=(delay_col, "std"),
        Energy_mean=(energy_col, "mean"),
        Energy_std=(energy_col, "std"),
    )
    .reset_index()
)

stats = stats.sort_values([nodes_col, scenario_col])

print("\n===== MÉDIAS E DESVIOS-PADRÃO =====")
print(stats.round(3))
#SAVE MEDIA
stats.to_csv(output_dir / "metricas_media_desvio_padrao.csv", index=False)
print("\nTabela salva em: metricas_media_desvio_padrao.csv")


# 5. FUNÇÃO AUXILIAR PARA PEGAR MÉDIA E DESVIO
# ============================================================

def get_values(metric_mean, metric_std, scenario):
    means = []
    stds = []

    for n in node_order:
        row = stats[
            (stats[nodes_col] == n)
            & (stats[scenario_col].str.lower() == scenario.lower())
        ]

        if row.empty:
            means.append(np.nan)
            stds.append(np.nan)
        else:
            means.append(float(row[metric_mean].iloc[0]))
            stds.append(float(row[metric_std].iloc[0]))

    return np.array(means), np.array(stds)


# 6. GRÁFICO 1 — PDR + THROUGHPUT
# ============================================================

pdr_normal, pdr_normal_std = get_values("PDR_mean", "PDR_std", "Normal")
pdr_flood, pdr_flood_std = get_values("PDR_mean", "PDR_std", "Flooding")

thr_normal, thr_normal_std = get_values("Throughput_mean", "Throughput_std", "Normal")
thr_flood, thr_flood_std = get_values("Throughput_mean", "Throughput_std", "Flooding")

plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 15,
})

fig, axes = plt.subplots(1, 2, figsize=(18, 9), dpi=220)

bar_width = 0.24
x = np.array([0, 1])
offsets = np.array([-bar_width, 0, bar_width])

labels = ["N30", "N50", "N100"]
colors = ["#d9eaf7", "#6f98c4", "#3f3f3f"]

# ---------------- PDR ----------------
ax = axes[0]

pdr_values = np.vstack([pdr_normal, pdr_flood]).T
pdr_stds = np.vstack([pdr_normal_std, pdr_flood_std]).T

for i, (label, color) in enumerate(zip(labels, colors)):
    vals = pdr_values[i]
    errs = pdr_stds[i]

    ax.bar(
        x + offsets[i],
        vals,
        bar_width,
        label=label,
        color=color,
        edgecolor="black",
        linewidth=1.2,
        yerr=errs,
        capsize=5,
        alpha=0.95,
    )

    for j, v in enumerate(vals):
        ax.text(
            x[j] + offsets[i],
            v + 1.1,
            f"{v:.1f}",
            ha="center",
            va="bottom",
            fontsize=13,
        )

ax.set_title("Figure (a): PDR (%) vs Topology", fontweight="bold")
ax.set_ylabel("PDR (%)")
ax.set_xlabel("Scenario and Topology")
ax.set_xticks(x)
ax.set_xticklabels(["Normal", "Flooding"])
ax.set_ylim(0, 105)
ax.grid(axis="y", linewidth=1, alpha=0.45)
ax.legend(
    loc="lower center",
    ncol=3,
    frameon=True,
    fancybox=True,
    bbox_to_anchor=(0.5, 0.02),
)

ax.annotate(
    "",
    xy=(1.30, np.nanmean(pdr_flood) - 3),
    xytext=(0.55, np.nanmean(pdr_normal) - 8),
    arrowprops=dict(
        arrowstyle="-|>",
        lw=2.2,
        color="gray",
        alpha=0.85,
        mutation_scale=22,
    ),
)

# ---------------- Throughput ----------------
ax = axes[1]

thr_values = np.vstack([thr_normal, thr_flood]).T
thr_stds = np.vstack([thr_normal_std, thr_flood_std]).T

for i, (label, color) in enumerate(zip(labels, colors)):
    vals = thr_values[i]
    errs = thr_stds[i]

    ax.bar(
        x + offsets[i],
        vals,
        bar_width,
        label=label,
        color=color,
        edgecolor="black",
        linewidth=1.2,
        yerr=errs,
        capsize=5,
        alpha=0.95,
    )

    for j, v in enumerate(vals):
        ax.text(
            x[j] + offsets[i],
            v + 3,
            f"{v:.1f}",
            ha="center",
            va="bottom",
            fontsize=13,
        )

ax.set_title("Figure (b): Throughput (Kbps) vs Topology", fontweight="bold")
ax.set_ylabel("Throughput (Kbps)")
ax.set_xlabel("Scenario and Topology")
ax.set_xticks(x)
ax.set_xticklabels(["Normal", "Flooding"])
ax.set_ylim(0, np.nanmax(thr_values + thr_stds) * 1.15)
ax.grid(axis="y", linewidth=1, alpha=0.45)
ax.legend(
    loc="lower center",
    ncol=3,
    frameon=True,
    fancybox=True,
    bbox_to_anchor=(0.5, 0.02),
)

ax.annotate(
    "",
    xy=(1.33, np.nanmean(thr_flood) - 10),
    xytext=(0.55, np.nanmean(thr_normal) - 5),
    arrowprops=dict(
        arrowstyle="-|>",
        lw=2.2,
        color="gray",
        alpha=0.85,
        mutation_scale=22,
    ),
)

plt.tight_layout(w_pad=3)
plt.savefig(output_dir /"PDR_Throughput_calculated_dataset.png", dpi=300, bbox_inches="tight")
plt.savefig(output_dir /"PDR_Throughput_calculated_dataset.pdf", bbox_inches="tight")
plt.show()


# 7. GRÁFICO 2 — DELAY + ENERGY COM BOXPLOT
# ============================================================

plot_df = df[[nodes_col, scenario_col, delay_col, energy_col]].copy()
plot_df.columns = ["Nodes", "Scenario", "Delay", "Energy"]
plot_df["Topo"] = plot_df["Nodes"].map({30: "N30", 50: "N50", 100: "N100"})

fig, axes = plt.subplots(1, 2, figsize=(18, 9), dpi=220)

base = np.array([0, 1])
offsets = [-0.28, 0, 0.28]
width = 0.22

topo_order = ["N30", "N50", "N100"]
colors_box = {
    "N30": "#f2ecc0",
    "N50": "#e6b46b",
    "N100": "#7a1235",
}

legend_handles = [
    Patch(facecolor=colors_box[t], edgecolor="black", label=t)
    for t in topo_order
]

# ---------------- Delay ----------------
ax = axes[0]

for i, topo in enumerate(topo_order):
    data = []

    for scen in scenario_order:
        vals = plot_df[
            (plot_df["Topo"] == topo)
            & (plot_df["Scenario"].str.lower() == scen.lower())
        ]["Delay"].dropna().values

        data.append(vals)

    positions = base + offsets[i]

    bp = ax.boxplot(
        data,
        positions=positions,
        widths=width,
        patch_artist=True,
        showfliers=True,
    )

    for patch in bp["boxes"]:
        patch.set(
            facecolor=colors_box[topo],
            alpha=0.9,
            edgecolor="black",
            linewidth=1.2,
        )

    for med in bp["medians"]:
        med.set(color="black", linewidth=1.8)

    for item in ["whiskers", "caps"]:
        for line in bp[item]:
            line.set(color="black", linewidth=1.0)

ax.set_yscale("log")
ax.set_title("Figure (a): End-to-End Delay (ms) [Log Scale]")
ax.set_ylabel("End-to-End Delay (ms) [Log Scale]")
ax.set_xlabel("Scenario & Topology")
ax.set_xticks(base)
ax.set_xticklabels(["Normal", "Flooding"])
ax.grid(axis="y", alpha=0.4)
ax.legend(
    handles=legend_handles,
    title="Topology (Nodes):",
    loc="upper left",
    frameon=True,
)

# ---------------- Energy ----------------
ax = axes[1]

for i, topo in enumerate(topo_order):
    data = []

    for scen in scenario_order:
        vals = plot_df[
            (plot_df["Topo"] == topo)
            & (plot_df["Scenario"].str.lower() == scen.lower())
        ]["Energy"].dropna().values

        data.append(vals)

    positions = base + offsets[i]

    bp = ax.boxplot(
        data,
        positions=positions,
        widths=width,
        patch_artist=True,
        showfliers=True,
    )

    for patch in bp["boxes"]:
        patch.set(
            facecolor=colors_box[topo],
            alpha=0.9,
            edgecolor="black",
            linewidth=1.2,
        )

    for med in bp["medians"]:
        med.set(color="black", linewidth=1.8)

    for item in ["whiskers", "caps"]:
        for line in bp[item]:
            line.set(color="black", linewidth=1.0)

ax.set_title("Figure (b): Energy Consumption (J)")
ax.set_ylabel("Energy Consumption (J)")
ax.set_xlabel("Scenario & Topology")
ax.set_xticks(base)
ax.set_xticklabels(["Normal", "Flooding"])
ax.grid(axis="y", alpha=0.4)
ax.legend(
    handles=legend_handles,
    title="Topology (Nodes):",
    loc="upper left",
    frameon=True,
)

plt.tight_layout(w_pad=3)
plt.savefig(output_dir /"Delay_Energy_calculated_dataset.png", dpi=300, bbox_inches="tight")
plt.savefig(output_dir /"Delay_Energy_calculated_dataset.pdf", bbox_inches="tight")
plt.show()

print("\nGráficos salvos:")
print("1. PDR_Throughput_calculated_dataset.png")
print("2. PDR_Throughput_calculated_dataset.pdf")
print("3. Delay_Energy_calculated_dataset.png")
print("4. Delay_Energy_calculated_dataset.pdf")
