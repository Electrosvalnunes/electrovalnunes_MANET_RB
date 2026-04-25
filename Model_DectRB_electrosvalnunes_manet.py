import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

warnings.filterwarnings("ignore")



# 1) PADRONIZAÇÃO DO DATASET
# ============================================================
#def padronizar_Dataset_electrosvalnunes_manet(df):
def padronizar_Dataset_electrosvalnunes_manet(df):

    """
    Converte o dataset enviado para os nomes esperados pelo modelo bayesiano.

    Mapeamento aplicado:
    - Scenario -> scenario
    - Total_Nodes -> topologyNodes
    - PDR_Percentage -> PDR
    - E2E_Delay_ms -> delayMean_ms
    - Throughput_Kbps -> throughput_bps (convertido para bps)
    - Energy_Consumed_J -> energyMean_J

    Colunas extras preservadas:
    - Rep_ID, Node_ID, Queue_Drops, Is_Attacker
    """
    df = df.copy()

    rename_map = {
        "Scenario": "scenario",
        "Total_Nodes": "topologyNodes",
        "PDR_Percentage": "PDR",
        "E2E_Delay_ms": "delayMean_ms",
        "Energy_Consumed_J": "energyMean_J",
    }
    df = df.rename(columns=rename_map)

    # Throughput: no dataset está em Kbps; o modelo original espera bps.
    if "Throughput_Kbps" in df.columns:
        df["throughput_bps"] = pd.to_numeric(df["Throughput_Kbps"], errors="coerce") * 1000.0
    elif "throughput_bps" not in df.columns:
        raise ValueError("Não encontrei nem 'Throughput_Kbps' nem 'throughput_bps' no dataset.")

    required = ["scenario", "topologyNodes", "PDR", "delayMean_ms", "throughput_bps", "energyMean_J"]
    faltando = [c for c in required if c not in df.columns]
    if faltando:
        raise ValueError(f"Faltam colunas obrigatórias no dataset: {faltando}")

    # Garantir tipos numéricos
    for col in ["topologyNodes", "PDR", "delayMean_ms", "throughput_bps", "energyMean_J"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Limpeza básica
    df = df.dropna(subset=required).copy()
    df["scenario"] = df["scenario"].astype(str).str.strip()
    df["topologyNodes"] = df["topologyNodes"].astype(int)

    # Remover linhas absurdas
    df = df[df["PDR"] >= 0].copy()
    df = df[df["delayMean_ms"] >= 0].copy()
    df = df[df["energyMean_J"] >= 0].copy()
    df = df[df["throughput_bps"] >= 0].copy()

    return df



# 2) UTILITÁRIOS NUMÉRICOS
# ============================================================
def softmax(logits):
    logits = np.asarray(logits, dtype=float)
    m = np.max(logits)
    exps = np.exp(logits - m)
    return exps / np.sum(exps)


def mean_std(values):
    arr = np.asarray(values, dtype=float)
    return float(np.mean(arr)), float(np.std(arr))



# 3) PRÉ-PROCESSAMENTO
# ============================================================
def adicionar_atributos_relativos_ao_baseline(
    df,
    metricas_base,
    topo_col="topologyNodes",
    classe_col="scenario",
    baseline_label="Normal",
):
    df = df.copy()

    baseline_df = df[df[classe_col] == baseline_label].copy()
    if baseline_df.empty:
        raise ValueError(
            f"Não existe cenário baseline '{baseline_label}' no dataset. "
            "Verifique os nomes da coluna Scenario."
        )

    baseline = baseline_df.groupby(topo_col)[metricas_base].agg(["mean", "std"])

    for m in metricas_base:
        medias = baseline[(m, "mean")]
        desvios = baseline[(m, "std")].replace(0, 1e-9)

        df[f"{m}_zbase"] = (
            (df[m] - df[topo_col].map(medias)) / df[topo_col].map(desvios)
        ).astype(float)

    eps = 1e-9
    df["efficiency_tp_energy"] = df["throughput_bps"] / (df["energyMean_J"] + eps)
    df["delay_over_pdr"] = df["delayMean_ms"] / (df["PDR"] + eps)
    df["energy_over_pdr"] = df["energyMean_J"] / (df["PDR"] + eps)
    df["delay_log"] = np.log1p(df["delayMean_ms"])
    df["energy_log"] = np.log1p(df["energyMean_J"])

    return df



# 4) DISCRETIZAÇÃO SEM LEAKAGE
# ============================================================
def construir_rotulos_bins(n_bins):
    if n_bins == 3:
        return ["Low", "Medium", "High"]
    if n_bins == 5:
        return ["VeryLow", "Low", "Medium", "High", "VeryHigh"]
    return [f"B{i + 1}" for i in range(n_bins)]


def _quantis_monotonicos(valores):
    valores = np.asarray(valores, dtype=float).copy()
    for i in range(1, len(valores)):
        if valores[i] <= valores[i - 1]:
            valores[i] = valores[i - 1] + 1e-12
    return valores


def discretizar_por_topologia_sem_leakage(
    train_df,
    test_df,
    metrics,
    topo_col="topologyNodes",
    n_bins=5,
):
    train_df = train_df.copy()
    test_df = test_df.copy()
    labels = construir_rotulos_bins(n_bins)
    quantis = np.linspace(0.0, 1.0, n_bins + 1)

    cortes = {}
    for topo in sorted(train_df[topo_col].unique()):
        subset = train_df[train_df[topo_col] == topo]
        cortes[topo] = {}
        for m in metrics:
            bordas = subset[m].quantile(quantis).values
            cortes[topo][m] = _quantis_monotonicos(bordas)

    def discretizar_valor(valor, bordas):
        idx = np.searchsorted(bordas[1:-1], valor, side="right")
        return labels[idx]

    for m in metrics:
        train_df[m + "_d"] = train_df.apply(
            lambda row: discretizar_valor(row[m], cortes[row[topo_col]][m]), axis=1
        )
        test_df[m + "_d"] = test_df.apply(
            lambda row: discretizar_valor(row[m], cortes[row[topo_col]][m]), axis=1
        )

    return train_df, test_df, labels



# 5) GRÁFICOS
# ============================================================
def plot_confusion(cm, labels, title, outpath):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap="Blues")

    limiar = cm.max() / 2 if cm.max() > 0 else 0
    for i in range(len(labels)):
        for j in range(len(labels)):
            cor = "white" if cm[i, j] > limiar else "black"
            ax.text(j, i, f"{cm[i, j]:.0f}", ha="center", va="center", color=cor, fontsize=11)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel("Real", fontsize=12)
    ax.set_xlabel("Predito", fontsize=12)
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(outpath, dpi=400, bbox_inches="tight")
    plt.close()


def roc_auc_ovr_macro(y_true, y_score, classes):
    """
    Calcula ROC-AUC por classe (OvR) e macro-average.
    Funciona para binário e multiclasse.

    Retorna:
    - roc_auc_macro
    - (all_fpr, mean_tpr, roc_auc, fpr, tpr, overlap_flags)
    """
    classes = list(classes)
    y_score = np.asarray(y_score, dtype=float)

    if y_score.ndim == 1:
        y_score = y_score.reshape(-1, 1)

    
    # CASO BINÁRIO
    # ========================================================
    if len(classes) == 2:
        # Se vier só uma coluna, reconstruímos a outra
        if y_score.shape[1] == 1:
            y_score = np.hstack([1 - y_score, y_score])
        elif y_score.shape[1] != 2:
            raise ValueError("Para problema binário, y_score deve ter 1 ou 2 colunas.")

        fpr = {}
        tpr = {}
        roc_auc = {}

        for i, cls in enumerate(classes):
            y_bin = (np.asarray(y_true) == cls).astype(int)
            fpr[i], tpr[i], _ = roc_curve(y_bin, y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(classes))]))
        mean_tpr = np.zeros_like(all_fpr)

        for i in range(len(classes)):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

        mean_tpr /= len(classes)
        roc_auc_macro = auc(all_fpr, mean_tpr)

        overlap_flags = {}
        for i in range(len(classes)):
            for j in range(i + 1, len(classes)):
                fpr_base = np.linspace(0, 1, 300)
                tpr_i_interp = np.interp(fpr_base, fpr[i], tpr[i])
                tpr_j_interp = np.interp(fpr_base, fpr[j], tpr[j])
                diff = np.max(np.abs(tpr_i_interp - tpr_j_interp))
                overlap_flags[(i, j)] = diff < 1e-3

        return roc_auc_macro, (all_fpr, mean_tpr, roc_auc, fpr, tpr, overlap_flags)

    
    # CASO MULTICLASSE
    # ========================================================
    y_true_bin = label_binarize(y_true, classes=classes)

    fpr = {}
    tpr = {}
    roc_auc = {}

    for i in range(len(classes)):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(classes))]))
    mean_tpr = np.zeros_like(all_fpr)

    for i in range(len(classes)):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    mean_tpr /= len(classes)
    roc_auc_macro = auc(all_fpr, mean_tpr)

    overlap_flags = {}
    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):
            fpr_base = np.linspace(0, 1, 300)
            tpr_i_interp = np.interp(fpr_base, fpr[i], tpr[i])
            tpr_j_interp = np.interp(fpr_base, fpr[j], tpr[j])
            diff = np.max(np.abs(tpr_i_interp - tpr_j_interp))
            overlap_flags[(i, j)] = diff < 1e-3

    return roc_auc_macro, (all_fpr, mean_tpr, roc_auc, fpr, tpr, overlap_flags)


def plot_roc_melhorada(
    all_fpr,
    mean_tpr,
    roc_auc_macro,
    roc_auc_ind,
    fpr_ind,
    tpr_ind,
    classes,
    titulo,
    outpath,
):
    """
    Gera curva ROC com melhor legibilidade visual.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Curva macro
    ax.plot(
        all_fpr,
        mean_tpr,
        linewidth=3.2,
        alpha=0.95,
        label=f"Macro-average (AUC = {roc_auc_macro:.3f})",
        zorder=10,
    )

    # Curvas por classe
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]

    for i, cls in enumerate(classes):
        ax.plot(
            fpr_ind[i],
            tpr_ind[i],
            linewidth=2.4,
            alpha=0.92,
            marker=markers[i % len(markers)],
            markevery=max(1, len(fpr_ind[i]) // 10),
            markersize=4.5,
            label=f"{cls} (AUC = {roc_auc_ind[i]:.3f})",
            zorder=5 + i,
        )

    # Classificador aleatório
    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1.6,
        alpha=0.8,
        label="Random classifier",
        zorder=1,
    )

    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.03)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(titulo, fontsize=14, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.35)

    ax.legend(
        fontsize=9,
        loc="lower right",
        frameon=True,
        fancybox=True,
        framealpha=0.93,
    )

    plt.tight_layout()
    plt.savefig(outpath, dpi=600, bbox_inches="tight")
    plt.close()



# 6) NÚCLEO DO MODELO BAYESIANO DISCRETO
# ============================================================
def treinar_cpts_discretas(
    train_df,
    disc_metrics,
    estados,
    classe_col="scenario",
    topo_col="topologyNodes",
    laplace=1.0,
):
    K = len(estados)
    cpts = {}

    for m_d in disc_metrics:
        counts = (
            train_df.groupby([classe_col, topo_col, m_d])
            .size()
            .unstack(fill_value=0)
        )

        for estado in estados:
            if estado not in counts.columns:
                counts[estado] = 0

        counts = counts[estados]
        cpts[m_d] = (counts + laplace).div(counts.sum(axis=1) + laplace * K, axis=0)

    return cpts


def inferir_amostra(row, classes, disc_metrics, cpts, prior_attack, topo_observada, estados):
    K = len(estados)
    log_post = []

    for classe in classes:
        logp = np.log(prior_attack[classe])
        chave = (classe, topo_observada)

        for m_d in disc_metrics:
            valor = row[m_d]
            if chave in cpts[m_d].index:
                prob = cpts[m_d].loc[chave, valor]
            else:
                prob = 1.0 / K
            logp += np.log(prob)

        log_post.append(logp)

    probas = softmax(log_post)
    pred = classes[int(np.argmax(log_post))]
    return pred, probas.tolist()



# 7) EXPERIMENTO PRINCIPAL
# ============================================================
def executar_experimento_rb_otimizado(
    file_path,
    n_iteracoes=100,
    train_per_combo=40,
    test_per_combo=10,
    n_bins=5,
    seed=42,
    usar_atributos_derivados=True,
    laplace=1.0,
    output_dir="resultados_rb_otimizada",
):
    file_path = Path(file_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    df_raw = pd.read_csv(file_path)
    #df = padronizar_dataset_manet(df_raw)
    df = padronizar_Dataset_electrosvalnunes_manet(df_raw)

    metricas_originais = ["PDR", "delayMean_ms", "throughput_bps", "energyMean_J"]

    if usar_atributos_derivados:
        df = adicionar_atributos_relativos_ao_baseline(df, metricas_originais)
        metrics = [
            "PDR",
            "delayMean_ms",
            "throughput_bps",
            "energyMean_J",
            "PDR_zbase",
            "delayMean_ms_zbase",
            "throughput_bps_zbase",
            "energyMean_J_zbase",
            "efficiency_tp_energy",
            "delay_over_pdr",
            "energy_over_pdr",
            "delay_log",
            "energy_log",
        ]
    else:
        metrics = metricas_originais.copy()

    topologias = sorted(df["topologyNodes"].unique().tolist())
    cenarios = sorted(df["scenario"].unique().tolist())

    print("=" * 90)
    print("DATASET PADRONIZADO COM SUCESSO")
    print(f"Arquivo lido: {file_path}")
    print(f"Topologias encontradas: {topologias}")
    print(f"Cenários encontrados: {cenarios}")
    print(f"Total de amostras válidas: {len(df)}")
    print("=" * 90)

    resultados = {
        n: {
            "acc": [],
            "precision_macro": [],
            "recall_macro": [],
            "f1_macro": [],
            "roc_auc_macro": [],
            "cm_iter": [],
            "y_real_all": [],
            "y_proba_all": [],
        }
        for n in topologias
    }

    for it in range(n_iteracoes):
        print(f"\nIteração {it + 1}/{n_iteracoes}")

        train_idx = []
        test_idx = []

        for n in topologias:
            df_n = df[df["topologyNodes"] == n]
            for s in cenarios:
                subset = df_n[df_n["scenario"] == s].index.to_numpy().copy()
                rng.shuffle(subset)

                minimo = train_per_combo + test_per_combo
                if len(subset) < minimo:
                    corte = max(1, len(subset) // 2)
                    train_idx.extend(subset[:corte])
                    test_idx.extend(subset[corte:])
                else:
                    train_idx.extend(subset[:train_per_combo])
                    test_idx.extend(subset[train_per_combo:train_per_combo + test_per_combo])

        train_df = df.loc[train_idx].copy()
        test_df = df.loc[test_idx].copy()

        train_df, test_df, estados = discretizar_por_topologia_sem_leakage(
            train_df=train_df,
            test_df=test_df,
            metrics=metrics,
            topo_col="topologyNodes",
            n_bins=n_bins,
        )
        disc_metrics = [m + "_d" for m in metrics]

        cpts = treinar_cpts_discretas(
            train_df=train_df,
            disc_metrics=disc_metrics,
            estados=estados,
            classe_col="scenario",
            topo_col="topologyNodes",
            laplace=laplace,
        )

        prior_attack = {s: 1.0 / len(cenarios) for s in cenarios}

        for n in topologias:
            test_n = test_df[test_df["topologyNodes"] == n]
            y_real = []
            y_pred = []
            y_proba = []

            for _, row in test_n.iterrows():
                pred, probas = inferir_amostra(
                    row=row,
                    classes=cenarios,
                    disc_metrics=disc_metrics,
                    cpts=cpts,
                    prior_attack=prior_attack,
                    topo_observada=n,
                    estados=estados,
                )
                y_real.append(row["scenario"])
                y_pred.append(pred)
                y_proba.append(probas)

            cm = confusion_matrix(y_real, y_pred, labels=cenarios)
            acc = accuracy_score(y_real, y_pred)
            prec = precision_score(y_real, y_pred, labels=cenarios, average="macro", zero_division=0)
            rec = recall_score(y_real, y_pred, labels=cenarios, average="macro", zero_division=0)
            f1m = f1_score(y_real, y_pred, labels=cenarios, average="macro", zero_division=0)
            roc_macro, _ = roc_auc_ovr_macro(y_real, y_proba, classes=cenarios)

            resultados[n]["acc"].append(acc)
            resultados[n]["precision_macro"].append(prec)
            resultados[n]["recall_macro"].append(rec)
            resultados[n]["f1_macro"].append(f1m)
            resultados[n]["roc_auc_macro"].append(roc_macro)
            resultados[n]["cm_iter"].append(cm)
            resultados[n]["y_real_all"].extend(y_real)
            resultados[n]["y_proba_all"].extend(y_proba)

    linhas_resumo = []
    print("\n" + "=" * 90)
    print("RESULTADOS FINAIS | RB OTIMIZADA | DATASET MANET ADAPTADO")
    print(f"Arquivo: {file_path}")
    print(f"Iterações: {n_iteracoes} | treino={train_per_combo} | teste={test_per_combo} | bins={n_bins}")
    print(f"Seed global: {seed} | Atributos derivados: {usar_atributos_derivados}")
    print("=" * 90)

    for n in topologias:
        acc_m, acc_s = mean_std(resultados[n]["acc"])
        prec_m, prec_s = mean_std(resultados[n]["precision_macro"])
        rec_m, rec_s = mean_std(resultados[n]["recall_macro"])
        f1_m, f1_s = mean_std(resultados[n]["f1_macro"])
        auc_m, auc_s = mean_std(resultados[n]["roc_auc_macro"])

        roc_macro_global, roc_pack = roc_auc_ovr_macro(
            resultados[n]["y_real_all"],
            resultados[n]["y_proba_all"],
            classes=cenarios,
        )

        print(f"\nTOPOLOGIA {n} nós")
        print(f"Accuracy:        {acc_m:.4f} ± {acc_s:.4f}")
        print(f"Precision Macro: {prec_m:.4f} ± {prec_s:.4f}")
        print(f"Recall Macro:    {rec_m:.4f} ± {rec_s:.4f}")
        print(f"F1 Macro:        {f1_m:.4f} ± {f1_s:.4f}")
        print(f"ROC-AUC Macro:   {auc_m:.4f} ± {auc_s:.4f}")
        print(f"ROC-AUC Global:  {roc_macro_global:.4f}")

        f1_list = np.asarray(resultados[n]["f1_macro"], dtype=float)
        idx_rep = int(np.argmin(np.abs(f1_list - f1_m)))
        cm_rep = resultados[n]["cm_iter"][idx_rep]

        out_cm = output_dir / f"matriz_confusao_{n}_nos_representativa.png"
        plot_confusion(
            cm_rep,
            labels=cenarios,
            title=f"Matriz de Confusão - {n} nós",
            outpath=out_cm,
        )

        all_fpr, mean_tpr, roc_auc_ind, fpr_ind, tpr_ind, overlap_flags = roc_pack

        out_roc = output_dir / f"roc_auc_{n}_nos_global.png"
        plot_roc_melhorada(
            all_fpr=all_fpr,
            mean_tpr=mean_tpr,
            roc_auc_macro=roc_macro_global,
            roc_auc_ind=roc_auc_ind,
            fpr_ind=fpr_ind,
            tpr_ind=tpr_ind,
            classes=cenarios,
            titulo=f"Curva ROC (OvR) - {n} nós",
            outpath=out_roc,
        )

        for (i, j), is_overlap in overlap_flags.items():
            if is_overlap:
                print(
                    f"[Aviso] Em {n} nós, as curvas ROC de '{cenarios[i]}' e '{cenarios[j]}' "
                    f"estão praticamente sobrepostas."
                )

        linhas_resumo.append(
            {
                "topologyNodes": n,
                "accuracy_mean": acc_m,
                "accuracy_std": acc_s,
                "precision_macro_mean": prec_m,
                "precision_macro_std": prec_s,
                "recall_macro_mean": rec_m,
                "recall_macro_std": rec_s,
                "f1_macro_mean": f1_m,
                "f1_macro_std": f1_s,
                "roc_auc_macro_mean": auc_m,
                "roc_auc_macro_std": auc_s,
                "roc_auc_global": roc_macro_global,
            }
        )

    resumo_df = pd.DataFrame(linhas_resumo)
    resumo_csv = output_dir / "resumo_metricas_por_topologia.csv"
    resumo_df.to_csv(resumo_csv, index=False)

    print(f"\nResumo salvo em: {resumo_csv}")
    print("\nConcluído.")

    return resumo_df


if __name__ == "__main__":
    executar_experimento_rb_otimizado(
        file_path="Dataset_electrosvalnunes_manet.csv",
        n_iteracoes=100,#50
        train_per_combo=2000,#40,
        test_per_combo=1000,#10,
        n_bins=5,
        seed=42,
        usar_atributos_derivados=True,
        laplace=1.0,
        output_dir="resultados_RB_electrosvalnunes_manet",
    )
