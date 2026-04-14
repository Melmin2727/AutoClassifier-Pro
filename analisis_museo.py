import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
import tkinter as tk
from tkinter import filedialog, messagebox
import csv
import io

def ejecutar_analisis_completo_robusto_v16():
    root = tk.Tk()
    root.withdraw()
    
    file_path = filedialog.askopenfilename(
        title="Seleccionar archivo (Excel o CSV) - Análisis Multivariado V16 COMPLETO",
        filetypes=[("Archivos de datos", "*.xlsx *.xls *.csv")]
    )
    
    if not file_path: 
        return

    try:
        # 🔥 CARGA ULTRA-FLEXIBLE (INTACTA)
        print("🔍 Analizando archivo...")
        df = None
        
        if file_path.lower().endswith('.csv'):
            print("🔍 CSV: Detectando separador...")
            separador_usado = None
            
            try:
                with open(file_path, 'r', encoding='utf-8-sig', errors='ignore') as f:
                    muestra = f.read(4096)
                    dialect = csv.Sniffer().sniff(muestra)
                    separador_usado = dialect.delimiter
                    print(f"✅ Sniffer detectó: '{separador_usado}'")
                    
                    df_preview = pd.read_csv(file_path, sep=separador_usado, 
                                           encoding='utf-8-sig', nrows=1000,
                                           on_bad_lines='skip')
                    if len(df_preview.columns) > 1:
                        df = pd.read_csv(file_path, sep=separador_usado, 
                                       encoding='utf-8-sig', on_bad_lines='skip')
                        print(f"✅ CSV completo: shape={df.shape}")
                    else:
                        raise ValueError("Sniffer falló")
                        
            except Exception as e_sniff:
                print(f"⚠️ Sniffer falló: {e_sniff}")
                for sep in [';', ',']:
                    for encoding in ['utf-8-sig', 'latin1']:
                        try:
                            df_preview = pd.read_csv(file_path, sep=sep, 
                                                   encoding=encoding, nrows=1000,
                                                   on_bad_lines='skip')
                            if len(df_preview.columns) > 1:
                                df = pd.read_csv(file_path, sep=sep, encoding=encoding,
                                               on_bad_lines='skip')
                                separador_usado = sep
                                print(f"✅ Fallback OK: sep='{sep}', shape={df.shape}")
                                break
                        except:
                            continue
        else:
            df = pd.read_excel(file_path)
            print(f"✅ Excel: shape={df.shape}")
        
        if df.empty or len(df.columns) < 2:
            raise ValueError("Archivo mal leído o vacío")
        
        print(f"📊 Shape final: {df.shape}")
        df.columns = df.columns.str.strip().str.upper()

        # 🔥 MUESTRA INTELIGENTE
        print(f"Filas originales: {len(df)}")
        MAX_MUESTRA = 5000
        
        if len(df) > MAX_MUESTRA:
            df_muestra = df.sample(n=MAX_MUESTRA, random_state=42).reset_index(drop=True)
            print(f"⚠️ MUESTRA: {MAX_MUESTRA} filas")
            df_completo = df
        else:
            df_muestra = df.copy()
            df_completo = df.copy()
            print("✅ Usando datos completos")

        # FILTRO MEJORADO
        X_num = df_muestra.select_dtypes(include=[np.number]).copy()
        columnas_prohibidas = [
            'ID', 'ID_MES', 'CASO', 'NRO', 'NRO_CASO', 'ANIO', 'AÑO', 'MES', 'FECHA',
            'CORTE', 'PAIS', 'CONTINENTE', 'ASIA', 'PUNO', 'EXCURSIONISTAS', 'COD'
        ]
        cols_omitir = [c for c in X_num.columns if any(palabra in c.upper() for palabra in columnas_prohibidas)]
        X_muestra = X_num.drop(columns=cols_omitir)
        
        X_muestra = X_muestra.replace([np.inf, -np.inf], np.nan).dropna()
        columnas_con_varianza = X_muestra.columns[X_muestra.std() > 0.01].tolist()
        X_muestra = X_muestra[columnas_con_varianza]
        
        print(f"🔍 Variables: {len(columnas_con_varianza)}")
        print(f"📊 Muestra procesada: {X_muestra.shape}")

        if X_muestra.shape[1] < 1:
            messagebox.showerror("Error", "❌ Sin variables numéricas válidas")
            return

        # ESTANDARIZACIÓN MUESTRA
        scaler_muestra = StandardScaler()
        X_scaled_muestra = scaler_muestra.fit_transform(X_muestra)
        
        print("🔍 Filas después de limpieza:", len(X_scaled_muestra))  # ✅ VERIFICACIÓN

        # 🔥 0. SCREE PLOT ACUMULADO (PRO) ✅ REEMPLAZADO
        print("📊 0. SCREE PLOT (Varianza Acumulada)...")
        pca_full = PCA()
        pca_full.fit(X_scaled_muestra)

        varianza_acumulada = np.cumsum(pca_full.explained_variance_ratio_)

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(
            range(1, len(varianza_acumulada) + 1),
            varianza_acumulada,
            marker='o',
            linewidth=3,
            markersize=8,
            color='darkblue'
        )

        # Líneas de referencia PROFESIONALES
        ax.axhline(y=0.80, color='red', linestyle='--', linewidth=2, label='80% Varianza')
        ax.axhline(y=0.90, color='orange', linestyle='--', linewidth=2, label='90% Varianza')

        ax.set_title("SCREE PLOT - VARIANZA EXPLICADA ACUMULADA",
                     fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel("Número de Componentes Principales")
        ax.set_ylabel("Varianza Acumulada")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        # 🔥 1. DENDROGRAMA ✅ FIXED con fig, ax
        print("📊 1. DENDROGRAMA...")
        fig, ax = plt.subplots(figsize=(14, 6))
        Z = linkage(X_scaled_muestra, method='ward')
        dendrogram(Z, leaf_rotation=90, leaf_font_size=10, truncate_mode='lastp', p=30, ax=ax)
        ax.set_title(f"DENDROGRAMA (muestra {len(X_muestra)} casos)", fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Índice del Caso')
        ax.set_ylabel('Distancia')
        plt.tight_layout()
        plt.show()

        # 🔥 2. MATRIZ CORRELACIÓN ✅ FIXED con fig, ax
        print("📊 2. MATRIZ DE CORRELACIÓN...")
        fig, ax = plt.subplots(figsize=(12, 10))
        corr = X_muestra.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, cmap='coolwarm', fmt=".2f", 
                   linewidths=0.5, center=0, cbar_kws={"shrink": .8}, square=True, ax=ax)
        ax.set_title('MATRIZ DE CORRELACIÓN (Triangular Profesional)', fontsize=16, fontweight='bold', pad=20)
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        plt.show()

        # 🔥 3. ANÁLISIS CLUSTERS ✅ CORREGIDO CON SOLUCIÓN INTELIGENTE
        print("📊 3. ANÁLISIS DE CLUSTERS...")
        
        # 🚀 SOLUCIÓN INTELIGENTE (MEJOR)
        max_k = min(10, len(X_scaled_muestra) - 1)
        if max_k < 3:
            max_k = 3
        K_range = range(2, max_k + 1)  # +1 para incluir max_k
        
        print("🔍 K evaluados:", list(K_range))  # ✅ VERIFICACIÓN
        
        inertia = []
        sil_scores = []
        
        for k in K_range:
            kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels_temp = kmeans_temp.fit_predict(X_scaled_muestra)
            inertia.append(kmeans_temp.inertia_)
            sil_scores.append(silhouette_score(X_scaled_muestra, labels_temp))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.plot(K_range, inertia, marker='o', linewidth=2, markersize=8)
        ax1.set_title("MÉTODO DEL CODO", fontsize=16, fontweight='bold', pad=15)
        ax1.set_xlabel("Número de clusters")
        ax1.set_ylabel("Inercia")
        ax1.grid(True, alpha=0.3)

        ax2.plot(K_range, sil_scores, marker='s', color='orange', linewidth=2, markersize=8)
        ax2.set_title("MÉTODO DE SILUETA", fontsize=16, fontweight='bold', pad=15)
        ax2.set_xlabel("Número de clusters")
        ax2.set_ylabel("Coeficiente de Silueta")
        ax2.grid(True, alpha=0.3)
        
        fig.suptitle("ANÁLISIS DE CLUSTERS - Selección Óptima de K", fontsize=18, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.show()

        # 🔥 CLUSTERIZACIÓN FORZADA A 3 GRUPOS
        print("🎯 CLUSTERIZACIÓN: 3 GRUPOS...")
        n_clusters = 3
        
        kmeans_muestra = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42, n_init=20)
        clusters_muestra = kmeans_muestra.fit_predict(X_scaled_muestra)
        
        X_completo = df_completo.select_dtypes(include=[np.number])
        X_completo = X_completo.drop(columns=cols_omitir, errors='ignore')
        X_completo = X_completo[columnas_con_varianza].replace([np.inf, -np.inf], np.nan).dropna()
        
        scaler_completo = StandardScaler()
        X_scaled_completo = scaler_completo.fit_transform(X_completo)
        
        kmeans_completo = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42, n_init=20)
        clusters = kmeans_completo.fit_predict(X_scaled_completo)

        # 🔥 4. CENTROIDES ✅ FIXED con fig, ax
        print("📊 4. DIAGRAMA DE CENTROIDES...")
        centroides_df = pd.DataFrame(kmeans_completo.cluster_centers_, columns=columnas_con_varianza)
        centroides_df.index = [f'Grupo {i}' for i in range(n_clusters)]
        
        fig, ax = plt.subplots(figsize=(14, 7))
        centroides_df.T.plot(kind='bar', width=0.8, colormap='viridis', edgecolor='black', linewidth=0.5, ax=ax)
        ax.axhline(0, color='black', linewidth=1, alpha=0.5)
        ax.set_title('DIAGRAMA DE CENTROIDES (Perfil por Grupo)', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Variables')
        ax.set_ylabel('Valor Estandarizado')
        ax.legend(title='Grupos', fontsize=11)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        # 🔥 6. DIAGRAMA DE PERFIL - NUEVO 📌 DESPUÉS DE CENTROIDES
        print("📊 6. DIAGRAMA DE PERFIL...")
        df_perfil = X_completo.copy()
        df_perfil['GRUPO'] = clusters
        
        perfil = df_perfil.groupby('GRUPO').mean()
        
        fig, ax = plt.subplots(figsize=(14, 7))
        perfil.T.plot(
            kind='line',
            marker='o',
            linewidth=3,
            markersize=8,
            ax=ax
        )
        ax.set_title("DIAGRAMA DE PERFIL DE CLÚSTERES", fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel("Variables")
        ax.set_ylabel("Valor Promedio")
        ax.legend(title="Grupos", fontsize=11)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        plt.show()

        # 🔥 5. DISPERSIÓN PCA ✅ FIXED con fig, ax
        print("📊 5. DISPERSIÓN PCA...")
        max_puntos = min(1000, len(X_scaled_muestra))
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled_muestra[:max_puntos])
        clusters_pca = clusters_muestra[:max_puntos]
        
        fig, ax = plt.subplots(figsize=(14, 9))
        
        colores = sns.color_palette("Set1", n_clusters)
        marcadores = ['o', 'X', 's']
        
        for i in range(n_clusters):
            idx = clusters_pca == i
            ax.scatter(X_pca[idx, 0], X_pca[idx, 1],
                      c=[colores[i]], marker=marcadores[i], s=120,
                      edgecolor='black', linewidth=0.8, label=f'Grupo {i}', alpha=0.85)
        
        for i in range(min(40, len(X_pca))):
            ax.text(X_pca[i, 0]*1.015, X_pca[i, 1]*1.015, str(i+1),
                   fontsize=9, color='darkred', fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
        
        ax.set_title("DISPERSIÓN DE CLÚSTERES: Análisis de Ejes Canónicos", fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel(f"Eje Canónico 1 ({pca.explained_variance_ratio_[0]*100:.1f}% varianza)")
        ax.set_ylabel(f"Eje Canónico 2 ({pca.explained_variance_ratio_[1]*100:.1f}% varianza)")
        ax.legend(title="Segmentos Identificados", fontsize=11)
        ax.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()

        # RESUMEN FINAL
        print(f"\n🎯 COMPLETADO: {n_clusters} grupos forzados")
        print(f"Variables: {len(columnas_con_varianza)}")
        print(f"Distribución: {pd.Series(clusters).value_counts().sort_index()}")

        messagebox.showinfo("✅ V16 COMPLETO (7 GRÁFICOS)", 
                           f"7 Gráficos generados:\n"
                           f"📊 0.Scree Plot ACUMULADO\n"
                           f"🌳 1.Dendrograma\n"
                           f"🔗 2.Correlación\n"
                           f"📈 3.Codo+Silueta ({len(K_range)} K's)\n"
                           f"📊 4.Centroides\n"
                           f"📉 6.Perfil\n"
                           f"🔍 5.PCA\n"
                           f"{n_clusters} Grupos | {len(columnas_con_varianza)} Variables")

    except Exception as e:
        error_msg = str(e)
        if "demasiado grande" in error_msg.lower():
            messagebox.showerror("❌ Muy Grande", f"{error_msg}\n💡 Usa muestra pequeña.")
        else:
            messagebox.showerror("❌ Error", error_msg)

if __name__ == "__main__":
    ejecutar_analisis_completo_robusto_v16()