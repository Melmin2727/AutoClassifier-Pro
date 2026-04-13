import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
import tkinter as tk
from tkinter import filedialog, messagebox

def ejecutar_analisis_completo():
    root = tk.Tk()
    root.withdraw()
    
    file_path = filedialog.askopenfilename(
        title="Seleccionar Excel para Análisis Multivariado",
        filetypes=[("Archivos de Excel", "*.xlsx *.xls")]
    )
    
    if not file_path: return

    try:
        # 1. Carga y limpieza automática
        df = pd.read_excel(file_path)
        df.columns = df.columns.str.strip()
        
        # Identificar columnas numéricas (omitir IDs para el cálculo)
        X_num = df.select_dtypes(include=[np.number])
        cols_omitir = [c for c in X_num.columns if 'id' in c.lower() or 'caso' in c.lower()]
        X = X_num.drop(columns=cols_omitir)

        # 2. Estandarización (Paso fundamental en Análisis Multivariado)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # --- 1er DIAGRAMA: DENDROGRAMA (Aparece primero) ---
        plt.figure(figsize=(12, 6))
        Z = linkage(X_scaled, method='ward')
        dendrogram(Z, leaf_rotation=90, leaf_font_size=10)
        plt.title('DIAGRAMA DENDROGRAMA: Agrupación Jerárquica', fontsize=14, fontweight='bold')
        plt.xlabel('Índice del Caso / Estudiante')
        plt.ylabel('Distancia Euclídea (Similitud)')
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()

        # 3. Cálculo de Clústeres (K-Means)
        n_clusters = 3 # Puedes ajustar este número según el dendrograma
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        df['cluster_resultado'] = kmeans.fit_predict(X_scaled)

        # --- 2do DIAGRAMA: DISPERSIÓN DE CLÚSTERES (Ejes Canónicos / PCA) ---
        pca = PCA(n_components=2)
        pca_data = pca.fit_transform(X_scaled)
        df['PCA1'] = pca_data[:, 0]
        df['PCA2'] = pca_data[:, 1]

        plt.figure(figsize=(12, 8))
        sns.scatterplot(
            data=df, x='PCA1', y='PCA2', hue='cluster_resultado', 
            palette='Set1', s=130, style='cluster_resultado', markers=True, edgecolor='black'
        )

        # Etiquetas numéricas para cada punto (Solicitado)
        for i in range(df.shape[0]):
            plt.text(
                df.PCA1[i] + 0.12, df.PCA2[i] + 0.12, 
                str(i + 1), # Representa el número de fila o caso
                fontsize=10, fontweight='bold', color='darkred'
            )

        plt.title('DISPERSIÓN DE CLÚSTERES: Análisis de Ejes Canónicos', fontsize=14, fontweight='bold')
        plt.xlabel(f'Eje Canónico 1 ({pca.explained_variance_ratio_[0]:.2%} de varianza)')
        plt.ylabel(f'Eje Canónico 2 ({pca.explained_variance_ratio_[1]:.2%} de varianza)')
        plt.legend(title="Segmentos Identificados", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.show()

        # --- 3er DIAGRAMA: PERFIL DE CLÚSTERES ---
        plt.figure(figsize=(11, 6))
        df_perfil = df.groupby('cluster_resultado')[X.columns].mean()
        
        for cluster_id in df_perfil.index:
            plt.plot(X.columns, df_perfil.loc[cluster_id], marker='o', label=f'Clúster {cluster_id}', linewidth=2.5)
        
        plt.title('DIAGRAMA DE PERFIL: Comportamiento por Grupo', fontsize=14, fontweight='bold')
        plt.ylabel('Valor Promedio Estandarizado')
        plt.xticks(rotation=20)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        messagebox.showinfo("Proceso Exitoso", f"Se analizaron las variables: {', '.join(X.columns)}")

    except Exception as e:
        messagebox.showerror("Error de Ejecución", f"Hubo un problema con el archivo: {e}")

if __name__ == "__main__":
    ejecutar_analisis_completo()gaaaaaaa