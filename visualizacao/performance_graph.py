

# visualizacao/performance_graph.py
import matplotlib.pyplot as plt

def plot_performance_graph(results, filename):
    """
    Gerar gráfico de barras de desempenho baseado nas métricas (F1-macro, F1-weighted).
    """
    labels = [f"{r['Representation']} + {r['Classifier']} ({r['Train Size']*100}%)" for r in results]
    f1_macro_scores = [r['F1-macro'] for r in results]
    f1_weighted_scores = [r['F1-weighted'] for r in results]

    x = range(len(labels))

    plt.figure(figsize=(10, 6))
    plt.bar(x, f1_macro_scores, width=0.4, label='F1-macro', align='center', color='b')
    plt.bar(x, f1_weighted_scores, width=0.4, label='F1-weighted', align='edge', color='r')

    plt.xlabel("Configuração do Classificador")
    plt.ylabel("Métrica F1")
    plt.xticks(x, labels, rotation=90)
    plt.legend()

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
