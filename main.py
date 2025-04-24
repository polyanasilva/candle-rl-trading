import pandas as pd
from scripts.data_handler import DataHandler
from scripts.clustering_model import ClusteringModel
from scripts.reinforcement_agent import ReinforcementAgent
from scripts.plotter import Plotter

def main():
    print("Iniciando pipeline do projeto...")

    # ========== 1. Pré-processamento ==========
    print("1. Pré-processando os dados...")
    preprocessor = DataHandler(filepath='data/dados.csv')
    preprocessor.load_data()
    preprocessor.calculate_candle_features()
    preprocessor.remove_outliers()
    preprocessor.normalize_features()
    preprocessor.combine_features()
    
    data = preprocessor.get_normalized_data()
    data_original = preprocessor.get_original_data()

    # ========== 2. Clusterização ==========
    print("2. Aplicando clusterização...")
    clusterer = ClusteringModel(data)
    clusterer.apply_kmeans(['body', 'lower_shadow', 'upper_shadow'], n_clusters=3)
    clusterer.combine_clusters()
    clustered_data = clusterer.get_data()

    # ========== 3. Treinamento de Aprendizado por Reforço ==========
    print("3. Treinando agente de RL...")
    agent = ReinforcementAgent(data=clustered_data)
    agent.train()
    df_with_actions = agent.get_results()
    df_with_actions.to_csv('saida_com_acoes.csv', index=False)

    # ========== 4. Visualização ==========
    print("4. Gerando visualizações...")
    plotter = Plotter()

    # Gráficos dos clusters individuais
    plotter.plot_cluster_distribution(clustered_data, features=['body', 'lower_shadow', 'upper_shadow'])

    # Gráfico candlestick com clusters
    plotter.plot_candlestick_with_clusters(data_original.iloc[415:425])

    # Candles por cluster
    data_limited = plotter.sort_data_for_cluster_plot(clustered_data)
    samples_per_cluster = [min(count, 7) for count in clustered_data.sort_values('cluster')['cluster'].value_counts().sort_index()]
    cumulative_samples = [0] + list(pd.Series(samples_per_cluster).cumsum())

    for i in range(len(samples_per_cluster)):
        start_idx = cumulative_samples[i]
        end_idx = cumulative_samples[i + 1]
        cluster_data = data_limited.iloc[start_idx:end_idx]
        plotter.plot_cluster_candles(cluster_data, f'Cluster {i + 1}')

    # Ações tomadas pelo agente
    plotter.plot_rl_actions('saida_com_acoes.csv')

    print("Pipeline concluído com sucesso!")

if __name__ == "__main__":
    main()
