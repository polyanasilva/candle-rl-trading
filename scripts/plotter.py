import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import itertools
import mplfinance as mpf

class Plotter:
    def __init__(self, custom_palette='Set1'):
        self.palette = custom_palette

    def plot_cluster_distribution(self, data, features, figsize=(14, 12)):
        plt.figure(figsize=figsize)

        for i, feature in enumerate(features, 1):
            plt.subplot(len(features), 1, i)
            sns.stripplot(
                x=f'{feature}_cluster',
                y=feature,
                data=data,
                hue=f'{feature}_cluster',
                palette=self.palette,
                jitter=True,
                alpha=0.8
            )
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.title(f'Clustering of {feature}')
            plt.xlabel('')
            plt.legend(title=f'{feature} cluster', bbox_to_anchor=(1, 1), loc='upper left')

        plt.tight_layout()
        plt.show()

    def plot_candlestick_with_clusters(self, data, title='Candlestick chart with candles clustering'):
        mc = mpf.make_marketcolors(up='g', down='r', wick='black', edge='inherit')
        style = mpf.make_mpf_style(marketcolors=mc)

        fig, ax = plt.subplots(figsize=(12, 6))
        mpf.plot(data, type='candle', ax=ax, style=style, show_nontrading=True)
        plt.grid(True, linestyle='--', alpha=0.7)

        for i, row in data.iterrows():
            ax.text(row.name, row['High'] + 0.1, int(row['cluster']), 
                    fontsize=8, ha='center', va='bottom', color='black')

        ax.set_title(title, fontsize=14)
        ax.set_ylabel('Price', fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def sort_data_for_cluster_plot(self, data, samples_per_cluster=7):
        data_sorted = data.sort_values(by='cluster').reset_index(drop=True)
        return data_sorted.groupby('cluster').head(samples_per_cluster)

    def plot_cluster_candles(self, data, cluster_label):
        fig, ax = plt.subplots(figsize=(12, 6))
        plt.grid(True, linestyle='--', alpha=0.7)

        candle_width, shadow_width = 0.7, 0.02
        color_up, color_down, color_wick = 'green', 'red', 'black'

        if len(data) <= 3:
            candle_width, shadow_width = 0.5, 0.004

        up = data[data['body'] >= 0]
        down = data[data['body'] < 0]

        ax.bar(up.index, up['body'], candle_width, color=color_up, label='Up')
        ax.bar(up.index, up['upper_shadow'], shadow_width, bottom=up['body'], color=color_wick)
        ax.bar(up.index, up['lower_shadow'], shadow_width, bottom=-up['lower_shadow'], color=color_wick)

        ax.bar(down.index, -down['body'], candle_width, color=color_down, label='Down')
        ax.bar(down.index, down['upper_shadow'], shadow_width, bottom=-down['body'], color=color_wick)
        ax.bar(down.index, down['lower_shadow'], shadow_width, bottom=-down['lower_shadow'], color=color_wick)

        ax.set_ylim(-15, 15)
        ax.set_title(cluster_label, fontsize=14)
        ax.set_ylabel('Price', fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def plot_rl_actions(self, df_path, n_samples=50):
        df = pd.read_csv(df_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.iloc[1:n_samples]

        plt.figure(figsize=(16, 8))
        plt.plot(df['Date'], df['Close'], label='Preço de Fechamento', color='black', linewidth=1)

        buy_signals = df[df['best_action'] == "Comprar"]
        sell_signals = df[df['best_action'] == "Vender"]
        hold_signals = df[df['best_action'] == "Segurar"]

        plt.scatter(buy_signals['Date'], buy_signals['Close'], label='Compra', marker='^', color='green', s=100)
        plt.scatter(sell_signals['Date'], sell_signals['Close'], label='Venda', marker='v', color='red', s=100)
        plt.scatter(hold_signals['Date'], hold_signals['Close'], label='Manter', marker='o', color='blue', s=60)

        plt.xticks(rotation=45)
        plt.xlabel('Data')
        plt.ylabel('Preço de Fechamento')
        plt.title('Ações Tomadas pelo Agente de RL ao Longo do Tempo')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

