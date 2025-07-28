# business_impact_metrics.py (ATUALIZADO)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

## --- Modelagem de Perdas (LGD, EAD, Expected Loss) ---

def simulate_lgd_ead(n_samples, random_seed=42):
    """
    Simula valores de Loss Given Default (LGD) e Exposure at Default (EAD).
    Em um cenário real, estes seriam modelos próprios treinados com dados históricos.

    Args:
        n_samples (int): Número de amostras para simular.
        random_seed (int): Semente para reprodutibilidade.

    Returns:
        pd.DataFrame: DataFrame com as colunas 'lgd' e 'ead'.
    """
    np.random.seed(random_seed)
    # LGD: Valor entre 0 e 1, representando a % perdida (Ex: Beta distribution)
    # Ajustando para simular um LGD médio um pouco menor para valores mais realistas no EL
    lgd = np.random.beta(a=1.5, b=8, size=n_samples) # Mais concentrado em LGDs menores
    lgd = np.clip(lgd, 0.1, 0.9) # Garante LGD entre 10% e 90%

    # EAD: Exposição no momento do default (simula um valor em Reais)
    ead = np.random.normal(loc=12000, scale=4000, size=n_samples) # EAD médio um pouco menor
    ead = np.maximum(1000, ead).round(2) # EAD mínimo de R$1000

    return pd.DataFrame({'lgd': lgd, 'ead': ead})

def calculate_expected_loss(df_with_pd, pd_column='prob_default', random_seed=42):
    """
    Calcula a Perda Esperada (Expected Loss) para cada cliente.
    EL = PD * LGD * EAD

    Args:
        df_with_pd (pd.DataFrame): DataFrame que deve conter a coluna de PD.
        pd_column (str): Nome da coluna com as probabilidades de default (PD).
        random_seed (int): Semente para reprodutibilidade ao simular LGD/EAD.

    Returns:
        pd.DataFrame: O DataFrame original com novas colunas 'lgd', 'ead' e 'expected_loss'.
    """
    # Simular LGD e EAD para os mesmos clientes que temos PD
    simulated_lgd_ead = simulate_lgd_ead(len(df_with_pd), random_seed=random_seed)
    df_with_pd['lgd'] = simulated_lgd_ead['lgd']
    df_with_pd['ead'] = simulated_lgd_ead['ead']

    df_with_pd['expected_loss'] = df_with_pd[pd_column] * df_with_pd['lgd'] * df_with_pd['ead']
    return df_with_pd

def plot_expected_loss_distribution(df, expected_loss_column='expected_loss'):
    """
    Plota a distribuição da Perda Esperada.

    Args:
        df (pd.DataFrame): DataFrame contendo a coluna de Expected Loss.
        expected_loss_column (str): Nome da coluna com os valores de Expected Loss.

    Returns:
        matplotlib.figure.Figure: Objeto Figure do Matplotlib.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df[expected_loss_column], bins=50, kde=True, ax=ax)
    ax.set_title('Distribuição da Perda Esperada por Cliente')
    ax.set_xlabel('Perda Esperada (R$)')
    ax.set_ylabel('Frequência')
    return fig

## --- Métricas de Rentabilidade Ajustada ao Risco (RORAC/RAROC) ---

def calculate_rorac_raroc(total_expected_loss, capital_allocated, revenue):
    """
    Calcula o RAROC (Risk-Adjusted Return on Capital).
    RAROC = (Receita - Perda Esperada) / Capital Alocado para Risco

    Args:
        total_expected_loss (float): Soma das perdas esperadas do portfólio.
        capital_allocated (float): Capital alocado para cobrir os riscos (Regulatório/Econômico).
        revenue (float): Receita bruta gerada pelo portfólio.

    Returns:
        float: O valor do RAROC. Retorna 0 se capital_allocated for <= 0.
    """
    if capital_allocated <= 0:
        return 0.0 # Evitar divisão por zero, retornar 0.0 para clareza

    raroc = (revenue - total_expected_loss) / capital_allocated
    return raroc

## --- Backtesting do Modelo (Simulado) ---

def simulate_backtesting_performance(df_with_pd, n_periods=12, random_seed=42):
    """
    Simula o desempenho do modelo ao longo de múltiplos períodos (ex: meses)
    e compara a probabilidade de default prevista com a taxa de default real simulada.

    Args:
        df_with_pd (pd.DataFrame): DataFrame com 'prob_default' e 'default' (observado).
        n_periods (int): Número de períodos históricos para simular.
        random_seed (int): Semente para reprodutibilidade.

    Returns:
        pd.DataFrame: DataFrame com as colunas 'Periodo', 'PD_Media_Prevista', 'Taxa_Default_Real_Simulada'.
    """
    np.random.seed(random_seed)
    results = []
    total_customers = len(df_with_pd)

    for i in range(1, n_periods + 1):
        # Simular um subconjunto de clientes para o período (como se fosse um novo mês de concessão)
        period_customers = df_with_pd.sample(n=min(1000, total_customers), random_state=random_seed + i, replace=True)

        # PD média prevista para o período
        avg_predicted_pd = period_customers['prob_default'].mean()

        # Simular defaults reais para o período
        # Adicionar um pouco de ruído para simular flutuações reais
        simulated_defaults = (period_customers['prob_default'] + np.random.normal(0, 0.02, len(period_customers))).apply(lambda x: np.clip(x, 0, 1))
        actual_defaulted = (np.random.rand(len(period_customers)) < simulated_defaults).astype(int)
        actual_default_rate = actual_defaulted.mean()

        results.append({
            'Periodo': f'Mês {i}',
            'PD_Media_Prevista': avg_predicted_pd,
            'Taxa_Default_Real_Simulada': actual_default_rate
        })
    return pd.DataFrame(results)

def plot_backtesting_results(df_results):
    """
    Plota os resultados do backtesting simulado.

    Args:
        df_results (pd.DataFrame): DataFrame com 'Periodo', 'PD_Media_Prevista', 'Taxa_Default_Real_Simulada'.

    Returns:
        matplotlib.figure.Figure: Objeto Figure do Matplotlib.
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(df_results['Periodo'], df_results['PD_Media_Prevista'] * 100, marker='o', label='PD Média Prevista (%)')
    ax.plot(df_results['Periodo'], df_results['Taxa_Default_Real_Simulada'] * 100, marker='x', linestyle='--', label='Taxa de Default Real Simulada (%)')
    ax.set_title('Backtesting do Modelo: PD Prevista vs. Default Real Simulada ao Longo do Tempo')
    ax.set_xlabel('Período')
    ax.set_ylabel('Taxa (%)')
    ax.grid(True)
    ax.legend()
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig

## --- VaR (Value at Risk) de Crédito (Simulado) ---

def calculate_credit_var(df_with_losses, confidence_level=0.99, n_simulations=10000, random_seed=42):
    """
    Calcula o VaR de Crédito usando simulação Monte Carlo.
    Simula perdas para o portfólio e encontra o percentil de perdas.

    Args:
        df_with_losses (pd.DataFrame): DataFrame com 'prob_default', 'lgd', 'ead'.
        confidence_level (float): Nível de confiança para o VaR (ex: 0.99 para 99% VaR).
        n_simulations (int): Número de simulações Monte Carlo.
        random_seed (int): Semente para reprodutibilidade.

    Returns:
        tuple: (VaR_value, fig_loss_distribution)
    """
    np.random.seed(random_seed)
    simulated_total_losses = []
    num_clients = len(df_with_losses)

    for _ in range(n_simulations):
        # Para cada cliente, simular se houve default (com base na prob_default)
        # e calcular a perda se houver default (LGD * EAD)
        random_numbers = np.random.rand(num_clients)
        # 1 se random_number < PD, 0 caso contrário
        defaulted = (random_numbers < df_with_losses['prob_default']).astype(int)
        
        # Calcular a perda para cada cliente: LGD * EAD se default, 0 caso contrário
        losses = defaulted * df_with_losses['lgd'] * df_with_losses['ead']
        
        # Somar as perdas de todos os clientes nesta simulação
        simulated_total_losses.append(losses.sum())

    # Calcular o VaR como o percentil da distribuição de perdas simuladas
    var_value = np.percentile(simulated_total_losses, confidence_level * 100)

    # Plotar a distribuição das perdas simuladas
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.histplot(simulated_total_losses, bins=100, kde=True, ax=ax, color='lightcoral')
    ax.axvline(x=np.mean(simulated_total_losses), color='green', linestyle='--', label=f'Perda Média: {np.mean(simulated_total_losses):,.0f} R$')
    ax.axvline(x=var_value, color='red', linestyle='-', label=f'VaR {int(confidence_level*100)}%: {var_value:,.0f} R$')
    ax.set_title(f'Distribuição de Perdas Totais Simuladas do Portfólio (Monte Carlo)')
    ax.set_xlabel('Perda Total do Portfólio (R$)')
    ax.set_ylabel('Frequência')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()

    return var_value, fig

# Exemplo de uso (para teste local)
if __name__ == '__main__':
    print("Testando business_impact_metrics.py com novas funções...")

    # Gerar dados fictícios com PD
    np.random.seed(42)
    n_test_samples = 1000
    df_test_pd = pd.DataFrame({
        'prob_default': np.random.beta(a=1, b=20, size=n_test_samples), # PDs mais realistas
        'default': np.random.randint(0, 2, n_test_samples) # Dummy default para teste
    })

    # Teste de LGD/EAD e Expected Loss
    df_test_el = calculate_expected_loss(df_test_pd.copy())
    print(f"Expected Loss (head):\n{df_test_el.head()}")
    print(f"Soma Total Perda Esperada: {df_test_el['expected_loss'].sum():,.2f} R$")

    # Teste de RORAC/RAROC
    total_el = df_test_el['expected_loss'].sum()
    total_revenue = 500000 # Exemplo de receita
    total_capital = 100000  # Exemplo de capital alocado
    raroc_val = calculate_rorac_raroc(total_el, total_capital, total_revenue)
    print(f"RAROC calculado: {raroc_val:.2%}")

    # Teste de Backtesting
    backtest_results = simulate_backtesting_performance(df_test_pd, n_periods=6)
    print(f"\nResultados do Backtesting (head):\n{backtest_results.head()}")
    fig_backtest = plot_backtesting_results(backtest_results)
    plt.show()

    # Teste de VaR
    var_value_test, fig_var_test = calculate_credit_var(df_test_el, confidence_level=0.95, n_simulations=5000)
    print(f"\nVaR de Crédito 95%: {var_value_test:,.2f} R$")
    plt.show()