# mancredit_streamlit.py (ATUALIZADO com Backtesting e VaR)

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report, precision_recall_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from streamlit_option_menu import option_menu

# Importar as funções dos arquivos de métricas e impacto de negócio
# Certifique-se que o nome do arquivo corresponde ao que você salvou (mancredit_metrics.py e mancredit_businessimpact.py)
from metrics_and_monitoring import calculate_ks_gini_ap, plot_precision_recall_curve, monitor_psi, plot_vintage_analysis
from business_impact_metrics import (
    calculate_expected_loss, 
    plot_expected_loss_distribution,
    calculate_rorac_raroc, 
    simulate_backtesting_performance, 
    calculate_credit_var,
    plot_backtesting_results
)

# --- Configurações Iniciais e Branding ---
st.set_page_config(
    layout="wide",
    page_title="MAN Consulting - MVP de Risco de Crédito Avançado",
    initial_sidebar_state="expanded"
)

# Adicionar o logo da MAN Consulting
# Verifique se o arquivo 'logo_man_consulting.png' está na mesma pasta do seu script
st.sidebar.image("logo_man_consulting.png", use_column_width=True, caption="MAN Consulting")
st.sidebar.title("MENU PRINCIPAL")

# Menu de Navegação na Sidebar
with st.sidebar:
    selected = option_menu(
        menu_title=None,
        options=["Análise de Risco", "Sobre a MAN Consulting"],
        icons=["bar-chart-fill", "info-circle-fill"],
        menu_icon="cast",
        default_index=0,
    )

# --- Funções do Aplicativo (Mantidas do código anterior) ---

@st.cache_data
def generate_fictitious_data(n_samples=10000):
    np.random.seed(42)
    idade = np.random.randint(18, 70, n_samples)
    renda_mensal = np.random.normal(5000, 2000, n_samples).round(2)
    divida_existente = np.random.normal(15000, 7000, n_samples).round(2)
    score_serasa = np.random.randint(300, 900, n_samples)
    tempo_emprego_meses = np.random.randint(0, 240, n_samples)
    num_emprestimos_ativos = np.random.randint(0, 5, n_samples)
    genero = np.random.choice(['Masculino', 'Feminino'], n_samples)
    estado_civil = np.random.choice(['Solteiro', 'Casado', 'Divorciado', 'Viúvo'], n_samples)
    escolaridade = np.random.choice(['Fundamental', 'Médio', 'Superior', 'Pós-graduação'], n_samples, p=[0.1, 0.3, 0.4, 0.2])
    tipo_moradia = np.random.choice(['Alugada', 'Propria', 'Financiada'], n_samples)
    prob_default = (1 / (1 + np.exp(
        - (
            0.0005 * (3000 - renda_mensal) +
            0.0001 * divida_existente +
            0.005 * (500 - score_serasa) +
            np.random.normal(0, 0.5, n_samples)
        )
    )))
    default = (np.random.rand(n_samples) < prob_default).astype(int)
    df = pd.DataFrame({
        'idade': idade, 'renda_mensal': np.maximum(500, renda_mensal),
        'divida_existente': np.maximum(0, divida_existente), 'score_serasa': score_serasa,
        'tempo_emprego_meses': tempo_emprego_meses, 'num_emprestimos_ativos': num_emprestimos_ativos,
        'genero': genero, 'estado_civil': estado_civil, 'escolaridade': escolaridade,
        'tipo_moradia': tipo_moradia, 'default': default
    })
    return df

@st.cache_resource
def train_model(df_input):
    numerical_cols = ['idade', 'renda_mensal', 'divida_existente', 'score_serasa', 'tempo_emprego_meses', 'num_emprestimos_ativos']
    categorical_cols = ['genero', 'estado_civil', 'escolaridade', 'tipo_moradia']
    X = df_input.drop('default', axis=1)
    y = df_input['default']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                     ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))])
    model_pipeline.fit(X_train, y_train)
    ohe = model_pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
    cat_feature_names = ohe.get_feature_names_out(categorical_cols)
    all_feature_names = numerical_cols + list(cat_feature_names)
    return model_pipeline, X_test, y_test, all_feature_names

# --- Conteúdo da Página "Análise de Risco" ---
if selected == "Análise de Risco":
    st.header("Análise de Risco de Crédito")
    st.markdown("""
    Esta seção apresenta o ciclo completo de modelagem de risco, desde o carregamento dos dados até a explicabilidade das previsões.
    """)

    st.sidebar.subheader("Carregar Dados")
    data_source = st.sidebar.radio("Escolha a fonte dos dados:", ("Gerar Dados Fictícios", "Upload de Arquivo CSV"))

    df = None
    if data_source == "Gerar Dados Fictícios":
        n_samples_gen = st.sidebar.slider("Número de clientes fictícios:", 1000, 50000, 10000)
        df = generate_fictitious_data(n_samples_gen)
        st.success(f"Dados fictícios gerados com sucesso! ({n_samples_gen} registros)")
    else:
        uploaded_file = st.sidebar.file_uploader("Faça o upload do seu arquivo CSV", type=["csv"])
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success("Arquivo CSV carregado com sucesso!")
            except Exception as e:
                st.error(f"Erro ao carregar o arquivo: {e}")
        else:
            st.info("Por favor, carregue um arquivo CSV para continuar ou selecione 'Gerar Dados Fictícios'.")

    if df is not None:
        st.subheader("Visualização dos Dados Carregados")
        st.write(df.head())
        st.write(f"Dimensões do dataset: {df.shape[0]} linhas, {df.shape[1]} colunas.")
        st.write("Distribuição da variável alvo ('default'):")
        st.write(df['default'].value_counts(normalize=True).apply(lambda x: f"{x:.2%}"))

        st.sidebar.subheader("Treinamento do Modelo")
        if st.sidebar.button("Treinar Modelo de Risco"):
            with st.spinner("Treinando modelo XGBoost e calculando métricas..."):
                model_pipeline, X_test, y_test, all_feature_names = train_model(df.copy())
                st.session_state['model_pipeline'] = model_pipeline
                st.session_state['X_test'] = X_test
                st.session_state['y_test'] = y_test
                st.session_state['all_feature_names'] = all_feature_names
                # Calcular prob_default para todo o df original (para PSI, EL, VaR e Backtesting)
                df['prob_default'] = model_pipeline.predict_proba(df.drop('default', axis=1))[:, 1]
                st.session_state['df_with_prob'] = df # Salva o df com as probabilidades no session_state
            st.sidebar.success("Modelo treinado e pronto para avaliação!")

        if 'model_pipeline' in st.session_state:
            st.subheader("Resultados da Modelagem")
            model_pipeline = st.session_state['model_pipeline']
            X_test = st.session_state['X_test']
            y_test = st.session_state['y_test']
            all_feature_names = st.session_state['all_feature_names']
            df_with_prob = st.session_state['df_with_prob']

            y_pred_proba = model_pipeline.predict_proba(X_test)[:, 1]

            # --- Novo: Slider para Threshold de Corte ---
            st.markdown("---")
            st.subheader("Ajuste o Limiar de Classificação de Risco")
            st.info("""
            O limiar (threshold) define a partir de qual probabilidade um cliente é classificado como 'Default' (inadimplente).
            Ajustar este valor muda o equilíbrio entre detectar inadimplentes reais e evitar classificar bons clientes erroneamente.
            """)
            classification_threshold = st.slider(
                'Defina o Limiar de Probabilidade para Classificação como "Default"',
                min_value=0.01, max_value=0.99, value=0.5, step=0.01,
                help="Clientes com probabilidade de default acima deste valor serão classificados como 'Default'."
            )
            y_pred_thresholded = (y_pred_proba >= classification_threshold).astype(int)

            # Métrica AUC-ROC (não muda com o threshold)
            auc_score = roc_auc_score(y_test, y_pred_proba)
            st.metric(label="AUC-ROC Score", value=f"{auc_score:.4f}")

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Curva ROC")
                fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
                fpr, tpr, thresholds_roc = roc_curve(y_test, y_pred_proba) # Capturar thresholds
                ax_roc.plot(fpr, tpr, color='blue', lw=2, label=f'Curva ROC (AUC = {auc_score:.2f})')
                ax_roc.plot([0, 1], [0, 1], color='gray', linestyle='--')
                # Adicionar o ponto do threshold na curva ROC
                idx = np.searchsorted(thresholds_roc, classification_threshold, side="right") -1
                if idx >= 0 and idx < len(fpr):
                    ax_roc.plot(fpr[idx], tpr[idx], 'o', color='red', markersize=8, label=f'Threshold={classification_threshold:.2f}')
                ax_roc.set_xlim([0.0, 1.0])
                ax_roc.set_ylim([0.0, 1.05])
                ax_roc.set_xlabel('Taxa de Falsos Positivos (FPR)')
                ax_roc.set_ylabel('Taxa de Verdadeiros Positivos (TPR)')
                ax_roc.set_title('Curva ROC')
                ax_roc.legend(loc="lower right")
                ax_roc.grid(True)
                st.pyplot(fig_roc)

            with col2:
                st.subheader(f"Matriz de Confusão (Threshold: {classification_threshold:.2f})")
                cm = confusion_matrix(y_test, y_pred_thresholded) # Usar y_pred_thresholded
                fig_cm, ax_cm = plt.subplots(figsize=(6, 4))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                            xticklabels=['Não Default', 'Default'],
                            yticklabels=['Não Default', 'Default'], ax=ax_cm)
                ax_cm.set_xlabel('Previsto')
                ax_cm.set_ylabel('Real')
                ax_cm.set_title(f'Matriz de Confusão (Threshold: {classification_threshold:.2f})')
                st.pyplot(fig_cm)

            st.subheader(f"Relatório de Classificação (Threshold: {classification_threshold:.2f})")
            st.text(classification_report(y_test, y_pred_thresholded)) # Usar y_pred_thresholded

            st.markdown("---")
            # --- Novas Métricas de Performance e Discriminação ---
            st.header("📈 Métricas de Performance e Discriminação Detalhadas")
            advanced_metrics = calculate_ks_gini_ap(y_test, y_pred_proba)

            col_met1, col_met2, col_met3 = st.columns(3)
            with col_met1:
                st.metric(label="KS Statistic", value=f"{advanced_metrics['KS_Statistic']:.4f}")
            with col_met2:
                st.metric(label="Gini Coefficient", value=f"{advanced_metrics['Gini_Coefficient']:.4f}")
            with col_met3:
                st.metric(label="Average Precision (AP)", value=f"{advanced_metrics['Average_Precision']:.4f}")

            st.subheader("Curva Precision-Recall")
            fig_prc = plot_precision_recall_curve(y_test, y_pred_proba)
            st.pyplot(fig_prc)

            st.markdown("---")
            # --- Métricas de Monitoramento e Robustez ---
            st.header("📊 Métricas de Monitoramento e Robustez")

            st.subheader("Population Stability Index (PSI) - Simulado")
            st.info("""
            O PSI mede o quanto a distribuição dos scores (ou features) mudou ao longo do tempo.
            Valores de PSI:
            < 0.1: Sem mudança significativa (OK)
            0.1 - 0.25: Pequena mudança (Revisão Necessária)
            > 0.25: Grande mudança (Re-desenvolvimento Provável)
            Abaixo, uma simulação de como o PSI pode se comportar ao longo de diferentes períodos.
            """)
            psi_results = monitor_psi(df_with_prob, 'prob_default')
            for period, psi_val in psi_results.items():
                st.write(f"**{period}:** PSI = **{psi_val:.4f}**")
                if psi_val < 0.1:
                    st.success("✅ Estável")
                elif 0.1 <= psi_val < 0.25:
                    st.warning("⚠️ Atenção: Mudança Moderada")
                else:
                    st.error("🚨 Alerta: Mudança Significativa")


            st.subheader("Análise de Vintage - Simulado")
            st.info("A Análise de Vintage acompanha a performance (e.g., inadimplência acumulada) de diferentes 'safras' de clientes (coortes de concessão) ao longo do tempo. Isso revela tendências de qualidade da carteira.")
            fig_vintage = plot_vintage_analysis(n_vintages=5)
            st.pyplot(fig_vintage)

            # --- Novo: Seção de Backtesting ---
            st.markdown("---")
            st.header("🔄 Backtesting do Modelo")
            st.info("""
            O Backtesting avalia a performance histórica do modelo, comparando as previsões com os resultados reais ao longo do tempo.
            Aqui, simulamos a taxa de default real para diferentes períodos e comparamos com a PD média prevista.
            """)
            n_periods_backtest = st.slider("Número de Períodos para Backtest", 3, 24, 12)
            backtest_results_df = simulate_backtesting_performance(df_with_prob, n_periods=n_periods_backtest)
            fig_backtest = plot_backtesting_results(backtest_results_df)
            st.pyplot(fig_backtest)

            st.markdown("---")
            # --- Métricas de Negócio e Impacto Financeiro ---
            st.header("💰 Métricas de Negócio e Impacto Financeiro")

            st.subheader("Perda Esperada (Expected Loss)")
            st.info("""
            A Perda Esperada (EL) é um componente crítico para precificação de produtos e gestão de capital.
            EL = Probabilidade de Default (PD) * Perda Dado Default (LGD) * Exposição no Default (EAD).
            """)
            # Passar o seed para garantir reprodutibilidade da simulação de LGD/EAD se houver 'Treinar Modelo' de novo
            df_with_el = calculate_expected_loss(df_with_prob.copy(), 'prob_default', random_seed=42)
            st.write(f"Soma total da Perda Esperada para a amostra: **R$ {df_with_el['expected_loss'].sum():,.2f}**")
            fig_el_dist = plot_expected_loss_distribution(df_with_el)
            st.pyplot(fig_el_dist)

            st.subheader("RAROC (Risk-Adjusted Return on Capital) - Simulado")
            st.info("""
            O RAROC mede o retorno gerado em relação ao capital alocado, ajustado pelo risco.
            É fundamental para decisões estratégicas de investimento e alocação de recursos.
            Quanto maior o RAROC, mais eficiente é o uso do capital em relação ao risco assumido.
            """)
            total_revenue_sim = st.slider("Receita Total Simulada (R$)", 100000.0, 50000000.0, 10000000.0, step=10000.0) # Ajustado valor inicial
            total_capital_sim = st.slider("Capital Alocado Simulada (R$)", 10000.0, 10000000.0, 2000000.0, step=1000.0) # Ajustado valor inicial

            total_expected_loss_for_raroc = df_with_el['expected_loss'].sum()
            raroc_value = calculate_rorac_raroc(total_expected_loss_for_raroc, total_capital_sim, total_revenue_sim)
            st.metric(label="RAROC", value=f"{raroc_value:.2%}")

            # --- Novo: Seção de VaR de Crédito ---
            st.markdown("---")
            st.header("📉 VaR (Value at Risk) de Crédito")
            st.info("""
            O VaR de Crédito estima a perda máxima potencial de um portfólio de crédito em um determinado horizonte de tempo,
            com um certo nível de confiança. Ele ajuda a quantificar o capital necessário para cobrir perdas inesperadas.
            """)
            col_var_input, col_var_output = st.columns(2)
            with col_var_input:
                var_confidence = st.slider("Nível de Confiança do VaR (%)", 90, 99, 99, step=1)
                n_simulations_var = st.slider("Número de Simulações Monte Carlo", 1000, 20000, 10000)

            # Reutilizar df_with_el que já tem prob_default, lgd, ead
            var_value, fig_var_dist = calculate_credit_var(
                df_with_el[['prob_default', 'lgd', 'ead']],
                confidence_level=var_confidence/100,
                n_simulations=n_simulations_var,
                random_seed=42 # Para reprodutibilidade
            )
            with col_var_output:
                st.metric(label=f"VaR de Crédito {var_confidence}%", value=f"R$ {var_value:,.2f}")
            st.pyplot(fig_var_dist)


            # --- Explicabilidade (SHAP) ---
            st.markdown("---")
            st.header("🧠 Explicabilidade do Modelo (SHAP)")
            st.markdown("""
            **SHAP (SHapley Additive exPlanations)** nos ajuda a entender como cada característica (feature) impacta a previsão do modelo.
            """)
            explainer = shap.TreeExplainer(model_pipeline.named_steps['classifier'])
            X_test_transformed = model_pipeline.named_steps['preprocessor'].transform(X_test)
            if isinstance(X_test_transformed, (np.ndarray, pd.DataFrame)):
                X_test_transformed_df = pd.DataFrame(X_test_transformed, columns=all_feature_names)
            else:
                X_test_transformed_df = pd.DataFrame(X_test_transformed.toarray(), columns=all_feature_names)
            shap_values = explainer.shap_values(X_test_transformed_df)

            col_shap1, col_shap2 = st.columns(2)
            with col_shap1:
                st.write("##### Importância Global das Features (SHAP Bar Plot)")
                fig_shap_bar, ax_shap_bar = plt.subplots(figsize=(10, 6))
                shap.summary_plot(shap_values, X_test_transformed_df, plot_type="bar", show=False)
                st.pyplot(fig_shap_bar)
            with col_shap2:
                st.write("##### Impacto e Direção das Features (SHAP Beeswarm Plot)")
                fig_shap_beeswarm, ax_shap_beeswarm = plt.subplots(figsize=(10, 6))
                shap.summary_plot(shap_values, X_test_transformed_df, show=False)
                st.pyplot(fig_shap_beeswarm)

            st.markdown("---")
            st.subheader("Simulador de Risco para um Novo Cliente")
            st.markdown("Ajuste as características abaixo para ver a probabilidade de default para um cliente hipotético.")

            input_idade = st.slider('Idade', 18, 70, 30)
            input_renda_mensal = st.slider('Renda Mensal (R$)', 500.0, 20000.0, 5000.0)
            input_divida_existente = st.slider('Dívida Existente (R$)', 0.0, 50000.0, 10000.0)
            input_score_serasa = st.slider('Score Serasa', 300, 900, 700)
            input_tempo_emprego_meses = st.slider('Tempo de Emprego (Meses)', 0, 240, 60)
            input_num_emprestimos_ativos = st.slider('Número de Empréstimos Ativos', 0, 5, 1)
            input_genero = st.selectbox('Gênero', ['Masculino', 'Feminino'])
            input_estado_civil = st.selectbox('Estado Civil', ['Solteiro', 'Casado', 'Divorciado', 'Viúvo'])
            input_escolaridade = st.selectbox('Escolaridade', ['Fundamental', 'Médio', 'Superior', 'Pós-graduação'])
            input_tipo_moradia = st.selectbox('Tipo de Moradia', ['Alugada', 'Propria', 'Financiada'])

            new_client_data = pd.DataFrame({
                'idade': [input_idade], 'renda_mensal': [input_renda_mensal],
                'divida_existente': [input_divida_existente], 'score_serasa': [input_score_serasa],
                'tempo_emprego_meses': [input_tempo_emprego_meses], 'num_emprestimos_ativos': [input_num_emprestimos_ativos],
                'genero': [input_genero], 'estado_civil': [input_estado_civil], 'escolaridade': [input_escolaridade],
                'tipo_moradia': [input_tipo_moradia],
            })

            new_client_proba = model_pipeline.predict_proba(new_client_data)[:, 1][0]
            st.write(f"##### Probabilidade de Inadimplência para este cliente: **{new_client_proba:.2%}**")

            st.write("##### Explicabilidade da Previsão para este Cliente (SHAP Force Plot)")
            explainer_single = shap.TreeExplainer(model_pipeline.named_steps['classifier'])
            single_client_transformed = model_pipeline.named_steps['preprocessor'].transform(new_client_data)
            if isinstance(single_client_transformed, (np.ndarray, pd.DataFrame)):
                single_client_transformed_df = pd.DataFrame(single_client_transformed, columns=all_feature_names)
            else:
                single_client_transformed_df = pd.DataFrame(single_client_transformed.toarray(), columns=all_feature_names)
            shap_values_single = explainer_single.shap_values(single_client_transformed_df)[0]

            fig_force_plot = shap.force_plot(explainer_single.expected_value, shap_values_single, single_client_transformed_df.iloc[0], matplotlib=True, show=False)
            st.pyplot(fig_force_plot, bbox_inches='tight')
            st.markdown("---")
            st.markdown("**Observação:** No gráfico acima, valores em vermelho empurram a previsão para cima (maior risco), enquanto valores em azul a empurram para baixo (menor risco). A base (E[f(x)]) é a previsão média do modelo.")
        else:
            st.warning("Por favor, clique em 'Treinar Modelo de Risco' na barra lateral para iniciar a análise.")
    else:
        st.info("Carregue seus dados ou gere dados fictícios para começar.")

# --- Conteúdo da Página "Sobre a MAN Consulting" ---
elif selected == "Sobre a MAN Consulting":
    st.header("Sobre a MAN Consulting")
    st.markdown("""
    A **MAN Consulting** é uma consultoria independente especializada em **modelagem preditiva, avaliação de risco de crédito e precificação dinâmica** para instituições financeiras, fintechs e plataformas de crédito na América Latina, EUA e Europa.
    """)

    st.subheader("Nossa Missão")
    st.markdown("""
    Entregar inteligência analítica aplicada ao crédito, com soluções que combinam:
    * **Rigor estatístico e compliance regulatório;**
    * **Rapidez na implementação com entregas modulares;**
    * **Resultados mensuráveis, como redução de inadimplência e otimização de pricing por risco.**
    """)

    st.subheader("Sobre o Fundador")
    st.markdown("""
    Fundada por **Matheus Nascimento**, profissional com mais de 7 anos de experiência no mercado financeiro, a MAN Consulting une visão estratégica de negócios com técnicas avançadas de análise quantitativa, oferecendo soluções customizadas para originadores de crédito, plataformas P2P, SCDs e gestoras de carteiras.

    Matheus atua como Credit and Data Analyst em gestora com mais de USD 300 milhões sob gestão, sendo responsável por:
    * Desenvolvimento de modelos de risco de crédito e ferramentas de gestão para carteiras de crédito e ativos financeiros;
    * Estruturação de modelos macroeconômicos preditivos para avaliação de condições de crédito;
    * Implementação de pipelines de dados e automação de rotinas analíticas com Python, SQL e modelos quantitativos.

    Matheus acumula passagens por research houses e tesourarias de investimentos, com track record de geração de alfa superior ao benchmark em diversas estratégias. É Mestre em Métodos Quantitativos e Pesquisa Operacional pelo ITA, com Pós-graduação em Engenharia Financeira pela FIA Business School.
    """)

    st.subheader("Conecte-se Conosco")
    st.markdown("""
    Ficaremos felizes em discutir suas necessidades e como a MAN Consulting pode ajudar sua empresa a otimizar a gestão de risco e as decisões de crédito.
    """)
    st.markdown("📩 **Contato:** matheus.nascimento@manconsulting.ai")
    st.markdown("🔗 **LinkedIn:** [Matheus Nascimento](https://www.linkedin.com/in/matheus-az-nascimento/)")