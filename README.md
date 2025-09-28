# 📊 Dashboard de Análise de Vídeos Virais (TikTok & YouTube)

[🚀 Acesse a minha análise aqui!](https://tiktok-and-youtube-analysis-d5l4tgkxmevvehe9xyogta.streamlit.app/)

## Descrição do Projeto

Este projeto consiste em um dashboard interativo construído com Python e Streamlit para a análise exploratória de dados de vídeos virais das plataformas TikTok e YouTube Shorts. O painel permite a visualização de tendências, análise de fatores de engajamento, exploração de conteúdo e performance geográfica.

Este dashboard foi desenvolvido como parte de um projeto de análise de dados para demonstrar habilidades em ETL, criação de gráficos, visualização de dados, modelagem de machine learning e desenvolvimento de aplicações web interativas para a cadeira de Business Inteligence do curso de Adminitração da UFC.

### Principais Funcionalidades

- **Visão Geral:** Métricas de performance, tendências temporais de visualizações e engajamento.
- **Análise de Fatores:** Análise de como a duração, hora de postagem, dia da semana e categoria impactam o engajamento.
- **Análise de Conteúdo:** Extração de palavras-chave, análise de sentimento de comentários e teste A/B sobre o uso de emojis.
- **Análise Geográfica:** Visualização de performance por país e região.
- **Filtros Interativos:** Permite filtrar os dados por País, Plataforma e Tipo de Dispositivo.

### Screenshot do Dashboard

![Banner do Dashboard](https://github.com/jaimejrs/tiktok-and-youtube-analysis/blob/7dc68c01cf11fea09fafa87e9da2126eb0c1f00e/geral.jpg)
![Banner do Dashboard](https://github.com/jaimejrs/tiktok-and-youtube-analysis/blob/7dc68c01cf11fea09fafa87e9da2126eb0c1f00e/geografico.jpg)


### 🛠️ Tecnologias Utilizadas

* **Linguagem:** Python 3
* **Análise de Dados:** Pandas, NumPy
* **Visualização de Dados:** Plotly, Matplotlib, Seaborn
* **Dashboard Interativo:** Streamlit
* **Machine Learning:** Scikit-learn (Random Forest), Transformers (Hugging Face para análise de sentimento)
* **Ambiente de Desenvolvimento:** Google Colab, Jupyter

### Como Executar o Projeto

Para executar este projeto localmente, siga os passos abaixo:
1. **Clone o repositório:**

git clone [Tiktok and Youtube Analysis](https://github.com/jaimejrs/tiktok-and-youtube-analysis.git)

2. **Navegue até o diretório do projeto:**

cd tiktok-and-youtube-analysis

3. **Crie um ambiente virtual (recomendado):**

python -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate

4. **Instale as dependências:**

pip install -r requirements.txt

5. **Execute o aplicativo Streamlit:**

streamlit run dashboard_colab.py
