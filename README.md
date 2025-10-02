[![author](https://img.shields.io/badge/author-jaimejrs-red.svg)](https://www.linkedin.com/in/jaimejrs) [![](https://img.shields.io/badge/python-3.0+-blue.svg)](https://www.python.org/downloads/release/python-3137/) [![CC0: Domínio Público](https://img.shields.io/badge/License-CC0-white.svg)](https://creativecommons.org/publicdomain/zero/1.0/)

<p align="center">
  <img src="https://github.com/jaimejrs/tiktok-and-youtube-analysis/blob/891cc3f18ed911f619c91b77eaa3337fd584d3fe/tiktokbanner.jpeg" >
</p>

# 📊 Dashboard de Análise de Vídeos Virais (TikTok & YouTube)

[🚀 Acesse a minha análise aqui!](https://tiktok-and-youtube-analysis-d5l4tgkxmevvehe9xyogta.streamlit.app/)

## Descrição do Projeto

Este projeto consiste em um dashboard interativo construído com Python e Streamlit para a análise exploratória de dados de vídeos virais das plataformas TikTok e YouTube Shorts. O painel permite a visualização de tendências, análise de fatores de engajamento, exploração de conteúdo e performance geográfica.

Este dashboard foi desenvolvido como parte de um projeto de análise de dados para demonstrar habilidades em ETL, criação de gráficos, visualização de dados, modelagem de machine learning e desenvolvimento de aplicações web interativas para a cadeira de Business Inteligence do curso de Adminitração da UFC.

### Sobre o Conjunto de Dados 

[Acesse os dados aqui!](https://www.kaggle.com/datasets/tarekmasryo/youtube-shorts-and-tiktok-trends-2025)

Para esta análise, foi utilizado o dataset youtube_shorts_tiktok_trends_2025.csv, que agrega uma rica coleção de informações sobre vídeos de formato curto, abrangendo as plataformas TikTok e YouTube Shorts. A estrutura do dataset foi projetada para permitir uma investigação multifacetada dos fatores que impulsionam o sucesso de um vídeo. As colunas disponíveis podem ser agrupadas em três categorias principais: métricas de performance, como total de visualizações, taxa de engajamento, curtidas e compartilhamentos; atributos de conteúdo, que descrevem o vídeo em si, incluindo sua categoria (ex: Gaming, Food, Art), duração, palavras-chave do título e uso de emojis; e dados contextuais, que fornecem informações sobre a origem e distribuição, como plataforma, país, região, tipo de dispositivo e dados temporais (dia da semana e hora da publicação).

Essa combinação de dados quantitativos e qualitativos oferece uma base sólida para aplicar técnicas de análise exploratória, modelagem estatística e machine learning, com o objetivo final de extrair insights acionáveis sobre o que define um conteúdo viral e como otimizar a performance de vídeos curtos

### Principais Funcionalidades

- **Visão Geral:** Métricas de performance, tendências temporais de visualizações e engajamento.
- **Análise de Fatores:** Análise de como a duração, hora de postagem, dia da semana e categoria impactam o engajamento.
- **Análise de Conteúdo:** Extração de palavras-chave, análise de sentimento de comentários e teste A/B sobre o uso de emojis.
- **Análise Geográfica:** Visualização de performance por país e região.
- **Filtros Interativos:** Permite filtrar os dados por País, Plataforma e Tipo de Dispositivo.

### Screenshot do Dashboard

![Print do Dashboard](https://github.com/jaimejrs/tiktok-and-youtube-analysis/blob/7dc68c01cf11fea09fafa87e9da2126eb0c1f00e/geral.jpg)
![Print do Dashboard](https://github.com/jaimejrs/tiktok-and-youtube-analysis/blob/7dc68c01cf11fea09fafa87e9da2126eb0c1f00e/geografico.jpg)


### 🛠️ Tecnologias Utilizadas

Este projeto foi construído utilizando um ecossistema moderno de ferramentas de Python para análise e visualização de dados.

* **Linguagem:** `Python 3`
* **Análise e Manipulação de Dados:** `Pandas`, `NumPy`
* **Cálculo Científico:** `SciPy` (para testes estatísticos)
* **Visualização de Dados:** `Plotly`
* **Dashboard Interativo:** `Streamlit`
* **Machine Learning e NLP:**
    * `Scikit-learn` (para o modelo de *Random Forest*)
    * `Transformers` (Hugging Face para Análise de Sentimento)
    * `NLTK` (para processamento de texto e *stopwords*)

### 🚀 Como Executar o Projeto

Para executar este dashboard interativo em sua máquina local, siga os passos abaixo:

1.  **Clone o repositório:**
    ```bash
    git clone [https://github.com/jaimejrs/tiktok-and-youtube-analysis.git](https://github.com/jaimejrs/tiktok-and-youtube-analysis.git)
    ```

2.  **Navegue até o diretório do projeto:**
    ```bash
    cd tiktok-and-youtube-analysis
    ```

3.  **Crie e ative um ambiente virtual (recomendado):**

    * **No Linux ou macOS:**
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```
    * **No Windows:**
        ```powershell
        python -m venv venv
        .\venv\Scripts\activate
        ```

4.  **Instale as dependências necessárias:**
    ```bash
    pip install -r requirements.txt
    ```

5.  **Adicione o arquivo de dados:**
    Certifique-se de que o arquivo `youtube_shorts_tiktok_trends_2025.csv` esteja localizado na raiz do diretório do projeto.

6.  **Execute o aplicativo Streamlit:**
    ```bash
    streamlit run dashboard_colab.py
    ```

Após executar o último comando, uma aba no seu navegador será aberta com o dashboard em funcionamento.
