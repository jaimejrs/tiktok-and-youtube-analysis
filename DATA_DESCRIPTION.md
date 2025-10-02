# 📄 Dicionário de Dados - Análise de Vídeos Virais

Este documento detalha a estrutura e o conteúdo do conjunto de dados `youtube_shorts_tiktok_trends_2025.csv`, utilizado como base para o [Dashboard de Análise de Vídeos Virais](https://github.com/jaimejrs/tiktok-and-youtube-analysis).

-   **Fonte Original:** [YouTube Shorts and TikTok Trends 2025 no Kaggle](https://www.kaggle.com/datasets/tarekmasryo/youtube-shorts-and-tiktok-trends-2025)
-   **Formato:** CSV (Comma-Separated Values)

## Visão Geral do DataFrame

O conjunto de dados é composto por **48.079 registros** e **58 colunas**, sem a presença de valores nulos, o que indica um dataset limpo e pronto para análise.

-   **Tipos de Dados:**
    -   `object` (strings): 29 colunas
    -   `int64` (inteiros): 14 colunas
    -   `float64` (decimais): 15 colunas
-   **Uso de Memória:** Aproximadamente 21.3+ MB.

---

## Estrutura do Dataset (Dicionário de Colunas)

As 58 colunas foram agrupadas em categorias lógicas para facilitar a compreensão.

### 1. Identificadores e Atributos Gerais

| Coluna | Tipo de Dado | Descrição |
| :--- | :--- | :--- |
| `row_id` | `object` | Identificador único para cada linha (vídeo) no dataset. |
| `platform` | `object` | Plataforma onde o vídeo foi publicado (`TikTok` ou `YouTube`). |
| `notes` | `object` | Anotações ou observações adicionais sobre o registro (coluna genérica).|

### 2. Atributos de Conteúdo

| Coluna | Tipo de Dado | Descrição |
| :--- | :--- | :--- |
| `title` | `object` | O título completo do vídeo. |
| `title_length` | `int64` | Número de caracteres no título do vídeo. |
| `title_keywords` | `object` | Principais palavras-chave extraídas do título. |
| `has_emoji` | `int64` | Booleano (`1` ou `0`) indicando se o título contém emojis. |
| `category` | `object` | Categoria principal do conteúdo (ex: `Gaming`, `Food`, `Art`). |
| `genre` | `object` | Gênero ou subcategoria mais específica do conteúdo. |
| `hashtag` / `tags` | `object` | Hashtags associadas ao vídeo. |
| `duration_sec` | `int64` | Duração do vídeo em segundos. |
| `sound_type` | `object` | Tipo de som utilizado (ex: `trending`, `licensed`, `original`). |
| `music_track` | `object` | Nome ou identificador da faixa de música utilizada. |
| `sample_comments`| `object` | Uma amostra de um comentário do vídeo para análise de sentimento. |

### 3. Métricas de Performance e Engajamento

| Coluna | Tipo de Dado | Descrição |
| :--- | :--- | :--- |
| `views` | `int64` | Número total de visualizações. |
| `likes` | `int64` | Número total de curtidas. |
| `dislikes` | `int64` | Número total de "não curtidas" (principalmente YouTube). |
| `comments` | `int64` | Número total de comentários. |
| `shares` | `int64` | Número total de compartilhamentos. |
| `saves` | `int64` | Número total de vezes que o vídeo foi salvo por usuários. |
| `engagement_total` | `int64` | Soma total das interações (curtidas + comentários + compartilhamentos). |
| `avg_watch_time_sec`| `float64`| Tempo médio de visualização do vídeo em segundos. |
| `completion_rate` | `float64` | Taxa percentual de conclusão do vídeo (assistido até o fim). |

### 4. Métricas Derivadas (Taxas e Rácios)

| Coluna | Tipo de Dado | Descrição |
| :--- | :--- | :--- |
| `engagement_rate`| `float64` | Taxa de engajamento (`engagement_total / views`). |
| `like_rate` | `float64` | Taxa de curtidas (`likes / views`). |
| `dislike_rate` | `float64` | Taxa de "não curtidas" (`dislikes / views`). |
| `share_rate` | `float64` | Taxa de compartilhamentos (`shares / views`). |
| `save_rate` | `float64` | Taxa de salvamentos (`saves / views`). |
| `comment_ratio` | `float64` | Proporção de comentários em relação a outras interações. |
| `like_dislike_ratio`| `float64`| Proporção entre curtidas e "não curtidas". |
| `engagement_per_1k`| `float64`| Engajamento total a cada 1000 visualizações. |
| `engagement_like_rate`|`float64` | Proporção de curtidas dentro do engajamento total. |
| `engagement_comment_rate`| `float64`| Proporção de comentários dentro do engajamento total. |
| `engagement_share_rate`| `float64`| Proporção de compartilhamentos dentro do engajamento total. |
| `engagement_velocity`| `float64`| Medida da velocidade com que o vídeo ganha engajamento. |

### 5. Atributos do Criador e da "Trend"

| Coluna | Tipo de Dado | Descrição |
| :--- | :--- | :--- |
| `author_handle` | `object` | O nome de usuário ou "handle" do criador do conteúdo. |
| `creator_tier` | `object` | Classificação do criador com base em seu alcance (ex: `Micro`, `Macro`). |
| `creator_avg_views`| `float64`| Média de visualizações dos vídeos deste criador. |
| `trend_label` | `object` | Rótulo que classifica a tendência associada ao vídeo. |
| `trend_duration_days`| `int64` | Duração estimada da tendência em dias. |
| `trend_type` | `object` | Tipo de tendência (ex: `Emerging`, `Seasonal`). |

### 6. Dados Contextuais (Geográficos, Temporais e Técnicos)

| Coluna | Tipo de Dado | Descrição |
| :--- | :--- | :--- |
| `country` | `object` | Código do país de publicação (ex: `US`, `BR`). |
| `region` | `object` | Região geográfica mais ampla (ex: `North America`). |
| `language` | `object` | Código do idioma principal do vídeo (ex: `en`, `pt`). |
| `publish_date_approx`| `object` | Data aproximada da publicação (requer conversão para `datetime`). |
| `upload_hour` | `int64` | Hora do dia (0-23) da publicação. |
| `publish_dayofweek`| `object` | Dia da semana da publicação (ex: `Monday`). |
| `is_weekend` | `int64` | Booleano (`1` ou `0`) indicando se a publicação foi no fim de semana. |
| `week_of_year` | `int64` | Semana do ano em que o vídeo foi publicado. |
| `year_month` | `object` | Ano e mês da publicação (coluna criada no pré-processamento). |
| `publish_period` | `object` | Período do dia (ex: `Morning`, `Afternoon`). |
| `season` / `event_season`| `object` | Estação do ano ou evento sazonal associado. |
| `device_type` | `object` | Tipo de dispositivo de visualização (ex: `Mobile`). |
| `device_brand` | `object` | Marca do dispositivo de visualização. |
| `traffic_source` | `object` | Origem do tráfego para o vídeo (ex: `ForYou`, `Search`, `External`). |
| `source_hint` | `object` | Pista sobre a fonte ou origem dos dados do registro. |

---

### Estatísticas Descritivas (Resumo)

Abaixo, um resumo das principais métricas numéricas do dataset:

| Métrica | Média | Desvio Padrão | Mínimo | Mediana (50%) | Máximo |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `views` | 99.292 | 131.852 | 794 | 59.620 | 3.080.686 |
| `likes` | 5.737 | 8.639 | 33 | 3.167 | 310.916 |
| `duration_sec` | 34.5 | 16.4 | 5 | 31 | 90 |
| `engagement_rate`| 0.075 | 0.030 | 0.014 | 0.071 | 0.235 |
| `completion_rate`| 0.635 | 0.112 | 0.400 | 0.635 | 0.850 |
