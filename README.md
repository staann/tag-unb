# Projeto de Análise de Grafos - TAG UNB
## Análise de Rede Social do Facebook

### Integrantes:
- [Gustavo Choueiri] - [232014010]
- [Giovanni Daldegan] - [232002520]

### Objetivo:
Analisar uma rede social do Facebook extraindo 2000 nós aleatórios e aplicando algoritmos de detecção de comunidades e medidas de centralidade.

## Estrutura do Projeto

### Arquivos:
- `main.py` - Código principal com todas as funções de análise
- `notebook_analise.py` - Versão para Jupyter Notebook
- `notebook_analise.ipynb` - **Jupyter Notebook pronto para usar**
- `main_completo.ipynb` - **Notebook completo e organizado do main.py**
- `requirements.txt` - Dependências do projeto
- `dados_facebook/` - Diretório com dados do Facebook (ego-networks)

### Funcionalidades Implementadas:

#### 1. Coleta de Dados
- Extração de 2000 nós aleatórios do conjunto de dados do Facebook
- Filtragem de arestas para manter apenas conexões entre nós selecionados
- Análise do componente gigante para garantir conectividade

#### 2. Detecção de Comunidades
- Implementação do algoritmo de Louvain
- Fallback para label propagation em caso de erro
- Análise do tamanho e distribuição das comunidades

#### 3. Medidas de Centralidade
- **Centralidade de Grau**: Mede conexões diretas
- **Centralidade de Intermediação**: Mede importância como ponte
- **Centralidade de Proximidade**: Mede proximidade a todos os nós
- **Centralidade de Autovetor**: Mede importância baseada em vizinhos importantes

#### 4. Visualizações
- Grafo com comunidades coloridas
- Visualizações destacando nós com maior centralidade
- Gráficos de barras com top nós
- Heatmap de correlação entre medidas

#### 5. Relatório de Análise
- Análise detalhada dos nós mais influentes
- Interpretações das diferentes medidas
- Recomendações para aplicações práticas

## Como Executar

### Pré-requisitos:
```bash
pip install -r requirements.txt
```

### Execução:
```bash
python main.py
```

### Para Jupyter Notebook:
```bash
# Opção 1: Executar arquivo Python no notebook
python notebook_analise.py

# Opção 2: Abrir notebook Jupyter
jupyter notebook notebook_analise.ipynb

# Opção 3: Abrir notebook completo organizado
jupyter notebook main_completo.ipynb
```

## Resultados Obtidos

### Estatísticas da Rede:
- **Nós analisados**: 1765 (componente gigante)
- **Arestas**: 21,625
- **Densidade**: 0.0139
- **Comunidades detectadas**: 21

### Nós Mais Influentes:

#### Centralidade de Grau (Top 3):
1. Nó 2347: valor=0.0828, grau=146
2. Nó 2543: valor=0.0816, grau=144
3. Nó 1888: valor=0.0760, grau=134

#### Centralidade de Intermediação (Top 3):
1. Nó 1718: valor=0.3405, grau=65
2. Nó 1085: valor=0.2388, grau=29
3. Nó 1405: valor=0.1787, grau=25

#### Centralidade de Proximidade (Top 3):
1. Nó 1376: valor=0.2740, grau=96
2. Nó 1718: valor=0.2734, grau=65
3. Nó 1509: valor=0.2728, grau=73

#### Centralidade de Autovetor (Top 3):
1. Nó 2206: valor=0.1221, grau=103
2. Nó 2233: valor=0.1219, grau=111
3. Nó 1985: valor=0.1202, grau=111

## Conclusões

### Principais Descobertas:
1. **Estrutura da Rede**: Características típicas de redes sociais reais
2. **Detecção de Comunidades**: Algoritmo de Louvain eficaz
3. **Medidas de Centralidade**: Diferentes medidas identificam diferentes tipos de importância
4. **Aplicações Práticas**: Útil para marketing, identificação de influenciadores e estratégias de disseminação

### Aplicações:
- Campanhas de marketing direcionadas
- Identificação de influenciadores
- Estratégias de disseminação de informações
- Análise de conectividade da rede

## Dependências

- networkx>=3.0
- pandas>=1.5.0
- numpy>=1.21.0
- matplotlib>=3.5.0
- seaborn>=0.11.0
- scipy>=1.9.0

## Licença

Projeto acadêmico - TAG UNB
