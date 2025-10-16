"""
Notebook Python para Análise de Grafos - TAG UNB
Este arquivo pode ser executado no Jupyter Notebook ou convertido para .ipynb

Integrantes:
- [Gustavo Choueiri] - [232014010]
- [Giovanni Daldegan] - [232002520]

Objetivo: Analisar uma rede social do Facebook extraindo 2000 nós aleatórios 
e aplicando algoritmos de detecção de comunidades e medidas de centralidade.
"""

# =============================================================================
# 1. IMPORTATION DE BIBLIOTECAS E CONFIGURAÇÕES
# =============================================================================

print("# Projeto de Análise de Grafos - TAG UNB")
print("## Análise de Rede Social do Facebook")
print()
print("**Integrantes:**")
print("- [Nome] - [Matrícula]")
print("- [Nome] - [Matrícula]")
print()
print("**Objetivo:** Analisar uma rede social do Facebook extraindo 2000 nós aleatórios e aplicando algoritmos de detecção de comunidades e medidas de centralidade.")
print()
print("---")

# Importação das bibliotecas necessárias
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import os
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Configurações para visualização
plt.style.use('default')
sns.set_palette("husl")

print("=== PROJETO DE ANÁLISE DE GRAFOS - TAG UNB ===")
print("Análise de Rede Social do Facebook")
print("=" * 50)

# =============================================================================
# 2. EXECUÇÃO DA ANÁLISE COMPLETA
# =============================================================================

print("\n## 2. Execução da Análise Completa")
print("\nExecutamos todo o código desenvolvido no arquivo `main.py` para realizar a análise completa.")
print()

# Executa todo o código do main.py
exec(open('main.py').read())

# =============================================================================
# 3. VISUALIZAÇÕES ADICIONAIS
# =============================================================================

print("\n## 3. Visualizações Adicionais")
print("\nCriamos visualizações específicas para o notebook com maior detalhamento.")
print()

# Visualização detalhada das comunidades
plt.figure(figsize=(15, 10))

# Subplot 1: Grafo com comunidades
plt.subplot(2, 2, 1)
num_comunidades = len(set(comunidades_dict.values()))
cores = plt.cm.Set3(np.linspace(0, 1, num_comunidades))
cores_nos = [cores[comunidades_dict.get(no, 0)] for no in G.nodes()]
pos = nx.spring_layout(G, k=1, iterations=50)
nx.draw(G, pos, node_color=cores_nos, node_size=30, edge_color='gray', alpha=0.6, with_labels=False)
plt.title('Grafo com Comunidades')
plt.axis('off')

# Subplot 2: Distribuição de graus
plt.subplot(2, 2, 2)
graus = [G.degree(n) for n in G.nodes()]
plt.hist(graus, bins=30, alpha=0.7, color='skyblue')
plt.title('Distribuição de Graus')
plt.xlabel('Grau')
plt.ylabel('Frequência')

# Subplot 3: Tamanhos das comunidades
plt.subplot(2, 2, 3)
tamanhos_comunidades = [len(comunidade) for comunidade in comunidades_list]
plt.bar(range(len(tamanhos_comunidades)), tamanhos_comunidades, color='lightcoral')
plt.title('Tamanhos das Comunidades')
plt.xlabel('ID da Comunidade')
plt.ylabel('Número de Nós')

# Subplot 4: Correlação entre medidas
plt.subplot(2, 2, 4)
df_medidas = pd.DataFrame(medidas)
correlacao = df_medidas.corr()
im = plt.imshow(correlacao, cmap='coolwarm', aspect='auto')
plt.colorbar(im)
plt.title('Correlação entre Medidas')
plt.xticks(range(len(medidas.keys())), list(medidas.keys()), rotation=45)
plt.yticks(range(len(medidas.keys())), list(medidas.keys()))

plt.tight_layout()
plt.show()

# =============================================================================
# 4. ANÁLISE DETALHADA DOS TOP NÓS
# =============================================================================

print("\n## 4. Análise Detalhada dos Top Nós")
print("\nAnálise detalhada dos nós mais influentes em cada medida de centralidade.")
print()

# Tabela resumo dos top nós
print("📊 TABELA RESUMO DOS NÓS MAIS INFLUENTES")
print("=" * 80)

# Cria DataFrame com os top 5 de cada medida
dados_resumo = []
for medida, nome in [('grau', 'Grau'), ('intermediacao', 'Intermediação'), 
                     ('proximidade', 'Proximidade'), ('autovetor', 'Autovetor')]:
    top_5 = nos_influentes[medida][:5]
    for i, (no, valor) in enumerate(top_5, 1):
        dados_resumo.append({
            'Medida': nome,
            'Posição': i,
            'Nó': no,
            'Valor': valor,
            'Grau': G.degree(no),
            'Comunidade': comunidades_dict.get(no, 'N/A')
        })

df_resumo = pd.DataFrame(dados_resumo)
print(df_resumo.to_string(index=False))

# Estatísticas gerais
print(f"\n📈 ESTATÍSTICAS GERAIS:")
print(f"• Total de nós analisados: {G.number_of_nodes()}")
print(f"• Total de arestas: {G.number_of_edges()}")
print(f"• Densidade: {nx.density(G):.4f}")
print(f"• Número de comunidades: {len(comunidades_list)}")
print(f"• Tamanho médio das comunidades: {np.mean(tamanhos_comunidades):.1f}")
print(f"• Coeficiente de clustering médio: {nx.average_clustering(G):.4f}")

# =============================================================================
# 5. CONCLUSÕES E INTERPRETAÇÕES
# =============================================================================

print("\n## 5. Conclusões e Interpretações")
print()
print("### Principais Descobertas:")
print()
print("1. **Estrutura da Rede**: O grafo extraído apresenta características típicas de redes sociais reais, com baixa densidade e estrutura de comunidade bem definida.")
print()
print("2. **Detecção de Comunidades**: O algoritmo de Louvain foi eficaz em identificar grupos coesos de usuários, revelando a estrutura social subjacente.")
print()
print("3. **Medidas de Centralidade**: Diferentes medidas identificaram diferentes tipos de importância:")
print("   - **Grau**: Identifica \"hubs\" da rede")
print("   - **Intermediação**: Identifica \"pontes\" entre comunidades")
print("   - **Proximidade**: Identifica nós centralmente posicionados")
print("   - **Autovetor**: Identifica influenciadores de influenciadores")
print()
print("4. **Aplicações Práticas**: Os resultados podem ser utilizados para:")
print("   - Campanhas de marketing direcionadas")
print("   - Identificação de influenciadores")
print("   - Estratégias de disseminação de informações")
print("   - Análise de conectividade da rede")
print()
print("### Limitações e Melhorias Futuras:")
print()
print("- O projeto trabalha com uma amostra de 2000 nós, que pode não representar completamente a rede original")
print("- Análises temporais poderiam revelar como a influência evolui ao longo do tempo")
print("- Incorporação de atributos dos usuários poderia enriquecer a análise")
print()
print("---")
print()
print("**Projeto concluído com sucesso!** ✅")
