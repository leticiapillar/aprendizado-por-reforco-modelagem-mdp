# FloodGuard - Relatorio da Etapa 6

Gerado em `2026-09-19 13:23:53` por `experiments/final_experiments.py`.

## Resumo

Esta etapa treinou DQN, PPO e A2C com os melhores hiperparametros da Etapa 5. Cada algoritmo foi treinado com as mesmas sementes e avaliado nos mesmos episodios de teste.

O maior retorno medio foi obtido por **PPO**. A tabela abaixo apresenta os resultados completos e permite comparar estabilidade, sucesso e seguranca.

## Configuracao

- Sementes de treino: `42, 123, 2024`.
- Passos de treino por modelo: `500000`.
- Episodios de teste por modelo: `100`.
- Avaliacao deterministica: `nao`.
- A avaliacao usa a recompensa original do MDP, sem o reforco de progresso usado no treino.
- A politica estocastica foi mantida para seguir o mesmo criterio usado na Etapa 5.

## Resultados

Os valores mostram a media entre sementes. O termo apos `+/-` e o desvio padrao entre os modelos treinados.

| Algoritmo | Retorno | Sucesso | Perda do animal | Truncamento | Bateria usada | Passos ate o sucesso |
|---|---:|---:|---:|---:|---:|---:|
| PPO | 1.53 +/- 1.13 | 99.33% +/- 1.15% | 0.00% | 0.67% | 15.42 | 17.80 |
| DQN | -61.25 +/- 4.22 | 49.33% +/- 4.04% | 3.33% | 47.33% | 32.73 | 43.62 |
| A2C | -97.41 +/- 2.25 | 0.00% +/- 0.00% | 36.67% | 63.33% | 1.67 | n/a |

## Leitura dos resultados

- **PPO ficou em primeiro lugar no retorno medio**, com 1.53.
- **PPO foi o mais confiavel:** sucesso em 99.33% dos episodios e baixa variacao entre sementes.
- **DQN teve desempenho intermediario:** sucesso em 49.33% dos episodios, mas 47.33% terminaram no limite de passos.
- **A2C nao concluiu resgates no experimento final.** O bom resultado da Etapa 5 nao se manteve com o treino mais longo, indicando instabilidade.
- Os tres algoritmos usaram poucas barreiras efetivas. O comportamento aprendido priorizou o resgate direto em vez da protecao do mapa.

## Graficos

### Curvas de aprendizado

A curva usa a recompensa de treino, que inclui o reforco de progresso. As metricas da tabela usam somente a recompensa original do MDP.

![Curvas de aprendizado](figures/stage6_learning_curves.png)

### Retorno nos episodios de teste

![Distribuicao do retorno](figures/stage6_return_boxplot.png)

### Taxas finais

![Metricas finais](figures/stage6_final_metrics.png)

## Arquivos gerados

- Modelos: `results/models/final`.
- Logs de treino: `results/logs/final`.
- Resultados por seed: `results/final/final_seed_results.csv`.
- Resultados por episodio: `results/final/final_episode_results.csv`.
- Resumo em JSON: `results/final/final_summary.json`.

## Conclusao

A Etapa 6 foi concluida com uma comparacao controlada entre os tres algoritmos. **PPO** apresentou o maior retorno medio. A decisao sobre o modelo final deve considerar tambem a taxa de sucesso, a perda do animal e a variacao entre sementes, e nao apenas o retorno.
