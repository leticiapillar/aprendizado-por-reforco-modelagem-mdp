# FloodGuard - Relatorio da Etapa 5

Gerado em `2026-09-19 12:27:14` por `experiments/tune.py`.

## Resumo

A Etapa 5 implementou e executou a busca de hiperparametros dos algoritmos DQN, PPO e A2C com Optuna. A rodada identifica configuracoes candidatas para o treino final e registra todos os resultados para auditoria.

Os resultados servem para selecionar configuracoes candidatas ao treino final. Eles nao definem, isoladamente, o melhor algoritmo, pois os estudos acumulam trials de execucoes com orcamentos diferentes.

## Objetivo

Encontrar configuracoes promissoras para cada algoritmo, usando o retorno medio de avaliacao como criterio de otimizacao. O treino usa o sinal denso de progresso da Etapa 4, enquanto a avaliacao usa a recompensa original do MDP.

## Configuracao da busca

- O `TPESampler` escolhe novas configuracoes com base nos resultados anteriores.
- O `MedianPruner` encerra antecipadamente trials com baixo desempenho.
- Todos os trials, inclusive os podados, permanecem registrados em SQLite e CSV.

A tabela abaixo mostra a configuracao da execucao mais recente. Os totais da secao de resultados tambem incluem trials de execucoes anteriores.

| Algoritmo | Limite de passos nos novos trials | Episodios de avaliacao |
|---|---:|---:|
| A2C | 100000 | 30 |
| DQN | 100000 | 30 |
| PPO | 100000 | 30 |

Os parametros avaliados foram:

- **DQN:** taxa de aprendizado, buffer, lote, desconto, exploracao e frequencia de atualizacao.
- **PPO:** taxa de aprendizado, passos, lote, epocas, desconto, GAE, clipping e coeficientes de perda.
- **A2C:** taxa de aprendizado, passos, desconto, GAE, entropia, valor e limite do gradiente.

## Resultados

| Algoritmo | Trials totais | Trials completos | Trials podados | Melhor retorno | Sucesso | Perda do animal | CSV de trials |
|---|---:|---:|---:|---:|---:|---:|---|
| A2C | 30 | 8 | 22 | 2.55 | 100.00% | 0.00% | `results/logs/optuna/a2c_trials.csv` |
| DQN | 30 | 9 | 21 | -59.17 | 53.33% | 0.00% | `results/logs/optuna/dqn_trials.csv` |
| PPO | 30 | 12 | 18 | 2.40 | 100.00% | 0.00% | `results/logs/optuna/ppo_trials.csv` |

## Leitura dos resultados

- **A2C:** maior retorno medio de 2.55, com 100.00% de sucesso e 0.00% de episodios truncados.
- **PPO:** retorno medio de 2.40, com 100.00% de sucesso e 0.00% de episodios truncados.
- **DQN:** retorno medio de -59.17, com 53.33% de sucesso e 46.67% de episodios truncados.
- A diferenca entre os dois maiores retornos foi 0.15. A comparacao deve ser confirmada com o mesmo orcamento e mais sementes.

## Melhores configuracoes

### A2C

Trial selecionado: `1`. Retorno medio: `2.55`.

| Metrica | Valor |
|---|---:|
| Retorno medio | 2.55 +/- 2.24 |
| Taxa de sucesso | 100.00% |
| Taxa de perda do animal | 0.00% |
| Taxa de truncamento | 0.00% |
| Uso medio de bateria | 15.50 |

| Hiperparametro | Valor |
|---|---:|
| `ent_coef` | `8.93172e-07` |
| `gae_lambda` | `0.886389` |
| `gamma` | `0.952423` |
| `learning_rate` | `0.00252688` |
| `max_grad_norm` | `0.397646` |
| `n_steps` | `5` |
| `vf_coef` | `0.70889` |

### DQN

Trial selecionado: `15`. Retorno medio: `-59.17`.

| Metrica | Valor |
|---|---:|
| Retorno medio | -59.17 +/- 38.65 |
| Taxa de sucesso | 53.33% |
| Taxa de perda do animal | 0.00% |
| Taxa de truncamento | 46.67% |
| Uso medio de bateria | 36.07 |

| Hiperparametro | Valor |
|---|---:|
| `batch_size` | `256` |
| `buffer_size` | `50000` |
| `exploration_final_eps` | `0.11829` |
| `exploration_fraction` | `0.363364` |
| `gamma` | `0.985519` |
| `gradient_steps` | `2` |
| `learning_rate` | `2.41644e-05` |
| `learning_starts` | `1000` |
| `target_update_interval` | `250` |
| `train_freq` | `4` |

### PPO

Trial selecionado: `17`. Retorno medio: `2.40`.

| Metrica | Valor |
|---|---:|
| Retorno medio | 2.40 +/- 2.22 |
| Taxa de sucesso | 100.00% |
| Taxa de perda do animal | 0.00% |
| Taxa de truncamento | 0.00% |
| Uso medio de bateria | 15.57 |

| Hiperparametro | Valor |
|---|---:|
| `batch_size` | `32` |
| `clip_range` | `0.149217` |
| `ent_coef` | `0.000593434` |
| `gae_lambda` | `0.999156` |
| `gamma` | `0.953873` |
| `learning_rate` | `4.56699e-05` |
| `n_epochs` | `20` |
| `n_steps` | `64` |
| `vf_coef` | `0.478818` |

## Arquivos gerados

- Bancos SQLite e CSVs de trials: `results/logs/optuna`.
- Graficos de historico e importancia: `results/figures`.
- JSONs `*_best_params.json`: configuracoes escolhidas para a Etapa 6.

## Proximos passos

Antes do treino final da Etapa 6, recomenda-se comparar as configuracoes vencedoras em varias sementes, usando o mesmo orcamento de treino e avaliacao.

1. Treinar novamente as configuracoes selecionadas de A2C, PPO e DQN.
2. Usar varias sementes para reduzir o efeito do acaso nos resultados.
3. Comparar retorno, sucesso, perda do animal e consumo de bateria.
4. Selecionar o modelo final com base nessa avaliacao controlada.

## Conclusao

A Etapa 5 foi concluida. A busca e reproduzivel, os trials sao auditaveis e as melhores configuracoes estao salvas para a Etapa 6. A2C e PPO apresentaram os melhores resultados acumulados. A escolha final deve ser confirmada em uma comparacao controlada, com o mesmo orcamento e multiplas sementes.
