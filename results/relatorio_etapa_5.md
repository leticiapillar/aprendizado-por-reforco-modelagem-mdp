# FloodGuard - Relatorio da Etapa 5

Gerado em `2026-09-20 18:09:07` por `experiments/tune.py`.

## Resumo

A Etapa 5 executou a busca de hiperparametros de DQN, PPO e A2C com Optuna. Cada estudo tem o mesmo orcamento em todos os trials, avalia a politica **deterministica** e inclui os hiperparametros default do SB3 como trial 0, de modo que o ganho da otimizacao e medido contra o default.

## Objetivo

Encontrar configuracoes promissoras para cada algoritmo. O criterio e o retorno medio de avaliacao **ao longo do treino parcial** (area sob a curva de aprendizado): como o cenario inicial e fixo, varias configuracoes atingem o retorno otimo ao final do treino, e a media ao longo do treino as desempata pela velocidade de convergencia. O retorno da avaliacao final tambem e registrado (`final_return_mean`). O treino usa o sinal denso de progresso da Etapa 4; a avaliacao usa a recompensa original do MDP.

## Protocolo da busca

- `TPESampler` propoe novas configuracoes com base nos resultados anteriores.
- `MedianPruner` encerra trials abaixo da mediana, somente apos um periodo de aquecimento (`--pruner-warmup-evals` avaliacoes intermediarias).
- **Seed de treino e episodios de avaliacao fixos** em todos os trials, para que a diferenca entre trials venha dos hiperparametros.
- **Avaliacao deterministica**: acao de maior valor/probabilidade. No DQN a avaliacao estocastica seria epsilon-greedy com o `exploration_final_eps` do proprio modelo, o que faria a busca otimizar o modo de avaliacao.
- O banco SQLite guarda o orcamento do estudo; tentar reutiliza-lo com outro orcamento gera erro, evitando misturar trials incomparaveis.
- Todos os trials, inclusive os podados, ficam no SQLite, em `*_trials.csv` e (retornos intermediarios) em `*_intermediate.csv`.

| Algoritmo | Passos por trial | Episodios de avaliacao final | Episodios de avaliacao do pruning | Avaliacao |
|---|---:|---:|---:|---|
| A2C | 100000 | 50 | 20 | deterministic |
| DQN | 100000 | 50 | 20 | deterministic |
| PPO | 100000 | 50 | 20 | deterministic |

## Resultados

| Algoritmo | Trials | Completos | Podados | Score do default | Melhor score | Retorno final (melhor) | Sucesso (melhor) | Perda do animal (melhor) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A2C | 30 | 15 | 15 | 4.14 | 4.14 | 4.18 | 100.00% | 0.00% |
| DQN | 30 | 20 | 10 | -5.40 | 5.00 | 5.00 | 100.00% | 0.00% |
| PPO | 30 | 18 | 12 | -8.31 | 2.88 | 5.00 | 100.00% | 0.00% |

## Leitura dos resultados

- **A2C:** melhor score 4.14 (trial 0, 100.00% de sucesso); o default do SB3 obteve 4.14 (ganho de 0.00). O default foi o melhor trial: a busca nao superou o default.
- **DQN:** melhor score 5.00 (trial 24, 100.00% de sucesso); o default do SB3 obteve -5.40 (ganho de 10.39).
- **PPO:** melhor score 2.88 (trial 25, 100.00% de sucesso); o default do SB3 obteve -8.31 (ganho de 11.19).
- Todos os trials usam a mesma seed de treino, os mesmos episodios de avaliacao e o mesmo orcamento; diferencas pequenas de retorno estao dentro do ruido de 1 seed e devem ser confirmadas na Etapa 6 com varias seeds.

## Graficos

### A2C

![Historico A2C](figures/optuna_a2c_history.png)

![Importancia A2C](figures/optuna_a2c_param_importance.png)

### DQN

![Historico DQN](figures/optuna_dqn_history.png)

![Importancia DQN](figures/optuna_dqn_param_importance.png)

### PPO

![Historico PPO](figures/optuna_ppo_history.png)

![Importancia PPO](figures/optuna_ppo_param_importance.png)

## Melhores configuracoes

### A2C

Trial selecionado: `0`. Score: `4.14`.

| Metrica | Valor |
|---|---:|
| Retorno medio | 4.18 +/- 5.80 |
| Taxa de sucesso | 100.00% |
| Taxa de perda do animal | 0.00% |
| Taxa de truncamento | 0.00% |
| Uso medio de bateria | 15.00 |

| Hiperparametro | Valor |
|---|---:|
| `ent_coef` | `1e-08` |
| `gae_lambda` | `1` |
| `gamma` | `0.99` |
| `learning_rate` | `0.0007` |
| `max_grad_norm` | `0.5` |
| `n_steps` | `5` |
| `vf_coef` | `0.5` |

### DQN

Trial selecionado: `24`. Score: `5.00`.

| Metrica | Valor |
|---|---:|
| Retorno medio | 5.00 +/- 0.00 |
| Taxa de sucesso | 100.00% |
| Taxa de perda do animal | 0.00% |
| Taxa de truncamento | 0.00% |
| Uso medio de bateria | 15.00 |

| Hiperparametro | Valor |
|---|---:|
| `batch_size` | `256` |
| `buffer_size` | `50000` |
| `exploration_final_eps` | `0.175249` |
| `exploration_fraction` | `0.400396` |
| `gamma` | `0.924781` |
| `gradient_steps` | `4` |
| `learning_rate` | `0.00170501` |
| `learning_starts` | `1000` |
| `target_update_interval` | `1000` |
| `train_freq` | `8` |

### PPO

Trial selecionado: `25`. Score: `2.88`.

| Metrica | Valor |
|---|---:|
| Retorno medio | 5.00 +/- 0.00 |
| Taxa de sucesso | 100.00% |
| Taxa de perda do animal | 0.00% |
| Taxa de truncamento | 0.00% |
| Uso medio de bateria | 15.08 |

| Hiperparametro | Valor |
|---|---:|
| `batch_size` | `64` |
| `clip_range` | `0.179857` |
| `ent_coef` | `3.92469e-06` |
| `gae_lambda` | `0.917995` |
| `gamma` | `0.94478` |
| `learning_rate` | `0.000938135` |
| `n_epochs` | `5` |
| `n_steps` | `512` |
| `vf_coef` | `0.427176` |

## Todas as configuracoes testadas

Score = media dos retornos de avaliacao ao longo do treino. Trials podados mostram o ultimo retorno intermediario (com o passo em que foram encerrados). O trial `DEFAULT` usa os hiperparametros padrao do SB3.

### A2C

| Trial | Estado | Score | Sucesso final | Hiperparametros |
|---:|---|---:|---:|---|
| 0 | DEFAULT | 4.14 | 100.00% | ent_coef=1e-08, gae_lambda=1, gamma=0.99, learning_rate=0.0007, max_grad_norm=0.5, n_steps=5, vf_coef=0.5 |
| 1 | COMPLETE | -60.91 | 100.00% | ent_coef=0.000106378, gae_lambda=0.973235, gamma=0.905803, learning_rate=8.46801e-05, max_grad_norm=0.314409, n_steps=5, vf_coef=0.781054 |
| 2 | COMPLETE | -14.79 | 72.00% | ent_coef=8.93172e-07, gae_lambda=0.886389, gamma=0.952423, learning_rate=0.00252688, max_grad_norm=0.397646, n_steps=5, vf_coef=0.70889 |
| 3 | COMPLETE | -66.53 | 4.00% | ent_coef=0.000117469, gae_lambda=0.80929, gamma=0.959182, learning_rate=5.29271e-05, max_grad_norm=0.345536, n_steps=20, vf_coef=0.377893 |
| 4 | COMPLETE | -31.69 | 80.00% | ent_coef=2.07575e-05, gae_lambda=0.824408, gamma=0.943971, learning_rate=0.00224132, max_grad_norm=0.936524, n_steps=5, vf_coef=0.275791 |
| 5 | PRUNED | -100.00 | n/a | ent_coef=1.37843e-08, gae_lambda=0.987104, gamma=0.996857, learning_rate=1.32041e-05, max_grad_norm=0.725209, n_steps=50, vf_coef=0.53522 |
| 6 | PRUNED | -100.00 | n/a | ent_coef=0.0216746, gae_lambda=0.918095, gamma=0.999137, learning_rate=0.000380481, max_grad_norm=0.616588, n_steps=100, vf_coef=0.936728 |
| 7 | COMPLETE | -6.88 | 98.00% | ent_coef=1.18721e-08, gae_lambda=0.906344, gamma=0.902558, learning_rate=0.00047024, max_grad_norm=0.951288, n_steps=10, vf_coef=0.539724 |
| 8 | PRUNED | -100.00 | n/a | ent_coef=0.0354185, gae_lambda=0.869527, gamma=0.974464, learning_rate=1.1623e-05, max_grad_norm=0.53996, n_steps=50, vf_coef=0.985525 |
| 9 | PRUNED | -4.40 | n/a | ent_coef=7.76286e-07, gae_lambda=0.941797, gamma=0.93711, learning_rate=0.000409869, max_grad_norm=0.781761, n_steps=10, vf_coef=0.514874 |
| 10 | PRUNED | -100.00 | n/a | ent_coef=0.00157558, gae_lambda=0.85198, gamma=0.921342, learning_rate=5.86138e-05, max_grad_norm=0.50257, n_steps=100, vf_coef=0.727953 |
| 11 | PRUNED | -1.50 | n/a | ent_coef=1.18247e-08, gae_lambda=0.933552, gamma=0.975728, learning_rate=0.000571854, max_grad_norm=0.956293, n_steps=10, vf_coef=0.544573 |
| 12 | COMPLETE | -9.29 | 100.00% | ent_coef=1.37294e-07, gae_lambda=0.999753, gamma=0.910376, learning_rate=0.000943904, max_grad_norm=0.81316, n_steps=10, vf_coef=0.448397 |
| 13 | PRUNED | -100.00 | n/a | ent_coef=8.57334e-08, gae_lambda=0.903206, gamma=0.92797, learning_rate=0.000158745, max_grad_norm=0.666502, n_steps=20, vf_coef=0.627004 |
| 14 | PRUNED | -100.00 | n/a | ent_coef=1.34837e-07, gae_lambda=0.962329, gamma=0.977866, learning_rate=0.000197037, max_grad_norm=0.869796, n_steps=10, vf_coef=0.328704 |
| 15 | COMPLETE | -5.05 | 100.00% | ent_coef=1.73768e-06, gae_lambda=0.954779, gamma=0.963541, learning_rate=0.00112281, max_grad_norm=0.471325, n_steps=5, vf_coef=0.432138 |
| 16 | COMPLETE | -7.16 | 16.00% | ent_coef=5.09172e-06, gae_lambda=0.953747, gamma=0.960849, learning_rate=0.00111751, max_grad_norm=0.467384, n_steps=5, vf_coef=0.421074 |
| 17 | COMPLETE | -17.61 | 100.00% | ent_coef=2.25874e-06, gae_lambda=0.980234, gamma=0.987345, learning_rate=0.00116643, max_grad_norm=0.595464, n_steps=5, vf_coef=0.631379 |
| 18 | PRUNED | -100.00 | n/a | ent_coef=4.22615e-05, gae_lambda=0.999532, gamma=0.966987, learning_rate=0.000220726, max_grad_norm=0.429893, n_steps=5, vf_coef=0.444981 |
| 19 | COMPLETE | -9.28 | 100.00% | ent_coef=0.000855865, gae_lambda=0.952584, gamma=0.984458, learning_rate=0.00139399, max_grad_norm=0.55274, n_steps=5, vf_coef=0.250591 |
| 20 | PRUNED | 4.55 | n/a | ent_coef=7.1801e-06, gae_lambda=0.931758, gamma=0.987775, learning_rate=0.000689689, max_grad_norm=0.394837, n_steps=5, vf_coef=0.362611 |
| 21 | PRUNED | 3.85 | n/a | ent_coef=2.97328e-08, gae_lambda=0.916202, gamma=0.946496, learning_rate=0.000401403, max_grad_norm=0.667113, n_steps=10, vf_coef=0.611215 |
| 22 | COMPLETE | 2.12 | 100.00% | ent_coef=2.50319e-07, gae_lambda=0.890861, gamma=0.916812, learning_rate=0.00173994, max_grad_norm=0.90163, n_steps=100, vf_coef=0.479301 |
| 23 | PRUNED | -16.30 | n/a | ent_coef=3.17287e-07, gae_lambda=0.874311, gamma=0.920977, learning_rate=0.00161489, max_grad_norm=0.481288, n_steps=100, vf_coef=0.47979 |
| 24 | COMPLETE | -0.24 | 100.00% | ent_coef=6.576e-08, gae_lambda=0.855222, gamma=0.931841, learning_rate=0.00288736, max_grad_norm=0.843597, n_steps=100, vf_coef=0.396851 |
| 25 | COMPLETE | 3.24 | 100.00% | ent_coef=4.22953e-08, gae_lambda=0.848838, gamma=0.912783, learning_rate=0.00298628, max_grad_norm=0.879904, n_steps=100, vf_coef=0.318101 |
| 26 | COMPLETE | 1.69 | 100.00% | ent_coef=2.85727e-07, gae_lambda=0.82596, gamma=0.914883, learning_rate=0.00182738, max_grad_norm=0.894807, n_steps=100, vf_coef=0.314737 |
| 27 | PRUNED | -100.00 | n/a | ent_coef=4.97936e-08, gae_lambda=0.845519, gamma=0.913952, learning_rate=0.00071698, max_grad_norm=0.991903, n_steps=100, vf_coef=0.581942 |
| 28 | PRUNED | -30.10 | n/a | ent_coef=3.61072e-07, gae_lambda=0.801585, gamma=0.900381, learning_rate=0.00184133, max_grad_norm=0.780542, n_steps=100, vf_coef=0.325225 |
| 29 | PRUNED | -100.00 | n/a | ent_coef=3.40988e-08, gae_lambda=0.8877, gamma=0.922555, learning_rate=0.00294991, max_grad_norm=0.736777, n_steps=100, vf_coef=0.860304 |

### DQN

| Trial | Estado | Score | Sucesso final | Hiperparametros |
|---:|---|---:|---:|---|
| 0 | DEFAULT | -5.40 | 100.00% | batch_size=32, buffer_size=100000, exploration_final_eps=0.05, exploration_fraction=0.1, gamma=0.99, gradient_steps=1, learning_rate=0.0001, learning_starts=100, target_update_interval=10000, train_freq=4 |
| 1 | COMPLETE | 1.21 | 100.00% | batch_size=128, buffer_size=10000, exploration_final_eps=0.194283, exploration_fraction=0.059263, gamma=0.970736, gradient_steps=4, learning_rate=8.46801e-05, learning_starts=100, target_update_interval=250, train_freq=8 |
| 2 | COMPLETE | -23.28 | 78.00% | batch_size=32, buffer_size=100000, exploration_final_eps=0.0331873, exploration_fraction=0.248069, gamma=0.968355, gradient_steps=2, learning_rate=1.30336e-05, learning_starts=500, target_update_interval=1000, train_freq=16 |
| 3 | COMPLETE | -2.20 | 100.00% | batch_size=128, buffer_size=100000, exploration_final_eps=0.197509, exploration_fraction=0.0835478, gamma=0.980139, gradient_steps=1, learning_rate=1.2943e-05, learning_starts=2000, target_update_interval=2000, train_freq=1 |
| 4 | COMPLETE | 1.41 | 100.00% | batch_size=128, buffer_size=10000, exploration_final_eps=0.0148296, exploration_fraction=0.242393, gamma=0.952221, gradient_steps=2, learning_rate=0.000379583, learning_starts=100, target_update_interval=10000, train_freq=8 |
| 5 | COMPLETE | 2.25 | 100.00% | batch_size=256, buffer_size=50000, exploration_final_eps=0.112779, exploration_fraction=0.48293, gamma=0.916919, gradient_steps=2, learning_rate=0.00245178, learning_starts=1000, target_update_interval=10000, train_freq=8 |
| 6 | COMPLETE | 3.41 | 100.00% | batch_size=256, buffer_size=50000, exploration_final_eps=0.112945, exploration_fraction=0.496107, gamma=0.90773, gradient_steps=2, learning_rate=0.00281825, learning_starts=1000, target_update_interval=5000, train_freq=8 |
| 7 | PRUNED | -3.05 | n/a | batch_size=256, buffer_size=25000, exploration_final_eps=0.128743, exploration_fraction=0.487999, gamma=0.906601, gradient_steps=4, learning_rate=0.00235202, learning_starts=1000, target_update_interval=5000, train_freq=4 |
| 8 | PRUNED | -35.40 | n/a | batch_size=64, buffer_size=50000, exploration_final_eps=0.0976669, exploration_fraction=0.359703, gamma=0.935281, gradient_steps=2, learning_rate=0.00050825, learning_starts=1000, target_update_interval=500, train_freq=16 |
| 9 | COMPLETE | 1.38 | 100.00% | batch_size=256, buffer_size=50000, exploration_final_eps=0.154062, exploration_fraction=0.367217, gamma=0.929395, gradient_steps=2, learning_rate=6.73273e-05, learning_starts=500, target_update_interval=5000, train_freq=1 |
| 10 | COMPLETE | -0.80 | 100.00% | batch_size=64, buffer_size=25000, exploration_final_eps=0.0728269, exploration_fraction=0.193943, gamma=0.997629, gradient_steps=1, learning_rate=0.000415248, learning_starts=2000, target_update_interval=5000, train_freq=8 |
| 11 | COMPLETE | 4.97 | 100.00% | batch_size=256, buffer_size=50000, exploration_final_eps=0.112011, exploration_fraction=0.496334, gamma=0.900883, gradient_steps=2, learning_rate=0.0029171, learning_starts=1000, target_update_interval=10000, train_freq=8 |
| 12 | PRUNED | -4.70 | n/a | batch_size=256, buffer_size=50000, exploration_final_eps=0.147921, exploration_fraction=0.414198, gamma=0.903317, gradient_steps=2, learning_rate=0.00127183, learning_starts=1000, target_update_interval=250, train_freq=8 |
| 13 | COMPLETE | 2.11 | 100.00% | batch_size=256, buffer_size=50000, exploration_final_eps=0.0872436, exploration_fraction=0.436286, gamma=0.920116, gradient_steps=2, learning_rate=0.00114097, learning_starts=1000, target_update_interval=2000, train_freq=8 |
| 14 | PRUNED | 1.10 | n/a | batch_size=256, buffer_size=50000, exploration_final_eps=0.13179, exploration_fraction=0.499559, gamma=0.900208, gradient_steps=2, learning_rate=0.00106319, learning_starts=1000, target_update_interval=500, train_freq=8 |
| 15 | COMPLETE | 4.62 | 100.00% | batch_size=256, buffer_size=50000, exploration_final_eps=0.17087, exploration_fraction=0.410645, gamma=0.942954, gradient_steps=4, learning_rate=0.00272864, learning_starts=1000, target_update_interval=1000, train_freq=8 |
| 16 | COMPLETE | 4.08 | 100.00% | batch_size=256, buffer_size=50000, exploration_final_eps=0.178521, exploration_fraction=0.347737, gamma=0.950035, gradient_steps=4, learning_rate=0.000877977, learning_starts=1000, target_update_interval=1000, train_freq=4 |
| 17 | PRUNED | 3.35 | n/a | batch_size=64, buffer_size=25000, exploration_final_eps=0.168501, exploration_fraction=0.420089, gamma=0.941418, gradient_steps=4, learning_rate=0.00298793, learning_starts=500, target_update_interval=1000, train_freq=1 |
| 18 | PRUNED | -22.20 | n/a | batch_size=32, buffer_size=10000, exploration_final_eps=0.0685703, exploration_fraction=0.319568, gamma=0.958668, gradient_steps=4, learning_rate=0.000252519, learning_starts=2000, target_update_interval=1000, train_freq=16 |
| 19 | COMPLETE | 4.78 | 66.00% | batch_size=256, buffer_size=50000, exploration_final_eps=0.147146, exploration_fraction=0.437018, gamma=0.922858, gradient_steps=4, learning_rate=0.00129894, learning_starts=1000, target_update_interval=10000, train_freq=8 |
| 20 | COMPLETE | -4.90 | 98.00% | batch_size=256, buffer_size=50000, exploration_final_eps=0.139017, exploration_fraction=0.45041, gamma=0.920901, gradient_steps=4, learning_rate=0.000173645, learning_starts=1000, target_update_interval=10000, train_freq=8 |
| 21 | COMPLETE | 4.49 | 100.00% | batch_size=256, buffer_size=50000, exploration_final_eps=0.167598, exploration_fraction=0.39963, gamma=0.930967, gradient_steps=4, learning_rate=0.00166773, learning_starts=1000, target_update_interval=10000, train_freq=8 |
| 22 | COMPLETE | 4.49 | 100.00% | batch_size=256, buffer_size=50000, exploration_final_eps=0.117828, exploration_fraction=0.448386, gamma=0.914045, gradient_steps=4, learning_rate=0.000701025, learning_starts=1000, target_update_interval=10000, train_freq=8 |
| 23 | COMPLETE | 4.20 | 92.00% | batch_size=256, buffer_size=50000, exploration_final_eps=0.152589, exploration_fraction=0.307117, gamma=0.941377, gradient_steps=4, learning_rate=0.00156056, learning_starts=1000, target_update_interval=10000, train_freq=8 |
| 24 | COMPLETE | 5.00 | 100.00% | batch_size=256, buffer_size=50000, exploration_final_eps=0.175249, exploration_fraction=0.400396, gamma=0.924781, gradient_steps=4, learning_rate=0.00170501, learning_starts=1000, target_update_interval=1000, train_freq=8 |
| 25 | PRUNED | 2.85 | n/a | batch_size=256, buffer_size=50000, exploration_final_eps=0.178125, exploration_fraction=0.375033, gamma=0.926757, gradient_steps=1, learning_rate=0.000682329, learning_starts=1000, target_update_interval=10000, train_freq=8 |
| 26 | PRUNED | -7.75 | n/a | batch_size=128, buffer_size=10000, exploration_final_eps=0.0980636, exploration_fraction=0.456009, gamma=0.911298, gradient_steps=4, learning_rate=0.00163362, learning_starts=100, target_update_interval=250, train_freq=16 |
| 27 | COMPLETE | 4.69 | 100.00% | batch_size=64, buffer_size=25000, exploration_final_eps=0.156725, exploration_fraction=0.316399, gamma=0.923513, gradient_steps=4, learning_rate=0.00175094, learning_starts=2000, target_update_interval=2000, train_freq=4 |
| 28 | PRUNED | -0.45 | n/a | batch_size=32, buffer_size=100000, exploration_final_eps=0.139322, exploration_fraction=0.389096, gamma=0.912852, gradient_steps=1, learning_rate=0.000736499, learning_starts=500, target_update_interval=500, train_freq=1 |
| 29 | PRUNED | -9.40 | n/a | batch_size=32, buffer_size=100000, exploration_final_eps=0.189495, exploration_fraction=0.463124, gamma=0.9347, gradient_steps=4, learning_rate=0.000289269, learning_starts=100, target_update_interval=10000, train_freq=4 |

### PPO

| Trial | Estado | Score | Sucesso final | Hiperparametros |
|---:|---|---:|---:|---|
| 0 | DEFAULT | -8.31 | 100.00% | batch_size=64, clip_range=0.2, ent_coef=1e-08, gae_lambda=0.95, gamma=0.99, learning_rate=0.0003, n_epochs=10, n_steps=2048, vf_coef=0.5 |
| 1 | COMPLETE | -68.70 | 0.00% | batch_size=32, clip_range=0.257427, ent_coef=7.82684e-06, gae_lambda=0.860848, gamma=0.918322, learning_rate=1.39277e-05, n_epochs=3, n_steps=128, vf_coef=0.468422 |
| 2 | COMPLETE | -79.60 | 82.00% | batch_size=256, clip_range=0.129302, ent_coef=0.000383404, gae_lambda=0.860923, gamma=0.980759, learning_rate=3.12332e-05, n_epochs=20, n_steps=2048, vf_coef=0.580114 |
| 3 | COMPLETE | -57.05 | 62.00% | batch_size=256, clip_range=0.158795, ent_coef=2.00898e-08, gae_lambda=0.817699, gamma=0.992095, learning_rate=5.91761e-05, n_epochs=5, n_steps=512, vf_coef=0.493998 |
| 4 | COMPLETE | -78.71 | 0.00% | batch_size=128, clip_range=0.122213, ent_coef=2.51971e-06, gae_lambda=0.954254, gamma=0.972828, learning_rate=2.23402e-05, n_epochs=10, n_steps=256, vf_coef=0.336902 |
| 5 | COMPLETE | -14.30 | 100.00% | batch_size=64, clip_range=0.387931, ent_coef=0.0219652, gae_lambda=0.981533, gamma=0.903418, learning_rate=0.00152958, n_epochs=10, n_steps=2048, vf_coef=0.960221 |
| 6 | COMPLETE | -6.79 | 100.00% | batch_size=64, clip_range=0.284652, ent_coef=1.27001e-08, gae_lambda=0.927255, gamma=0.943392, learning_rate=0.000601887, n_epochs=10, n_steps=64, vf_coef=0.825719 |
| 7 | COMPLETE | 1.99 | 96.00% | batch_size=64, clip_range=0.35936, ent_coef=0.035389, gae_lambda=0.899208, gamma=0.946824, learning_rate=0.00189477, n_epochs=5, n_steps=64, vf_coef=0.990747 |
| 8 | PRUNED | -100.00 | n/a | batch_size=32, clip_range=0.398444, ent_coef=0.0438483, gae_lambda=0.899379, gamma=0.95224, learning_rate=0.000132144, n_epochs=5, n_steps=1024, vf_coef=0.733505 |
| 9 | COMPLETE | -9.76 | 100.00% | batch_size=128, clip_range=0.323055, ent_coef=0.000413651, gae_lambda=0.807248, gamma=0.938715, learning_rate=0.00246184, n_epochs=5, n_steps=64, vf_coef=0.908632 |
| 10 | PRUNED | -98.05 | n/a | batch_size=64, clip_range=0.217264, ent_coef=8.24223e-07, gae_lambda=0.996537, gamma=0.963979, learning_rate=0.000577994, n_epochs=3, n_steps=64, vf_coef=0.686883 |
| 11 | PRUNED | 0.05 | n/a | batch_size=64, clip_range=0.316584, ent_coef=0.000298087, gae_lambda=0.909145, gamma=0.938575, learning_rate=0.000716322, n_epochs=20, n_steps=64, vf_coef=0.839704 |
| 12 | COMPLETE | -8.00 | 98.00% | batch_size=64, clip_range=0.320056, ent_coef=2.03613e-05, gae_lambda=0.908019, gamma=0.939069, learning_rate=0.00104917, n_epochs=5, n_steps=64, vf_coef=0.997415 |
| 13 | COMPLETE | 2.48 | 98.00% | batch_size=64, clip_range=0.273121, ent_coef=1.57989e-07, gae_lambda=0.932337, gamma=0.951196, learning_rate=0.000284295, n_epochs=10, n_steps=64, vf_coef=0.836898 |
| 14 | PRUNED | -50.05 | n/a | batch_size=64, clip_range=0.357759, ent_coef=1.42313e-07, gae_lambda=0.868039, gamma=0.957323, learning_rate=0.000235527, n_epochs=10, n_steps=1024, vf_coef=0.824785 |
| 15 | COMPLETE | 1.37 | 90.00% | batch_size=64, clip_range=0.25617, ent_coef=0.0070657, gae_lambda=0.8818, gamma=0.919745, learning_rate=0.00291015, n_epochs=5, n_steps=256, vf_coef=0.740367 |
| 16 | PRUNED | -63.30 | n/a | batch_size=32, clip_range=0.365252, ent_coef=4.67166e-05, gae_lambda=0.94045, gamma=0.923988, learning_rate=7.6959e-05, n_epochs=20, n_steps=128, vf_coef=0.90897 |
| 17 | PRUNED | -100.00 | n/a | batch_size=256, clip_range=0.268713, ent_coef=0.00181012, gae_lambda=0.831724, gamma=0.965972, learning_rate=0.000295783, n_epochs=10, n_steps=512, vf_coef=0.67795 |
| 18 | PRUNED | -1.00 | n/a | batch_size=128, clip_range=0.211969, ent_coef=1.12376e-07, gae_lambda=0.92545, gamma=0.951388, learning_rate=0.00159786, n_epochs=3, n_steps=64, vf_coef=0.896969 |
| 19 | PRUNED | -100.00 | n/a | batch_size=64, clip_range=0.349136, ent_coef=8.59212e-05, gae_lambda=0.886494, gamma=0.928217, learning_rate=0.000153373, n_epochs=5, n_steps=64, vf_coef=0.987489 |
| 20 | COMPLETE | 2.79 | 100.00% | batch_size=64, clip_range=0.291832, ent_coef=7.80821e-07, gae_lambda=0.965838, gamma=0.909493, learning_rate=0.000512648, n_epochs=10, n_steps=64, vf_coef=0.280518 |
| 21 | COMPLETE | -7.56 | 100.00% | batch_size=64, clip_range=0.288845, ent_coef=3.10932e-07, gae_lambda=0.963441, gamma=0.913309, learning_rate=0.000406877, n_epochs=10, n_steps=64, vf_coef=0.304085 |
| 22 | COMPLETE | -5.54 | 76.00% | batch_size=64, clip_range=0.237086, ent_coef=2.00499e-06, gae_lambda=0.972054, gamma=0.900074, learning_rate=0.00108301, n_epochs=10, n_steps=64, vf_coef=0.397003 |
| 23 | PRUNED | -100.00 | n/a | batch_size=64, clip_range=0.287688, ent_coef=5.63575e-08, gae_lambda=0.934347, gamma=0.932101, learning_rate=0.000156231, n_epochs=10, n_steps=64, vf_coef=0.609436 |
| 24 | PRUNED | -0.10 | n/a | batch_size=64, clip_range=0.332029, ent_coef=4.7207e-07, gae_lambda=0.992475, gamma=0.910065, learning_rate=0.000453353, n_epochs=10, n_steps=64, vf_coef=0.258877 |
| 25 | COMPLETE | 2.88 | 100.00% | batch_size=64, clip_range=0.179857, ent_coef=3.92469e-06, gae_lambda=0.917995, gamma=0.94478, learning_rate=0.000938135, n_epochs=5, n_steps=512, vf_coef=0.427176 |
| 26 | PRUNED | -0.50 | n/a | batch_size=256, clip_range=0.172971, ent_coef=5.59334e-06, gae_lambda=0.914991, gamma=0.979387, learning_rate=0.00100533, n_epochs=20, n_steps=512, vf_coef=0.404147 |
| 27 | PRUNED | -78.65 | n/a | batch_size=128, clip_range=0.104859, ent_coef=1.1951e-06, gae_lambda=0.949298, gamma=0.959998, learning_rate=0.000218559, n_epochs=3, n_steps=512, vf_coef=0.25513 |
| 28 | COMPLETE | -5.04 | 100.00% | batch_size=32, clip_range=0.178169, ent_coef=5.61313e-08, gae_lambda=0.966679, gamma=0.929957, learning_rate=0.000742258, n_epochs=10, n_steps=512, vf_coef=0.383235 |
| 29 | COMPLETE | -3.91 | 100.00% | batch_size=64, clip_range=0.237302, ent_coef=1.21553e-05, gae_lambda=0.95319, gamma=0.969266, learning_rate=0.000378361, n_epochs=10, n_steps=1024, vf_coef=0.529897 |

## Arquivos gerados

- Bancos SQLite, CSVs de trials e retornos intermediarios: `results/logs/optuna`.
- Graficos de historico e importancia: `results/figures`.
- JSONs `*_best_params.json`: configuracoes candidatas para a Etapa 6.

## Limitacoes

- Cada trial usa uma unica seed de treino: o retorno de um trial e uma estimativa ruidosa. A Etapa 6 reavalia as configuracoes com varias seeds.
- O orcamento por trial e menor que o do treino final.
