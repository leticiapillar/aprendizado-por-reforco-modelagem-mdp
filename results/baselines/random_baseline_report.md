# FloodGuard - Etapa 3: baseline aleatorio

Relatorio gerado por `python experiments/random_baseline.py`.

## Parametros do ambiente

| Parametro | Valor |
|---|---:|
| Episodios | 1000 |
| Seed inicial | 42 |
| Grid | 6x6 |
| Max steps | 100 |
| Bateria maxima | 100 |
| Kits de barreira | 2 |
| Prob. avanco agua rasa (`p`) | 0.05 |
| Prob. aprofundamento (`q`) | 0.1 |

## Resultados do agente aleatorio

| Metrica | Valor |
|---|---:|
| Retorno medio | -95.95 +/- 14.06 |
| Taxa de sucesso | 1.10% |
| Taxa de perda do animal | 48.00% |
| Taxa de truncamento por steps | 49.80% |
| Taxa de bateria zerada fora da base | 1.10% |
| Taxa de coleta do animal | 26.30% |
| Steps medios por episodio | 86.48 |
| Steps medios ate sucesso | 75.45 |
| Uso medio de bateria | 73.33 |
| Barreiras instaladas por episodio | 2.00 |
| Barreiras efetivas por episodio | 0.05 |

## Sanity check

A calibracao adotada nesta etapa reduziu `flood_advance_prob` para `0.05` e manteve `flood_deepen_prob` em `0.1`. Com isso, o agente aleatorio tem sucesso baixo, mas nao nulo, enquanto a perda do animal continua frequente o suficiente para o ambiente nao ficar trivial.

## Metricas fixadas para as proximas etapas

- Retorno medio por episodio (`return_mean`).
- Taxa de sucesso: animal entregue na zona segura (`success_rate`).
- Taxa de perda do animal por agua profunda (`animal_lost_rate`).
- Taxa de episodios truncados pelo limite de steps (`truncated_rate`).
- Taxa de bateria zerada fora da base (`battery_depleted_rate`).
- Uso medio de bateria: energia acumulada gasta em movimento/barreiras (`battery_spent_mean`).
- Numero medio de steps ate sucesso, medido apenas nos episodios bem-sucedidos (`steps_to_success_mean`).
- Numero medio de barreiras instaladas e de barreiras efetivas (`barriers_installed_mean`, `effective_barriers_installed_mean`).

## Dicionario das metricas extras do `info`

| Metrica | Significado |
|---|---|
| `battery_spent` | Energia acumulada gasta em movimentos e instalacao de barreiras. Difere da bateria final porque o robo pode recarregar na base. |
| `termination_reason` | Motivo do fim do episodio: `success`, `animal_lost`, `battery_depleted`, `max_steps` ou `running` enquanto o episodio ainda nao terminou. |
| `barriers_installed` | Numero de barreiras realmente instaladas no episodio. Acoes invalidas de instalacao nao entram nessa contagem. |
| `effective_barriers_installed` | Numero de barreiras instaladas em celulas com risco imediato: celula rasa ou celula seca adjacente a agua. |
| `pickup_step` | Step em que o robo pegou o animal pela primeira vez. Fica `None` se o animal nunca foi coletado. |
| `delivery_step` | Step em que o robo entregou o animal na zona segura. Fica `None` se nao houve sucesso. |

## Arquivos gerados

- `random_baseline_episodes.csv`: metricas por episodio.
- `random_baseline_summary.json`: resumo agregado em formato legivel por codigo.
- `random_baseline_report.md`: este relatorio.
