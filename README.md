# FloodGuard

Ambiente Gymnasium e experimentos de aprendizado por reforço profundo para o
problema de contenção de enchente e resgate de animais (versão simplificada:
1 animal, grid 6×6). Ver `proposta-validacao-enchente-simplificada.md` para a
formalização completa do MDP e `plano-execucao.md` para o plano de execução
do trabalho por etapas.

## Estrutura do projeto

```
floodguard/
  envs/flood_guard_env.py   # MDP / ambiente Gymnasium (Etapa 1 — implementado)
  rendering/renderer.py     # visualização via matplotlib (Etapa 2 — implementado)
  utils/seeding.py          # utilitário de reprodutibilidade
experiments/                # scripts de treino, renderização de exemplo e otimização de hiperparâmetros
  random_baseline.py        # baseline aleatório / sanity check do ambiente (Etapa 3)
  train.py                  # treino default DQN/PPO/A2C com stable-baselines3 (Etapa 4)
  tune.py                   # busca Optuna de hiperparâmetros (Etapa 5)
notebooks/                  # relatório final em literate programming (Etapa 7)
results/
  baselines/                # métricas do agente aleatório (Etapa 3)
  models/                   # checkpoints treinados (não versionado)
  logs/                     # logs de treino / tensorboard / optuna (não versionado)
  figures/                  # gráficos e imagens usados no relatório (versionado)
tests/                      # testes automatizados (pytest)
```

## Setup

**Requisito:** Python **3.10, 3.11 ou 3.12** (`stable-baselines3`/`torch`
ainda não têm suporte estável a versões mais novas — se `python3 --version`
no seu sistema já mostra 3.13+, use o `uv` como abaixo em vez do `python3`
do sistema).

Este projeto usa [`uv`](https://docs.astral.sh/uv/) para criar o ambiente
virtual com a versão correta do Python, mesmo que o Python padrão do sistema
seja outra versão:

```bash
# instala uv, se ainda não tiver
curl -LsSf https://astral.sh/uv/install.sh | sh

# cria o venv com Python 3.11 e instala as dependências
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install -e .   # instala o pacote `floodguard` em modo editável
```

Alternativamente, com `venv`/`pip` puro (garanta antes que `python3
--version` está entre 3.10 e 3.12):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

`requirements.txt` fixa faixas de versão compatíveis; `requirements-lock.txt`
registra as versões exatas instaladas no ambiente usado para gerar os
resultados do relatório (útil para reprodutibilidade exata, não para
instalação do zero).

## Rodando os testes

```bash
source .venv/bin/activate
pytest
```

`tests/test_setup.py` (Etapa 0) confirma que as dependências estão
instaladas, que o ambiente `FloodGuard-v0` está registrado no Gymnasium e
que a seed global é determinística. `tests/test_env.py` (Etapa 1) cobre a
dinâmica do MDP em si: reset determinístico, custos de movimento, bloqueio
por água profunda/borda, barreiras, resgate/entrega, perda do animal,
bateria zerada, truncamento por steps e conformidade com a API do
Gymnasium (`check_env`). `tests/test_rendering.py` (Etapa 2) cobre a
renderização: formato/dtype do frame RGB, os diferentes status do animal e
comportamento de `render()` sem `render_mode`.

## Visualização

`env.render()` (com `FloodGuardEnv(render_mode="rgb_array")`) retorna um
frame RGB do estado atual do grid — robô, animal, base, zona segura, água
por nível de inundação e barreiras instaladas, no estilo do mockup em
`assets/floodguard_mockup_simplificado.png`. Para gerar um episódio de
exemplo (imagens + GIF) em `results/figures/`:

```bash
source .venv/bin/activate
python experiments/render_example_episode.py
```

Episódio de exemplo (política heurística fixa, só para ilustrar as
mecânicas do ambiente — não é um agente treinado):

![Episódio de exemplo do FloodGuardEnv: o robô instala uma barreira, resgata o animal e o entrega na zona segura enquanto a água avança](results/figures/example_episode.gif)

| Estado inicial | Barreira instalada | Animal resgatado | Entrega concluída |
|---|---|---|---|
| ![Estado inicial](results/figures/example_episode_initial.png) | ![Barreira instalada](results/figures/example_episode_barrier.png) | ![Animal resgatado](results/figures/example_episode_rescue.png) | ![Entrega concluída](results/figures/example_episode_final.png) |

## Baseline aleatório / sanity check

A Etapa 3 roda uma política uniformemente aleatória para verificar se o
ambiente está calibrado antes dos treinos com DQN, PPO e A2C. O script salva
métricas por episódio, resumo agregado e um relatório curto:

```bash
source .venv/bin/activate
python experiments/random_baseline.py --episodes 1000 --seed 42
```

Arquivos gerados em `results/baselines/`:

- `random_baseline_episodes.csv` — métricas por episódio.
- `random_baseline_summary.json` — resumo agregado para reuso em scripts.
- `random_baseline_report.md` — leitura do sanity check e métricas
  fixadas para as próximas etapas.

Nesta calibração, `flood_advance_prob` foi reduzido para `0.05` e
`flood_deepen_prob` permanece em `0.10`, deixando a taxa de sucesso aleatória
baixa, mas não nula, sem tornar a perda do animal irrelevante.

### Métricas da Etapa 3

O ambiente inclui campos extras em `info` para tornar a avaliação dos
episódios auditável. Essas métricas não alteram a dinâmica do MDP; apenas
registram o que aconteceu durante o episódio.

| Métrica | Significado |
|---|---|
| `battery_spent` | Energia acumulada gasta em movimentos e instalação de barreiras. É diferente da bateria final, porque o robô pode recarregar na base. |
| `termination_reason` | Motivo do fim do episódio: `success`, `animal_lost`, `battery_depleted`, `max_steps` ou `running` enquanto o episódio ainda não terminou. |
| `barriers_installed` | Número de barreiras realmente instaladas no episódio. Ações inválidas de instalação não entram nessa contagem. |
| `effective_barriers_installed` | Número de barreiras instaladas em células com risco imediato: célula rasa ou célula seca adjacente à água. |
| `pickup_step` | Step em que o robô pegou o animal pela primeira vez. Fica `None` se o animal nunca foi coletado. |
| `delivery_step` | Step em que o robô entregou o animal na zona segura. Fica `None` se não houve sucesso. |

No resumo agregado do baseline, esses campos viram métricas como
`battery_spent_mean`, `battery_depleted_rate`,
`barriers_installed_mean`, `effective_barriers_installed_mean` e
`steps_to_success_mean`, usadas depois para comparar DQN, PPO e A2C.

## Etapa 4 - Treino com Stable-Baselines3

A Etapa 4 treina e avalia DQN, PPO e A2C com 500.000 timesteps, seed 42 e
hiperparâmetros padrão do Stable-Baselines3.

```bash
source .venv/bin/activate
python experiments/train.py --algo dqn
python experiments/train.py --algo ppo
python experiments/train.py --algo a2c
```

Os três algoritmos superaram o agente aleatório e a etapa foi validada. A
metodologia, os resultados e os links para os artefatos estão no
[relatório consolidado da Etapa 4](results/relatorio_etapa_4.md).

## Etapa 5 - Otimizacao com Optuna

A Etapa 5 busca hiperparametros melhores para DQN, PPO e A2C. Cada execucao
salva o banco SQLite do Optuna, um CSV com todos os trials, o JSON com a melhor
configuracao, graficos em `results/figures/` e atualiza o
[relatorio consolidado da Etapa 5](results/relatorio_etapa_5.md).

Busca completa sugerida:

```bash
source .venv/bin/activate
python experiments/tune.py --algo dqn --n-trials 20 --total-timesteps 100000 --eval-episodes 30
python experiments/tune.py --algo ppo --n-trials 20 --total-timesteps 100000 --eval-episodes 30
python experiments/tune.py --algo a2c --n-trials 20 --total-timesteps 100000 --eval-episodes 30
```

Smoke test rapido:

```bash
source .venv/bin/activate
python experiments/tune.py --algo a2c --n-trials 1 --total-timesteps 32 --eval-episodes 1 --eval-freq 0
```

## Reprodutibilidade

Use `floodguard.utils.set_global_seed(seed)` no início de qualquer script de
treino/avaliação/otimização para fixar as seeds de `random`, `numpy` e
`torch` de forma consistente entre os experimentos.
