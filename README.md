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
notebooks/                  # relatório final em literate programming (Etapa 7)
results/
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

## Reprodutibilidade

Use `floodguard.utils.set_global_seed(seed)` no início de qualquer script de
treino/avaliação/otimização para fixar as seeds de `random`, `numpy` e
`torch` de forma consistente entre os experimentos.
