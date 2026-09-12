# FloodGuard

Ambiente Gymnasium e experimentos de aprendizado por reforço profundo para o
problema de contenção de enchente e resgate de animais (versão simplificada:
1 animal, grid 6×6). Ver `proposta-validacao-enchente-simplificada.md` para a
formalização completa do MDP e `plano-execucao.md` para o plano de execução
do trabalho por etapas.

## Estrutura do projeto

```
floodguard/
  envs/flood_guard_env.py   # MDP / ambiente Gymnasium (Etapa 1)
  rendering/                # visualização do ambiente (Etapa 2)
  utils/seeding.py          # utilitário de reprodutibilidade
experiments/                # scripts de treino e otimização de hiperparâmetros
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

Na Etapa 0 (setup), `tests/test_setup.py` apenas confirma que as
dependências estão instaladas, que o ambiente `FloodGuard-v0` está
registrado no Gymnasium e que a seed global é determinística. Os testes da
dinâmica do MDP em si (`tests/test_env.py`) são adicionados na Etapa 1.

## Reprodutibilidade

Use `floodguard.utils.set_global_seed(seed)` no início de qualquer script de
treino/avaliação/otimização para fixar as seeds de `random`, `numpy` e
`torch` de forma consistente entre os experimentos.
