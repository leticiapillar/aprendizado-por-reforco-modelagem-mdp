# FloodGuard - Relatorio da Etapa 4

## Objetivo

Verificar se o ambiente FloodGuard funciona com DQN, PPO e A2C do
Stable-Baselines3. Para validar a etapa, os tres algoritmos deviam aprender
durante o treino e superar um agente que escolhe acoes aleatoriamente.

## Como os testes foram feitos

Cada algoritmo foi treinado com:

- 500.000 timesteps;
- seed de treino igual a 42;
- politica `MlpPolicy`;
- hiperparametros padrao do Stable-Baselines3.

Depois do treino, cada modelo foi avaliado em 100 episodios (seeds 10.042 a
10.141) com a **politica deterministica** (acao de maior valor/probabilidade).
O agente aleatorio foi avaliado nos mesmos episodios.

Comandos usados:

```bash
python experiments/train.py --algo dqn --seed 42
python experiments/train.py --algo ppo --seed 42
python experiments/train.py --algo a2c --seed 42
```

## Ajustes para o aprendizado

**Normalizacao da observacao.** Os valores da observacao tem escalas diferentes
(mapas 0/1, bateria e passos ate 100), entao sao normalizados para `[0, 1]` em
`float32` (`NormalizeObservation`). Uma versao anterior usava o
`RescaleObservation` do Gymnasium, que mantem o dtype inteiro do `Box` achatado
e truncava posicao, bateria e passos para 0/1: o agente quase nao via o estado.
Isso foi corrigido (com teste de regressao) e **todos os resultados abaixo foram
regerados apos a correcao**; os numeros da versao anterior nao sao validos.

**Reward shaping no treino.** O agente recebe um sinal de progresso (`+5` ao se
aproximar do proximo objetivo, `-5` ao se afastar). Ele e usado somente no
treino; a avaliacao usa apenas a recompensa original do ambiente.

**Avaliacao deterministica.** E a politica que o algoritmo aprendeu e a unica
comparavel entre os tres (no DQN o modo estocastico do SB3 e epsilon-greedy).
O relatorio mede tambem travamentos: passos consecutivos sem nenhum efeito no
robo.

## Resultados

| Politica | Retorno medio | Sucesso | Perda do animal | Sem conclusao | Passos sem efeito | Episodios travados |
|---|---:|---:|---:|---:|---:|---:|
| Aleatoria | -95,67 | 0% | 44% | 54% | n/a | n/a |
| DQN | 5,00 | 100% | 0% | 0% | 0.0% | 0% |
| PPO | 5,00 | 100% | 0% | 0% | 0.0% | 0% |
| A2C | 3,95 | 99% | 0% | 1% | 5.4% | 1% |

O ambiente tem cenario inicial fixo (robo, animal e zona segura nas mesmas
celulas) e enchente lenta; por isso a politica que segue o caminho mais curto
(15 passos: 5 acima, 5 abaixo, 5 a esquerda) obtem sempre retorno `20 - 15 = 5`.
Retorno 5,00 com 100% de sucesso corresponde ao **resgate direto**. Ainda e
possivel superar 5,00 instalando barreiras efetivas no caminho (+5 cada, ao
custo de 1 passo e de bateria): com os 2 kits o retorno chega a ate 13,00. A
Etapa 6 mostra quais algoritmos aprendem esse comportamento.

Retorno de treino (inclui o reward shaping) por quarto do treinamento:

| Algoritmo | 1o quarto | 2o quarto | 3o quarto | 4o quarto |
|---|---:|---:|---:|---:|
| DQN | 69,15 | 78,54 | 78,64 | 79,13 |
| PPO | 76,06 | 79,79 | 79,46 | 79,35 |
| A2C | 76,67 | 78,44 | 77,09 | 77,77 |

## Conclusao

Com a observacao corrigida, os tres algoritmos superam o agente aleatorio em
retorno e taxa de sucesso na avaliacao deterministica, e a etapa esta validada.
DQN e PPO fazem o resgate direto em todos os episodios (retorno 5,00); o A2C
fica ligeiramente abaixo por um episodio que nao terminou no limite de passos. Como os algoritmos default ja
resolvem a tarefa, a comparacao entre eles (e o efeito da otimizacao de
hiperparametros) depende de outras medidas alem do retorno final, como a
velocidade de convergencia e a estabilidade entre seeds, tratadas nas Etapas 5
e 6.

## Artefatos

- DQN: [`logs/dqn/dqn_seed_42_evaluation.json`](logs/dqn/dqn_seed_42_evaluation.json)
- PPO: [`logs/ppo/ppo_seed_42_evaluation.json`](logs/ppo/ppo_seed_42_evaluation.json)
- A2C: [`logs/a2c/a2c_seed_42_evaluation.json`](logs/a2c/a2c_seed_42_evaluation.json)
