# FloodGuard - Relatorio da Etapa 4

## Objetivo

O objetivo desta etapa foi verificar se o ambiente FloodGuard funciona com os
algoritmos DQN, PPO e A2C do Stable-Baselines3. Para validar a etapa, os tres
algoritmos deveriam aprender durante o treino e superar um agente que escolhe
acoes aleatoriamente.

## Como os testes foram feitos

Cada algoritmo foi treinado com:

- 500.000 timesteps;
- seed de treino igual a 42;
- politica `MlpPolicy`;
- hiperparametros padrao do Stable-Baselines3.

Depois do treino, cada modelo foi avaliado em 100 episodios, usando as seeds
de 10.042 a 10.141. O agente aleatorio foi avaliado nos mesmos episodios para
que a comparacao fosse justa.

Comandos usados:

```bash
python experiments/train.py --algo dqn --seed 42
python experiments/train.py --algo ppo --seed 42
python experiments/train.py --algo a2c --seed 42
```

## Ajustes para o aprendizado

A observacao original possui valores em escalas diferentes. Por exemplo, os
mapas usam valores pequenos, enquanto bateria e quantidade de steps chegam a
100. Por isso, todos os valores foram normalizados para o intervalo de 0 a 1.

Durante o treino, o agente tambem recebeu um sinal de progresso:

- `+5` ao se aproximar do proximo objetivo;
- `-5` ao se afastar do proximo objetivo.

Esse sinal ajuda o agente a descobrir o caminho ate o animal e depois ate a
zona segura. Ele e usado somente no treino. Na avaliacao, foi utilizada apenas
a recompensa original do ambiente.

A avaliacao foi feita de forma estocastica. Isso permite que o agente tente
outra acao quando o avanco da agua bloqueia o caminho que ele escolheria com
maior frequencia.

## Resultados

| Politica | Retorno medio | Sucesso | Coleta do animal | Perda do animal | Episodios sem conclusao |
|---|---:|---:|---:|---:|---:|
| Aleatoria | -95,67 | 0% | 27% | 44% | 54% |
| DQN | -78,80 | 26% | 98% | 1% | 73% |
| PPO | 2,97 | 100% | 100% | 0% | 0% |
| A2C | 2,36 | 100% | 100% | 0% | 0% |

PPO e A2C salvaram o animal em todos os episodios avaliados. Os dois tambem
concluiram a tarefa rapidamente, em cerca de 17 steps.

O DQN superou o agente aleatorio e quase sempre chegou ao animal, mas teve mais
dificuldade para completar a entrega. Por isso, terminou somente 26% dos
episodios com sucesso e atingiu o limite de steps em 73% deles.

Os tres algoritmos melhoraram ao longo do treino:

| Algoritmo | Retorno no primeiro quarto | Retorno no ultimo quarto |
|---|---:|---:|
| DQN | -27,17 | -13,27 |
| PPO | 72,81 | 76,43 |
| A2C | 75,82 | 76,35 |

Esses valores incluem o sinal de progresso usado no treino. Por esse motivo,
eles servem apenas para comparar o inicio e o final de cada treinamento.

## Conclusao

Todos os algoritmos tiveram retorno e taxa de sucesso maiores que o agente
aleatorio. Portanto, a integracao com o Stable-Baselines3 funcionou e a Etapa 4
foi validada.

PPO e A2C apresentaram os melhores resultados. O DQN tambem aprendeu, mas ainda
precisa de ajustes para concluir a entrega com maior frequencia.

## Arquivos completos

- DQN: [`logs/dqn/dqn_seed_42_evaluation.json`](logs/dqn/dqn_seed_42_evaluation.json)
- PPO: [`logs/ppo/ppo_seed_42_evaluation.json`](logs/ppo/ppo_seed_42_evaluation.json)
- A2C: [`logs/a2c/a2c_seed_42_evaluation.json`](logs/a2c/a2c_seed_42_evaluation.json)
- Curvas de treino: `logs/<algoritmo>/tensorboard/`
- Modelos treinados: `models/<algoritmo>/`
