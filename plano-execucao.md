# Plano de Execução — FloodGuard (versão simplificada)

> Baseado na proposta já validada com o professor
> (`proposta-validacao-enchente-simplificada.md`): 1 animal, grid 6×6, bateria,
> barreiras, avanço estocástico da água; algoritmos DQN, PPO e A2C via
> stable-baselines3.

Este documento organiza o trabalho em **etapas independentes**, cada uma com
objetivo, entradas, saídas/entregáveis e critério de "pronto". A ordem sugerida
respeita as dependências reais do projeto (ex.: não dá pra otimizar
hiperparâmetros antes de o ambiente existir), mas cada etapa é escrita para ser
retomada isoladamente — por outra pessoa do grupo, em outra sessão, ou por mim
em uma conversa futura — sem precisar reconstruir o contexto das etapas
anteriores além do que está registrado nos entregáveis.

Mapeamento com a tarefa da disciplina:
- Etapas 0–2 → "Problema" (MDP + ambiente Gymnasium + visualização)
- Etapas 3–5 → "Algoritmos" e otimização de hiperparâmetros
- Etapa 6 → "Experimentos"
- Etapa 7 → "Relatório do trabalho"

---

## Etapa 0 — Setup do projeto

**Objetivo:** preparar a estrutura de repositório e o ambiente de
desenvolvimento para que as etapas seguintes possam ser codificadas sem
retrabalho de infraestrutura.

**Tarefas:**
- Criar estrutura de diretórios:
  ```
  floodguard/
    envs/flood_guard_env.py      # implementação do MDP/Gymnasium
    rendering/                   # renderização matplotlib/pygame
  experiments/
    train.py                     # treino de um algoritmo dado config
    tune_<algo>.py (ou 1 script parametrizado)  # busca Optuna
  notebooks/
    relatorio.ipynb              # notebook final (literate programming)
  results/
    models/                      # checkpoints treinados
    logs/                        # tensorboard / csv de treino
    figures/                     # gráficos gerados para o relatório
  tests/
    test_env.py
  ```
- Definir gerenciador de ambiente (venv/conda) e `requirements.txt` ou
  `pyproject.toml` com: `gymnasium`, `stable-baselines3[extra]`, `optuna`,
  `matplotlib`, `numpy`, `pytest`, `jupyter`, `pygame` (opcional, para
  renderização animada), `tensorboard`.
- Configurar seed global e utilitário de reprodutibilidade (semente única
  registrada nos experimentos).
- `git` já está inicializado; criar `.gitignore` (venv, `results/models/*`,
  `results/logs/*`, checkpoints grandes) para não versionar artefatos
  pesados — versionar só código, configs e figuras finais usadas no relatório.

**Entregáveis:** repositório com estrutura acima, ambiente instalável (`pip
install -r requirements.txt` funcionando), `README.md` curto explicando como
rodar.

**Pronto quando:** um membro do grupo consegue clonar o repo, instalar
dependências e rodar `pytest` (mesmo que sem testes ainda) sem erro.

---

## Etapa 1 — Implementação do MDP como ambiente Gymnasium

**Objetivo:** implementar `FloodGuardEnv` seguindo exatamente a formalização
da Seção 3 da proposta (estado, ação, transição, recompensa, terminação).

**Tarefas:**
- Definir `observation_space` (provavelmente `Dict` ou vetor achatado
  combinando: posição do robô, bateria, mapa de inundação 6×6, mapa de
  barreiras 6×6, kits restantes, status/posição do animal, steps restantes).
- Definir `action_space = Discrete(6)`.
- Implementar `reset()`: posiciona robô na base, animal na posição fixa,
  grid seco exceto foco de enchente, bateria cheia, kits iniciais, contador
  de steps.
- Implementar `step()`:
  - Custo de bateria por movimento conforme célula (seca/rasa/profunda
    intransponível).
  - Ação `instalar_barreira`: consome kit + custo de bateria, marca célula
    como protegida.
  - Ação `esperar/carregar`: recarrega bateria só se sobre a base.
  - Resgate/entrega automáticos ao ocupar a célula do animal / zona segura.
  - Avanço estocástico da água (probabilidades `p` e `q`), respeitando
    células protegidas por barreira.
  - Cálculo de recompensa conforme Seção 3 (resgate, barreira efetiva, custo
    por step, perda do animal, bateria zerada fora da base).
  - Condições de terminação (`terminated`/`truncated` separados, conforme
    API do Gymnasium).
- Registrar o ambiente via `gymnasium.register` para permitir
  `gym.make("FloodGuard-v0")`.
- Escrever testes unitários (`tests/test_env.py`) cobrindo: reset determinístico,
  transições de bateria, instalação de barreira bloqueando avanço da água,
  terminação por resgate/perda/bateria/steps, e `gymnasium.utils.env_checker.check_env`.

**Entregáveis:** `floodguard/envs/flood_guard_env.py` + testes passando.

**Pronto quando:** `check_env(env)` não acusa erros e os testes cobrem todos
os caminhos de terminação descritos na proposta.

---

## Etapa 2 — Renderização / visualização

**Objetivo:** implementar os mecanismos de visualização exigidos pelo escopo
da tarefa, reaproveitando o mockup já desenhado na proposta.

**Tarefas:**
- Implementar `render(mode="rgb_array")` com matplotlib reproduzindo os
  elementos do mockup (`assets/floodguard_mockup_simplificado.png`): robô
  (triângulo + indicador de bateria), animal, zona segura, base, barreiras,
  células por nível de inundação.
- (Opcional, se sobrar tempo) versão animada com pygame ou
  `matplotlib.animation` para gerar um GIF/vídeo de um episódio completo —
  útil como material de apresentação e para o relatório.
- Função utilitária para salvar frames de um episódio (para ilustrar o
  relatório com um "episódio exemplo": estados inicial, intermediário e
  final).

**Entregáveis:** `floodguard/rendering/`, função `render()` integrada ao env,
script/exemplo que gera 2–3 imagens de exemplo em `results/figures/`.

**Pronto quando:** dá pra rodar um episódio com política aleatória e obter
imagens/gif legíveis do que está acontecendo no grid.

---

## Etapa 3 — Baseline e sanity check do ambiente

**Objetivo:** confirmar que o ambiente tem uma dinâmica sensata antes de
gastar tempo de computação com os algoritmos de RL profundo.

**Tarefas:**
- Rodar agente aleatório por N episódios, coletar métricas básicas: retorno
  médio, taxa de resgate bem-sucedido, taxa de perda do animal, taxa de
  episódios truncados por steps, taxa de "bateria zerada".
- Verificar se a taxa de resgate aleatória é baixa mas não nula (ambiente
  nem trivial demais nem impossível) — ajustar `p`, `q`, custos de bateria e
  magnitudes de recompensa se necessário nesta etapa (é o momento mais barato
  para recalibrar o MDP).
- Documentar as métricas escolhidas para os experimentos (serão as mesmas
  usadas na Etapa 6): retorno médio por episódio, taxa de sucesso, taxa de
  perda do animal, uso médio de bateria, número médio de steps até o
  resgate, número de barreiras instaladas com efeito.

**Entregáveis:** notebook/script curto com resultados do agente aleatório e
justificativa de eventuais ajustes no MDP; definição final das métricas de
avaliação.

**Pronto quando:** o grupo concorda que o ambiente está "calibrado" (nem
trivial, nem impossível) e as métricas de avaliação estão fixadas.

---

## Etapa 4 — Integração com stable-baselines3 e validação dos 3 algoritmos

**Objetivo:** treinar uma primeira versão (sem otimização) de DQN, PPO e A2C
no ambiente, com hiperparâmetros padrão, para validar a integração.

**Tarefas:**
- Envolver o ambiente com wrappers necessários (`Monitor`, `FlattenObservation`
  se a observação for `Dict`, etc.).
- Script único parametrizável `experiments/train.py --algo {dqn,ppo,a2c}`
  que treina com hiperparâmetros default do SB3 por um número fixo de
  timesteps, salvando logs em `results/logs/<algo>/` (formato tensorboard).
- Rodar os três algoritmos com hiperparâmetros default e comparar curvas de
  aprendizado (retorno médio ao longo do treino) — só para confirmar que
  todos aprendem algo acima do baseline aleatório.

**Entregáveis:** três treinos default rodando, logs no tensorboard,
confirmação de que a integração SB3 ↔ ambiente está correta.

**Pronto quando:** as três curvas de aprendizado mostram progresso
consistente acima do baseline aleatório (senão, voltar à Etapa 1/3 para
revisar o MDP ou os wrappers).

---

## Etapa 5 — Otimização de hiperparâmetros (Optuna)

**Objetivo:** buscar boas configurações de hiperparâmetros para cada um dos
três algoritmos, registrando todas as configurações testadas (não só a
melhor), conforme exigido pela tarefa.

**Tarefas:**
- Definir, por algoritmo, o espaço de busca (ex.: DQN — learning rate,
  buffer size, batch size, exploration fraction, target update interval;
  PPO — learning rate, n_steps, batch_size, n_epochs, gamma, gae_lambda,
  clip_range, ent_coef; A2C — learning rate, n_steps, gamma, ent_coef,
  vf_coef).
- Implementar `experiments/tune.py --algo {dqn,ppo,a2c}` com Optuna:
  função objetivo = retorno médio de avaliação em N episódios após treino
  parcial (usar `optuna.pruners` para descartar trials ruins cedo e economizar
  tempo de computação).
- Rodar um número de trials viável dentro do tempo disponível (definir
  orçamento de trials/tempo por algoritmo antes de começar, ex.: 20–30
  trials cada).
- Salvar todos os trials (não só o melhor) em CSV/`optuna` storage (SQLite)
  para permitir montar depois a tabela "outras configurações testadas"
  exigida no relatório.
- Gerar gráficos do Optuna (`plot_optimization_history`,
  `plot_param_importances`) por algoritmo.

**Entregáveis:** `results/logs/optuna_<algo>.db` (ou CSV equivalente),
gráficos de importância de hiperparâmetros, tabela comparativa de trials por
algoritmo.

**Pronto quando:** há, para cada algoritmo, uma configuração final escolhida
+ registro de pelo menos ~10–20 configurações alternativas testadas com seus
resultados.

---

## Etapa 6 — Experimentos finais e avaliação comparativa

**Objetivo:** treinar a versão final de cada algoritmo com os melhores
hiperparâmetros encontrados e avaliar de forma justa e comparável.

**Tarefas:**
- Treinar cada algoritmo com a melhor configuração da Etapa 5, usando
  múltiplas seeds (ex.: 3–5 seeds por algoritmo) para reportar média ±
  desvio padrão e reduzir o risco de conclusões baseadas em sorte de seed.
- Avaliar cada modelo final em N episódios de teste (política determinística
  quando aplicável) usando as métricas definidas na Etapa 3.
- Gerar tabela comparativa final (algoritmo × métricas) e gráficos:
  curvas de aprendizado sobrepostas, boxplot/barras de retorno final,
  possivelmente um "episódio ilustrado" por algoritmo usando a renderização
  da Etapa 2.
- Discutir trade-offs observados (ex.: DQN mais lento para convergir vs.
  PPO mais estável vs. A2C mais rápido porém mais instável), relacionando
  com o comportamento aprendido em relação ao trade-off proteção-vs-resgate
  do MDP.

**Entregáveis:** tabelas e gráficos finais em `results/figures/`, modelos
treinados salvos em `results/models/`.

**Pronto quando:** há dados suficientes (múltiplas seeds, métricas
consistentes) para escrever a seção de Resultados do relatório com
confiança estatística mínima.

---

## Etapa 7 — Relatório (literate programming)

**Objetivo:** consolidar tudo em um notebook único, no estilo literate
programming, cobrindo as seções exigidas pela tarefa.

**Tarefas:**
- Montar `notebooks/relatorio.ipynb` integrando texto explicativo e código
  (não apenas células de código soltas — cada célula deve ser precedida por
  texto que a contextualiza e seguida de discussão do resultado, quando
  aplicável).
- Estrutura obrigatória:
  1. **Introdução** — contexto do problema (enchente/El Niño em Porto
     Alegre), motivação do MDP, visão geral dos experimentos e principais
     descobertas (resumo dos resultados da Etapa 6).
  2. **Modelagem do problema** — apresentar o MDP completo (estado, ação,
     transição, recompensa, terminação, γ), reaproveitando/expandindo o
     texto da proposta já validada, com o código do `FloodGuardEnv` e
     exemplos de renderização (Etapas 1–2).
  3. **Metodologia de experimentos** — algoritmos empregados e por que
     (Etapa 4), métricas adotadas (Etapa 3), espaço e processo de busca de
     hiperparâmetros incluindo configurações alternativas testadas (Etapa 5).
  4. **Resultados** — tabelas/gráficos da Etapa 6, discussão numérica.
  5. **Conclusões** — visão geral, dificuldades encontradas (calibração do
     MDP, custo computacional da busca de hiperparâmetros, etc.),
     limitações (ex.: grid pequeno, 1 único animal, observabilidade total),
     trabalhos futuros (ex.: extensão para múltiplos animais da proposta
     original, observabilidade parcial, etc.).
- Revisar organização/clareza (a tarefa pontua isso explicitamente):
  sumário no topo, seções numeradas, legendas em figuras/tabelas.
- Exportar o notebook executado (HTML ou PDF) como entregável final, além do
  `.ipynb`.

**Entregáveis:** `notebooks/relatorio.ipynb` executado do início ao fim sem
erros + versão exportada (HTML/PDF).

**Pronto quando:** alguém de fora do grupo consegue entender o problema, a
metodologia e os resultados lendo só o relatório, sem precisar consultar o
código-fonte separadamente.

---

## Observações gerais de execução

- **Divisão entre integrantes:** as Etapas 1–3 (ambiente) formam uma trilha;
  as Etapas 4–6 (algoritmos/experimentos) outra — mas 4–6 dependem de 1–3
  estarem "prontas" (ver critérios de pronto de cada etapa) antes de
  começar de verdade, então vale paralelizar apenas a preparação (ex.:
  alguém já estuda a API do SB3 e escreve `train.py` genérico enquanto o
  ambiente é finalizado).
- **Checkpoints de validação com o professor:** o escopo do problema e os
  três algoritmos já foram validados (conforme a proposta). Se a calibração
  da Etapa 3 exigir mudanças relevantes na função de recompensa ou nas
  probabilidades `p`/`q`, considerar uma checagem rápida com o professor
  antes de seguir para a Etapa 4.
- **Controle de tempo computacional:** a Etapa 5 (Optuna) é a que mais
  consome tempo de máquina — definir orçamento de trials antes de começar
  para não comprometer o cronograma das etapas seguintes.
