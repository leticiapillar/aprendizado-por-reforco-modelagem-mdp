# Plano de Execução — EcoDelivery (versão simplificada)

Plano passo-a-passo para desenvolvimento do trabalho, com base na
[proposta de validação simplificada](proposta-validacao-simplificada.md) (grid
6×6, 1 pacote, 1 estação de recarga, congestionamento estocástico). As etapas
seguem uma ordem de dependência lógica, mas algumas podem ser paralelizadas
(indicado onde relevante).

---

## Etapa 0 — Validação com o professor

**O que fazer:** Apresentar `proposta-validacao-simplificada.md` (MDP + algoritmos
propostos) ao professor e coletar feedback/aprovação.

**Por quê:** É um requisito explícito do trabalho — tanto o problema quanto os
algoritmos escolhidos precisam ser validados antes de começar a implementação.
Começar a codificar antes disso é arriscado: qualquer ajuste pedido pelo
professor (ex.: mudar a função de recompensa, o tamanho do grid, trocar um
algoritmo) é muito mais barato de fazer agora do que depois de já ter código e
experimentos rodados em cima da versão errada.

**Entregável:** Aprovação (ou lista de ajustes) registrada, e o documento de
proposta atualizado caso haja mudanças.

**Critério de conclusão:** Professor validou o escopo do MDP e os três
algoritmos (DQN, PPO, A2C).

---

## Etapa 1 — Configuração do ambiente de desenvolvimento

**O que fazer:**
- Inicializar repositório git (controle de versão desde o início).
- Criar ambiente virtual Python e instalar dependências: `gymnasium`,
  `stable-baselines3`, `optuna`, `matplotlib`, `numpy`, `pandas`, `jupyter`
  (ou `quarto`, se optar por ele para o relatório).
- Definir estrutura de pastas do projeto (ex.: `envs/`, `experiments/`,
  `notebooks/`, `assets/`, `reports/`).

**Por quê:** Ter o setup pronto e versionado desde o início evita retrabalho e
permite reproduzir os experimentos depois (importante para o relatório, que
precisa justificar cada escolha). Um `requirements.txt`/`pyproject.toml` fixando
versões também evita que resultados mudem por causa de uma atualização de
biblioteca no meio do trabalho.

**Entregável:** Repositório com estrutura de pastas, `requirements.txt` (ou
equivalente), e primeiro commit.

**Critério de conclusão:** `import gymnasium, stable_baselines3, optuna` funciona
sem erros no ambiente criado.

---

## Etapa 2 — Implementação do ambiente Gymnasium (`EcoDeliveryEnv`)

**O que fazer:** Implementar a classe do ambiente seguindo a interface padrão do
Gymnasium (`reset()`, `step()`, `observation_space`, `action_space`), conforme a
formalização do MDP já definida:
- Estado: posição do agente, bateria, status do pacote, deadline restante, mapa
  de congestionamento, steps restantes.
- Ação: 5 ações discretas (mover em 4 direções + esperar/carregar).
- Transição: custo de bateria variável por célula, evolução estocástica do
  congestionamento, recarga na estação.
- Recompensa e condições de terminação conforme especificado na proposta.

**Por quê:** É o núcleo do trabalho — sem o ambiente correto, nenhum experimento
de RL é válido. Implementar isso isoladamente (antes de conectar com os
algoritmos) facilita testar e depurar a lógica do MDP sem a complexidade
adicional do treinamento.

**Entregável:** Classe `EcoDeliveryEnv` registrada no Gymnasium (via
`gymnasium.register`), com `step()` e `reset()` funcionais.

**Critério de conclusão:** Rodar um loop com ações aleatórias
(`env.action_space.sample()`) por N episódios sem erros, e observar que
recompensas, terminação e transições de estado fazem sentido.

---

## Etapa 3 — Testes de sanidade do ambiente

**O que fazer:** Escrever pequenos scripts/testes que verificam casos específicos:
- Bateria chega a 0 fora da estação → episódio termina com penalidade.
- Agente entra na estação → bateria recupera corretamente.
- Pacote é entregue → recompensa positiva e episódio termina (ou segue, se
  houver mais lógica).
- Deadline expira → penalidade aplicada corretamente.
- Congestionamento afeta o custo de bateria como esperado.

**Por quê:** Bugs na lógica de recompensa/transição são difíceis de detectar só
olhando curvas de treinamento (um agente pode "aprender" a explorar um bug em
vez de resolver o problema real). Validar cada regra isoladamente agora evita
gastar tempo de treinamento em cima de um ambiente com erro.

**Entregável:** Conjunto de testes (mesmo que informais, tipo `assert`) cobrindo
as regras centrais do MDP.

**Critério de conclusão:** Todos os casos testados se comportam como especificado
na proposta.

---

## Etapa 4 — Mecanismo de renderização

**O que fazer:** Implementar o método `render()` do ambiente, reaproveitando o
estilo visual já prototipado nos mockups (`assets/ecodelivery_mockup_simplificado.png`):
agente, ponto de coleta/entrega, estação de recarga, células congestionadas,
indicador de bateria.

**Por quê:** É um requisito explícito do escopo do trabalho ("mecanismos de
visualização/renderização"), além de ser essencial para depurar visualmente o
comportamento do agente treinado e para ilustrar trajetórias no relatório final.

**Entregável:** Método de renderização (modo estático `rgb_array` e,
idealmente, geração de GIF de um episódio completo).

**Critério de conclusão:** Rodar um episódio com uma política qualquer e gerar
imagens/gif que representam corretamente a trajetória do agente.

*(Esta etapa pode ser feita em paralelo com a Etapa 3.)*

---

## Etapa 5 — Baseline com hiperparâmetros padrão

**O que fazer:** Treinar cada um dos três algoritmos (DQN, PPO, A2C) com os
hiperparâmetros padrão do stable-baselines3, sem otimização ainda, apenas para
confirmar que o ambiente é "treinável" (ou seja, que existe uma política melhor
que aleatória e os algoritmos conseguem encontrá-la).

**Por quê:** Antes de investir tempo em otimização de hiperparâmetros, é
importante confirmar que não há um problema fundamental no ambiente (reward
esparso demais, episódios longos demais, etc.) que impeça qualquer algoritmo de
aprender. Isso também gera uma primeira curva de aprendizado de referência.

**Entregável:** Curvas de treinamento (recompensa média por episódio) para os
três algoritmos com config padrão.

**Critério de conclusão:** Ao menos um algoritmo mostra melhora clara de
desempenho em relação a uma política aleatória.

---

## Etapa 6 — Otimização de hiperparâmetros

**O que fazer:** Definir, para cada algoritmo, o espaço de busca de
hiperparâmetros relevantes (ex.: learning rate, tamanho do buffer/batch,
gamma, arquitetura da rede, `n_steps`/`n_epochs` para PPO/A2C, etc.) e rodar uma
busca com Optuna (ou grid/random search, se preferir algo mais simples de
justificar no relatório). Registrar **todas** as configurações testadas, não só
a melhor.

**Por quê:** É um requisito explícito do trabalho — o relatório precisa mostrar
não apenas a melhor configuração, mas o processo de busca. Isso também é o que
demonstra domínio do assunto: mostrar que a escolha final não foi arbitrária.

**Entregável:** Tabela/log de todas as configurações testadas por algoritmo,
com a métrica de desempenho de cada uma, e a configuração final escolhida.

**Critério de conclusão:** Cada um dos três algoritmos tem uma configuração
final justificada por comparação com alternativas testadas.

---

## Etapa 7 — Experimentos finais

**O que fazer:** Com a configuração final de cada algoritmo, rodar múltiplas
sementes (seeds) de treinamento (ex.: 3 a 5 seeds por algoritmo) para capturar
variância, e avaliar a política final em episódios de teste (sem exploração).
Coletar métricas: recompensa média, taxa de entregas concluídas dentro do
prazo, consumo médio de bateria, taxa de episódios "encalhados", etc.

**Por quê:** Um único run de treinamento pode ser sorte ou azar (RL profundo é
notoriamente sensível a seed). Rodar múltiplas seeds e reportar média ± desvio
padrão é o que torna a comparação entre algoritmos estatisticamente
defensável no relatório.

**Entregável:** Dataset de resultados (CSV ou similar) com métricas por
algoritmo/seed, pronto para análise e geração de gráficos.

**Critério de conclusão:** Resultados suficientes para comparar os três
algoritmos de forma justa (mesmo orçamento de treinamento/steps para todos).

---

## Etapa 8 — Análise dos resultados

**O que fazer:** Gerar gráficos comparativos (curvas de aprendizado, boxplots
de recompensa final, etc.) e discutir: qual algoritmo teve melhor desempenho,
qual foi mais estável, qual convergiu mais rápido, e possíveis explicações
ligadas às características do ambiente (ex.: reward esparso favorece
on-policy vs off-policy).

**Por quê:** É a parte que transforma números em insight — o relatório pede
explicitamente discussão dos resultados, não só apresentação de números.

**Entregável:** Conjunto de gráficos e texto de análise, já em formato
aproveitável para o relatório final.

**Critério de conclusão:** Você consegue responder "qual algoritmo é melhor
para este problema, e por quê" com base nos dados coletados.

---

## Etapa 9 — Redação do relatório (literate programming)

**O que fazer:** Converter o notebook de implementação/experimentação em
relatório final, integrando texto e código, com as seções obrigatórias:
introdução, modelagem do problema (MDP), metodologia de experimentos,
resultados, conclusões (visão geral, dificuldades, limitações, trabalhos
futuros).

**Por quê:** É a entrega final do trabalho e onde a organização e clareza da
comunicação também são avaliadas, não só o código.

**Entregável:** Notebook/relatório final exportado (HTML/PDF), seguindo a
estrutura pedida.

**Critério de conclusão:** Relatório revisado, com todas as seções obrigatórias
preenchidas e código executável de ponta a ponta.

---

## Etapa 10 — Revisão final

**O que fazer:** Revisar o relatório completo (ortografia, clareza, gráficos
legendados, referências ao SB3/Gymnasium citadas corretamente), garantir que o
notebook roda do zero sem erros, e conferir se todos os requisitos do enunciado
foram atendidos.

**Por quê:** Última chance de pegar inconsistências antes da entrega (ex.:
número incorreto de algoritmos, seção faltando, gráfico sem legenda).

**Entregável:** Versão final pronta para entrega.

**Critério de conclusão:** Checklist do enunciado 100% conferido.

---

## Resumo visual das dependências

```
Etapa 0 (validação)
   │
   ▼
Etapa 1 (setup)
   │
   ▼
Etapa 2 (ambiente) ──► Etapa 3 (testes) 
   │                        │
   └──► Etapa 4 (render) ◄──┘
   │
   ▼
Etapa 5 (baseline)
   │
   ▼
Etapa 6 (otimização de hiperparâmetros)
   │
   ▼
Etapa 7 (experimentos finais)
   │
   ▼
Etapa 8 (análise)
   │
   ▼
Etapa 9 (relatório)
   │
   ▼
Etapa 10 (revisão final)
```
