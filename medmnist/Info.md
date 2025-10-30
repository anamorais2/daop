## Compreensão da Framework DAOP

---

### 1. `DA/data_augmentation_albumentations.py`

Define o **espaço de pesquisa** (catálogo de transformações) que o Algoritmo Evolucionário (EA) utiliza para otimizar o *data augmentation*.

#### Notas:
- É importante considerar o **tamanho das imagens**.  
  Embora o ficheiro de configuração defina o tamanho (por exemplo, 28×28 no MNIST), algumas funções internas têm parâmetros específicos para outras dimensões (ex: 32×32).
- **Solução:** tornar o código mais genérico, utilizando `img_size` dentro da função `da_funcs_probs`.

---

### 2. `EA.py`

É o **núcleo principal** do sistema.  
É invocado através do ficheiro `main_do.py` e controla todo o **ciclo de vida do EA**.

#### Principais funcionalidades:
1. Verifica se deve carregar um estado anterior (se `start_parent` for diferente de `None`) para continuar uma execução interrompida.  
2. Gera o **cromossomo Downstream** aleatoriamente (`config['create_chromosome']`) na primeira geração.  
3. Um **indivíduo** tem a seguinte estrutura:
   ```
   [genotype, fitness, pretext_accuracy, training_time, training_history]
   ```

#### Métricas utilizadas

| Fase | Medição (Accuracy) | Objetivo da Métrica |
|:-----|:-------------------|:--------------------|
| **Pretext** | *Pretext Test Accuracy* (Precisão no Teste RotNet) | Métrica auxiliar. Confirma se o pré-treino foi bem-sucedido e se o *backbone* (camadas iniciais da ResNet) está pronto para a tarefa *Downstream*. |
| **Downstream** | *Downstream Test Accuracy* (Precisão no Teste de Classificação) | Métrica principal. Mede a precisão do modelo na tarefa de classificação **BreastMNIST** (conjunto de teste). É o **valor de fitness** que o EA tenta maximizar. |

---

### 3. `evolution_mod_functions.py`

Contém funções que **modificam o comportamento** do EA em gerações específicas.  
Normalmente é usado para:
- Alternar entre fases (pré-treino → downstream);
- Aplicar *fine-tuning* mais prolongado em determinados momentos.

---

### 4. `main_do.py`

Atua como **launcher** do processo evolutivo.  
Garante que o EA é executado para os *seeds* pretendidos, podendo:
- Iniciar uma execução do zero;
- Recuperar um estado guardado de uma execução anterior.

---

### 5. `net_models_torch.py`

Define a **CNN (ResNet18)** utilizada e gere a transição entre fases:

- **Pré-treino (RotNet):** ajusta a rede para a tarefa de predição de rotações.  
- **Downstream:** congela as camadas iniciais e treina apenas o novo classificador.

#### Função principal:
`switch_to_downstream()`  
Prepara o modelo para a tarefa final (**BreastMNIST**):
- Congela as camadas iniciais (`layer3`, `layer4`);
- Substitui a cabeça de classificação (`self.model.fc`) por novas camadas (`ProjectorBlockResNet18`) adaptadas ao número de classes *Downstream*.

---

### 6. `rotnet_torch.py`

Módulo responsável pelas **operações de treino e avaliação** de baixo nível.  
Inclui funções para treino, cálculo de *loss* e precisão.

#### Funções principais:
- **`rotnet_collate_fn`**  
  Aplica-se a cada imagem antes da formação do *batch*.  
  Faz a **replicação de canais (1 → 3)**, garantindo que os tensores enviados para `torch.rot90` e para a ResNet estão sempre no formato `(C, H, W)` com `C = 3`, independentemente de a imagem ser *grayscale* ou *RGB*.

- **`train_pretext()` e `train_downstream()`**  
  Executam os ciclos de treino:
  - `train_pretext`: treina para prever 4 rotações (*RotNet*);
  - `train_downstream`: treina para prever as classes reais (ex: 2 classes no BreastMNIST).

#### Mecânica do RotNet:
O RotNet cria **4 cópias** de cada imagem, rodadas em:  
`0°`, `90°`, `180°`, e `270°`.  
A tarefa do modelo é **classificar qual rotação foi aplicada**.  
Isto força o modelo a aprender *features* robustas e invariantes à rotação, melhorando o desempenho na tarefa *Downstream*.

#### Função de avaliação:
- **`evaluate_rotnet()`**  
  Coordena todo o processo:
  1. Treina o RotNet  
  2. Testa o RotNet  
  3. Muda para *Downstream*  
  4. Treina *Downstream*  
  5. Testa *Downstream*  
  Devolve a **precisão final (fitness)**.

---

### Nota sobre replicação de canal

A replicação de canal foi adotada por ser uma solução **pragmática e robusta**, permitindo utilizar a arquitetura ResNet (originalmente desenhada para 3 canais) **sem modificações complexas**.  
Isto manteve o foco da pesquisa na otimização dos *data augmentations*.

- **Precisão:** não se espera melhoria significativa ao alterar o modelo para 1 canal, pois a informação visual é preservada.  
- **Eficiência:** modificar o `conv1` da ResNet para aceitar 1 canal traria ganhos de velocidade, evitando cálculos redundantes nos canais R, G e B.

---


### 7. `Metricas`

- **`graphs.py`** 
    - Evolução do EA -> Gera gráficos da Max Fitness (sua Accuracy máxima) e Average Fitness ao longo das gerações. Essencial para verificar a convergência do algoritmo.

-  **`train_acc_loss.py`** 
    - Histórico de Treino -> Lê o histórico de Accuracy e perda de treino (Loss) dos modelos (guardada nos ficheiros pickle, se configurado). Útil para diagnosticar overfitting ou underfitting em indivíduos específicos.

-  **`da_distribution.py`** 
    - Frequência de Aumentos -> Calcula a frequência com que cada transformação específica (ex: HorizontalFlip, Rotate, Equalize) foi selecionada pelo Algoritmo Evolucionário no pool de melhores indivíduos.
- **`da_pr_distribution.py`** 
    - Parâmetros Otimizados -> Analisa a distribuição dos parâmetros (PR, valores $p_1$ a $p_4$) que o EA atribuiu aos aumentos de dados mais utilizados. Isto revela o grau de intensidade ideal das transformações (ex: Rotation Limit ideal).
- **`visualize_DA.py`** 
    - Visualização da Aplicação -> Utiliza a política de DA do best_individual (o string complexo no CSV) e aplica-a a imagens de teste para que possa ver o resultado dos aumentos otimizados.

- **`statistic_tests.py`**  
    - Validação Estatística	 -> Permite aplicar testes como ANOVA ou testes post-hoc para determinar se a melhor política de DA do DAOP é estatisticamente superior a um baseline manual (ex: um HorizontalFlip fixo).

- **`grad_cam.py`**  
	- Interpretabilidade do Modelo -> Implementa o Grad-CAM (Gradient-weighted Class Activation Mapping) para gerar mapas de calor, mostrando onde o modelo está a olhar na imagem para tomar a decisão de classificação. Útil para entender se os aumentos de dados estão a forçar o modelo a olhar para as características corretas.


---

### Nota para o cálculo da curva de ROC

    1. rotnet_torch.py: Adicionar o cálculo da AUC e os scores de previsão (outputs) na função test_downstream.

    2. utils.py: Adicionar uma nova função para gerar e guardar o gráfico ROC/AUC.

---

### Nota para o Resnet 50

 - Primeiro de tudo alterar no ficheiro config_base qual é o modelo que queremos resnet18 ou resnet50