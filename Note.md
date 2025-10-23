### Notas sobre adpatação do DAOP com medmnist

- Primeiro, ter a GPU sem estar em modo de poupança de energia (ECO e/ou optimizado)

- In order to execute the framework, please create a conda environment with the dependencies in the [conda_env_short_medmnist.yml](conda_env_short_medmnist.yml) file:

```bash
conda env create -f conda_env_short_medmnist.yml
conda activate daop_medmnist
```

- De seguida executar o seguinte comando 

```bash
pip install medmnist pympler memory_profiler
```

Para usar a TrainResNet18 temos de modificar no Data Loader, uma vez que as imagens do dataset breast tem um canal ($28\times28\times1$) e temos de replir três vezes, transformando-a numa imagem "falsa RGB" ($28\times28\times3$).

Se quisermos modificar o modelo, altera-se o primeiro filtro convolucional (conv1) da ResNet para aceitar 1 canal de entrada em vez de 3