import torch
import torchvision

# 1. Verificar a versão do PyTorch
print(f"PyTorch Version: {torch.__version__}")

# 2. Verificar se o CUDA está disponível
cuda_disponivel = torch.cuda.is_available()
print(f"CUDA está disponível? {cuda_disponivel}")

if cuda_disponivel:
    # 3. Qual o dispositivo CUDA (GPU) ativo?
    print(f"Dispositivo CUDA (GPU) ativo: {torch.cuda.get_device_name(0)}")
    
    # 4. Teste de alocação simples (deve funcionar sem erros)
    x = torch.rand(5, 3).cuda()
    print("Tensor alocado na GPU com sucesso!")

    # 5. Múltiplo teste (para ver se a GPU está realmente a calcular)
    a = torch.tensor([1.0, 2.0], device='cuda')
    b = torch.tensor([3.0, 4.0], device='cuda')
    c = a * b
    print(f"Resultado do cálculo na GPU: {c}")

else:
    print("O PyTorch está a correr apenas na CPU.")

exit()