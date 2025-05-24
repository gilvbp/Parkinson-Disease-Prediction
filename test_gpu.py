import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

# Verifique se o CUDA está disponível
if not torch.cuda.is_available():
    raise Exception("CUDA não está disponível no seu sistema!")

# Defina o dispositivo para múltiplas GPUs
device = torch.device('cuda' if torch.cuda.device_count() > 1 else 'cpu')

# Carregue o modelo ResNet-50
model = models.resnet50(pretrained=False)

# Mova o modelo para múltiplas GPUs, se disponíveis
if torch.cuda.device_count() > 1:
    print(f"Usando {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)

# Mova o modelo para o dispositivo com CUDA
model.to(device)

# Defina um otimizador e uma função de perda
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Exemplo de dados de entrada grandes (tamanho de imagem aumentado)
batch_size = 64
inputs = torch.randn(batch_size, 3, 512, 512).to(device)  # Batch de imagens de 512x512 pixels
labels = torch.randint(0, 1, (batch_size,)).to(device)  # Exemplo de rótulos (classificação de 1000 classes)

# Loop de treinamento
model.train()
for epoch in range(5):
    # Zere os gradientes do otimizador
    optimizer.zero_grad()

    # Passe os dados pelo modelo
    outputs = model(inputs)

    # Calcule a perda
    loss = criterion(outputs, labels)

    # Propague o erro para trás
    loss.backward()

    # Atualize os pesos
    optimizer.step()

    print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

print("Treinamento finalizado!")
