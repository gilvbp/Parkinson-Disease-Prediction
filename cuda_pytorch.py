import torch
import torch.nn as nn
import torch.optim as optim

# Verifique se o CUDA está disponível
if not torch.cuda.is_available():
    raise Exception("CUDA não está disponível no seu sistema!")

# Defina um dispositivo para múltiplas GPUs
device = torch.device('cuda' if torch.cuda.device_count() > 1 else 'cpu')


# Defina uma rede neural simples
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(784, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# Instancie o modelo
model = SimpleNN()

# Mova o modelo para múltiplas GPUs
if torch.cuda.device_count() > 1:
    print(f"Usando {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)

# Mova o modelo para o dispositivo com CUDA
model.to(device)

# Defina um otimizador e uma função de perda
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Exemplo de um conjunto de dados (use um dataset real para treinamento)
inputs = torch.randn(64, 784).to(device)  # Exemplo de batch de dados de entrada
labels = torch.randint(0, 10, (64,)).to(device)  # Exemplo de rótulos

# Loop de treinamento
model.train()
for epoch in range(100000):
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
