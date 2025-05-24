import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
from pycuda import gpuarray
import time

# Inicializa driver
cuda.init()

# Verifica número de GPUs
num_gpus = cuda.Device.count()
print(f"Número de GPUs detectadas: {num_gpus}")

if num_gpus < 2:
    print("É necessário pelo menos duas GPUs para testar o NVLink.")
    exit()

# Cria contexto na GPU 0
gpu0 = cuda.Device(0)
ctx0 = gpu0.make_context()

# Cria contexto na GPU 1
gpu1 = cuda.Device(1)
ctx1 = gpu1.make_context()

# Habilita acesso peer-to-peer (se suportado)
if cuda.Device(0).can_access_peer(1):
    ctx0.push()
    cuda.Context.enable_peer_access(1)
    ctx0.pop()

if cuda.Device(1).can_access_peer(0):
    ctx1.push()
    cuda.Context.enable_peer_access(0)
    ctx1.pop()

# Define tamanho dos dados
size_mb = 500
size = size_mb * 1024 * 1024 // np.dtype(np.float32).itemsize

# Cria array na GPU 0
ctx0.push()
a_gpu0 = gpuarray.to_gpu(np.random.randn(size).astype(np.float32))
ctx0.pop()

# Mede tempo de cópia peer-to-peer
ctx1.push()

start = cuda.Event()
end = cuda.Event()

start.record()

# Aloca espaço na GPU 1
a_gpu1 = gpuarray.empty_like(a_gpu0)

# Copia dados diretamente da GPU0 -> GPU1 via peer-to-peer
cuda.memcpy_peer(a_gpu1.gpudata, 1, a_gpu0.gpudata, 0, a_gpu0.nbytes)

end.record()
end.synchronize()

time_ms = start.time_till(end)
ctx1.pop()

# Mostra resultados
print(f"Tempo de transferência peer-to-peer (NVLink) de {size_mb} MB: {time_ms:.2f} ms")
print(f"Velocidade: {(size_mb / (time_ms / 1000)):.2f} MB/s")

# Libera contexto
ctx0.pop()
ctx1.pop()
ctx0.detach()
ctx1.detach()
