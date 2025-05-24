import cupy as cp
import time

# Verifica número de GPUs
n_gpus = cp.cuda.runtime.getDeviceCount()
print(f"Número de GPUs: {n_gpus}")

if n_gpus < 2:
    raise RuntimeError("Precisa de pelo menos 2 GPUs para testar NVLink")

# Verifica e habilita P2P
for i in range(n_gpus):
    for j in range(n_gpus):
        if i == j:
            continue
        can_access = cp.cuda.runtime.deviceCanAccessPeer(i, j)
        print(f"GPU {i} pode acessar GPU {j} via P2P: {'Sim' if can_access else 'Não'}")
        if can_access:
            with cp.cuda.Device(i):
                try:
                    cp.cuda.runtime.deviceEnablePeerAccess(j)
                    print(f"P2P habilitado entre GPU {i} -> GPU {j}")
                except cp.cuda.runtime.CUDARuntimeError as e:
                    if 'PeerAccessAlreadyEnabled' not in str(e):
                        print(f"Erro ao habilitar P2P: {e}")

# Dados
size_mb = 500
size = size_mb * 1024 * 1024 // 4  # float32

# Aloca na GPU 0
with cp.cuda.Device(0):
    a = cp.random.rand(size, dtype=cp.float32)

# Mede tempo de cópia para GPU 1
start = time.time()
with cp.cuda.Device(1):
    b = cp.asarray(a)  # Cópia direta GPU0 -> GPU1 via NVLink
cp.cuda.Stream.null.synchronize()
end = time.time()

elapsed = end - start
speed = size_mb / elapsed

print(f"Tempo de transferência: {elapsed * 1000:.2f} ms")
print(f"Velocidade: {speed:.2f} MB/s")
