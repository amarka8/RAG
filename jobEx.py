"""Commands:
module purge ????
module load anaconda3/2024.6 ???
??? conda activate /scratch/network/jdh4/.gpu_workshop/envs/cupy-env

python svd.py ???

$ sbatch job.slurm

$ cat slurm-*.out
"""

from time import perf_counter
import torch

N = 1000

cuda0 = torch.device('cuda')
x = torch.randn(N, N, dtype=torch.float64, device=cuda0)
t0 = perf_counter()
u, s, v = torch.svd(x)
elapsed_time = perf_counter() - t0

print("Execution time: ", elapsed_time)
print("Result: ", torch.sum(s).cpu().numpy())
print("PyTorch version: ", torch.__version__)
