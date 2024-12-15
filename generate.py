def generate_task(T, AffinityT, Nx, Ny):
    task = f"""#BSUB -J "t{T}_{Nx}"
#BSUB -W 0:20
#BSUB -o t{T}_{Nx}
#BSUB -e a.out.%J.err
#BSUB -R "affinity[core({AffinityT})]"
OMP_NUM_THREADS={T} /polusfs/lsf/openmp/launchOpenMP.py ./main {Nx} {Ny} 10 15 {T} 0.00001 no_debug
"""
    return task

# Список размеров сетки
grid_sizes = [
    (99, 99),
    (316, 316),
    (999, 999),
    (3162, 3162)
]

# Генерация отдельных файлов для каждой задачи
for T in [1, 2, 4, 8, 16, 32]:
    for Nx, Ny in grid_sizes:
        AffinityT = T
        if T == 32:
            AffinityT = 16
        task = generate_task(T, AffinityT, Nx, Ny)

        filename = f'input/input_{T}_{Nx}.lsf'
        with open(filename, 'w') as f:
            f.write(task)