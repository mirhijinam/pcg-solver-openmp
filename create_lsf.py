def generate_task(T, AffinityT, Nx, Ny):
    task = f"""
        #BSUB -J "polusIsDoingVRRRR{T}_{Nx}"
        #BSUB -W 0:20
        #BSUB -o res/{Nx}_{T}
        #BSUB -e err/a.out.%J.err
        #BSUB -R "affinity[core({AffinityT})]"
        OMP_NUM_THREADS={AffinityT} /polusfs/lsf/openmp/launchOpenMP.py ./main {Nx} {Ny} 7 10 {T} 0.00001 no_debug
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

        filename = f'inp/{Nx}_{T}.lsf'
        with open(filename, 'w') as f:
            f.write(task)