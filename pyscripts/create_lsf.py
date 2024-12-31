import os

def generate_task(T, AffinityT, Nx, Ny, num):
    task = f"""#BSUB -J "VRRRR_polus_is_working_VRRRR_{T}_{Nx}_{num}"
#BSUB -W 0:20
#BSUB -o res/{Nx}_{T}/{num}.out
#BSUB -e err/{Nx}_{T}/{num}.err
#BSUB -R "affinity[core({AffinityT})]"
OMP_NUM_THREADS={T} /polusfs/lsf/openmp/launchOpenMP.py ./main {Nx} {Ny} 7 10 {T} 0.00001 no_debug
"""
    return task

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

        # Создаём папки для каждого Nx и T
        folder_path = f'inp/{Nx}_{T}'
        res_folder = f'res/{Nx}_{T}'
        err_folder = f'err/{Nx}_{T}'
        os.makedirs(folder_path, exist_ok=True)
        os.makedirs(res_folder, exist_ok=True)
        os.makedirs(err_folder, exist_ok=True)

        for num in range(1, 101):
            task = generate_task(T, AffinityT, Nx, Ny, num)
            filename = f'{folder_path}/{num}.lsf'
            with open(filename, 'w') as f:
                f.write(task)
