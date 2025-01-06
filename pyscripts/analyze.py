import os
import csv
import re
import matplotlib.pyplot as plt

def parse_log_file(filepath):
    data = {}
    with open(filepath, 'r') as file:
        for line in file:
            if "Generate:" in line:
                data['Generate'] = float(line.split(":")[1])
            elif "Fill:" in line:
                data['Fill'] = float(line.split(":")[1])
            elif "SPMV:" in line:
                data['SPMV'] = float(line.split(":")[1])
            elif "AXPY:" in line:
                data['AXPY'] = float(line.split(":")[1])
            elif "DOT:" in line:
                data['DOT'] = float(line.split(":")[1])
            elif "NNZ:" in line:
                data['NNZ'] = int(line.split(":")[1])
            elif "Iterations:" in line:
                data['Iterations'] = int(line.split(":")[1])
            elif "Residual:" in line:
                data['Residual'] = float(line.split(":")[1])
    return data

def calculate_gflops(data, N):
    gflops = {}
    if 'SPMV' in data and 'NNZ' in data:
        gflops['SPMV'] = (2.0 * data['NNZ']) / (data['SPMV'] * 1e9) if data['SPMV'] > 0 else 0
    if 'DOT' in data:
        gflops['DOT'] = (2.0 * N) / (data['DOT'] * 1e9) if data['DOT'] > 0 else 0
    if 'AXPY' in data:
        gflops['AXPY'] = (2.0 * N) / (data['AXPY'] * 1e9) if data['AXPY'] > 0 else 0
    return gflops

def process_logs(log_folder):
    data_by_function = {"Generate": {}, "Fill": {}, "SPMV": {}, "AXPY": {}, "DOT": {}}
    gflops_by_function = {"SPMV": {}, "AXPY": {}, "DOT": {}}

    for filename in os.listdir(log_folder):
        if filename.endswith(".log"):
            match = re.match(r"(\d+)_(\d+)\.log", filename)
            if match:
                Nx = int(match.group(1))
                T = int(match.group(2))
                N = (Nx + 1) * (Nx + 1)

                filepath = os.path.join(log_folder, filename)
                log_data = parse_log_file(filepath)

                if "Iterations" in log_data:
                    iters = log_data["Iterations"]
                    if "SPMV" in log_data and iters > 0:
                        spmv_calls = iters + 1
                        log_data["SPMV"] = log_data["SPMV"] / spmv_calls
                    
                    if "AXPY" in log_data and iters > 0:
                        axpy_calls = 3 * iters
                        if axpy_calls > 0:
                            log_data["AXPY"] = log_data["AXPY"] / axpy_calls

                    if "DOT" in log_data and iters >= 0:
                        dot_calls = 2 * iters + 1
                        log_data["DOT"] = log_data["DOT"] / dot_calls

                gflops_data = calculate_gflops(log_data, N)

                for key in data_by_function:
                    if key in log_data:
                        if T not in data_by_function[key]:
                            data_by_function[key][T] = {}
                        data_by_function[key][T][N] = log_data[key]

                for key in gflops_by_function:
                    if key in gflops_data:
                        if key not in gflops_by_function:
                            gflops_by_function[key] = {}
                        if N not in gflops_by_function[key]:
                            gflops_by_function[key][N] = {}
                        gflops_by_function[key][N][T] = gflops_data[key]

    return data_by_function, gflops_by_function

def save_tables(data_by_function, gflops_by_function, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for function, data in data_by_function.items():
        output_file = os.path.join(output_folder, f"{function}_time_table.csv")
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)

            all_N = sorted({N for T_data in data.values() for N in T_data})
            writer.writerow(["T / N"] + all_N)

            for T, N_data in sorted(data.items()):
                row = [T] + [N_data.get(N, "") for N in all_N]
                writer.writerow(row)

    for function, gflops_data in gflops_by_function.items():
        output_file = os.path.join(output_folder, f"{function}_gflops_table.csv")
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)

            all_N = sorted(gflops_data.keys())
            writer.writerow(["T / N"] + all_N)

            for T in sorted({T for N_data in gflops_data.values() for T in N_data}):
                row = [T] + [gflops_data[N].get(T, "") for N in all_N]
                writer.writerow(row)

def plot_graphs(data_by_function, gflops_by_function, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for function, data in data_by_function.items():
        plt.figure()

        for T, N_data in sorted(data.items()):
            sorted_data = sorted(N_data.items())
            N_values, times = zip(*sorted_data)
            plt.plot(N_values, times, marker='o', label=f"T = {T}")

        plt.xlabel("Размерность матрицы (N)")
        plt.ylabel("Время выполнения, с (среднее за 1 вызов)")
        plt.title(f"Время выполнения (среднее) для {function}")
        plt.legend()
        plt.grid(True)

        output_file = os.path.join(output_folder, f"{function}_time.png")
        plt.savefig(output_file)
        plt.close()

    for function, gflops_data in gflops_by_function.items():
        plt.figure()

        for N, T_data in sorted(gflops_data.items()):
            sorted_data = sorted(T_data.items())
            T_values, gflops = zip(*sorted_data)
            plt.plot(T_values, gflops, marker='o', label=f"N = {N}")

        plt.xlabel("Количество потоков (T)")
        plt.ylabel("GFLOPS (на один вызов)")
        plt.title(f"Производительность (GFLOPS) для {function}")
        plt.legend()
        plt.grid(True)

        output_file = os.path.join(output_folder, f"{function}_gflops.png")
        plt.savefig(output_file)
        plt.close()

def main():
    log_folder = "summary"
    common_output_folder = "output"
    table_output_folder = os.path.join(common_output_folder, "tables")
    graph_output_folder = os.path.join(common_output_folder, "plots")

    data_by_function, gflops_by_function = process_logs(log_folder)

    save_tables(data_by_function, gflops_by_function, table_output_folder)

    plot_graphs(data_by_function, gflops_by_function, graph_output_folder)

if __name__ == "__main__":
    main()
