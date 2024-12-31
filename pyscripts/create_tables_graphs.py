import os
import csv
import matplotlib.pyplot as plt

def load_summary_data(summary_folder):
    sequential_data = {"fill": {}, "generate": {}, "spmv": {}, "axpy": {}, "dot": {}}
    parallel_data = {"fill": {}, "generate": {}, "spmv": {}, "axpy": {}, "dot": {}}

    for filename in os.listdir(summary_folder):
        if filename.endswith(".log"):
            Nx, T = map(int, filename.replace(".log", "").split("_"))
            filepath = os.path.join(summary_folder, filename)

            with open(filepath, 'r') as file:
                for line in file:
                    if line.startswith("Generate:"):
                        time = float(line.split(":")[1])
                        (sequential_data if T == 1 else parallel_data)["generate"].setdefault(Nx, {})[T] = time
                    elif line.startswith("Fill:"):
                        time = float(line.split(":")[1])
                        (sequential_data if T == 1 else parallel_data)["fill"].setdefault(Nx, {})[T] = time
                    elif "\tSPMV:" in line:
                        time = float(line.split(":")[1])
                        (sequential_data if T == 1 else parallel_data)["spmv"].setdefault(Nx, {})[T] = time
                    elif "\tAXPY:" in line:
                        time = float(line.split(":")[1])
                        (sequential_data if T == 1 else parallel_data)["axpy"].setdefault(Nx, {})[T] = time
                    elif "\tDOT:" in line:
                        time = float(line.split(":")[1])
                        (sequential_data if T == 1 else parallel_data)["dot"].setdefault(Nx, {})[T] = time

    return sequential_data, parallel_data

def save_csv(data, output_folder, filename, keys):
    filepath = os.path.join(output_folder, filename)
    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        header = ["Nx"] + keys
        writer.writerow(header)
        for Nx, values in data.items():
            row = [Nx] + [values.get(key, "") for key in keys]
            writer.writerow(row)

def plot_graph(data, output_folder, filename, ylabel, title):
    for Nx, times in data.items():
        keys = sorted(times.keys())
        values = [times[key] for key in keys]
        plt.plot(keys, values, label=f"Nx={Nx}")

    plt.xlabel("Threads (T)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.savefig(os.path.join(output_folder, filename))
    plt.clf()

def generate_tables_and_graphs(summary_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    sequential_data, parallel_data = load_summary_data(summary_folder)

    # Save sequential data
    save_csv(sequential_data, output_folder, "sequential_results.csv", list(sequential_data.keys()))

    # Save parallel data and plot graphs
    for key, data in parallel_data.items():
        save_csv(data, output_folder, f"parallel_{key}.csv", sorted(data.keys()))
        plot_graph(data, output_folder, f"{key}_parallel.png", ylabel=f"{key} Time (s)", title=f"{key} Parallel Performance")

summary_folder = "summary"
output_folder = "output"
generate_tables_and_graphs(summary_folder, output_folder)
