import os
import re
import csv

def parse_output_file(filepath):
    data = {
        "Generate": [],
        "Fill": [],
        "SPMV": [],
        "AXPY": [],
        "DOT": [],
        "Iterations": [],
        "Residual": [],
        "NNZ": []
    }
    with open(filepath, 'r') as file:
        for line in file:
            if "Generate:" in line:
                data["Generate"].append(float(line.split(":")[1]))
            elif "Fill:" in line:
                data["Fill"].append(float(line.split(":")[1]))
            elif "\tSPMV:" in line:
                data["SPMV"].append(float(line.split(":")[1]))
            elif "\tAXPY:" in line:
                data["AXPY"].append(float(line.split(":")[1]))
            elif "\tDOT:" in line:
                data["DOT"].append(float(line.split(":")[1]))
            elif "Iterations:" in line:
                data["Iterations"].append(int(line.split(":")[1]))
            elif "Residual:" in line:
                data["Residual"].append(float(line.split(":")[1]))
            elif "NNZ:" in line:
                data["NNZ"].append(int(line.split(":")[1]))
    return data

def calculate_averages(data):
    averages = {}
    for key, values in data.items():
        averages[key] = sum(values) / len(values) if values else 0
    return averages

def write_summary_file(output_folder, Nx, T, averages):
    output_filepath = os.path.join(output_folder, f"{Nx}_{T}.log")
    with open(output_filepath, 'w') as file:
        file.write(f"Nx_T:{Nx}_{T}\n\n")
        file.write(f"Generate:{averages['Generate']:.6f}\n")
        file.write(f"\tNNZ:{int(averages['NNZ']):d}\n")
        file.write(f"Fill:{averages['Fill']:.6f}\n")
        file.write(f"Solve:{sum([averages['SPMV'], averages['AXPY'], averages['DOT']]):.6f}\n")
        file.write(f"\tSPMV:{averages['SPMV']:.6f}\n")
        file.write(f"\tAXPY:{averages['AXPY']:.6f}\n")
        file.write(f"\tDOT:{averages['DOT']:.6f}\n\n")
        file.write("Solution completed.\n")
        file.write(f"\tIterations:{averages['Iterations']:.0f}\n")
        file.write(f"\tResidual:{averages['Residual']:.6e}\n")

def write_nnz_file(output_folder, nnz_data):
    output_filepath = os.path.join(output_folder, "nnz_summary.csv")
    with open(output_filepath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Nx", "T", "NNZ"])
        for (Nx, T), nnz in sorted(nnz_data.items()):
            writer.writerow([Nx, T, nnz])

def process_results(result_base_folder, summary_output_folder):
    os.makedirs(summary_output_folder, exist_ok=True)

    nnz_data = {}

    for root, dirs, files in os.walk(result_base_folder):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            Nx, T = map(int, re.match(r"(\d+)_(\d+)", dir_name).groups())

            aggregated_data = {
                "Generate": [],
                "Fill": [],
                "SPMV": [],
                "AXPY": [],
                "DOT": [],
                "Iterations": [],
                "Residual": [],
                "NNZ": []
            }

            for file_name in os.listdir(dir_path):
                if file_name.endswith(".out"):
                    file_path = os.path.join(dir_path, file_name)
                    file_data = parse_output_file(file_path)
                    for key in aggregated_data:
                        aggregated_data[key].extend(file_data[key])

            averages = calculate_averages(aggregated_data)
            write_summary_file(summary_output_folder, Nx, T, averages)
            nnz_data[(Nx, T)] = averages['NNZ']

    write_nnz_file(summary_output_folder, nnz_data)

result_base_folder = "res"
summary_output_folder = "summary"
process_results(result_base_folder, summary_output_folder)
