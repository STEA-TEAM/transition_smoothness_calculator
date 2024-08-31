from glob import glob
from os import cpu_count, path

import torch
import multiprocessing as mp
import numpy as np


def calculate_edge_transition_cost(
    _from_file_paths,
    _to_file_paths,
    _finished_count,
    _total_file_count,
    _cost_metrics_queue,
):
    for from_file_index, from_file_path in enumerate(_from_file_paths):
        for to_file_index, to_file_path in enumerate(_to_file_paths):
            from_last_frame = torch.load(from_file_path, weights_only=False)[:, :, -1]
            to_first_frame = torch.load(to_file_path, weights_only=False)[:, :, 0]
            transition_matrix = to_first_frame - from_last_frame

            # translation_matrix = transition_matrix[0:3] / transition_matrix[0:3].max()
            # rotation_matrix = transition_matrix[3:6] / transition_matrix[3:6].max()
            # scale_matrix = transition_matrix[6:9] / transition_matrix[6:9].max()
            # regular_motion_transition_cost = (
            #     torch.sum(translation_matrix)
            #     + torch.sum(rotation_matrix)
            #     + torch.sum(scale_matrix)
            # )
            regular_motion_transition_cost = torch.sum(transition_matrix)
            euclidean_distance_matrix = torch.tensor(np.full((1, 32), 0.5))
            lambda_8 = 0.5
            lambda_9 = torch.tensor(np.full((1, 32), 0.5))
            edge_transition_cost = (
                lambda_8 * regular_motion_transition_cost
                + torch.sum(euclidean_distance_matrix * lambda_9)
            )
            # if math.isnan(abs(edge_transition_cost.item())):
            #     print(f"{path.basename(from_file_path)} To {path.basename(to_file_path)}: NaN")
            # print(f"{path.basename(from_file_path)} To {path.basename(to_file_path)}: {abs(edge_transition_cost.item())}")
            _cost_metrics_queue.put(
                {
                    "from_file_name": from_file_path,
                    "to_file_name": to_file_path,
                    "cost": edge_transition_cost,
                }
            )

        _finished_count.value += 1
        print(f"[{_finished_count.value + 1}/{_total_file_count}] file: {from_file_path}")


if __name__ == "__main__":
    input_folder = "simplified_tensor"
    all_models_files = glob(f"{input_folder}/*.pt")
    print(f"Found {len(all_models_files)} models")
    processes_count = cpu_count()
    print(f"Using {processes_count} processes")

    ctx = mp.get_context("spawn")
    finished_count = ctx.Value("i", 0)
    cost_metrics_queue = ctx.Queue()

    processes = []
    for i in range(processes_count):
        start = i * len(all_models_files) // processes_count
        end = (i + 1) * len(all_models_files) // processes_count
        process = ctx.Process(
            target=calculate_edge_transition_cost,
            args=(
                all_models_files[start:end],
                all_models_files,
                finished_count,
                len(all_models_files),
                cost_metrics_queue,
            ),
        )
        processes.append(process)
        process.start()

    for process in processes:
        process.join()

    cost_metrics = torch.tensor(
        np.full((len(all_models_files), len(all_models_files)), 0.0)
    )

    print(f"Total entries: {cost_metrics_queue.qsize()}")

    for index in range(cost_metrics_queue.qsize()):
        if index % 1000 == 0:
            print(f"Processing {index + 1} entries")
        cost_metric = cost_metrics_queue.get()
        from_file_index = all_models_files.index(cost_metric['from_file_name'])
        to_file_index = all_models_files.index(cost_metric['to_file_name'])
        cost_metrics[from_file_index, to_file_index] = cost_metric["cost"]

    torch.save(cost_metrics, "saved_cost_metrics.pt")
