from glob import glob
from typing import AnyStr, List

import torch
from multiprocessing import cpu_count, Manager, Pool, Value
import numpy as np


def calculate_edge_transition_cost(
    _from_file_paths: List[AnyStr],
    _to_file_paths: List[AnyStr],
    _finished_file_count: Value,
):
    _results = torch.tensor(np.full((len(_from_file_paths), len(_to_file_paths)), 0.0))
    for from_file_index, from_file_path in enumerate(_from_file_paths):
        for to_file_index, to_file_path in enumerate(_to_file_paths):
            from_last_frame = torch.load(from_file_path, weights_only=False)[:, :, -1]
            to_first_frame = torch.load(to_file_path, weights_only=False)[:, :, 0]
            transition_matrix = to_first_frame - from_last_frame

            regular_motion_transition_cost = torch.sum(transition_matrix)
            euclidean_distance_matrix = torch.tensor(np.full((1, 32), 0.5))
            lambda_8 = 0.5
            lambda_9 = torch.tensor(np.full((1, 32), 0.5))
            edge_transition_cost = (
                lambda_8 * regular_motion_transition_cost
                + torch.sum(euclidean_distance_matrix * lambda_9)
            )
            _results[from_file_index, to_file_index] = edge_transition_cost
        _finished_file_count.value += 1
        print(
            f"[{_finished_file_count.value}/{len(_to_file_paths)}] Finished processing {from_file_path}"
        )
    return _results


if __name__ == "__main__":
    input_folder = "simplified_tensor"
    all_models_files = glob(f"{input_folder}/*.pt")
    print(f"Found {len(all_models_files)} models")
    processes_count = cpu_count()
    print(f"Using {processes_count} processes")

    with Manager() as manager:
        finished_file_count = manager.Value("i", 0)
        with Pool(processes=processes_count) as pool:
            job_list = []
            step_count = len(all_models_files) // processes_count
            for index in range(processes_count):
                start_index = index * step_count
                end_index = (index + 1) * step_count
                job_list.append(
                    pool.apply_async(
                        calculate_edge_transition_cost,
                        args=(
                            all_models_files[start_index:end_index],
                            all_models_files,
                            finished_file_count,
                        ),
                    )
                )
            pool.close()
            pool.join()

            cost_list = [job.get() for job in job_list]

            cost_metrics = torch.vstack(cost_list)
            torch.save(cost_metrics, "saved_cost_metrics.pt")
