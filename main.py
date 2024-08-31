import torch
import numpy as np

if __name__ == "__main__":
    action_000_last_frame = torch.load(
        "simplified_tensor/0071@000.pt", weights_only=False
    )[:, :, -1]
    action_001_first_frame = torch.load(
        "simplified_tensor/0071@001.pt", weights_only=False
    )[:, :, 0]
    transition_matrix = action_001_first_frame - action_000_last_frame

    translation_matrix = transition_matrix[0:3] / transition_matrix[0:3].max()
    rotation_matrix = transition_matrix[3:6] / transition_matrix[3:6].max()
    scale_matrix = transition_matrix[6:9] / transition_matrix[6:9].max()
    regular_motion_transition_cost = (
        torch.sum(translation_matrix)
        + torch.sum(rotation_matrix)
        + torch.sum(scale_matrix)
    )
    # regular_motion_transition_cost = torch.sum(transition_matrix)
    euclidean_distance_matrix = torch.tensor(np.full((1, 32), 0.5))
    lambda_8 = 0.5
    lambda_9 = torch.tensor(np.full((1, 32), 0.5))
    edge_transition_cost = lambda_8 * regular_motion_transition_cost + torch.sum(
        euclidean_distance_matrix * lambda_9
    )
    print(edge_transition_cost.item())
