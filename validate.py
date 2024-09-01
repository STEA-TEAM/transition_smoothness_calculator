import torch

if __name__ == "__main__":
    abs_cost_metrics = torch.load("cost_metrics.pt", weights_only=False).abs()
    min_row_index, min_column_index = torch.unravel_index(abs_cost_metrics.argmin(), abs_cost_metrics.shape)
    pose_index_list = [min_row_index.item(), min_column_index.item()]
    print(f"#{pose_index_list[0]} -> #{pose_index_list[1]} costs: {abs_cost_metrics[min_row_index][min_column_index]}")
    while len(pose_index_list) < 60:
        row = abs_cost_metrics[min_column_index]
        min_column_index = row.argmin().item()
        pose_index_list.append(min_column_index)
        print(f"#{pose_index_list[-2]} -> #{pose_index_list[-1]} costs: {row[min_column_index]}")