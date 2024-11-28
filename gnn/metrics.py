import torch
import torch.nn.functional as F


import torch


import torch

import torch

import torch

import torch

import torch


class CustomR2Score:
    def __init__(self, dtype=torch.float32):
        """
        Initialize the R2Score class. The dtype parameter allows setting the data type for computations.
        Default is torch.float32.
        """
        self.dtype = dtype
        self.reset()

    def reset(self):
        """
        Reset the internal state of the R2Score class (i.e., sum of squared residuals and total sum of squares).
        """
        self.squared_residual_sum = 0.0
        self.total_sum_squares = 0.0
        self.count = 0
        self.y_true_values = (
            []
        )  # Collect all true values across batches for mean computation

    def update(self, y_true, y_pred):
        """
        Update the metric with a new batch of ground truth (y_true) and predictions (y_pred).
        """
        y_true = y_true.to(self.dtype)
        y_pred = y_pred.to(self.dtype)

        # Collect the true values to compute the mean in `compute`
        self.y_true_values.append(y_true)

        # Compute the residuals (error) and squared residuals
        residual = y_true - y_pred
        squared_residual = residual**2

        # Accumulate the squared residuals
        self.squared_residual_sum += squared_residual.sum()

        # Accumulate the total sum of squares (but use the mean later in compute)
        self.count += y_true.numel()

    def compute(self):
        """
        Compute the R² score based on the accumulated values.
        """
        if self.count == 0:
            return torch.tensor(0.0, dtype=self.dtype)

        # Compute the global mean of y_true across all updates
        all_y_true = torch.cat(self.y_true_values, dim=0)
        mean_y_true = all_y_true.mean()

        # Compute the total sum of squares based on the global mean of y_true
        total_sum_squares = ((all_y_true - mean_y_true) ** 2).sum()

        # Compute the R² score
        r2 = 1 - (self.squared_residual_sum / total_sum_squares)
        return r2

    def __call__(self, y_true, y_pred):
        """
        Update the metric and compute the R² score.
        """
        self.update(y_true, y_pred)
        return self.compute()


if __name__ == "__main__":
    metric = CustomR2Score()
    # metric.update(torch.tensor([1, 1, 2, 3]), torch.tensor([2, 1, 1, 2]))
    metric.update(
        torch.tensor([1, 1, 2, 2, 1, 2, 3]), torch.tensor([2, 1, 1, 2, 1, 1, 2])
    )
    # metric.update(torch.tensor([2, 1, 2]), torch.tensor([2, 1, 1]))
    result = metric.compute()
    print(result)
