import jax
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from box_domain import union_box_constraint
from matplotlib.patches import Rectangle
from torch.autograd.functional import hessian, jacobian

torch.manual_seed(0)


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)  # one scalar output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, input_dim)
        h = self.act(self.fc1(x))
        y = self.fc2(h)  # shape (batch_size, 1)
        return y.squeeze(-1)  # shape (batch_size,)


class ParetoApproximator:
    def __init__(self, Y: np.ndarray, box_centers: np.ndarray, radii: np.ndarray) -> None:
        dim = Y.shape[1]
        self.n_obj = dim
        self.n_var = dim - 1
        self.xl = np.min(Y[:, 0:-1], axis=0)
        self.xu = np.max(Y[:, 0:-1], axis=0)
        self._centers = box_centers
        self._radius = radii
        self.n_eq_constr = 0
        self.n_ieq_constr = self.n_var
        self._fit_pareto_front(Y)
        self._objective = lambda x: self._model(x.unsqueeze(0))[0]
        self._ieq_jacobian = jax.jacobian(union_box_constraint, argnums=0)
        self._ieq_hessian = jax.hessian(union_box_constraint, argnums=0)

    def _fit_pareto_front(self, Y: np.ndarray) -> None:
        """interpolate the Pareto front with NN

        Args:
            Y (np.ndarray): finite approximation to the Pareto front
        """
        X, y_true = torch.tensor(Y[:, 0:-1], dtype=torch.float32), torch.tensor(Y[:, -1], dtype=torch.float32)
        self._model = MLP(self.n_var)
        criterion = nn.MSELoss()
        optimizer = optim.SGD(self._model.parameters(), lr=0.01)
        for epoch in range(500):
            self._model.train()
            optimizer.zero_grad()
            y_pred = self._model(X)
            loss = criterion(y_pred, y_true)
            loss.backward()
            optimizer.step()
            if epoch % 50 == 0:
                print(f"Epoch {epoch}, loss = {loss.item():.4f}")

    def objective(self, x: np.ndarray) -> float:
        x_ = torch.from_numpy(x).float()
        return self._objective(x_).detach().cpu().numpy()

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        x_ = torch.from_numpy(x).float().requires_grad_(True)
        return jacobian(self._objective, x_).detach().cpu().numpy()

    def hessian(self, x: np.ndarray) -> np.ndarray:
        x_ = torch.from_numpy(x).float().requires_grad_(True)
        return hessian(self._objective, x_).detach().cpu().numpy()

    def ieq_constraint(self, x: np.ndarray) -> np.ndarray:
        return np.array(union_box_constraint(x, self._centers, self._radius))

    def ieq_jacobian(self, x: np.ndarray) -> np.ndarray:
        return np.array(self._ieq_jacobian(x, self._centers, self._radius))

    def ieq_hessian(self, x: np.ndarray) -> np.ndarray:
        # TODO: get rid of it since the constraint is linear, the Hessian is always zero
        return np.array(self._ieq_hessian(x, self._centers, self._radius))

    def plot_domain(self, ax=None, edgecolor="blue", facecolor="r", alpha=0.5, **rect_kwargs) -> None:
        if ax is None:
            fig, ax = plt.subplots()
        for (cx, cy), (rx, ry) in zip(self._centers, self._radius):
            # Compute bottom‐left corner
            x0 = cx - rx
            y0 = cy - ry
            width = 2 * rx
            height = 2 * ry
            rect = Rectangle(
                (x0, y0), width, height, edgecolor=edgecolor, facecolor=facecolor, alpha=alpha, **rect_kwargs
            )
            ax.add_patch(rect)
        ax.set_aspect("equal", "box")
        # Adjust limits to show all boxes nicely
        all_x = self._centers[:, 0]
        all_y = self._centers[:, 1]
        all_rx = self._radius[:, 0]
        all_ry = self._radius[:, 1]
        xmin = (all_x - all_rx).min()
        xmax = (all_x + all_rx).max()
        ymin = (all_y - all_ry).min()
        ymax = (all_y + all_ry).max()
        padding_x = (xmax - xmin) * 0.1
        padding_y = (ymax - ymin) * 0.1
        ax.set_xlim(xmin - padding_x, xmax + padding_x)
        ax.set_ylim(ymin - padding_y, ymax + padding_y)
        ax.set_xlabel("$f_1$")
        ax.set_ylabel("$f_2$")
        plt.tight_layout()
        return ax
