import math
from functools import partial
from itertools import combinations, combinations_with_replacement

import jax.numpy as jnp
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.art3d as art3d
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from box_domain import union_box_constraint
from jax import jacfwd, jacrev, jit
from matplotlib.patches import Rectangle
from torch.autograd.functional import hessian, jacobian

__author__ = "Hao Wang"

torch.manual_seed(0)


def my_hessian(fun):
    return jit(jacfwd(jacrev(fun)))


def poly_features(t: torch.Tensor, degree: int = 2, interaction_only: bool = False) -> torch.Tensor:
    """
    Build polynomial features of input t up to specified degree, including cross-terms if interaction_only=False.
    t: shape (batch, n_features)
    returns: shape (batch, n_output_features)
    """
    # ensure tracking
    t = t.requires_grad_(True)
    batch, n_features = t.shape
    # we will collect monomials of each combination of feature-columns
    # first get list of column tensors
    cols = [t[:, i : i + 1] for i in range(n_features)]  # list of (batch,1) tensors
    prods = []
    # include lower‐order terms (degree=1)
    if degree >= 1:
        for c in cols:
            prods.append(c)

    # for higher degrees:
    for deg in range(2, degree + 1):
        if interaction_only:
            combs = combinations(cols, deg)
        else:
            combs = combinations_with_replacement(cols, deg)
        for comb in combs:
            # horizontally stack the deg columns → then take prod along last dim
            stacked = torch.cat(comb, dim=1)  # shape (batch, deg)
            monom = torch.prod(stacked, dim=1, keepdim=True)  # (batch,1)
            prods.append(monom)
    # now concatenate all features
    out = torch.cat(prods, dim=1)  # shape (batch, #monomials)
    return out


class PolynomialRegression(nn.Module):
    def __init__(self, input_dim: int, degree: int = 2) -> None:
        super().__init__()
        self.degree = degree
        self.hidden_dim = math.comb(input_dim + degree - 1, degree) + input_dim
        self.fc = nn.Linear(self.hidden_dim, 1)  # one scalar output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, input_dim)
        h = poly_features(x, self.degree)
        y = self.fc(h)  # shape (batch_size, 1)
        return y.squeeze(-1)  # shape (batch_size,)


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 16)
        self.fc5 = nn.Linear(16, 1)  # one scalar output
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, input_dim)
        h1 = self.act(self.fc1(x))
        h2 = self.act(self.fc2(h1))
        h3 = self.act(self.fc3(h2))
        h4 = self.act(self.fc4(h3))
        y = self.fc5(h4)  # shape (batch_size, 1)
        return y.squeeze(-1)  # shape (batch_size,)


def train_mlp(data: np.ndarray, epochs: int = 5000):
    c = 1e-5
    dim = data.shape[1] - 1
    X = torch.tensor(data[:, 0:-1], dtype=torch.float32)
    y_true = torch.tensor(data[:, -1], dtype=torch.float32)
    model = MLP(dim)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.5, momentum=0.5)
    lr_lambda = lambda step: max(0.0, 1.0 - float(step) / float(epochs))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        y_pred = model(X)
        # compute L1 regularization:
        if 11 < 2:
            l1_norm = torch.tensor(0.0, device=y_pred.device)
            for name, param in model.named_parameters():
                if "bias" in name:
                    continue
                l1_norm += torch.abs(param).sum()
            loss = criterion(y_pred, y_true) + c * l1_norm
        loss = criterion(y_pred, y_true)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if epoch % 50 == 0:
            print(f"Epoch {epoch}, loss = {loss.item():.5e}")
    model.eval()
    return model


def train_poly_regressor(data: np.ndarray, degree: int = 2):
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.preprocessing import PolynomialFeatures

    X_train, y_train = data[:, 0:-1], data[:, -1]
    # Create polynomial features
    poly = PolynomialFeatures(degree=degree, include_bias=True)
    # include_bias=True adds the constant term (x^0 = 1)
    # so the LinearRegression intercept corresponds accordingly
    X_train_poly = poly.fit_transform(X_train)
    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    y_pred = model.predict(X_train_poly)
    mse = mean_squared_error(y_train, y_pred)
    r2 = r2_score(y_train, y_pred)
    print(f"Degree = {degree} → MSE = {mse:.3f}, R² = {r2:.3f}")
    return model


class ParetoApproximator:
    def __init__(self, data: np.ndarray, box_centers: np.ndarray, radii: np.ndarray) -> None:
        self.dim: int = data.shape[1]
        # self.xl: np.ndarray = np.min(data, axis=0)
        # self.xu: np.ndarray = np.max(data, axis=0)
        self.xl: np.ndarray = np.array([-1] * self.dim)
        self.xu: np.ndarray = np.array([0] * self.dim)
        self._centers: np.ndarray = box_centers
        self._radius: np.ndarray = radii
        self.n_eq_constr: int = 1
        self.n_ieq_constr: int = self.dim * 2 + 1
        self._fit_pareto_front_approximator(data)
        self._ieq: callable = jit(partial(self.__class__._ieq_constraint, self))
        self._ieq_jacobian: callable = jit(jacrev(self._ieq))
        self._ieq_hessian: callable = my_hessian(self._ieq)
        # self._ieq_jacobian: callable = jax.jacobian(union_box_constraint, argnums=0)
        # self._ieq_hessian: callable = jax.hessian(union_box_constraint, argnums=0)

    def _fit_pareto_front_approximator(self, data: np.ndarray) -> None:
        """interpolate the Pareto front

        Args:
            Y (np.ndarray): finite approximation to the Pareto front
            epochs (int): number of epochs to train the MLP
        """
        self._model = train_mlp(data)
        self._pareto_approximator = lambda x: self._model(x.unsqueeze(0))[0]

    def objective(self, x: np.ndarray) -> np.ndarray:
        return x

    def jacobian(self, _: np.ndarray) -> np.ndarray:
        return np.diag(np.ones(self.dim))

    def hessian(self, _: np.ndarray) -> np.ndarray:
        return np.zeros((self.dim, self.dim, self.dim))

    def eq_constraint(self, x: np.ndarray) -> np.ndarray:
        x_ = torch.from_numpy(x[0:-1]).float()
        return self._pareto_approximator(x_).detach().cpu().numpy() - x[-1]

    def eq_jacobian(self, x: np.ndarray) -> np.ndarray:
        x_ = torch.from_numpy(x[0:-1]).float().requires_grad_(True)
        return np.r_[jacobian(self._pareto_approximator, x_).detach().cpu().numpy(), -1]

    def eq_hessian(self, x: np.ndarray) -> np.ndarray:
        x_ = torch.from_numpy(x[0:-1]).float().requires_grad_(True)
        H = np.zeros((self.dim, self.dim))
        H[0:-1, 0:-1] = hessian(self._pareto_approximator, x_).detach().cpu().numpy()
        return H

    def ieq_jacobian(self, x: np.ndarray) -> np.ndarray:
        return np.array(self._ieq_jacobian(x))

    def ieq_hessian(self, x: np.ndarray) -> np.ndarray:
        return np.array(self._ieq_hessian(x))

    def _ieq_constraint(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.array(jnp.r_[jnp.sum(x[0:2] ** 2) - 1, self.xl - x, x - self.xu])
        # return np.array(union_box_constraint(x[0:-1], self._centers, self._radius))

    def ieq_constraint(self, x: np.ndarray) -> np.ndarray:
        return np.array(self._ieq_constraint(x))

    # def ieq_jacobian(self, x: np.ndarray) -> np.ndarray:
    #     J = np.array(self._ieq_jacobian(x[0:-1], self._centers, self._radius))
    #     return np.c_[J, np.zeros((self.n_ieq_constr, 1))]

    # def ieq_hessian(self, x: np.ndarray) -> np.ndarray:
    #     H = np.zeros((self.n_ieq_constr, self.dim, self.dim))
    #     H_ = self._ieq_hessian(x[0:-1], self._centers, self._radius)
    #     H[:, 0 : self.n_ieq_constr, 0 : self.n_ieq_constr] = H_
    #     return H

    def sample_from_domain(n: int = 100) -> np.ndarray:
        pass

    def plot_domain(self, ax=None, edgecolor="blue", facecolor="r", alpha=0.5, **rect_kwargs):
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
            if ax.name == "3d":  # add the patches to the x-y plane
                art3d.pathpatch_2d_to_3d(rect, z=0, zdir="z")
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
