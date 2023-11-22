import csv
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from absl import logging
from ml_collections import ConfigDict

# import utils


def train(
    device: torch.device,
    training_data: Iterable,
    x_test: np.array,
    q_test: np.array,
    model: torch.nn.Module,
    config: ConfigDict,
    csv_path: Path,
    model_dir: Path,
    lr_dir: Path,
):
    int_weights = torch.tensor(config.int_weights, requires_grad=True).to(device)
    ratio = torch.tensor(config.ratio, requires_grad=True, dtype=torch.float32).to(
        device
    )
    if config.optimizer == "Adam":
        optimizer = torch.optim.Adam(params=model.parameters(), lr=config.lr)
        if config.decay == "Exponential":
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=config.decay_rate
            )
        else:
            raise ValueError("other decay have not been implemented")
    else:
        raise ValueError("other optimizer have not been implemented")
    x_test = torch.tensor(x_test, dtype=torch.float32).to(device)
    q_test = torch.tensor(q_test, dtype=torch.float32).to(device)
    with open(csv_path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "epoch",
                "total",
                "res_loss",
                "flux_loss",
                "u_ic",
                "F_ic",
                "u_bc",
                "F_bc",
                "MAE",
                "L2RE",
            ]
        )
    logging.info("Created the csv file")
    logging.info("----------Start training-----------")
    for epoch in range(config.epochs):
        sdpath = model_dir / "model_{:02d}".format(epoch)
        lrpath = lr_dir / "model_{:02d}".format(epoch)
        # ----------- test step-----------#
        with torch.no_grad():
            q_pred = model(x_test)
        mae = torch.nn.L1Loss()(q_test, q_pred)
        # L2RR : L2 relative error
        L2RE = torch.nn.MSELoss()(q_test, q_pred) / torch.nn.MSELoss()(
            q_test, torch.zeros_like(q_test)
        )
        # --------- train step------------#
        data = next(training_data)
        x_int, x_ic, x_bc = data["interior"], data["initial"], data["boundary"]
        x_int = torch.tensor(x_int, requires_grad=True, dtype=torch.float32).to(device)
        x_ic = torch.tensor(x_ic, requires_grad=True, dtype=torch.float32).to(device)
        x_bc = torch.tensor(x_bc, requires_grad=True, dtype=torch.float32).to(device)
        res_loss, flux_loss = model.interior_loss(x_int, int_weights)
        u_ic_loss, F_ic_loss = model.init_loss(x_ic)
        u_bc_loss, F_bc_loss = model.bc_loss(x_bc)
        loss = (
            ratio[0] * res_loss
            + ratio[1] * flux_loss
            + ratio[2] * u_ic_loss
            + ratio[3] * u_bc_loss
            # + ratio[2] * F_ic_loss
            # + ratio[3] * F_bc_loss
        )
        learning_rate = optimizer.state_dict()["param_groups"][0]["lr"]
        if epoch % 1000 == 0:
            logging.info(
                "epoch : {}  |  res_loss :{}  | flux_loss :{} | u_ic : {}| F_ic : {} | lr : {} | MAE : {} |L2RE:{}".format(
                    epoch,
                    res_loss,
                    flux_loss,
                    u_ic_loss,
                    F_ic_loss,
                    learning_rate,
                    mae,
                    L2RE,
                )
            )
        if epoch % 1000 == 0:
            with open(csv_path, "a+") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        epoch,
                        loss.item(),
                        res_loss.item(),
                        flux_loss.item(),
                        u_ic_loss.item(),
                        F_ic_loss.item(),
                        u_bc_loss.item(),
                        F_bc_loss.item(),
                        mae.item(),
                        L2RE.item(),
                    ]
                )
            torch.save(model.state_dict(), sdpath)
            torch.save(optimizer.state_dict(), lrpath)
        if epoch == config.epochs - 1:
            continue
        else:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if epoch % 1000 == 0:
                scheduler.step()


def alternate(
    device: torch.device,
    training_data: Iterable,
    x_test: np.array,
    q_test: np.array,
    model: torch.nn.Module,
    config: ConfigDict,
    csv_path: Path,
    model_dir: Path,
    lr_dir: Path,
    step: int,
):
    if config.optimizer == "Adam":
        optimizer_u = torch.optim.Adam(params=model.u.parameters(), lr=config.lr)
        optimizer_F = torch.optim.Adam(params=model.Flux.parameters(), lr=config.lr)
    else:
        raise ValueError("other optimizer have not been implemented")
    # scheduler_u = torch.optim.lr_scheduler.ExponentialLR(
    #     optimizer_u, gamma=config.decay_rate
    # )
    # scheduler_F = torch.optim.lr_scheduler.ExponentialLR(
    #     optimizer_F, gamma=config.decay_rate
    # )
    if config.metric == "MAE":
        metric = torch.nn.L1Loss()
    x_test = torch.tensor(x_test, dtype=torch.float32).to(device)
    q_test = torch.tensor(q_test, dtype=torch.float32).to(device)
    with open(csv_path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "epoch",
                "total",
                "res_loss",
                "flux_loss",
                "u_ic",
                "F_ic",
                "u_bc",
                "F_bc",
                "MAE",
            ]
        )
    logging.info("Created the csv file")
    logging.info("----------Start training-----------")
    for epoch in range(config.epochs):
        sdpath = model_dir / "model_{:02d}".format(epoch)
        # ----------- test step-----------#
        with torch.no_grad():
            q_pred = model(x_test)
        mae = metric(q_test, q_pred)

        # --------- train step------------#
        data = next(training_data)
        x_int, x_ic, x_bc = data["interior"], data["initial"], data["boundary"]
        x_int = torch.tensor(x_int, requires_grad=True, dtype=torch.float32).to(device)
        x_ic = torch.tensor(x_ic, requires_grad=True, dtype=torch.float32).to(device)
        x_bc = torch.tensor(x_bc, requires_grad=True, dtype=torch.float32).to(device)
        res_loss, flux_loss = model.interior_loss(x_int)
        u_ic_loss, F_ic_loss = model.init_loss(x_ic)
        u_bc_loss, F_bc_loss = model.bc_loss(x_bc)
        loss = res_loss + flux_loss + u_ic_loss + F_ic_loss + u_bc_loss + F_bc_loss
        if epoch % 100 == 0:
            logging.info(
                "epoch : {}  |  res_loss :{}  | flux_loss :{} | u_ic : {}| F_ic : {} | MAE : {}".format(
                    epoch, res_loss, flux_loss, u_ic_loss, F_ic_loss, mae
                )
            )
            with open(csv_path, "a+") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        epoch,
                        loss.item(),
                        res_loss.item(),
                        flux_loss.item(),
                        u_ic_loss.item(),
                        F_ic_loss.item(),
                        u_bc_loss.item(),
                        F_bc_loss.item(),
                        mae.item(),
                    ]
                )
            torch.save(model.state_dict(), sdpath)
        if epoch == config.epochs - 1:
            continue
        else:
            if epoch % step == 0:
                optimizer_u.zero_grad()
                loss.backward()
                optimizer_u.step()
                # scheduler_u.step()

            else:
                optimizer_F.zero_grad()
                loss.backward()
                optimizer_F.step()
                # scheduler_F.step()


def gradnorm(
    device: torch.device,
    training_data: Iterable,
    x_test: np.array,
    q_test: np.array,
    model: torch.nn.Module,
    config: ConfigDict,
    csv_path: Path,
    model_dir: Path,
    lr_dir: Path,
):
    alpha = 0.12
    if config.optimizer == "Adam":
        optimizer_net = torch.optim.Adam(params=model.parameters(), lr=config.lr)
        if config.decay == "Exponential":
            scheduler_net = torch.optim.lr_scheduler.ExponentialLR(
                optimizer_net, gamma=config.decay_rate
            )
        else:
            raise ValueError("other decay have not been implemented")
    else:
        raise ValueError("other optimizer have not been implemented")
    if config.metric == "MAE":
        metric = torch.nn.L1Loss()
    x_test = torch.tensor(x_test, dtype=torch.float32).to(device)
    q_test = torch.tensor(q_test, dtype=torch.float32).to(device)
    with open(csv_path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "epoch",
                "total",
                "res_loss",
                "flux_loss",
                "u_ic",
                "F_ic",
                "u_bc",
                "F_bc",
                "MAE",
            ]
        )
    logging.info("Created the csv file")
    logging.info("----------Start training-----------")
    for epoch in range(config.epochs):
        sdpath = model_dir / "model_{:02d}".format(epoch)
        lrpath = lr_dir / "model_{:02d}".format(epoch)
        # ----------- test step-----------#
        with torch.no_grad():
            q_pred = model(x_test)
        mae = metric(q_test, q_pred)
        # --------- train step------------#
        data = next(training_data)
        x_int, x_ic, x_bc = data["interior"], data["initial"], data["boundary"]
        x_int = torch.tensor(x_int, requires_grad=True, dtype=torch.float32).to(device)
        x_ic = torch.tensor(x_ic, requires_grad=True, dtype=torch.float32).to(device)
        x_bc = torch.tensor(x_bc, requires_grad=True, dtype=torch.float32).to(device)
        losses = model.losses(x_int, x_ic, x_bc)
        if epoch == 0:
            weights = torch.ones_like(losses)
            weights = torch.nn.parameter.Parameter(weights)
            Total = weights.sum().detach()
            optimizer_w = torch.optim.Adam([weights], lr=1e-4)
            L0 = losses.detach()
        weighted_loss = weights * losses
        learning_rate = optimizer_net.state_dict()["param_groups"][0]["lr"]
        if epoch % 100 == 0:
            logging.info(
                "epoch : {}  |  res_loss :{}  | flux_loss :{} | u_ic : {}| F_ic : {} | lr : {} | MAE : {} ".format(
                    epoch,
                    weighted_loss[:, 0:1].item(),
                    weighted_loss[:, 1:2].item(),
                    weighted_loss[:, 2:3].item(),
                    weighted_loss[:, 3:4].item(),
                    learning_rate,
                    mae,
                )
            )
            with open(csv_path, "a+") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        epoch,
                        weighted_loss.sum(dim=1).item(),
                        weighted_loss[:, 0:1].item(),
                        weighted_loss[:, 1:2].item(),
                        weighted_loss[:, 2:3].item(),
                        weighted_loss[:, 3:4].item(),
                        weighted_loss[:, 4:5].item(),
                        weighted_loss[:, 5:6].item(),
                        mae.item(),
                    ]
                )
            torch.save(model.state_dict(), sdpath)
            torch.save(optimizer_net.state_dict(), lrpath)
        if epoch == config.epochs - 1:
            continue
        else:
            optimizer_net.zero_grad()
            weighted_loss.sum(dim=1).backward(retain_graph=True)
            # compute the L2 norm of the gradients for each task
            gradient_w = []
            for i in range(len(losses)):
                dl = torch.autograd.grad(
                    weighted_loss[0, i],
                    list(model.u.layers[-1].parameters())
                    + list(model.Flux.layers[-1].parameters()),
                    retain_graph=True,
                    create_graph=True,
                )[0]
                gradient_w.append(torch.norm(dl))
            gradient_w = torch.stack(gradient_w)
            losses_ratio = losses.detach() / L0
            rt = losses_ratio / losses_ratio.mean()
            gradient_w_avg = gradient_w.mean().detach()
            constant = (gradient_w_avg * rt**alpha).detach()
            gradnorm_loss = torch.abs(gradient_w - constant).sum()
            optimizer_w.zero_grad()
            gradnorm_loss.backward()
            optimizer_net.step()
            optimizer_w.step()
            weights = (weights / weights.sum() * Total).detach()
            weights = torch.nn.Parameter(weights)
            optimizer_w = torch.optim.Adam([weights], lr=1e-4)
            if epoch % 100 == 0:
                scheduler_net.step()


def weighted_flux(
    device: torch.device,
    training_data: Iterable,
    x_test: np.array,
    q_test: np.array,
    model: torch.nn.Module,
    config: ConfigDict,
    csv_path: Path,
    model_dir: Path,
    lr_dir: Path,
):
    ratio = torch.tensor(config.ratio, requires_grad=True, dtype=torch.float32).to(
        device
    )
    if config.optimizer == "Adam":
        optimizer = torch.optim.Adam(params=model.parameters(), lr=config.lr)
        if config.decay == "Exponential":
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=config.decay_rate
            )
        else:
            raise ValueError("other decay have not been implemented")
    else:
        raise ValueError("other optimizer have not been implemented")
    x_test = torch.tensor(x_test, dtype=torch.float32).to(device)
    q_test = torch.tensor(q_test, dtype=torch.float32).to(device)
    with open(csv_path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "epoch",
                "total",
                "res_loss",
                "flux_loss",
                "u_ic",
                "F_ic",
                "u_bc",
                "F_bc",
                "MAE",
                "L2RE",
            ]
        )
    logging.info("Created the csv file")
    logging.info("----------Start training-----------")
    for epoch in range(config.epochs):
        sdpath = model_dir / "model_{:02d}".format(epoch)
        lrpath = lr_dir / "model_{:02d}".format(epoch)
        # ----------- test step-----------#
        with torch.no_grad():
            q_pred = model(x_test)
        mae = torch.nn.L1Loss()(q_test, q_pred)
        # L2RR : L2 relative error
        L2RE = torch.nn.MSELoss()(q_test, q_pred) / torch.nn.MSELoss()(
            q_test, torch.zeros_like(q_test)
        )
        # --------- train step------------#
        data = next(training_data)
        x_int, x_ic, x_bc = data["interior"], data["initial"], data["boundary"]
        x_int = torch.tensor(x_int, requires_grad=True, dtype=torch.float32).to(device)
        x_ic = torch.tensor(x_ic, requires_grad=True, dtype=torch.float32).to(device)
        x_bc = torch.tensor(x_bc, requires_grad=True, dtype=torch.float32).to(device)
        res_loss, flux_loss = model.interior_loss(x_int)
        u_ic_loss, F_ic_loss = model.init_loss(x_ic)
        u_bc_loss, F_bc_loss = model.bc_loss(x_bc)
        loss = (
            ratio[0] * res_loss
            + ratio[1] * flux_loss
            + ratio[2] * u_ic_loss
            + ratio[3] * F_ic_loss
            + ratio[4] * u_bc_loss
            + ratio[5] * F_bc_loss
        )
        learning_rate = optimizer.state_dict()["param_groups"][0]["lr"]
        if epoch % 100 == 0:
            logging.info(
                "epoch : {}  |  res_loss :{}  | flux_loss :{} | u_ic : {}| F_ic : {} | lr : {} | MAE : {} |L2RE:{}".format(
                    epoch,
                    res_loss,
                    flux_loss,
                    u_ic_loss,
                    F_ic_loss,
                    learning_rate,
                    mae,
                    L2RE,
                )
            )
        if epoch % 1000 == 0:
            with open(csv_path, "a+") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        epoch,
                        loss.item(),
                        res_loss.item(),
                        flux_loss.item(),
                        u_ic_loss.item(),
                        F_ic_loss.item(),
                        u_bc_loss.item(),
                        F_bc_loss.item(),
                        mae.item(),
                        L2RE.item(),
                    ]
                )
            torch.save(model.state_dict(), sdpath)
            torch.save(optimizer.state_dict(), lrpath)
        if epoch == config.epochs - 1:
            continue
        else:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if epoch % 1000 == 0:
                scheduler.step()


def decouple(
    device: torch.device,
    training_data: Iterable,
    x_test: np.array,
    q_test: np.array,
    model: torch.nn.Module,
    config: ConfigDict,
    csv_path: Path,
    model_dir: Path,
    lr_dir: Path,
):
    ratio = torch.tensor(config.ratio, requires_grad=True, dtype=torch.float32).to(
        device
    )
    if config.optimizer == "Adam":
        optimizer = torch.optim.Adam(params=model.parameters(), lr=config.lr)
        if config.decay == "Exponential":
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=config.decay_rate
            )
        else:
            raise ValueError("other decay have not been implemented")
    else:
        raise ValueError("other optimizer have not been implemented")
    x_test = torch.tensor(x_test, dtype=torch.float32).to(device)
    q_test = torch.tensor(q_test, dtype=torch.float32).to(device)
    with open(csv_path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "epoch",
                "total",
                "res_loss",
                "flux_loss",
                "u_ic",
                "F_ic",
                "u_bc",
                "F_bc",
                "MAE",
                "L2RE",
            ]
        )
    logging.info("Created the csv file")
    logging.info("----------Start training-----------")
    for epoch in range(config.epochs):
        sdpath = model_dir / "model_{:02d}".format(epoch)
        lrpath = lr_dir / "model_{:02d}".format(epoch)
        # ----------- test step-----------#
        with torch.no_grad():
            q_pred = model(x_test)
        mae = torch.nn.L1Loss()(q_test, q_pred)
        # L2RR : L2 relative error
        L2RE = torch.nn.MSELoss()(q_test, q_pred) / torch.nn.MSELoss()(
            q_test, torch.zeros_like(q_test)
        )
        # --------- train step------------#
        data = next(training_data)
        x_int, x_ic, x_bc = data["interior"], data["initial"], data["boundary"]
        x_int = torch.tensor(x_int, requires_grad=True, dtype=torch.float32).to(device)
        x_ic = torch.tensor(x_ic, requires_grad=True, dtype=torch.float32).to(device)
        x_bc = torch.tensor(x_bc, requires_grad=True, dtype=torch.float32).to(device)
        res_loss, flux_loss = model.interior_loss(x_int)
        u_ic_loss, F_ic_loss = model.init_loss(x_ic)
        u_bc_loss, F_bc_loss = model.bc_loss(x_bc)
        loss = (
            ratio[0] * res_loss
            + ratio[1] * flux_loss
            + ratio[2] * u_ic_loss
            # + ratio[3] * F_ic_loss
            + ratio[4] * u_bc_loss
            # + ratio[5] * F_bc_loss
        )
        learning_rate = optimizer.state_dict()["param_groups"][0]["lr"]
        if epoch % 100 == 0:
            logging.info(
                "epoch : {}  |  res_loss :{}  | flux_loss :{} | u_ic : {}| F_ic : {} | lr : {} | MAE : {} |L2RE:{}".format(
                    epoch,
                    res_loss,
                    flux_loss,
                    u_ic_loss,
                    F_ic_loss,
                    learning_rate,
                    mae,
                    L2RE,
                )
            )
        if epoch % 1000 == 0:
            with open(csv_path, "a+") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        epoch,
                        loss.item(),
                        res_loss.item(),
                        flux_loss.item(),
                        u_ic_loss.item(),
                        F_ic_loss.item(),
                        u_bc_loss.item(),
                        F_bc_loss.item(),
                        mae.item(),
                        L2RE.item(),
                    ]
                )
            torch.save(model.state_dict(), sdpath)
            torch.save(optimizer.state_dict(), lrpath)
        if epoch == config.epochs - 1:
            continue
        else:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if epoch % 1000 == 0:
                scheduler.step()
