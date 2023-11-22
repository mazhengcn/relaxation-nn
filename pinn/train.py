import csv
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from absl import logging
from ml_collections import ConfigDict


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
                "u_ic",
                "u_bc",
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
        res_loss = model.interior_loss(x_int)
        u_ic_loss = model.init_loss(x_ic)
        u_bc_loss = model.bc_loss(x_bc)
        # loss = res_loss + u_ic_loss + u_bc_loss
        loss = (
            config.weights[0] * res_loss
            + config.weights[1] * u_ic_loss
            + config.weights[2] * u_bc_loss
        )
        learning_rate = optimizer.state_dict()["param_groups"][0]["lr"]
        if epoch % 1000 == 0:
            logging.info(
                "epoch : {}  |  res_loss :{} | u_ic : {}| lr : {} | MAE : {}".format(
                    epoch, res_loss, u_ic_loss, learning_rate, mae
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
                        u_ic_loss.item(),
                        u_bc_loss.item(),
                        mae.item(),
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


def train_sv(
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
                "u_ic",
                "u_bc",
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
        res_loss = model.interior_loss(x_int)
        u_ic_loss = model.init_loss(x_ic)
        u_bc_loss = model.bc_loss(x_bc)
        sv_loss = model.supervise(x_test, q_test)
        loss = sv_loss
        learning_rate = optimizer.state_dict()["param_groups"][0]["lr"]
        if epoch % 1000 == 0:
            logging.info(
                "epoch : {} |sv_loss:{} |  res_loss :{} | u_ic : {}| lr : {} | MAE : {}".format(
                    epoch, sv_loss, res_loss, u_ic_loss, learning_rate, mae
                )
            )
        if epoch % 500 == 0:
            with open(csv_path, "a+") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        epoch,
                        loss.item(),
                        res_loss.item(),
                        u_ic_loss.item(),
                        u_bc_loss.item(),
                        mae.item(),
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
            if epoch % 100 == 0:
                scheduler.step()


def svpinn(
    device: torch.device,
    training_data: Iterable,
    x_test: np.array,
    q_test: np.array,
    model: torch.nn.Module,
    config: ConfigDict,
    lrpath: Path,
    csv_path: Path,
    model_dir: Path,
    lr_dir: Path,
):
    optimizer = torch.optim.Adam(params=model.parameters())
    optimizer.load_state_dict(torch.load(lrpath))
    metric = torch.nn.L1Loss()
    x_test = torch.tensor(x_test, dtype=torch.float32).to(device)
    q_test = torch.tensor(q_test, dtype=torch.float32).to(device)
    with open(csv_path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "epoch",
                "total",
                "sv_loss",
                "res_loss",
                "u_ic",
                "u_bc",
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
        res_loss = model.interior_loss(x_int)
        u_ic_loss = model.init_loss(x_ic)
        u_bc_loss = model.bc_loss(x_bc)
        sv_loss = model.supervise(x_test, q_test)
        loss = res_loss + u_ic_loss + u_bc_loss
        learning_rate = optimizer.state_dict()["param_groups"][0]["lr"]
        if epoch % 100 == 0:
            logging.info(
                "epoch : {} | sv_loss:{} | res_loss :{} | u_ic : {}| lr : {} | MAE : {}".format(
                    epoch, sv_loss, res_loss, u_ic_loss, learning_rate, mae
                )
            )
            with open(csv_path, "a+") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        epoch,
                        loss.item(),
                        sv_loss.item(),
                        res_loss.item(),
                        u_ic_loss.item(),
                        u_bc_loss.item(),
                        mae.item(),
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


def hessian_analysis(
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
                "u_ic",
                "u_bc",
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
        res_loss = model.interior_loss(x_int)
        u_ic_loss = model.init_loss(x_ic)
        u_bc_loss = model.bc_loss(x_bc)
        loss = (
            config.weights[0] * res_loss
            + config.weights[1] * u_ic_loss
            + config.weights[2] * u_bc_loss
        )
        learning_rate = optimizer.state_dict()["param_groups"][0]["lr"]
        if epoch % 1000 == 0:
            logging.info(
                "epoch : {}  |  res_loss :{} | u_ic : {}| lr : {} | MAE : {}".format(
                    epoch, res_loss, u_ic_loss, learning_rate, mae
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
                        u_ic_loss.item(),
                        u_bc_loss.item(),
                        mae.item(),
                    ]
                )
            torch.save(model.state_dict(), sdpath)
            torch.save(optimizer.state_dict(), lrpath)
        if epoch == config.epochs - 1:
            continue
        else:
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            grads = [p.grad for p in model.parameters() if p.grad is not None]
            loss_grads = _flatten(grads)
            loss_grads.requires_grad_(True)
            num_params = loss_grads.shape[0]
            vmat = torch.eye(num_params, dtype=torch.float32).to(device)
            Hmat = torch.zeros_like(vmat)
            for i in range(vmat.shape[0]):
                v = vmat[i : i + 1, :].requires_grad_(False)
                vprod = torch.multiply(loss_grads, v)
                Hmat[i : i + 1, :] = _flatten(
                    torch.autograd.grad(vprod, model.parameters(), retain_graph=True)
                )
            print(torch.linalg.eigh(Hmat)[0])
            optimizer.step()
            if epoch % 1000 == 0:
                scheduler.step()


def _flatten(vectors):
    return torch.concat([torch.reshape(v, [-1]) for v in vectors], axis=0)
