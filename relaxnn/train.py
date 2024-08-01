import csv
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from absl import logging
from ml_collections import ConfigDict


def train(
    device: torch.device,
    datagenerator,
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
        L2RE = torch.sqrt(
            torch.nn.MSELoss()(q_test, q_pred)
            / torch.nn.MSELoss()(q_test, torch.zeros_like(q_test))
        )
        # --------- train step------------#
        x_int, x_ic, x_bc = datagenerator.samples()
        x_int.requires_grad = True
        x_ic.requires_grad = True
        x_bc.requires_grad = True
        res_loss, flux_loss = model.interior_loss(x_int, int_weights)
        u_ic_loss, F_ic_loss = model.init_loss(x_ic)
        u_bc_loss, F_bc_loss = model.bc_loss(x_bc)
        loss = (
            ratio[0] * res_loss
            + ratio[1] * flux_loss
            + ratio[2] * u_ic_loss
            + ratio[3] * u_bc_loss
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
