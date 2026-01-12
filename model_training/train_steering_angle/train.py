import os
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

import driving_data
import model


class DataLogger:
    def __init__(self, logs_path: str):
        self.logs_path = logs_path
        self.writer = SummaryWriter(logs_path)

    def log_scalar(self, tag: str, value: float, step: int):
        self.writer.add_scalar(tag, value, step)

    def close(self):
        self.writer.close()


class Trainer:
    def __init__(self, log_dir, logger, l2_norm_const=0.001, learning_rate=1e-4, device=None):
        self.log_dir = log_dir
        self.l2_norm_const = l2_norm_const
        self.logger = logger
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.model = model.SteeringAngleModel().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = torch.nn.MSELoss()

    def _prepare_batch(self, xs, ys):
        x_tensor = torch.from_numpy(np.asarray(xs, dtype=np.float32))
        x_tensor = x_tensor.permute(0, 3, 1, 2).contiguous().to(self.device)
        y_tensor = torch.from_numpy(np.asarray(ys, dtype=np.float32)).to(self.device)
        return x_tensor, y_tensor

    def _compute_loss(self, preds, targets):
        mse_loss = self.loss_fn(preds, targets)
        l2_loss = 0.5 * sum(torch.sum(p ** 2) for p in self.model.parameters())
        return mse_loss + self.l2_norm_const * l2_loss

    def train(self, epochs, batch_size):
        for epoch in range(epochs):
            self._train_one_epoch(epoch, batch_size)
            print(f"Epoch {epoch + 1}/{epochs} completed.")

    def _train_one_epoch(self, epoch, batch_size):
        self.model.train()
        steps_per_epoch = int(driving_data.num_images / batch_size)
        for step in range(steps_per_epoch):
            xs, ys = driving_data.LoadTrainBatch(batch_size)
            x_tensor, y_tensor = self._prepare_batch(xs, ys)

            self.optimizer.zero_grad()
            preds = self.model(x_tensor)
            loss = self._compute_loss(preds, y_tensor)
            loss.backward()
            self.optimizer.step()

            if step % 10 == 0:
                self._log_progress(epoch, step, batch_size)

            if step % batch_size == 0:
                self._save_checkpoint()

    def _log_progress(self, epoch, step, batch_size):
        self.model.eval()
        with torch.no_grad():
            xs, ys = driving_data.LoadValBatch(batch_size)
            x_tensor, y_tensor = self._prepare_batch(xs, ys)
            preds = self.model(x_tensor)
            loss_value = self._compute_loss(preds, y_tensor).item()
        self.model.train()

        print(f"Epoch: {epoch + 1}, Step: {step}, Loss: {loss_value:.5f}")
        global_step = epoch * int(driving_data.num_images / batch_size) + step
        self.logger.log_scalar("loss", loss_value, global_step)

    def _save_checkpoint(self):
        os.makedirs(self.log_dir, exist_ok=True)
        checkpoint_path = os.path.join(self.log_dir, "model.pth")
        torch.save(self.model.state_dict(), checkpoint_path)
        print(f"Model checkpoint saved at {checkpoint_path}")

    def close(self):
        self.logger.close()


if __name__ == "__main__":
    LOGDIR = "model_training/train_steering_angle/save"
    LOGS_PATH = "model_training/train_steering_angle/logs"
    EPOCHS = 30
    BATCH_SIZE = 100

    logger = DataLogger(LOGS_PATH)
    trainer = Trainer(LOGDIR, logger)

    try:
        trainer.train(EPOCHS, BATCH_SIZE)
    finally:
        trainer.close()

    print(
        "Run the command line:\n"
        "--> tensorboard --logdir=model_training/train_steering_angle/logs"
        "\nThen open http://0.0.0.0:6006/ into your web browser"
    )
