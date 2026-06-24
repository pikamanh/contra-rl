"""Custom SB3 callbacks for training Contra."""
import os
import zipfile
from datetime import datetime

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class ContraCallback(BaseCallback):
    """
    Combined callback that handles:
      - Saving the best model whenever mean episode reward improves.
      - Saving the latest model after every PPO rollout (epoch).
      - Saving a checkpoint every `save_freq_epochs` epochs.
      - Optionally bundling all saved models + logs into a zip archive.

    Directory layout inside `save_dir`:
        best/best_model.zip
        latest/latest_model.zip
        checkpoints/checkpoint_epoch_<N>.zip
        training_<timestamp>.zip   ← created when --zip is set
    """

    def __init__(
        self,
        save_dir: str,
        save_freq_epochs: int = 25,
        save_zip: bool = False,
        log_dir: str = None,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.save_dir = save_dir
        self.save_freq_epochs = save_freq_epochs
        self.save_zip = save_zip
        self.log_dir = log_dir

        self.best_dir = os.path.join(save_dir, "best")
        self.latest_dir = os.path.join(save_dir, "latest")
        self.ckpt_dir = os.path.join(save_dir, "checkpoints")
        for d in (self.best_dir, self.latest_dir, self.ckpt_dir):
            os.makedirs(d, exist_ok=True)

        self.best_mean_reward = -np.inf
        self.epoch = 0
        self.shoot_action_count = 0
        self.shoot_when_enemy_count = 0
        self.env_step_count = 0

    # ------------------------------------------------------------------
    # SB3 hooks
    # ------------------------------------------------------------------

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        self.env_step_count += len(infos)

        for info in infos:
            if info.get("is_shooting", False):
                self.shoot_action_count += 1
                if info.get("active_enemies", 0) > 0:
                    self.shoot_when_enemy_count += 1

        return True

    def _on_rollout_end(self) -> None:
        self.epoch += 1
        self._report_shoot_stats()
        self._save_latest()
        self._maybe_save_best()

        if self.epoch % self.save_freq_epochs == 0:
            self._save_checkpoint()
            if self.save_zip:
                self._create_zip(tag=f"epoch_{self.epoch}")

    def on_training_end(self) -> None:
        self._save_latest()
        if self.save_zip:
            self._create_zip(tag="final")
        if self.verbose >= 1:
            print(f"\nTraining finished after {self.epoch} epochs.")

    # ------------------------------------------------------------------
    # Save helpers
    # ------------------------------------------------------------------

    def _save_latest(self):
        path = os.path.join(self.latest_dir, "latest_model")
        self.model.save(path)
        if self.verbose >= 2:
            print(f"[Epoch {self.epoch:>5}] Saved latest → {path}.zip")

    def _maybe_save_best(self):
        buf = self.model.ep_info_buffer
        if not buf:
            return
        mean_reward = np.mean([ep["r"] for ep in buf])
        if mean_reward > self.best_mean_reward:
            self.best_mean_reward = mean_reward
            path = os.path.join(self.best_dir, "best_model")
            self.model.save(path)
            if self.verbose >= 1:
                print(
                    f"[Epoch {self.epoch:>5}] New best  → {path}.zip"
                    f"  (mean_reward={mean_reward:.2f})"
                )

    def _save_checkpoint(self):
        name = f"checkpoint_epoch_{self.epoch}"
        path = os.path.join(self.ckpt_dir, name)
        self.model.save(path)
        if self.verbose >= 1:
            print(f"[Epoch {self.epoch:>5}] Checkpoint → {path}.zip")

    def _create_zip(self, tag: str):
        """Bundle all saved models (and optionally logs) into a single zip."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_name = f"training_{tag}_{ts}.zip"
        zip_path = os.path.join(self.save_dir, zip_name)

        dirs_to_bundle = [self.best_dir, self.latest_dir, self.ckpt_dir]
        if self.log_dir and os.path.isdir(self.log_dir):
            dirs_to_bundle.append(self.log_dir)

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for src_dir in dirs_to_bundle:
                for root, _, files in os.walk(src_dir):
                    for fname in files:
                        full = os.path.join(root, fname)
                        arcname = os.path.relpath(full, self.save_dir)
                        zf.write(full, arcname)

        if self.verbose >= 1:
            size_mb = os.path.getsize(zip_path) / 1024 / 1024
            print(f"[Epoch {self.epoch:>5}] Archive    → {zip_path}  ({size_mb:.1f} MB)")
