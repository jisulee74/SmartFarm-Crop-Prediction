"""GPU execution adapter for the existing official TFT candidate."""
from __future__ import annotations

import tempfile

import pandas as pd
import torch

import tft_candidate as base


def train_tft(fit, validation, context, target, params, seed, fixed_epochs=None):
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
    from pytorch_forecasting import TemporalFusionTransformer
    from pytorch_forecasting.metrics import RMSE

    b = base.backend
    b.seed_everything(seed)
    medians = b.fit_imputation(fit)
    fit_i = b.apply_imputation(fit, medians)
    validation_i = b.apply_imputation(validation, medians) if validation is not None else None
    context_i = b.apply_imputation(context, medians)
    train_long = b.make_long_sequences(pd.DataFrame(columns=fit_i.columns), fit_i, target)
    if validation_i is None:
        validation_i = fit_i.tail(min(max(8, len(fit_i) // 10), len(fit_i))).copy()
        context_i = fit_i.iloc[:-len(validation_i)].copy()
    val_long = b.make_long_sequences(context_i, validation_i, target)
    train_ds, val_ds = b.make_datasets(train_long, val_long, target)
    batch_size = min(int(params["batch_size"]), len(train_ds))
    train_dl = train_ds.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_dl = val_ds.to_dataloader(train=False, batch_size=min(batch_size, len(val_ds)), num_workers=0)
    model = TemporalFusionTransformer.from_dataset(
        train_ds, learning_rate=float(params["learning_rate"]), hidden_size=int(params["hidden_size"]),
        attention_head_size=int(params["attention_head_size"]),
        hidden_continuous_size=int(params["hidden_continuous_size"]), dropout=float(params["dropout"]),
        output_size=1, loss=RMSE(), log_interval=-1, reduce_on_plateau_patience=4)
    with tempfile.TemporaryDirectory() as directory:
        checkpoint = ModelCheckpoint(dirpath=directory, monitor="val_loss", mode="min", save_top_k=1)
        callbacks = [checkpoint]
        if fixed_epochs is None:
            callbacks.append(EarlyStopping(monitor="val_loss", patience=10, mode="min", min_delta=1e-5))
        trainer = pl.Trainer(
            max_epochs=int(fixed_epochs or params["max_epochs"]),
            accelerator="gpu" if torch.cuda.is_available() else "cpu", devices=1,
            logger=False, enable_model_summary=False, enable_progress_bar=False,
            callbacks=callbacks, deterministic="warn")
        trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)
        best_epoch = int(trainer.current_epoch + 1)
        if fixed_epochs is None and checkpoint.best_model_path:
            model = TemporalFusionTransformer.load_from_checkpoint(checkpoint.best_model_path)
    model.to("cpu")
    return model, train_ds, medians, best_epoch


def _activate():
    base.backend.train_tft = train_tft


def tune(*args, **kwargs):
    _activate(); return base.tune(*args, **kwargs)


def validate(*args, **kwargs):
    _activate(); return base.validate(*args, **kwargs)


def fit_predict(*args, **kwargs):
    _activate(); return base.fit_predict(*args, **kwargs)
