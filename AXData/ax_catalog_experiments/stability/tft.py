"""TFT with fixed registered inputs, stable scaling and finite-gradient checks."""
from __future__ import annotations
import os
import tempfile
from contextlib import contextmanager, nullcontext
from pathlib import Path
import numpy as np
import pandas as pd
from .policy import StablePreprocessor


@contextmanager
def _trusted_local_checkpoint_load(torch):
    """PyTorch 2.6 compatibility for checkpoints produced by this campaign."""
    original = torch.load
    previous_force = os.environ.get("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD")
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    def load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original(*args, **kwargs)
    torch.load = load
    try:
        yield
    finally:
        torch.load = original
        if previous_force is None:
            os.environ.pop("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", None)
        else:
            os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = previous_force


class StableTFT:
    encoder_length=4
    def __init__(self,features,params,seed,entity,current):
        self.features=list(features);self.params=params;self.seed=seed;self.entity=entity;self.best_epoch=None
    def transform(self,frame):
        f=frame.copy(); f[self.features]=self.preprocessor.transform(frame);return f
    def sequences(self, context, evaluation, target):
        # Targets in encoder positions are placeholders, never model inputs.
        combined=pd.concat([context.assign(_eval=-1),evaluation.assign(_eval=np.arange(len(evaluation)))],ignore_index=True)
        combined=combined.sort_values([*self.entity,'feature_date','_eval'])
        feature_blocks=[]; sequence_ids=[]; time_indices=[]; targets=[]
        for _,group in combined.groupby(self.entity,dropna=False,sort=False):
            eval_values=group._eval.to_numpy(); target_values=group[target].to_numpy(float)
            feature_values=group[self.features].to_numpy(float)
            for i in np.flatnonzero(eval_values>=0):
                history=list(range(max(0,i-self.encoder_length),i))
                if not history:history=[i]
                take=[history[0]]*(self.encoder_length-len(history))+history+[i]
                feature_blocks.append(feature_values[take])
                sequence_ids.extend([f'seq_{int(eval_values[i])}']*(self.encoder_length+1))
                time_indices.extend(range(self.encoder_length+1))
                targets.extend([0.]*self.encoder_length+[float(target_values[i])])
        if not feature_blocks:
            return pd.DataFrame(columns=['sequence_id','time_idx',target,*self.features])
        result=pd.DataFrame(np.vstack(feature_blocks),columns=self.features)
        result.insert(0,target,targets);result.insert(0,'time_idx',time_indices);result.insert(0,'sequence_id',sequence_ids)
        return result
    def fit(self,train,target,validation=None,fixed_epochs=None):
        import torch
        import lightning.pytorch as pl
        from lightning.pytorch.callbacks import Callback,EarlyStopping,ModelCheckpoint
        from pytorch_forecasting import TimeSeriesDataSet,TemporalFusionTransformer
        from pytorch_forecasting.data.encoders import NaNLabelEncoder,TorchNormalizer
        from pytorch_forecasting.metrics import RMSE
        class Finite(Callback):
            def on_before_backward(self,trainer,model,loss):
                if not torch.isfinite(loss).all():raise FloatingPointError('Nonfinite TFT loss/output')
            def on_before_optimizer_step(self,trainer,model,optimizer):
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise FloatingPointError('Nonfinite TFT gradient')
            def on_validation_epoch_end(self,trainer,model):
                for value in trainer.callback_metrics.values():
                    if torch.is_tensor(value) and not torch.isfinite(value).all():raise FloatingPointError('Nonfinite TFT validation metric')
        pl.seed_everything(self.seed,workers=True)
        self.target=target;self.preprocessor=StablePreprocessor(self.features).fit(train)
        self.context=self.transform(train)
        long=self.sequences(self.context.iloc[:0],self.context,target)
        self.dataset=TimeSeriesDataSet(long,time_idx='time_idx',target=target,group_ids=['sequence_id'],
          min_encoder_length=self.encoder_length,max_encoder_length=self.encoder_length,
          min_prediction_length=1,max_prediction_length=1,time_varying_known_reals=self.features,
          time_varying_unknown_reals=[],scalers={f:None for f in self.features},
          target_normalizer=TorchNormalizer(method='identity'),
          categorical_encoders={'sequence_id':NaNLabelEncoder(add_nan=True)},
          add_relative_time_idx=False,add_encoder_length=False,add_target_scales=False)
        if set(self.dataset.reals)!=set(self.features):raise ValueError('TFT input expansion')
        val_ds=None
        if validation is not None:
            val_long=self.sequences(self.context,self.transform(validation),target)
            val_ds=TimeSeriesDataSet.from_dataset(self.dataset,val_long,predict=False,stop_randomization=True)
        self.model=TemporalFusionTransformer.from_dataset(self.dataset,learning_rate=self.params['learning_rate'],
          hidden_size=self.params['hidden_size'],attention_head_size=self.params['attention_head_size'],
          hidden_continuous_size=self.params['hidden_continuous_size'],dropout=self.params['dropout'],
          output_size=1,loss=RMSE(),log_interval=-1,reduce_on_plateau_patience=4)
        batch=min(self.params['batch_size'],len(self.dataset))
        persistent=self.params.get('_checkpoint_dir')
        context=nullcontext(str(Path(persistent))) if persistent else tempfile.TemporaryDirectory()
        with context as tmp:
            Path(tmp).mkdir(parents=True,exist_ok=True)
            checkpoint=ModelCheckpoint(dirpath=tmp,monitor='val_loss',mode='min',save_top_k=1,save_last=bool(persistent))
            callbacks=[Finite()]
            if val_ds is not None:callbacks.extend([checkpoint,EarlyStopping(monitor='val_loss',patience=10,mode='min',check_finite=True)])
            trainer=pl.Trainer(max_epochs=int(fixed_epochs or self.params['max_epochs']),
              accelerator='gpu' if torch.cuda.is_available() else 'cpu',devices=1,
              gradient_clip_val=1.,precision='bf16-mixed' if torch.cuda.is_available() else '32-true',logger=False,enable_progress_bar=False,enable_model_summary=False,
              enable_checkpointing=val_ds is not None,callbacks=callbacks,deterministic='warn')
            last=Path(tmp)/'last.ckpt'
            with _trusted_local_checkpoint_load(torch):
                trainer.fit(self.model,train_dataloaders=self.dataset.to_dataloader(train=True,batch_size=batch,num_workers=0,drop_last=False),
                  val_dataloaders=val_ds.to_dataloader(train=False,batch_size=batch,num_workers=0) if val_ds is not None else None,
                  ckpt_path=str(last) if persistent and last.exists() else None)
                self.best_epoch=int(fixed_epochs or trainer.current_epoch)
                if val_ds is not None:
                    if not checkpoint.best_model_path:raise ValueError('No finite TFT checkpoint')
                    payload=torch.load(checkpoint.best_model_path,map_location='cpu')
                    self.best_epoch=int(payload['epoch'])+1
                    self.model=TemporalFusionTransformer.load_from_checkpoint(checkpoint.best_model_path)
        self.model.to('cpu');return self
    def predict(self,frame):
        from pytorch_forecasting import TimeSeriesDataSet
        evaluation=frame.copy();evaluation[self.target]=0.
        long=self.sequences(self.context,self.transform(evaluation),self.target)
        ds=TimeSeriesDataSet.from_dataset(self.dataset,long,predict=False,stop_randomization=True)
        result=self.model.predict(ds.to_dataloader(train=False,batch_size=self.params['batch_size'],num_workers=0),
          mode='prediction',return_index=True,trainer_kwargs={'logger':False,'enable_progress_bar':False})
        idx=result.index.copy();idx['p']=result.output.detach().cpu().numpy().reshape(-1)
        idx['order']=idx.sequence_id.str.replace('seq_','',regex=False).astype(int)
        return idx.sort_values('order').p.to_numpy(float)
    def save(self,path):
        import torch
        torch.save(dict(state_dict=self.model.state_dict(),dataset=self.dataset.get_parameters(),
          preprocessor=self.preprocessor,features=self.features,params=self.params),path)
