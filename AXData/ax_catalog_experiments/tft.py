"""TFT adapter: registered features only, no lagged target or relative-time extras."""
from __future__ import annotations
import sys
import pandas as pd
from .catalog import AX


class TFTAdapter:
    def __init__(self,features,params,seed,entity,current):
        sys.path.insert(0,str(AX));sys.path.insert(0,str(AX/'tomato_flower_count_prediction'))
        import tft_candidate
        tft_candidate.configure(features,entity,current)
        self.backend=tft_candidate.backend
        self.features,self.params,self.seed=features,params,seed
        self.best_epoch=None
        def datasets(train_long,val_long,target):
            from pytorch_forecasting import TimeSeriesDataSet
            from pytorch_forecasting.data.encoders import NaNLabelEncoder,TorchNormalizer
            training=TimeSeriesDataSet(train_long,time_idx='time_idx',target=target,group_ids=['sequence_id'],min_encoder_length=self.backend.ENCODER_LENGTH,max_encoder_length=self.backend.ENCODER_LENGTH,min_prediction_length=1,max_prediction_length=1,time_varying_known_reals=features,time_varying_unknown_reals=[],target_normalizer=TorchNormalizer(method='standard'),categorical_encoders={'sequence_id':NaNLabelEncoder(add_nan=True)},add_relative_time_idx=False,add_encoder_length=False,add_target_scales=False)
            if set(training.reals)!=set(features):raise ValueError('TFT input schema expanded unexpectedly')
            return training,TimeSeriesDataSet.from_dataset(training,val_long,predict=False,stop_randomization=True)
        self.backend.make_datasets=datasets
    def fit(self,train,target,validation=None,fixed_epochs=None):
        self.target=target;self.context=train.copy()
        self.model,self.dataset,self.medians,self.best_epoch=self.backend.train_tft(train,validation,train.iloc[:0],target,self.params,self.seed,fixed_epochs=fixed_epochs)
        return self
    def predict(self,frame):
        evaluation=frame.copy();evaluation[self.target]=0.
        context=self.context.copy();context[self.target]=0.
        return self.backend.predict_tft(self.model,self.dataset,context,evaluation,self.target,self.medians,self.params['batch_size'])
    def save(self,path):
        import torch
        torch.save({'state_dict':self.model.state_dict(),'features':self.features,'params':self.params,'seed':self.seed,'dataset_parameters':self.dataset.get_parameters(),'medians':self.medians},path)
