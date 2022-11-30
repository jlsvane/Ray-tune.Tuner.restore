#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 10:06:05 2022

@author: lupus
"""


import ray
from ray import air, tune
import ray.rllib.algorithms.impala as impala
from ray.tune import Tuner
from ray.tune.schedulers.pb2 import PB2
from ray.tune import Stopper
from ray.rllib.utils.framework import try_import_tf

from ray.rllib.examples.env.random_env import RandomEnv
from ray.tune.registry import register_env

def env_creator(config):
        env = RandomEnv(config=config)
        return env

register_env("RandomEnv", env_creator)

class Impalaalgo(impala.Impala):
    def __init__(self, config, **kwargs):
        super(Impalaalgo, self).__init__(config, **kwargs)
        
    def reset_config(self, new_config):
        """ to enable reuse of actors """
        self.config = new_config
        return True    

pb2 = PB2(
    time_attr="timesteps_total",
    metric="episode_reward_mean",
    mode="max",
    perturbation_interval=500,
    quantile_fraction=0.25,  # copy bottom % with top %
    # Specifies the hyperparam search space
    hyperparam_bounds={
        "lr": [1e-9,1e-3],
        "entropy_coeff": [0.0, 1e-2],
    },
    log_config=True
)


ray.init(address=None)

config = impala.ImpalaConfig().to_dict()

tune_params = {
                "env": "RandomEnv",
                "num_gpus": 1, 
                "num_workers": 1, 
                "framework": "tf2",
                "eager_tracing": True,
                "recreate_failed_workers": True,
                "restart_failed_sub_environments": True,
                "vf_loss_coeff": 1.0,
                "lr": tune.uniform(1e-9,1e-3),
                "entropy_coeff": tune.uniform(0.0,1e-2),
            }

config = config | tune_params

# comment out when trying to test tune.Tuner.restore() below

# tuner = tune.Tuner(
#                     Impalaalgo,
#                     run_config=air.RunConfig(
#                         name="Impalaalgo_RAY2_1_0_pb2_tuner_restore_test",
#                         local_dir="impala_default_model",
#                         stop={"timesteps_total": 1e6,"episode_reward_mean":1.0}, 
#                         checkpoint_config=air.CheckpointConfig(checkpoint_at_end=True),
#                         ),
#                     param_space=config,
                        
#                     tune_config=tune.tune_config.TuneConfig(
#                         scheduler=pb2,
#                         num_samples=4,
#                         reuse_actors=True,
#                         )
#                     )

# results = tuner.fit()

# comment out when running tuner.Tuner.fit() above

tuner = tune.Tuner.restore("/home/lupus/tuner_restore_test/impala_default_model/Impalaalgo_RAY2_1_0_pb2_tuner_restore_test")

results = tuner.fit()