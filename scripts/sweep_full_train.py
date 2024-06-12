
import json
from datetime import datetime
import matplotlib.pyplot as plt
from pathlib import Path
from ConfigSpace import Categorical, Configuration, ConfigurationSpace, Float, UniformIntegerHyperparameter

from scripts.trainer import trainer

class optimize_prior():
    
    def __init__(self, processed, device, train_loader, model_path, tag, logger_cfg, 
                 model, blender_resrc_path, EXEC_ON_CLUSTER, debug, pretrained_infer):
        
        self.processed = processed
        self.device = device
        self.train_loader = train_loader
        self.model_path = model_path
        self.tag = tag
        self.logger_cfg = logger_cfg
        self.model = model
        self.blender_resrc_path = blender_resrc_path
        self.EXEC_ON_CLUSTER = EXEC_ON_CLUSTER
        self.debug = debug
        self.pretrained_infer = pretrained_infer
        
        with open(str(Path(self.processed.parents[1], f"configs/base.json")), "r") as f:
            self.config = json.load(f)
        self.sweep_trials = self.config["TRAIN_PARAM"]["motionprior"]["sweep_trials"]
        if self.config["TRAIN_PARAM"]["motionprior"]["emotional"]: 
            if "_fing" in self.config["TRAIN_PARAM"]["diffusion"]["lmdb_cache"]: self.cfg_name = "prior_emotional_fing"
            else: self.cfg_name = "prior_emotional"
        else: self.cfg_name = "prior"
        with open(str(Path(self.processed.parents[1], f"configs/{self.cfg_name}.json")), "r") as f:
            self.prior_cfg = json.load(f)
            
        self.sweep_cfgs_base = Path(self.processed.parents[1], "configs/sweeps", datetime.now().strftime("%Y%m%d-%H%M%S"))
        self.sweep_cfgs_base.mkdir(parents=True, exist_ok=True)
        self.sweep_count = 0
        
    @property
    def configspace(self) -> ConfigurationSpace:
        
        cs = ConfigurationSpace(seed=0)
        
        batch_size = Categorical("batch_size", [16, 32, 64, 128], default=64)
        learning_rate = Float("lr_base", (1e-8, 1e-1), default=1e-4, log=True)
        latent_dim = Categorical("latent_dim", [64, 128, 256, 512], default=256)
        ff_size = Categorical("ff_size", [64, 128, 256, 512, 1024], default=1024)
        num_layers = UniformIntegerHyperparameter("num_layers", lower=7, upper=13, default_value=9, q=2)
        num_heads = Categorical("num_heads", [4, 8], default=4)
        dropout = Float("dropout", (0.01, 0.3), default=0.1, log=True)
        lambda_kl = Float("lambda_kl", (1e-10, 1e-5), default=1e-7, log=True)
        use_recons_joints = Categorical("use_recons_joints", [True, False], default=True)
        
        cs.add_hyperparameters([batch_size, learning_rate, latent_dim, ff_size, num_layers, num_heads, dropout, lambda_kl, use_recons_joints])
        # cs.sample_configuration(2)
        return cs    
        
    def train(self, config: Configuration, seed: int = 0, budget: int = 25) -> float:
        print(f"[PRIOR-OPTIM] [{self.sweep_count}/{self.sweep_trials}] budget epochs: {budget}")
        self.sweep_count += 1
        self.sweep_cfgs = self.sweep_cfgs_base / f"{self.sweep_count}"
        self.sweep_cfgs.mkdir(parents=True, exist_ok=True)
        self.config_copy = self.config.copy()
        self.prior_cfg_copy = self.prior_cfg.copy()
        config_dict = config.get_dictionary()
        self.config_copy["TRAIN_PARAM"]["motionprior"]["batch_size"] = config_dict["batch_size"]
        self.config_copy["TRAIN_PARAM"]["motionprior"]["lr_base"] = config_dict["lr_base"]
        self.config_copy["TRAIN_PARAM"]["motionprior"]["sweep_given_budget"] = int(budget)
        self.prior_cfg_copy["arch_main"]["latent_dim"] = [1, config_dict["latent_dim"]]
        self.prior_cfg_copy["arch_main"]["ff_size"] = config_dict["ff_size"]
        self.prior_cfg_copy["arch_main"]["num_layers"] = config_dict["num_layers"]  
        self.prior_cfg_copy["arch_main"]["num_heads"] = config_dict["num_heads"]    
        self.prior_cfg_copy["arch_main"]["dropout"] = config_dict["dropout"]
        self.prior_cfg_copy["losses"]["LAMBDA_KL"] = config_dict["lambda_kl"]
        self.prior_cfg_copy["losses"]["use_recons_joints"] = config_dict["use_recons_joints"]
        with open(str(Path(self.sweep_cfgs, f"base.json")), "w") as f:
            json.dump(self.config_copy, f, indent=4)
        with open(str(Path(self.sweep_cfgs, f"{self.cfg_name}.json")), "w") as f:
            json.dump(self.prior_cfg_copy, f, indent=4)
        
        self.model.setup(self.processed, self.config_copy, self.prior_cfg_copy)
        self.model.to(self.device)
        T = trainer(self.config_copy, self.device, train_loader=self.train_loader, model_path=self.model_path, tag=self.tag, 
                    logger_cfg=self.logger_cfg, model=self.model, processed=self.processed, b_path=self.blender_resrc_path, 
                    EXEC_ON_CLUSTER=self.EXEC_ON_CLUSTER, debug=self.debug, pretrained_infer=self.pretrained_infer, sweep=self.sweep_cfgs)
        return T.sweep_motionprior(int(budget))

    def plot_trajectory(self, facades):
        plt.figure()
        plt.title("Trajectory")
        plt.xlabel("Wallclock time [s]")
        plt.ylabel(facades[0].scenario.objectives)
        plt.ylim(0, 0.4)
        for facade in facades:
            X, Y = [], []
            for item in facade.intensifier.trajectory:
                # Single-objective optimization
                assert len(item.config_ids) == 1
                assert len(item.costs) == 1
                y = item.costs[0]
                x = item.walltime
                X.append(x)
                Y.append(y)
            plt.plot(X, Y, label=facade.intensifier.__class__.__name__)
            plt.scatter(X, Y, marker="x")
        plt.legend()
        fig = Path(self.sweep_cfgs_base, "optimization_trajectory.png")
        plt.savefig(fig)