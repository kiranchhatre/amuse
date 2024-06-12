
import torch
import torch.nn as nn
from einops import rearrange
from torchmetrics import Metric

from dm.utils.ldm_evals import subject2genderbeta

class LatentPriorLosses(Metric):
    
    def __init__(self,cfg, device, smplx_models=None):
        super().__init__(dist_sync_on_step=False) 

        self.d = device
        self.cfg = cfg
        self.smplx_models = smplx_models
        
        self.train_lpdm_version = self.cfg["losses"]["train_lpdm"]["version"]
        
        self.stage = self.cfg["losses"]["stage"] 
        self.predict_epsilon = self.cfg["losses"]["predict_epsilon"] 
        self.use_recons_joints = self.cfg["losses"]["use_recons_joints"]
        
        self.LAMBDA_PRIOR = self.cfg["losses"]["LAMBDA_PRIOR"]
        self.LAMBDA_KL = self.cfg["losses"]["LAMBDA_KL"]
        self.LAMBDA_REC = self.cfg["losses"]["LAMBDA_REC"]
        self.LAMBDA_GEN = self.cfg["losses"]["LAMBDA_GEN"]
        self.LAMBDA_JOINT = self.cfg["losses"]["LAMBDA_JOINT"]
        self.LAMBDA_LATENT = self.cfg["losses"]["LAMBDA_LATENT"]
        self.vtex_displacement = self.cfg["losses"]["vtex_displacement"]
        if self.vtex_displacement: 
            assert self.smplx_models is not None, "[LatentPriorLosses] smplx_models must be provided if vtex_displacement is True"
            assert self.stage in ['vae_diffusion'], "[LatentPriorLosses] vtex_displacement loss is only in vae_diffusion, but {}".format(self.stage)
        
        losses = []

        if self.stage in ['diffusion', 'vae_diffusion']:                        
            losses.append("inst_loss")

        if self.stage in ['vae', 'vae_diffusion']:
            losses.append("recons_feature")
            losses.append("recons_joints") 
            losses.append("kl_motion") 
            
            if self.train_lpdm_version == "v0": 
                losses.append("gen_feature") 
                losses.append("gen_joints")                                    
            elif self.train_lpdm_version == "v1": 
                losses.append("latent_feature")  
            else: raise ValueError(f"train_lpdm_version {self.train_lpdm_version} not supported, choose: v0 or v1")  
            
            if self.vtex_displacement: 
                losses.append("rec_vtex_displacement")
                losses.append("gen_vtex_displacement")                   
        
        if self.stage not in ['vae', 'diffusion', 'vae_diffusion']:             
            raise ValueError(f"Stage {self.stage} not supported")

        losses.append("total")

        for loss in losses:
            self.add_state(loss,
                           default=torch.tensor(0.0).to(self.d),
                           dist_reduce_fx="sum")
        self.add_state("count", torch.tensor(0).to(self.d), dist_reduce_fx="sum")
        self.losses = losses

        self._losses_func = {}
        self._params = {}
        for loss in losses:
            
            if loss.split('_')[0] == 'inst':
                self._losses_func[loss] = nn.MSELoss(reduction='mean')
                self._params[loss] = 1
            
            if loss.split('_')[0] == 'kl':                                      
                if self.LAMBDA_KL != 0.0:
                    self._losses_func[loss] = KLLoss()
                    self._params[loss] = self.LAMBDA_KL
            elif loss.split('_')[0] == 'recons':                                
                self._losses_func[loss] = torch.nn.SmoothL1Loss(
                    reduction='mean')
                self._params[loss] = self.LAMBDA_REC
            elif loss.split('_')[-1] == 'displacement':
                self._losses_func[loss] = torch.nn.SmoothL1Loss(
                    reduction='mean')
                self._params[loss] = self.LAMBDA_REC
            elif loss.split('_')[0] == 'gen':                                   
                self._losses_func[loss] = torch.nn.SmoothL1Loss(
                    reduction='mean')
                self._params[loss] = self.LAMBDA_GEN
            elif loss.split('_')[0] == 'latent':                                   
                self._losses_func[loss] = torch.nn.SmoothL1Loss(
                    reduction='mean')
                self._params[loss] = self.LAMBDA_LATENT
            else: ValueError("This loss is not recognized.")
            
            if loss.split('_')[-1] == 'joints':                                 
                self._params[loss] = self.LAMBDA_JOINT

    def update(self, rs_set, audio_ablation):
        total: float = 0.0
        # Compute the losses
        # Compute instance loss
        if self.stage in ["vae", "vae_diffusion"]:   
            total += self._update_loss("recons_feature", rs_set['m_rst'],
                                       rs_set['m_ref'])
            if self.use_recons_joints:
                total += self._update_loss("recons_joints", rs_set['joints_rst'],    
                                        rs_set['joints_ref'])
            total += self._update_loss("kl_motion", rs_set['dist_m'], rs_set['dist_ref'])

        if self.stage in ["diffusion", "vae_diffusion"]:                       
            # predict noise
            if self.predict_epsilon:
                total += self._update_loss("inst_loss", rs_set['noise_pred'],
                                           rs_set['noise'])
            # predict x
            else:
                total += self._update_loss("x_loss", rs_set['pred'],
                                           rs_set['latent'])

        if self.stage in ["vae_diffusion"]:                                    
            if self.train_lpdm_version == "v0": 
                # noise+text_emb => diff_reverse => latent => decode => motion
                total += self._update_loss("gen_feature", rs_set['gen_m_rst'],
                                        rs_set['m_ref'])
                if self.use_recons_joints:
                    total += self._update_loss("gen_joints", rs_set['gen_joints_rst'],
                                            rs_set['joints_ref'])
            elif self.train_lpdm_version == "v1":
                total += self._update_loss("latent_feature", rs_set['lat_rm'],
                                           rs_set['lat_m'])
                
        if self.stage in ["vae_diffusion"] and self.vtex_displacement:                                    
            # vtex displacement
            genders = [x[1] for x in rs_set['attr']]
            persons = [x[0] for x in rs_set['attr']]
            m_ref_vertices = self._get_vertices(rs_set['m_ref_3D'], genders, persons, audio_ablation)
            m_rst_vertices = self._get_vertices(rs_set['m_rst_3D'], genders, persons, audio_ablation)
            gen_m_rst_vertices = self._get_vertices(rs_set['gen_m_rst_3D'], genders, persons, audio_ablation)
            
            total += self._update_loss("rec_vtex_displacement", m_rst_vertices,
                                        m_ref_vertices)
            total += self._update_loss("gen_vtex_displacement", gen_m_rst_vertices,
                                        m_ref_vertices)

        self.total += total.detach()
        self.count += 1
        
        return total

    def compute(self, split=None):
        count = getattr(self, "count")
        return {loss: getattr(self, loss) / count for loss in self.losses}

    def _update_loss(self, loss: str, outputs, inputs):
        # Update the loss
        val = self._losses_func[loss](outputs, inputs)
        getattr(self, loss).__iadd__(val.detach())
        # Return a weighted sum
        weighted_loss = self._params[loss] * val
        return weighted_loss

    def loss2logname(self, loss: str, split: str):
        if loss == "total":
            log_name = f"{loss}/{split}"
        else:
            loss_type, name = loss.split("_")
            log_name = f"{loss_type}/{name}/{split}"
        return log_name
    
    @torch.no_grad()
    def _get_vertices(self, joints, genders, persons, audio_ablation):
        
        poses, transes = joints["poses"].to(dtype=torch.float64), joints["trans"].to(dtype=torch.float64)
        assert len(genders) == len(persons), f"[LATPRIOR LOSS] len(genders) {len(genders)} len(persons) {len(persons)}"
        all_betas = []
        for person in persons:
            _, betas = subject2genderbeta(person)
            all_betas.append(betas)
        betas = torch.tensor(all_betas).to(self.d, dtype=torch.float64)

        if audio_ablation == "v1":
            # BEAT SMPLX FLAME 2020 Neutral model data
            
            old_bs, seq = poses.shape[:2]
            poses = rearrange(poses, "b f j c -> (b f) (j c)")
            transes = rearrange(transes, "b f c -> (b f) c")
            new_bs = poses.shape[0]
            expression = torch.zeros((new_bs, 10), dtype=torch.float64).to(self.d)
            betas = betas.unsqueeze(1).repeat(1, seq, 1).reshape(-1, 300)
            
            smplx_model = self.smplx_models["neutral"].to(self.d, dtype=torch.float64)
            output = self.smpl_forward(smplx_model, betas, poses, transes, expression)
            vertices_batch = output.vertices.reshape(old_bs, seq, -1, 3)
        
        else: 
            assert audio_ablation == "v0", "audio_ablation must be v1 or v2"
            # BEAT 3DV submission gendered data
            
            smplx_male, smplx_female = self.smplx_models["male"].to(self.d, dtype=torch.float64), \
                                       self.smplx_models["female"].to(self.d, dtype=torch.float64)
            
            male_count, female_count = genders.count("male"), genders.count("female")
            male_idx, female_idx = [i for i, x in enumerate(genders) if x == "male"], \
                                   [i for i, x in enumerate(genders) if x == "female"]
            
            male_vertices, female_vertices = None, None
            for count, idx, model, gender in zip([male_count, female_count], [male_idx, female_idx], \
                                                [smplx_male, smplx_female], ["male", "female"]):
                if count != 0:
                    poses_i, transes_i = poses[idx], transes[idx]
                    betas_i = betas[idx]
                    old_bs, seq = poses_i.shape[:2]
                    
                    poses_i = rearrange(poses_i, "b f j c -> (b f) (j c)")
                    transes_i = rearrange(transes_i, "b f c -> (b f) c")
                    new_bs = poses_i.shape[0]
                    expression = torch.zeros((new_bs, 10), dtype=torch.float64).to(self.d)
                    betas_i = betas_i.unsqueeze(1).repeat(1, seq, 1).reshape(-1, 300)
                            
                    smplx_model = self.smplx_models["neutral"].to(self.d, dtype=torch.float64)
                    output = self.smpl_forward(model, betas_i, poses_i, transes_i, expression)
                    if gender == "male": male_vertices = output.vertices.reshape(old_bs, seq, -1, 3)
                    elif gender == "female": female_vertices = output.vertices.reshape(old_bs, seq, -1, 3)

            # combine male_vertices and female_vertices to vertices_batch using male_idx, female_idx
            vertices_batch = []
            for i, g in enumerate(genders):
                if g == "male": vertices_batch.append(male_vertices[male_idx.index(i)])
                elif g == "female": vertices_batch.append(female_vertices[female_idx.index(i)])
            vertices_batch = torch.stack(vertices_batch, dim=0)
        
        return vertices_batch
    
    def smpl_forward(self, smplx_model, betas, poses, transes, expression):
        return  smplx_model(
                    betas=betas,
                    global_orient=poses[:, :3],
                    body_pose=poses[:, 3:66],
                    jaw_pose=poses[:, 66:69],
                    leye_pose=poses[:, 69:72],
                    reye_pose=poses[:, 72:75],
                    left_hand_pose=poses[:, 75:120],
                    right_hand_pose=poses[:, 120:165],
                    transl=transes,
                    expression=expression,
                    return_verts=True,
        )

class KLLoss:

    def __init__(self):
        pass

    def __call__(self, q, p):
        div = torch.distributions.kl_divergence(q, p)
        return div.mean()

    def __repr__(self):
        return "KLLoss()"

class KLLossMulti:

    def __init__(self):
        self.klloss = KLLoss()

    def __call__(self, qlist, plist):
        return sum([self.klloss(q, p) for q, p in zip(qlist, plist)])

    def __repr__(self):
        return "KLLossMulti()"
