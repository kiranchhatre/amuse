from .audio import EmotionNet, AutoEncoder2x, AST_EVP, Pretrained_AST_EVP
from .latent_diffusion import MotionPrior, LatentDiffusionModel

allmodels = {
    "wav_mfcc": EmotionNet(),
    # "wav_dtw_mfcc": AutoEncoder2x(),
    "wav_dtw_mfcc": AST_EVP(),
    "motionprior": MotionPrior(),
    "latent_diffusion": LatentDiffusionModel(),
}