"""modelファイルの実体化"""

from .csn import ConditionalSimNet2d, ConditionalSimNet1d

from .tripletnet import CS_Tripletnet

from .unet5.model_csn_640 import UNetcsn
from .unet5.model_csn_640_de5 import UNetcsnde5
from .unet5.model_normal import UNetNormal
from .unet5.model_notcsn_640_de5 import UNetnotcsnde5
from .unet5.model_unet5_1d_de5 import UNet5_1d_de5, main as model_unet5_1d_de5

from .waveunet.model_waveunet5 import WaveUNet, main as model_waveunet5

from .triplet.model_triplet_csn_640_de5 import UNetForTriplet640De5
from .triplet.model_triplet_csn_640_de1 import UNetForTriplet640De1
from .triplet.model_triplet_128_de1 import UNetForTriplet128De1
from .triplet.model_triplet_inst import UNetForTripletInst
from .triplet.model_triplet_1d_de5_embnet import UNetForTriplet_1d_de5_embnet
from .triplet.model_triplet_1d_de5_ae_embnet import main as model_triplet_1d_de5_ae_embnet, AEForTriplet_1d_de5_embnet
from .triplet.model_triplet_csn640_de5_to1d_embedding import main as model_triplet_csn640_de5_to1d_embedding, UNetForTriplet_2d_de5_embnet
from .triplet.model_triplet_to1d_embnet_silence import main as model_triplet_to1d_embnet_silence, UNetForTriplet_2d_de5_embnet_silence
from .triplet.model_triplet_2d_de5_to1d_embnet_lastmean import main as model_triplet_2d_de5_to1d_embnet_lastmean, UNetForTriplet_2d_de5_embnet_lastmean
from .triplet.model_nnet import main as model_nnet, NNet
from .triplet.model_triplet_2d_csn640de5_to1d640 import UNetForTriplet_2d_de5_to1d640
from .triplet.pretrain import PreTrain
from .triplet.pretrain_32 import PreTrain32
from .triplet.triplet import Triplet
from .triplet.model_triplet_csn640_to1d640_1dde5 import UNetForTriplet_to1d640_1dde5
from .triplet.model_triplet_to1d640_1dde1_embedding import UNetForTriplet_to1d640_1dde1_embnet

from .to1d.model_avgp import AVGPooling
from .to1d.model_linear import To1D640
from .to1d.model_embedding import EmbeddingNet

from .AE.model_ae import main as model_ae, AE

from .demucs.model_demucs import Demucs, main as model_demucs
from .demucs.model_demucs_v2_original import Demucs_v2_original, main as model_demucs_v2_original
from .demucs.model_hdemucs import HDemucs, main as model_hdemucs

from .jnet.model_jnet_128_embnet import JNet128Embnet
from .jnet.model_jnet_attention import JNet128Attention
from .jnet.jnet import JNet
from .jnet.jnet_validknn import JNetValidKnn

__all__ = ["ConditionalSimNet2d",
            "ConditionalSimNet1d"
            "UNetcsnde5",
            "UNetNormal",
            "UNetForTriplet640De5",
            "UNetForTriplet640De1",
            "UNetForTriplet128De1",
            "AVGPooling",
            "To1D",
            "UNetnotcsnde5",
            "UNetForTripletInst",
            "model_ae",
            "AE"]
