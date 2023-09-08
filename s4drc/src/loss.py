from functools import reduce
from typing import Literal, get_args

import numpy as np
import torch
import torch.nn as nn
from auraloss.freq import MultiResolutionSTFTLoss
from auraloss.perceptual import FIRFilter
from auraloss.time import DCLoss, ESRLoss
from scipy import linalg
from torch import Tensor
from tqdm import tqdm

LossType = Literal['MAE', 'MSE', 'ESR', 'DC', 'Multi-STFT',
                   'ESR+DC', 'ESR+DC+Multi-STFT', 'MAE+Multi-STFT', 'MAE+ESR+DC+Multi-STFT']


class Sum(nn.Module):
    losses: nn.ModuleList

    def __init__(self, *losses: nn.Module):
        super().__init__()
        self.losses = nn.ModuleList(losses)

    def forward(self, y_hat: Tensor, y: Tensor) -> Tensor:
        return reduce(
            lambda a, b: a + b,
            (loss(y_hat, y) for loss in self.losses),
        )


class PreEmphasisESRLoss(nn.Module):
    def __init__(self, filter_coef: float | None):
        super().__init__()
        if filter_coef is not None and 0 < filter_coef < 1:
            self.pre_emphasis_filter = FIRFilter('hp', filter_coef, 44100)
        else:
            self.pre_emphasis_filter = None
        self.esr = ESRLoss()

    def forward(self, y_hat: Tensor, y: Tensor) -> Tensor:
        if self.pre_emphasis_filter:
            y_hat, y = self.pre_emphasis_filter(y_hat, y)
        return self.esr(y_hat, y)
    

class FrechetAudioDistance:
    def __init__(
        self, 
        model_name="vggish", 
        use_pca=False, 
        use_activation=False, 
        verbose=False, 
    ):
        self.model_name = model_name
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.__get_model(model_name=model_name, use_pca=use_pca, use_activation=use_activation)
        self.verbose = verbose
    
    def __get_model(self, model_name="vggish", use_pca=False, use_activation=False):
        """
        There are two optional models vggish and PANN for embedding. Adopt vggish in this code.
        Params:
        -- x   : Either 
            (i) a string which is the directory of a set of audio files, or
            (ii) a np.ndarray of shape (num_samples, sample_length)
        """
        if model_name == "vggish":
            # S. Hershey et al., "CNN Architectures for Large-Scale Audio Classification", ICASSP 2017
            self.model = torch.hub.load('harritaylor/torchvggish', 'vggish')
            if not use_pca:
                self.model.postprocess = False
            if not use_activation:
                self.model.embeddings = nn.Sequential(*list(self.model.embeddings.children())[:-1])

        self.model.eval()
    
    def get_embeddings(self, x, sr: int = 44_100):
        """
        Get embeddings using VGGish model.
        Params:
        -- x    : a list of np.ndarray audio samples
        -- sr   : Sampling rate, if x is a list of audio samples. Default value is 16000.
        """
        embd_lst = []
        try:
            for audio in tqdm(x, disable=(not self.verbose)):

                if self.model_name == "vggish":
                    embd = self.model.forward(audio, sr)
                elif self.model_name == "pann":
                    with torch.no_grad():
                        out = self.model(torch.tensor(audio).float().unsqueeze(0), None)
                        embd = out['embedding'].data[0]
                if self.device == torch.device('cuda'):
                    embd = embd.cpu()
                embd = embd.detach().numpy()
                embd_lst.append(embd)
        except Exception as e:
            print("[Frechet Audio Distance] get_embeddings throw an exception: {}".format(str(e)))
        res = np.concatenate(embd_lst, axis=0)

        return res
    
    def calculate_embd_statistics(self, embd_lst):
        if isinstance(embd_lst, list):
            embd_lst = np.array(embd_lst)
        mu = np.mean(embd_lst, axis=0)
        sigma = np.cov(embd_lst, rowvar=False)
        return mu, sigma
    
    def calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        """
        Adapted from: https://github.com/mseitzer/pytorch-fid/blob/master/src/pytorch_fid/fid_score.py
        
        Numpy implementation of the Frechet Distance.
        The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
        and X_2 ~ N(mu_2, C_2) is
                d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
        Stable version by Dougal J. Sutherland.
        Params:
        -- mu1   : Numpy array containing the activations of a layer of the
                inception net (like returned by the function 'get_predictions')
                for generated samples.
        -- mu2   : The sample mean over activations, precalculated on an
                representative data set.
        -- sigma1: The covariance matrix over activations for generated samples.
        -- sigma2: The covariance matrix over activations, precalculated on an
                representative data set.
        Returns:
        --   : The Frechet Distance.
        """

        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)
        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape, \
            'Training and test mean vectors have different lengths'
        assert sigma1.shape == sigma2.shape, \
            'Training and test covariances have different dimensions'

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return (diff.dot(diff) + np.trace(sigma1)
                + np.trace(sigma2) - 2 * tr_covmean)

    def score(
        self, 
        original:np.ndarray, 
        target:np.ndarray, 
        store_embds=False,
        dtype="float32"
    ):
        original = original.squeeze()
        target = target.squeeze()
        assert len(original.shape) == 2 and len(target.shape) == 2
        
        
        embds_background = self.get_embeddings(original)
        
        embds_eval = self.get_embeddings(target)


        if len(embds_background) == 0:
            print("[Frechet Audio Distance] background set dir is empty, exitting...")
            return -1
        if len(embds_eval) == 0:
            print("[Frechet Audio Distance] eval set dir is empty, exitting...")
            return -1
        
        mu_background, sigma_background = self.calculate_embd_statistics(embds_background)
        mu_eval, sigma_eval = self.calculate_embd_statistics(embds_eval)
        fad_score = self.calculate_frechet_distance(
            mu_background, 
            sigma_background, 
            mu_eval, 
            sigma_eval
        )

        return fad_score


def forge_loss_criterion_by(loss_type: LossType, filter_coef: float) -> nn.Module:
    if loss_type == 'MAE':
        return nn.L1Loss()
    if loss_type == 'MSE':
        return nn.MSELoss()
    if loss_type == 'ESR':
        return PreEmphasisESRLoss(filter_coef)
    if loss_type == 'DC':
        return DCLoss()
    if loss_type == 'Multi-STFT':
        return MultiResolutionSTFTLoss()

    if loss_type == 'ESR+DC':
        return Sum(PreEmphasisESRLoss(filter_coef), DCLoss())

    if loss_type == 'MAE+Multi-STFT':
        return Sum(MultiResolutionSTFTLoss(), nn.L1Loss())
    if loss_type == 'ESR+DC+Multi-STFT':
        return Sum(PreEmphasisESRLoss(filter_coef), DCLoss(), MultiResolutionSTFTLoss())
    if loss_type == 'MAE+ESR+DC+Multi-STFT':
        return Sum(PreEmphasisESRLoss(filter_coef), DCLoss(), MultiResolutionSTFTLoss(), nn.L1Loss())

    raise ValueError(f'Unsupported loss type `{loss_type}`.')


def forge_validation_criterions_by(
    filter_coef: float, *loss_to_keep: LossType,
) -> nn.ModuleDict:
    return nn.ModuleDict({
        loss_type: forge_loss_criterion_by(
            loss_type, filter_coef).eval()
        for loss_type in get_args(LossType)
        if '+' not in loss_type or loss_type in loss_to_keep
    })
