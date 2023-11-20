import torch
import torchaudio.functional as Fa
import time
import librosa.core as lc
import librosa
import os
#import museval
import numpy as np
from torchmetrics.audio import SignalDistortionRatio as SDR, ScaleInvariantSignalDistortionRatio as SISDR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import dask.array as da

# GPUが使用可能かどうか判定、使用可能なら使用する
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"\n=== Using {device}({__name__}). ===\n")

def trackname(no):
    if no in range(1, 10):
        track_name = "Track0000{}".format(no)
    elif no in range(10, 100):
        track_name = "Track000{}".format(no)
    elif no in range(100, 1000):
        track_name = "Track00{}".format(no)
    elif no in range(1000, 2101):
        track_name = "Track0{}".format(no)
    return track_name

def detrackname(name):
    if name[6] != "0":
        no = name[6:]
    elif name[7] != "0":
        no = name[7:]
    elif name[8] != "0":
        no = name[8:]
    elif name[9] != "0":
        no = name[9:]
    return int(no)

def time2hms(time):
    hour = time//3600
    min = (time%3600)//60
    sec = (time%3600)%60
    if hour == 0:
        return f"{int(min):0>2}:{int(sec):0>2}"
    else:
        return f"{int(hour):0>2}:{int(min):0>2}:{int(sec):0>2}"

class progress_bar():
    def __init__(self, name: str, total: int) -> None:
        self.total = total
        self.progress = 0
        self.rate = 0
        print(f"{name} : [", end="")
    
    def update(self, n: int):
        self.progress += n
        while self.rate < self.progress / self.total:
            self.rate += 0.01
            print("#", end="")
            if self.rate >= 1:
                print("]")
                return

def standardize(spec):
    std, mean = torch.std_mean(spec) #平均、標準偏差
    if -1e-4 < std and std < 1e-4:
        transformed = torch.zeros_like(spec) # 分散が0 = 元音源が無音 -> NaNを0に
        mean = torch.zeros_like(mean); std = torch.zeros_like(std) #分散、平均も0に
    else:
        transformed = (spec - mean) / std
    #print(mean, std)
    return mean, std, transformed

def destandardize(spec, mean, std):
    #print(mean, std)
    spec_ = torch.zeros_like(spec)
    for i in range(len(mean)):
        spec_[i] = spec[i] * std[i].item() + mean[i].item()
    return spec_

def normalize(spec, max = None, min = None):
    if max is None and min is None:
        max = torch.max(spec)
        min = torch.min(spec)
    if -1e-4 < max - min and max - min < 1e-4:
        transformed = torch.zeros_like(spec) # 分散が0 = 元音源が無音 -> NaNを0に
        max = torch.zeros_like(max); min = torch.zeros_like(min) #分散、平均も0に
    else:
        transformed = (spec - min) / (max - min)
    return max, min, transformed

def denormalize(max, min, spec):
    return spec * (max - min) + min

def l2normalize(vec):
    norm = np.linalg.norm(vec, ord=2)
    if norm <= 0.001:
        norm = 1.0
    return vec / norm


def nan_checker(array):
    """
    input ndarray
    '''
    return
    配列にNaNが含まれない   -> True
    配列にNaNが含まれる     -> False
    """
    if torch.any(torch.isnan(array)):
        return "Have NaN."
    else:
        return "Not Have NaN."

def start():
    if device == "cuda":
        torch.cuda.synchronize()
    start_time = time.time() # 時間計測開始
    return start_time

def finish(start_time):
    if device == "cuda":
        torch.cuda.synchronize()
    finish_time = time.time() - start_time # 時間計測終了
    return finish_time

def get_inner_prod_md(u_s, v_s):
    
    # get parameters
    n, m = np.shape(u_s) # number and dimension of vector
    
    # prepare output array
    inner_prod = np.zeros([n]) # <1d array [n]>
    
    # loop for dimension
    for j in range(m):
        inner_prod += u_s[:,j] * v_s[:,j]
    
    # end 
    return inner_prod

def inner_prod(x, y):
    """ラストの2次元で内積を計算する場合のみ。xは2次元を仮定。"""
    x_shape = x.shape; y_shape = y.shape
    row = x_shape[-2]; col = y_shape[-1]
    y = y.reshape(-1, y_shape[-2], col) # ラスト2次元以外を一つにまとめる
    dx = da.from_array(x)
    #y = y.reshape(y_shape[-2], -1) # ラスト2次元以外を一つにまとめる
    out = []
    for i in range(y.shape[0]):
        dy = da.from_array(y)
        out.append(da.dot(dx, dy).compute())
    out = out.reshape(y_shape[:-2] + (row, col))
    return out

def complex_norm(
        complex_tensor,
        power: float = 1.0
):
    r"""Compute the norm of complex tensor input.

    Args:
        complex_tensor (Tensor): Tensor shape of `(..., complex=2)`
        power (float): Power of the norm. (Default: `1.0`).

    Returns:
        Tensor: Power of the normed input tensor. Shape of `(..., )`
    """

    # Replace by torch.norm once issue is fixed
    # https://github.com/pytorch/pytorch/issues/34279
    #out = torch.view_as_real(complex_tensor).pow(2.).sum(-1).pow(0.5 * power)
    out = torch.view_as_real(complex_tensor).pow(2.).sum(-1).pow(power)
    return out

def angle(
        complex_tensor
):
    r"""Compute the angle of complex tensor input.

    Args:
        complex_tensor (Tensor): Tensor shape of `(..., complex=2)`

    Return:
        Tensor: Angle of a complex tensor. Shape of `(..., )`
    """
    #real = torch.view_as_real(complex_tensor)
    return torch.atan2(complex_tensor[..., 1], complex_tensor[..., 0])
    #return complex_tensor.angle()

def magphase(
        complex_tensor,
        power: float = 1.0
):
    r"""Separate a complex-valued spectrogram with shape `(..., 2)` into its magnitude and phase.

    Args:
        complex_tensor (Tensor): Tensor shape of `(..., complex=2)`
        power (float): Power of the norm. (Default: `1.0`)

    Returns:
        (Tensor, Tensor): The magnitude and phase of the complex tensor
    """
    #mag = complex_norm(complex_tensor, power)
    mag = torch.abs(complex_tensor)
    zeros_to_ones = torch.where(mag == 0, 1.0, 0.0)
    mag_nonzero = mag + zeros_to_ones
    phase = torch.empty_like(complex_tensor, dtype=torch.complex64, device=mag.device)
    phase.real = complex_tensor.real / mag_nonzero + zeros_to_ones # 無音ならphaseは1+j0
    phase.imag = complex_tensor.imag / mag_nonzero
    return mag, phase

class TorchSTFT:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        if cfg.mel:
            self.melfilter = torch.from_numpy(np.squeeze(librosa.filters.mel(sr=cfg.sr, n_mels=cfg.n_mels, n_fft=cfg.f_size)))
        
    def stft(self, wave):
        *other, length = wave.shape
        x = wave.reshape(-1, length)
        z = torch.stft(
                x,
                n_fft=self.cfg.f_size,
                hop_length=self.cfg.hop_length,
                window=torch.hann_window(self.cfg.f_size).to(wave),
                win_length=self.cfg.f_size,
                normalized=False,
                center=True,
                return_complex=True,
                pad_mode='reflect')
        _, freqs, frame = z.shape
        return z.view(*other, freqs, frame)

    def normalize(self, data, max = None, min = None):
        if max is None:# and min is None:
            max = data.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
        min = data.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
        max_min = max - min
        max_min = torch.where(max_min == 0, 1, max_min) # 0なら1に変換
        transformed = (data - min) / max_min
        return transformed, max, min

    def transform(self, sound, param=None):
        transformed, phase = magphase(self.stft(sound)) #stftして振幅と位相に分解
        del sound
        if self.cfg.mel:
            transformed = torch.matmul(self.melfilter.to(transformed.device), transformed)
        if self.cfg.db:
            transformed = Fa.amplitude_to_DB(transformed, 20, amin=1e-05, db_multiplier=0)
        if param is None:
            transformed, max, min = self.normalize(transformed) #正規化
            params = torch.stack([max, min], dim=0)
        else:
            # instはmixのmax,minの値で正規化する
            transformed_list = []
            for i in range(len(self.cfg.inst_list)):
                #print(transformed_n.shape)
                transformed_per, _, _ = self.normalize(transformed[:,i], max=param[0], min=param[1]) #正規化
                transformed_list.append(transformed_per)
            transformed = torch.stack(transformed_list, dim=1)
            params = param
        return params, transformed, phase
    
    def denormalize(self, spec, max, min):
        return spec * (max - min) + min
    
    def detransform(self, spec, phase, max, min):
        #print(spec.shape)
        spec_denormal = self.denormalize(spec, max, min).to("cpu").numpy() #正規化を解除
        if self.cfg.db:
            spec_denormal = librosa.db_to_amplitude(spec_denormal) #dbを元の振幅に直す
        return lc.istft(spec_denormal * phase.to("cpu").numpy(), n_fft=self.cfg.f_size, hop_length=self.cfg.hop_length)



class STFT:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        if cfg.mel:
            self.melfilter = np.squeeze(librosa.filters.mel(sr=cfg.sr, n_mels=cfg.n_mels, n_fft=cfg.f_size))
    def transform(self, sound, param=None):
        if isinstance(sound, torch.Tensor):
            sound = sound.to("cpu").numpy()
        transformed_n, phase_n = lc.magphase(lc.stft(sound, n_fft=self.cfg.f_size, hop_length=self.cfg.hop_length)) #stftして振幅と位相に分解
        if self.cfg.mel:
            #print(transformed_n.shape)
            #print(self.melfilter.shape, transformed_n.shape)
            transformed_n = inner_prod(self.melfilter, transformed_n)
            #transformed_n = np.transpose(transformed_n, axes=-2)
            #print(transformed_n.shape)
            #transformed_n = self.melfilter @ transformed_n
            #print(transformed_n.shape)
            #transformed_n = librosa.feature.melspectrogram(y=transformed_n, sr=cfg.sr, win_length=cfg.f_size, hop_length=cfg.hop_length)
        if self.cfg.db:
            transformed_n = librosa.amplitude_to_db(transformed_n)
        #transformed_n = librosa.amplitude_to_db(transformed_n) #振幅をdb変換
        transformed = torch.from_numpy(transformed_n); phase = torch.from_numpy(phase_n)
        #mean, std, transformed = standardize(transformed) #スペクトログラムを標準化
        #params = torch.tensor([mean, std])
        if param is None:
            max, min, transformed = normalize(transformed) #正規化
        else:
            _, _, transformed = normalize(transformed, max=param[0], min=param[1]) #正規化
        params = torch.tensor([max, min])
        return params, transformed, phase


def stft(sound, cfg, param = None):
    """入力の音源波形をstft→magphaseで振幅と位相に分解→正規化

    Args:
        sound_t (_type_): _description_
        f_size (int, optional): _description_. Defaults to 1024.

    Returns:
        _type_: _description_
    """
    if isinstance(sound, torch.Tensor):
        sound = sound.numpy()
    transformed_n, phase_n = lc.magphase(lc.stft(sound, n_fft=cfg.f_size, hop_length=cfg.hop_length)) #stftして振幅と位相に分解
    if cfg.mel:
        #print(transformed_n.shape)
        transformed_n = librosa.filters.mel(sr=cfg.sr, n_mels=259, n_fft=cfg.f_size) @ transformed_n
        #print(transformed_n.shape)
        #transformed_n = librosa.feature.melspectrogram(y=transformed_n, sr=cfg.sr, win_length=cfg.f_size, hop_length=cfg.hop_length)
    if cfg.db:
        transformed_n = librosa.amplitude_to_db(transformed_n)
    #transformed_n = librosa.amplitude_to_db(transformed_n) #振幅をdb変換
    transformed = torch.from_numpy(transformed_n); phase = torch.from_numpy(phase_n)
    #mean, std, transformed = standardize(transformed) #スペクトログラムを標準化
    #params = torch.tensor([mean, std])
    if param is None:
        max, min, transformed = normalize(transformed) #正規化
    else:
        max, min, transformed = normalize(transformed, max=param[0], min=param[1]) #正規化
    params = torch.tensor([max, min])
    return params, transformed, phase

def istft(spec_normal, cfg, phase, max, min):
    #spec_destandard_db = destandardize(spec_standard, mean, std).to("cpu") #標準化を解除
    spec_denormal = denormalize(max, min, spec_normal).to("cpu").numpy() #正規化を解除
    if cfg.db:
        spec_denormal = librosa.db_to_amplitude(spec_denormal) #dbを元の振幅に直す
    #if torch.all(torch.from_numpy(spec_denormal_db) == 0.0):
    #    spec_denormal_amp = spec_denormal_db
    #else:
    #    spec_denormal_amp = librosa.db_to_amplitude(spec_denormal_db) #dbを元の振幅に直す
    return lc.istft(spec_denormal * phase.to("cpu").numpy(), n_fft=cfg.f_size, hop_length=cfg.hop_length)

def file_exist(dir_path):
    # ディレクトリがない場合、作成する
    if not os.path.exists(dir_path):
        print("ディレクトリを作成します")
        os.makedirs(dir_path)

def evaluate(reference, estimate, inst_list, writer, epoch):
    # assume mix as estimates
    B, C, S, T = reference.shape
    reference = torch.reshape(reference, (B, T, C*S))
    estimate  = torch.reshape(estimate, (B, T, C*S))
    scores = {}
    for inst in inst_list:
        scores[inst] = {"SDR":0, "ISR":0, "SIR":0, "SAR":0}
    for idx, inst in enumerate(inst_list):
        # Evaluate using museval
        score = museval.evaluate(references=reference[:,:,idx*S:(idx+1)*S], estimates=estimate[:,:,idx*S:(idx+1)*S])
        #print(score)
        for i,key in enumerate(list(scores[inst].keys())):
            #print(score[i].shape)
            scores[inst][key] = np.mean(score[i])
    # print nicely formatted and aggregated scores
    sdr="SDR"; isr="ISR"; sir="SIR"; sar="SAR"
    for inst in inst_list:
        writer.add_scalar(f"{sdr}/Test", scores[inst][sdr], global_step=epoch)
        writer.add_scalar(f"{isr}/Test", scores[inst][isr], global_step=epoch)
        writer.add_scalar(f"{sir}/Test", scores[inst][sir], global_step=epoch)
        writer.add_scalar(f"{sar}/Test", scores[inst][sar], global_step=epoch)
        print(f"{inst:<10}- SDR: {scores[inst][sdr]:.3f}, ISR: {scores[inst][isr]:.3f}, SIR: {scores[inst][sir]:.3f}, SAR: {scores[inst][sar]:.3f}")

"""
def evaluate_mine(reference, estimate, inst_list):
    scores = {}
    for inst in inst_list:
        scores[inst] = {"SDR":0, "ISR":0, "SIR":0, "SAR":0}
    for idx,inst in enumerate(inst_list):
        scores[inst]["SDR"] = SDR(preds=estimate[:,idx], target=reference[:,idx])
        scores[inst]["SI-SDR"] = SISDR(preds=estimate[:,idx], target=reference[:,idx])
"""

def knn_psd(label:np.ndarray, vec:np.ndarray, cfg):
    total_all   = 0
    correct_all = 0
    knn_sk = KNeighborsClassifier(n_neighbors=5, weights="uniform", n_jobs=cfg.num_workers)
    for idx in range(len(label)):
        reduce_idx = np.where((label[:,0]==label[idx,0]) & (label[:,1]==label[idx,1]))[0].tolist() # fitする擬似楽曲と構成が同じ曲を除く
        knn_sk.fit(np.delete(vec, reduce_idx, axis=0), np.delete(label[:,0], reduce_idx, axis=0))
        pred = knn_sk.predict(vec[idx].reshape(1, -1))
        #print(label[idx], pred)
        #print(f"{inst:<10}: {metrics.accuracy_score([info_list[idx]], pred)}%")
        if label[idx,0] == pred:
            correct_all += 1
        total_all += 1
    return correct_all / total_all

def tsne_psd(label:np.ndarray, vec:np.ndarray, mode: str, cfg, dir_path:str, current_epoch=0):
    tsne_start = start()
    print(f"= T-SNE...")
    counter = 0
    num_continue = 10
    markers = [",", "o", "v", "^", "p", "D", "<", ">", "8", "*"]
    colors = ["r", "g", "b", "c", "m", "y", "k", "#ffa500", "#00ff00", "gray"]
    #cmap = plt.cm.get_cmap("tab20")
    #label20 = []
    num_songs = 10
    color10 = []
    marker10 = []
    label_picked = []
    vec10 = []
    while counter <= num_songs:
        pick_label = np.random.choice(label[:,0])
        if num_continue > 500:
            break
        if pick_label in label_picked:
            num_continue += 1
            continue
        samesong_idx = np.where(label[:,0]==pick_label)[0]
        samesong_vec = vec[samesong_idx]
        samesong_label = label[samesong_idx]
        #label20.append(label[samesong_idx])
        # 色を指定
        color10 = color10 + [colors[counter] for i in range(samesong_idx.shape[0])]
        # マークを指定
        counter_m = -1 # 便宜上。本当は0にしたい
        log_ver = []
        for i in range(samesong_idx.shape[0]):
            if not samesong_label[i, 1] in log_ver:
                log_ver.append(samesong_label[i, 1])
                counter_m += 1
            marker10.append(markers[counter_m])
        vec10.append(samesong_vec)
        label_picked.append(pick_label)
        counter += 1
    #color20 = np.concatenate(color20, axis=0)
    vec10 = np.concatenate(vec10, axis=0)
    perplexity = [5, 15, 30, 50]
    for i in range(len(perplexity)):
        fig, ax = plt.subplots(1, 1)
        X_reduced = TSNE(n_components=2, random_state=0, perplexity=perplexity[i]).fit_transform(vec10)
        for j in range(len(vec10)):
            mappable = ax.scatter(X_reduced[j, 0], X_reduced[j, 1], c=color10[j], marker=marker10[j], s=30)
        #fig.colorbar(mappable, norm=BoundaryNorm(bounds,cmap.N))
        file_exist(dir_path)
        fig.savefig(dir_path + f"/emb_{mode}_e{current_epoch}_s{counter}_tsne_p{perplexity[i]}_m{cfg.margin}.png")
        plt.clf()
        plt.close()
    tsne_time = finish(tsne_start)
    print(f"T-SNE was finished!")
    print(f"= T-SNE time is {tsne_time} sec. =")


def tsne_psd_marker(label:np.ndarray, vec:np.ndarray, mode: str, cfg, dir_path:str, current_epoch=0):
    tsne_start = start()
    print(f"= T-SNE...")
    counter = 0
    num_continue = 10
    markers = [",", "o", "v", "^", "p", "D", "<", ">", "8", "*"]
    colors = ["r", "g", "b", "c", "m", "y", "k", "#ffa500", "#00ff00", "gray"]
    #cmap = plt.cm.get_cmap("tab20")
    #label20 = []
    num_songs = 10
    color10 = []
    marker10 = []
    label_picked = []
    vec10 = []
    id_list = []
    ver_list = []
    for id in label[:,0]:
        if not id in id_list:
            id_list.append(id)
    for ver in label[:,1]:
        if not ver in ver_list:
            ver_list.append(ver)
            
    #color20 = np.concatenate(color20, axis=0)
    #vec10 = np.concatenate(vec10, axis=0)
    perplexity = [5, 15, 30, 50]
    for i in range(len(perplexity)):
        fig, ax = plt.subplots(1, 1)
        X_reduced = TSNE(n_components=2, random_state=0, perplexity=perplexity[i]).fit_transform(vec)
        for j in range(len(vec)):
            mappable = ax.scatter(X_reduced[j, 0], X_reduced[j, 1], c=colors[id_list.index(label[j,0])], marker=markers[ver_list.index(label[j,1])], s=30)
        #fig.colorbar(mappable, norm=BoundaryNorm(bounds,cmap.N))
        file_exist(dir_path)
        fig.savefig(dir_path + f"/emb_{mode}_e{current_epoch}_s{counter}_tsne_p{perplexity[i]}_m{cfg.margin}.png")
        plt.clf()
        plt.close()
    tsne_time = finish(tsne_start)
    print(f"T-SNE was finished!")
    print(f"= T-SNE time is {tsne_time} sec. =")


if "__main__" == __name__:
    pass