from utils.func import TorchSTFT
from dataset.dataset_zume import TripletLoader
import hydra
import random
import matplotlib.pyplot as plt
import torch
import soundfile
import numpy as np
from torch.utils.data import Dataset, DataLoader

@hydra.main(version_base=None, config_path="./configs", config_name="kari")
def main(cfg):
    dataset = TripletLoader(cfg=cfg.train, mode="valid")
    loader = DataLoader(
            dataset=dataset,
            batch_size=10,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
        )
    stft_mel = hydra.utils.instantiate(cfg.stft_mel)
    stft_no_mel = hydra.utils.instantiate(cfg.stft_no_mel)
    mix_a, stems_a, mix_p, stems_p, mix_n, stems_n, c = next(iter(loader))
    #target = stems_a[2]
    target = mix_a; stems = stems_a
    param, spec_m, phase = stft_no_mel.transform(target)
    _, spec_s,_ = stft_no_mel.transform(stems, param)
    print(c)
    #param_m, mel_m, _ = stft_mel.transform(target)
    #_, mel_s, _ = stft_mel.transform(stems, param_m)
    #print(spec_s.shape, mel_s.shape)
    b = 0
    fig3, ax3 = plt.subplots(2, 3, layout="constrained")
    ax = ax3[0, 0].imshow(
        #torch.squeeze(spec_s[5, c[5]]),
        torch.squeeze(spec_m[b]),
        origin="lower",
        aspect="auto",
        cmap="plasma",
    )
    ax3[0, 0].set_title("mix")
    ax = ax3[0, 1].imshow(
        #torch.squeeze(mel_s[5, c[5]]),
        torch.squeeze(spec_s[b, 0]),
        origin="lower",
        aspect="auto",
        cmap="plasma",
    )
    ax3[0, 1].set_title("drums")
    ax = ax3[0, 2].imshow(
        #torch.squeeze(mel_s[5, c[5]]),
        torch.squeeze(spec_s[b, 1]),
        origin="lower",
        aspect="auto",
        cmap="plasma",
    )
    ax3[0, 2].set_title("bass")
    ax = ax3[1, 0].imshow(
        #torch.squeeze(mel_s[5, c[5]]),
        torch.squeeze(spec_s[b, 2]),
        origin="lower",
        aspect="auto",
        cmap="plasma",
    )
    ax3[1, 0].set_title("piano")
    ax = ax3[1, 1].imshow(
        #torch.squeeze(mel_s[5, c[5]]),
        torch.squeeze(spec_s[b, 3]),
        origin="lower",
        aspect="auto",
        cmap="plasma",
    )
    ax3[1, 1].set_title("guitar")
    ax = ax3[1, 2].imshow(
        #torch.squeeze(mel_s[5, c[5]]),
        torch.squeeze(spec_s[b, 4]),
        origin="lower",
        aspect="auto",
        cmap="plasma",
    )
    ax3[1, 2].set_title("residuals")
    fig3.savefig(f"./comfirm/ba4_1/mix_a_{c[b]}.png")
    print(torch.sum((spec_m[b] - torch.sum(spec_s[b], dim=1))**2))
    #print(spec_s.shape, phase.shape)
    #wave = stft_no_mel.detransform(spec_m[5], phase[5], param[0, 5], param[1, 5])
    #print(wave.shape)
    #soundfile.write(f"./comfirm/ba4_1/mix_a_{c[5]}.wav", np.squeeze(wave), samplerate=cfg.train.sr)

if __name__ == "__main__":
    main()
