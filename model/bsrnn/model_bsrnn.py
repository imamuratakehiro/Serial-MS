import torch
import torch.nn as nn
import typing as tp

def get_fftfreq(
        sr: int = 44100,
        n_fft: int = 2048
) -> torch.Tensor:
    """
    Torch workaround of librosa.fft_frequencies
    srとn_fftから求められるstftの結果の周波数メモリを配列にして出力。
    0から始まり、最後が22050。
    """
    out = sr * torch.fft.fftfreq(n_fft)[:n_fft // 2 + 1]
    out[-1] = sr // 2
    return out


def get_subband_indices(
        freqs: torch.Tensor,
        splits: tp.List[tp.Tuple[int, int]],
) -> tp.List[tp.Tuple[int, int]]:
    """
    Computes subband frequency indices with given bandsplits
    1. 入力で[end_freq, step]の組みが与えられる。
    2. stepの周波数幅でスペクトログラムをband splitする。
    3. end_freqがstepの値の区切り目で、達すると次のend_freqとstepに切り替わる。

    freqs_splits = [
        (1000, 100),
        (4000, 250),
        (8000, 500),
        (16000, 1000),
        (20000, 2000),
    ]
    上記のsplitsが与えられた場合、
    start=0(Hz)     -> end=1000(Hz)の間を100(Hz)でband split
    start=1000(Hz)  -> end=4000(Hz)の間を250(Hz)でband split

    (以下略)

    start=16000(Hz) -> end=20000(Hz)の間を2000(Hz)でband split
    最後はstart=20000(Hz)で、残りの余った周波数部分を1bandとする。
    """
    indices = []
    start_freq, start_index = 0, 0
    for end_freq, step in splits:
        bands = torch.arange(start_freq + step, end_freq + step, step)
        start_freq = end_freq
        for band in bands:
            end_index = freqs[freqs < band].shape[0]
            indices.append((start_index, end_index))
            start_index = end_index
    indices.append((start_index, freqs.shape[0]))
    return indices

def freq2bands(
        bandsplits: tp.List[tp.Tuple[int, int]],
        sr: int = 44100,
        n_fft: int = 2048
) -> tp.List[tp.Tuple[int, int]]:
    """
    Returns start and end FFT indices of given bandsplits
    1. 入力で[end_freq, step]の組みが与えられる。
    2. stepの周波数幅でスペクトログラムをband splitする。
    3. end_freqがstepの値の区切り目で、達すると次のend_freqとstepに切り替わる。

    freqs_splits = [
        (1000, 100),
        (4000, 250),
        (8000, 500),
        (16000, 1000),
        (20000, 2000),
    ]
    上記のsplitsが与えられた場合、
    start=0(Hz)     -> end=1000(Hz)の間を100(Hz)でband split
    start=1000(Hz)  -> end=4000(Hz)の間を250(Hz)でband split

    (以下略)

    start=16000(Hz) -> end=20000(Hz)の間を2000(Hz)でband split
    最後はstart=20000(Hz)で、残りの余った周波数部分を1bandとする。
    """
    freqs = get_fftfreq(sr=sr, n_fft=n_fft)
    band_indices = get_subband_indices(freqs, bandsplits)
    return band_indices

class BandSplitModule(nn.Module):
    """
    BandSplit (1st) Module of BandSplitRNN.
    Separates input in k subbands and runs through LayerNorm+FC layers.
    """

    def __init__(
            self,
            sr: int,
            n_fft: int,
            bandsplits: tp.List[tp.Tuple[int, int]],
            t_timesteps: int = 517,
            fc_dim: int = 128,
            complex_as_channel: bool = True,
            is_mono: bool = False,
    ):
        super(BandSplitModule, self).__init__()

        frequency_mul = 1
        if complex_as_channel:
            frequency_mul *= 2
        if not is_mono:
            frequency_mul *= 2

        self.cac = complex_as_channel
        self.is_mono = is_mono
        self.bandwidth_indices = freq2bands(bandsplits, sr, n_fft)
        self.layernorms = nn.ModuleList([
            nn.LayerNorm([(e - s) * frequency_mul, t_timesteps])
            for s, e in self.bandwidth_indices
        ])
        self.fcs = nn.ModuleList([
            nn.Linear((e - s) * frequency_mul, fc_dim)
            for s, e in self.bandwidth_indices
        ])

    def generate_subband(
            self,
            x: torch.Tensor
    ) -> tp.Iterator[torch.Tensor]:
        for start_index, end_index in self.bandwidth_indices:
            yield x[:, :, start_index:end_index]

    def forward(self, x: torch.Tensor):
        """
        Input: [batch_size, n_channels, freq, time]
        Output: [batch_size, k_subbands, time, fc_output_shape]
        """
        xs = []
        for i, x in enumerate(self.generate_subband(x)):
            B, C, F, T = x.shape
            # view complex as channels
            if x.dtype == torch.cfloat:
                x = torch.view_as_real(x).permute(0, 1, 4, 2, 3)
            # from channels to frequency
            x = x.reshape(B, -1, T) # [B, frequency_mul*subband_step, time]
            # run through model
            x = self.layernorms[i](x)
            x = x.transpose(-1, -2) # [B, time, frequency_mul*subband_step]
            x = self.fcs[i](x) # [B, time, fc_dim]
            xs.append(x)
        return torch.stack(xs, dim=1)# [B, n_subbands, time, fc_dim]

class RNNModule(nn.Module):
    """
    RNN submodule of BandSequence module
    """

    def __init__(
            self,
            input_dim_size: int,
            hidden_dim_size: int,
            rnn_type: str = 'lstm',
            bidirectional: bool = True
    ):
        super(RNNModule, self).__init__()
        self.groupnorm = nn.GroupNorm(input_dim_size, input_dim_size)
        self.rnn = getattr(nn, rnn_type)(
            input_dim_size, hidden_dim_size, batch_first=True, bidirectional=bidirectional
        )
        self.fc = nn.Linear(
            hidden_dim_size * 2 if bidirectional else hidden_dim_size,
            input_dim_size
        )

    def forward(
            self,
            x: torch.Tensor
    ):
        """
        Input shape:
            across T - [batch_size, k_subbands, time, n_features]
            OR
            across K - [batch_size, time, k_subbands, n_features]
        """
        B, K, T, N = x.shape  # across T      across K (keep in mind T->K, K->T)

        out = x.view(B * K, T, N)  # [BK, T, N]    [BT, K, N]

        out = self.groupnorm(
            out.transpose(-1, -2)
        ).transpose(-1, -2)  # [BK, T, N]    [BT, K, N]
        out = self.rnn(out)[0]  # [BK, T, H]    [BT, K, H]
        out = self.fc(out)  # [BK, T, N]    [BT, K, N]

        x = out.view(B, K, T, N) + x  # [B, K, T, N]  [B, T, K, N]

        x = x.permute(0, 2, 1, 3).contiguous()  # [B, T, K, N]  [B, K, T, N]
        return x


class BandSequenceModelModule(nn.Module):
    """
    BandSequence (2nd) Module of BandSplitRNN.
    Runs input through n BiLSTMs in two dimensions - time and subbands.
    """

    def __init__(
            self,
            input_dim_size: int,
            hidden_dim_size: int,
            rnn_type: str = 'lstm',
            bidirectional: bool = True,
            num_layers: int = 12,
    ):
        super(BandSequenceModelModule, self).__init__()

        self.bsrnn = nn.ModuleList([])

        for _ in range(num_layers):
            rnn_across_t = RNNModule(
                input_dim_size, hidden_dim_size, rnn_type, bidirectional
            )
            rnn_across_k = RNNModule(
                input_dim_size, hidden_dim_size, rnn_type, bidirectional
            )
            self.bsrnn.append(
                nn.Sequential(rnn_across_t, rnn_across_k)
            )

    def forward(self, x: torch.Tensor):
        """
        Input shape: [batch_size, k_subbands, time, n_features]
        Output shape: [batch_size, k_subbands, time, n_features]
        """
        for i in range(len(self.bsrnn)):
            x = self.bsrnn[i](x)
        return x

class GLU(nn.Module):
    """
    GLU Activation Module.
    """
    def __init__(self, input_dim: int):
        super(GLU, self).__init__()
        self.input_dim = input_dim
        self.linear = nn.Linear(input_dim, input_dim * 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        x = self.linear(x)
        x = x[..., :self.input_dim] * self.sigmoid(x[..., self.input_dim:])
        return x


class MLP(nn.Module):
    """
    Just a simple MLP with tanh activation (by default).
    """
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            activation_type: str = 'tanh',
    ):
        super(MLP, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            self.select_activation(activation_type)(),
            nn.Linear(hidden_dim, output_dim),
            GLU(output_dim)
        )

    @staticmethod
    def select_activation(activation_type: str) -> nn.modules.activation:
        if activation_type == 'tanh':
            return nn.Tanh
        elif activation_type == 'relu':
            return nn.ReLU
        elif activation_type == 'gelu':
            return nn.GELU
        else:
            raise ValueError("wrong activation function was selected")

    def forward(self, x: torch.Tensor):
        x = self.mlp(x)
        return x


class MaskEstimationModule(nn.Module):
    """
    MaskEstimation (3rd) Module of BandSplitRNN.
    Recreates from input initial subband dimensionality via running through LayerNorms+MLPs and forms the T-F mask.
    """

    def __init__(
            self,
            sr: int,
            n_fft: int,
            bandsplits: tp.List[tp.Tuple[int, int]],
            t_timesteps: int = 517,
            fc_dim: int = 128,
            mlp_dim: int = 512,
            complex_as_channel: bool = True,
            is_mono: bool = False,
    ):
        super(MaskEstimationModule, self).__init__()

        frequency_mul = 1
        if complex_as_channel:
            frequency_mul *= 2
        if not is_mono:
            frequency_mul *= 2

        self.cac = complex_as_channel
        self.is_mono = is_mono
        self.frequency_mul = frequency_mul

        self.bandwidths = [(e - s) for s, e in freq2bands(bandsplits, sr, n_fft)]
        self.layernorms = nn.ModuleList([
            nn.LayerNorm([t_timesteps, fc_dim])
            for _ in range(len(self.bandwidths))
        ])
        self.mlp = nn.ModuleList([
            MLP(fc_dim, mlp_dim, bw * frequency_mul, activation_type='tanh')
            for bw in self.bandwidths
        ])

    def forward(self, x: torch.Tensor):
        """
        Input: [batch_size, k_subbands, time, fc_shape]
        Output: [batch_size, freq, time]
        """
        outs = []
        for i in range(x.shape[1]):
            # run through model
            out = self.layernorms[i](x[:, i])
            out = self.mlp[i](out)
            B, T, F = out.shape
            # return to complex
            if self.cac:
                out = out.view(B, -1, 2, F//self.frequency_mul, T).permute(0, 1, 3, 4, 2)
                out = torch.view_as_complex(out.contiguous())
            else:
                out = out.view(B, -1, F//self.frequency_mul, T).contiguous()
            outs.append(out)

        # concat all subbands
        outs = torch.cat(outs, dim=-2)
        return outs

class BandSplitRNN(nn.Module):
    """
    BandSplitRNN as described in paper.
    """

    def __init__(
            self,
            sr: int,
            n_fft: int,
            bandsplits: tp.List[tp.Tuple[int, int]],
            complex_as_channel: bool,
            is_mono: bool,
            bottleneck_layer: str,
            t_timesteps: int,
            fc_dim: int,
            rnn_dim: int,
            rnn_type: str,
            bidirectional: bool,
            num_layers: int,
            mlp_dim: int,
            return_mask: bool = False
    ):
        super(BandSplitRNN, self).__init__()

        # encoder layer
        self.bandsplit = BandSplitModule(
            sr=sr,
            n_fft=n_fft,
            bandsplits=bandsplits,
            t_timesteps=t_timesteps,
            fc_dim=fc_dim,
            complex_as_channel=complex_as_channel,
            is_mono=is_mono,
        )

        # bottleneck layer
        if bottleneck_layer == 'rnn':
            self.bandsequence = BandSequenceModelModule(
                input_dim_size=fc_dim,
                hidden_dim_size=rnn_dim,
                rnn_type=rnn_type,
                bidirectional=bidirectional,
                num_layers=num_layers,
            )
        #elif bottleneck_layer == 'att':
        #    self.bandsequence = BandTransformerModelModule(
        #        input_dim_size=fc_dim,
        #        hidden_dim_size=rnn_dim,
        #        num_layers=num_layers,
        #    )
        else:
            raise NotImplementedError

        # decoder layer
        self.maskest = MaskEstimationModule(
            sr=sr,
            n_fft=n_fft,
            bandsplits=bandsplits,
            t_timesteps=t_timesteps,
            fc_dim=fc_dim,
            mlp_dim=mlp_dim,
            complex_as_channel=complex_as_channel,
            is_mono=is_mono,
        )
        self.cac = complex_as_channel
        self.return_mask = return_mask

    def wiener(self, x_hat: torch.Tensor, x_complex: torch.Tensor) -> torch.Tensor:
        """
        Wiener filtering of the input signal
        """
        # TODO: add Wiener Filtering
        return x_hat

    def compute_mask(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes complex-valued T-F mask.
        """
        x = self.bandsplit(x)  # [batch_size, k_subbands, time, fc_dim]
        x = self.bandsequence(x)  # [batch_size, k_subbands, time, fc_dim]
        x = self.maskest(x)  # [batch_size, freq, time]

        return x

    def forward(self, x: torch.Tensor):
        """
        Input and output are T-F complex-valued features.
        Input shape: batch_size, n_channels, freq, time]
        Output shape: batch_size, n_channels, freq, time]
        """
        # use only magnitude if not using complex input
        x_complex = None
        if not self.cac:
            x_complex = x
            x = x.abs()
        # normalize
        # TODO: Try to normalize in bandsplit and denormalize in maskest
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        std = x.std(dim=(1, 2, 3), keepdim=True)
        x = (x - mean) / (std + 1e-5)

        # compute T-F mask
        mask = self.compute_mask(x)

        # multiply with original tensor
        x = mask if self.return_mask else mask * x

        # denormalize
        x = x * std + mean

        if not self.cac:
            x = self.wiener(x, x_complex)

        return x


if __name__ == '__main__':
    """
    batch_size, n_channels, freq, time = 2, 2, 1025, 259
    in_features = torch.rand(batch_size, n_channels, freq, time, dtype=torch.cfloat)
    cfg = {
        "sr": 44100,
        "n_fft": 2048,
        "bandsplits": [
            (1000, 100),
            (4000, 250),
            (8000, 500),
            (16000, 1000),
            (20000, 2000),
        ],
        "complex_as_channel": True,
        "is_mono": n_channels == 1,
        "bottleneck_layer": 'rnn',
        "t_timesteps": 259,
        "fc_dim": 128,
        "rnn_dim": 256,
        "rnn_type": "LSTM",
        "bidirectional": True,
        "num_layers": 1,
        "mlp_dim": 512,
        "return_mask": False,
    }
    model = BandSplitRNN(**cfg)._load_from_state_dict()
    _ = model.eval()

    with torch.no_grad():
        out_features = model(in_features)

    print(model)
    print(f"Total number of parameters: {sum([p.numel() for p in model.parameters()])}")
    print(f"In shape: {in_features.shape}\nOut shape: {out_features.shape}")
    print(f"In dtype: {in_features.dtype}\nOut dtype: {out_features.dtype}")
    """
    freqs_splits = [
        (1000, 100),
        (4000, 250),
        (8000, 500),
        (16000, 1000),
        (20000, 2000),
    ]
    sr = 44100
    n_fft = 2048

    out = freq2bands(freqs_splits, sr, n_fft)

    sum_tuples = 0
    for tup in out:
        sum_tuples += tup[1]

    #assert sum_tuples == n_fft // 2 + 1

    print(f"Input:\n{freqs_splits}\n{sr}\n{n_fft}\nOutput:{out}")