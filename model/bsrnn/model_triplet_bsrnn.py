import torch
import torch.nn as nn
from torchinfo import summary
import typing as tp

from ..csn import ConditionalSimNet2d, ConditionalSimNet1d

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
            #bandsplits: tp.List[tp.Tuple[int, int]],
            bandwidth_indices,
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
        #print(is_mono)

        self.cac = complex_as_channel
        self.is_mono = is_mono
        #self.bandwidth_indices = freq2bands(bandsplits, sr, n_fft)
        self.bandwidth_indices = bandwidth_indices
        #self.layernorms = nn.ModuleList([
        #    nn.LayerNorm([(e - s) * frequency_mul, t_timesteps])
        #    for s, e in self.bandwidth_indices
        #])
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
            #x = self.layernorms[i](x)
            x = nn.LayerNorm(x.shape[-2:], elementwise_affine=False, device=x.device)(x)
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
            bidirectional: bool = True,
            mode_t: bool = False,
            last: bool = False,
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
        if last:
            self.rnn_triplet = getattr(nn, rnn_type)(
                input_dim_size, hidden_dim_size, batch_first=True, bidirectional=bidirectional
            )
        self.mode_t = mode_t
        self.last = last

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
        #print(out.dtype)

        out = self.groupnorm(
            out.transpose(-1, -2)
        ).transpose(-1, -2)  # [BK, T, N]    [BT, K, N]
        #out, (h_n, c_n) = self.rnn(out)  # [BK, T, H]    [BT, K, H]
        out, hc = self.rnn(out)  # [BK, T, H]    [BT, K, H]
        #print(out.shape)
        #H = out.shape[-1]
        out = self.fc(out)  # [BK, T, N]    [BT, K, N]

        x = out.view(B, K, T, N) + x  # [B, K, T, N]  [B, T, K, N]

        x = x.permute(0, 2, 1, 3).contiguous()  # [B, T, K, N]  [B, K, T, N]
        #if self.mode_t:
        #    c_n = c_n.view(B, K, H)
        return x

class RNNModuleWithTriplet(nn.Module):
    """
    RNN submodule of BandSequence module
    """

    def __init__(
            self,
            input_dim_size: int,
            hidden_dim_size: int,
            rnn_type: str = 'lstm',
            bidirectional: bool = True,
    ):
        super(RNNModuleWithTriplet, self).__init__()
        self.groupnorm = nn.GroupNorm(input_dim_size, input_dim_size)
        self.rnn = getattr(nn, rnn_type)(
            input_dim_size, hidden_dim_size, batch_first=True, bidirectional=bidirectional
        )
        self.fc = nn.Linear(
            hidden_dim_size * 2 if bidirectional else hidden_dim_size,
            input_dim_size
        )
        self.rnn_triplet = getattr(nn, rnn_type)(
            input_dim_size, hidden_dim_size, batch_first=True, bidirectional=bidirectional
        )
        self.fc_triplet = nn.Linear(
            hidden_dim_size * 2 if bidirectional else hidden_dim_size,
            input_dim_size
        )

    def forward(
            self,
            x: torch.Tensor,
            trp = None,
    ):
        """
        Input shape:
            across T - [batch_size, k_subbands, time, n_features]
            OR
            across K - [batch_size, time, k_subbands, n_features]
        """
        B, K, T, N = x.shape  # across T      across K (keep in mind T->K, K->T)

        out_mss = x.view(B * K, T, N)  # [BK, T, N]    [BT, K, N]
        #print(out.dtype)

        out_mss = self.groupnorm(
            out_mss.transpose(-1, -2)
        ).transpose(-1, -2)  # [BK, T, N]    [BT, K, N]
        if trp is not None:
            out_trp = trp.view(B * K, T, N)  # [BK, T, N]    [BT, K, N]
            out_trp = self.groupnorm(
                out_trp.transpose(-1, -2)
            ).transpose(-1, -2)  # [BK, T, N]    [BT, K, N]
        else:
            trp = x.clone()
            out_trp = out_mss.clone()
        #out, (h_n, c_n) = self.rnn(out)  # [BK, T, H]    [BT, K, H]
        out_mss, hc = self.rnn(out_mss)  # [BK, T, H]    [BT, K, H]
        out_trp     = self.rnn_triplet(out_trp, hc)[0]
        #print(out.shape)
        #H = out.shape[-1]
        out_mss = self.fc(out_mss)  # [BK, T, N]    [BT, K, N]
        out_trp = self.fc_triplet(out_trp)

        out_mss = out_mss.view(B, K, T, N) + x  # [B, K, T, N]  [B, T, K, N]
        out_trp = out_trp.view(B, K, T, N) + trp

        out_mss = out_mss.permute(0, 2, 1, 3).contiguous()  # [B, T, K, N]  [B, K, T, N]
        out_trp = out_trp.permute(0, 2, 1, 3).contiguous()  # [B, T, K, N]  [B, K, T, N]

        #print(c_n.shape)
        return out_mss, out_trp

class RNNModuleForTriplet(nn.Module):
    """
    RNN submodule of BandSequence module
    """

    def __init__(
            self,
            input_dim_size: int,
            hidden_dim_size: int,
            rnn_type: str = 'lstm',
            bidirectional: bool = True,
    ):
        super(RNNModuleForTriplet, self).__init__()
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
        _, (h, c) = self.rnn(out)  # [BK, T, H]    [BT, K, H]
        H = _.shape[-1]
        #print(out.shape)
        #out = self.fc(out)  # [BK, T, N]    [BT, K, N]

        #x = out.view(B, K, T, N) + x  # [B, K, T, N]  [B, T, K, N]
        #print(h.shape)
        out = h.permute(1, 2, 0).reshape(B*K, H).reshape(B, K, H)

        #x = x.permute(0, 2, 1, 3).contiguous()  # [B, T, K, N]  [B, K, T, N]
        #if self.mode_t:
        #    c_n = c_n.view(B, K, H)
        return out


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

        #self.bsrnn = nn.ModuleList([])
        self.rnn_t = nn.ModuleList([])
        self.rnn_k = nn.ModuleList([])

        for i in range(num_layers):
            #rnn_across_t = RNNModule(
            #    input_dim_size, hidden_dim_size, rnn_type, bidirectional
            #)
            #rnn_across_k = RNNModule(
            #    input_dim_size, hidden_dim_size, rnn_type, bidirectional
            #)
            #self.bsrnn.append(
            #    nn.Sequential(rnn_across_t, rnn_across_k)
            #)
            self.rnn_t.append(RNNModule(
                input_dim_size, hidden_dim_size, rnn_type, bidirectional, mode_t=True, last=i==(num_layers - 1)
                ))
            self.rnn_k.append(RNNModule(
                input_dim_size, hidden_dim_size, rnn_type, bidirectional
            ))
        self.num_layers = num_layers

    def forward(self, x: torch.Tensor):
        """
        Input shape: [batch_size, k_subbands, time, n_features]
        Output shape: [batch_size, k_subbands, time, n_features]
        """
        #cell_t = []
        #for i in range(len(self.bsrnn)):
        for i in range(self.num_layers):
            x = self.rnn_t[i](x)
            x = self.rnn_k[i](x)
            #cell_t.append(c)
            #x = self.bsrnn[i](x)
            #if i == self.num_layers - 1:
                #print(c.shape)
        return x #torch.stack(cell_t, dim=1) # [B, n_rnn, K, N*2]

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
            glu: bool = True
    ):
        super(MLP, self).__init__()

        if glu:
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                self.select_activation(activation_type)(),
                nn.Linear(hidden_dim, output_dim),
                GLU(output_dim)
            )
        else:
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                self.select_activation(activation_type)(),
                nn.Linear(hidden_dim, output_dim),
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
        #self.layernorms = nn.ModuleList([
        #    nn.LayerNorm([t_timesteps, fc_dim])
        #    for _ in range(len(self.bandwidths))
        #])
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
            #out = self.layernorms[i](x[:, i])
            out = nn.LayerNorm(x.shape[-2:], elementwise_affine=False, device=x.device)(x[:, i])
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

class EmbeddingModule(nn.Module):
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
            out_dim: int = 128,
            complex_as_channel: bool = True,
            is_mono: bool = False,
    ):
        super(EmbeddingModule, self).__init__()

        self.bandwidths = [(e - s) for s, e in freq2bands(bandsplits, sr, n_fft)]
        #self.layernorms = nn.ModuleList([
        #    nn.LayerNorm([t_timesteps, fc_dim])
        #    for _ in range(len(self.bandwidths))
        #])
        self.mlp = nn.ModuleList([
            MLP(fc_dim, mlp_dim, out_dim, activation_type='relu', glu=False)
            for bw in self.bandwidths
        ])
        self.mlp_last = MLP(out_dim*len(self.bandwidths), mlp_dim, out_dim, activation_type="relu", glu=False)

    def forward(self, x: torch.Tensor):
        """
        Input: [batch_size, k_subbands, time, fc_shape]
        Output: [batch_size, freq, time]
        """
        outs = []
        for i in range(x.shape[1]):
            # run through model
            #out = self.layernorms[i](x[:, i])
            out = nn.LayerNorm(x.shape[-2:], elementwise_affine=False, device=x.device)(x[:, i])
            out = self.mlp[i](out)
            B, T, F = out.shape
            outs.append(torch.mean(out, dim=1)) # [B, F]
        output_emb = self.mlp_last(torch.stack(outs, dim=1).view(B, -1))
        return output_emb

class BandSplitRNNForTriplet(nn.Module):
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
            inst_list,
            return_mask: bool = False
    ):
        super(BandSplitRNNForTriplet, self).__init__()
        
        # Cul Subband_width
        bandwidth_indices = freq2bands(bandsplits, sr, n_fft)

        # encoder layer
        self.bandsplit = BandSplitModule(
            sr=sr,
            n_fft=n_fft,
            #bandsplits=bandsplits,
            bandwidth_indices=bandwidth_indices,
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

        self.embnet = EmbeddingModule(
            sr=sr,
            n_fft=n_fft,
            bandsplits=bandsplits,
            t_timesteps=t_timesteps,
            fc_dim=fc_dim,
            mlp_dim=mlp_dim,
            out_dim=128,
            complex_as_channel=complex_as_channel,
            is_mono=is_mono,
        )
        self.cac = complex_as_channel
        self.return_mask = return_mask
        
        # to1d
        #self.to1d = nn.Linear(num_layers*len(bandwidth_indices)*rnn_dim*2, 640)
        self.inst_list = inst_list

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
        mask = self.maskest(x)  # [batch_size, freq, time]
        emb = self.embnet(x) # [B, 128]

        return mask, emb

    def forward(self, x: torch.Tensor):
        """
        Input and output are T-F complex-valued features.
        Input shape: batch_size, n_channels, freq, time]
        Output shape: batch_size, n_channels, freq, time]
        """
        # use only magnitude if not using complex input
        B = x.shape[0]
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
        mask, output_emb = self.compute_mask(x)

        # multiply with original tensor
        x = mask if self.return_mask else mask * x

        # denormalize
        x = x * std + mean

        if not self.cac:
            x = self.wiener(x, x_complex)
        
        # to1d
        #print(cell_t.shape)
        # 原点からのユークリッド距離にlogをかけてsigmoidしたものを無音有音の確率とする
        csn1d = ConditionalSimNet1d() # csnのモデルを保存されないようにするために配列に入れる
        #output_probability = {inst : torch.log(torch.sqrt(torch.sum(csn1d(output_emb, torch.tensor([i], device=output_emb.device))**2, dim=1))) for i,inst in enumerate(self.inst_list)} # logit
        output_probability = {inst : torch.log(torch.sqrt(torch.sum(output_emb**2, dim=1))) for i,inst in enumerate(self.inst_list)} # logit

        return output_emb, output_probability, x

def main():
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
        "is_mono": True,
        "bottleneck_layer": 'rnn',
        "t_timesteps": 259,
        "fc_dim": 128,
        "rnn_dim": 256,
        "rnn_type": "LSTM",
        "bidirectional": True,
        "num_layers": 12,
        "mlp_dim": 512,
        "inst_list": ["bass"],
        "return_mask": False,
    }
    model = BandSplitRNNForTriplet(**cfg)
    batchsize = 16
    summary(model=model,
            input_size=(batchsize, 1, 1025, 259),
            dtypes=[torch.complex64],
            col_names=["input_size", "output_size", "num_params", "mult_adds"],
            depth=4)

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
    """
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
        "is_mono": True,
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
    model = BandSplitRNNForTriplet(**cfg)
    batchsize = 16
    summary(model=model,
            input_size=(batchsize, 1, 1025, 259),
            dtypes=[torch.complex64],
            col_names=["input_size", "output_size", "num_params", "mult_adds"],
            depth=4)