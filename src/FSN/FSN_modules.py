import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn import functional
import numpy as np

###### from audio_zen.acoustics.feature
def mag_phase(complex_tensor):
    return torch.abs(complex_tensor), torch.angle(complex_tensor)

####### audio_zen.acoustics.mask
def decompress_cIRM(mask, K=10, limit=9.9):
    mask = limit * (mask >= limit) - limit * (mask <= -limit) + mask * (torch.abs(mask) < limit)
    mask = -K * torch.log((K - mask) / (K + mask))
    return mask


############################################

# from audio_zen.acoustics.feature import drop_band
def drop_band(input, num_groups=2):
    """
    Reduce computational complexity of the sub-band part in the FullSubNet model.

    Shapes:
        input: [B, C, F, T]
        return: [B, C, F // num_groups, T]
    """
    batch_size, _, num_freqs, _ = input.shape
    assert batch_size > num_groups, f"Batch size = {batch_size}, num_groups = {num_groups}. The batch size should larger than the num_groups."

    if num_groups <= 1:
        # No demand for grouping
        return input

    # Each sample must has the same number of the frequencies for parallel training.
    # Therefore, we need to drop those remaining frequencies in the high frequency part.
    if num_freqs % num_groups != 0:
        input = input[..., :(num_freqs - (num_freqs % num_groups)), :]
        num_freqs = input.shape[2]

    output = []
    for group_idx in range(num_groups):
        samples_indices = torch.arange(group_idx, batch_size, num_groups, device=input.device)
        freqs_indices = torch.arange(group_idx, num_freqs, num_groups, device=input.device)

        selected_samples = torch.index_select(input, dim=0, index=samples_indices)
        selected = torch.index_select(selected_samples, dim=2, index=freqs_indices)  # [B, C, F // num_groups, T]

        output.append(selected)

    return torch.cat(output, dim=0)

# from audio_zen.model.base_model import BaseModel
# from audio_zen.constant import EPSILON

EPSILON = np.finfo(np.float32).eps

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

    @staticmethod
    def unfold(input, num_neighbor):
        """
        Along with the frequency dim, split overlapped sub band units from spectrogram.

        Args:
            input: [B, C, F, T]
            num_neighbor:

        Returns:
            [B, N, C, F_s, T], F 为子频带的频率轴大小, e.g. [2, 161, 1, 19, 200]
        """
        assert input.dim() == 4, f"The dim of input is {input.dim()}. It should be four dim."
        batch_size, num_channels, num_freqs, num_frames = input.size()

        if num_neighbor < 1:
            # No change for the input
            return input.permute(0, 2, 1, 3).reshape(batch_size, num_freqs, num_channels, 1, num_frames)

        output = input.reshape(batch_size * num_channels, 1, num_freqs, num_frames)
        sub_band_unit_size = num_neighbor * 2 + 1

        # Pad to the top and bottom
        output = functional.pad(output, [0, 0, num_neighbor, num_neighbor], mode="reflect")

        output = functional.unfold(output, (sub_band_unit_size, num_frames))
        assert output.shape[-1] == num_freqs, f"n_freqs != N (sub_band), {num_freqs} != {output.shape[-1]}"

        # Split the dim of the unfolded feature
        output = output.reshape(batch_size, num_channels, sub_band_unit_size, num_frames, num_freqs)
        output = output.permute(0, 4, 1, 2, 3).contiguous()

        return output

    @staticmethod
    def _reduce_complexity_separately(sub_band_input, full_band_output, device):
        """

        Args:
            sub_band_input: [60, 257, 1, 33, 200]
            full_band_output: [60, 257, 1, 3, 200]
            device:

        Notes:
            1. 255 and 256 freq not able to be trained
            2. batch size 应该被 3 整除，否则最后一部分 batch 内的频率无法很好的训练

        Returns:
            [60, 85, 1, 36, 200]
        """
        batch_size = full_band_output.shape[0]
        n_freqs = full_band_output.shape[1]
        sub_batch_size = batch_size // 3
        final_selected = []

        for idx in range(3):
            # [0, 60) => [0, 20)
            sub_batch_indices = torch.arange(idx * sub_batch_size, (idx + 1) * sub_batch_size, device=device)
            full_band_output_sub_batch = torch.index_select(full_band_output, dim=0, index=sub_batch_indices)
            sub_band_output_sub_batch = torch.index_select(sub_band_input, dim=0, index=sub_batch_indices)

            # Avoid to use padded value (first freq and last freq)
            # i = 0, (1, 256, 3) = [1, 4, ..., 253]
            # i = 1, (2, 256, 3) = [2, 5, ..., 254]
            # i = 2, (3, 256, 3) = [3, 6, ..., 255]
            freq_indices = torch.arange(idx + 1, n_freqs - 1, step=3, device=device)
            full_band_output_sub_batch = torch.index_select(full_band_output_sub_batch, dim=1, index=freq_indices)
            sub_band_output_sub_batch = torch.index_select(sub_band_output_sub_batch, dim=1, index=freq_indices)

            # ([30, 85, 1, 33 200], [30, 85, 1, 3, 200]) => [30, 85, 1, 36, 200]

            final_selected.append(torch.cat([sub_band_output_sub_batch, full_band_output_sub_batch], dim=-2))

        return torch.cat(final_selected, dim=0)

    @staticmethod
    def sband_forgetting_norm(input, train_sample_length):
        """
        与 forgetting norm相同，但使用拼接后模型的中间频带来计算均值
        效果不好
        Args:
            input:
            train_sample_length:

        Returns:

        """
        assert input.ndim == 3
        batch_size, n_freqs, n_frames = input.size()

        eps = 1e-10
        alpha = (train_sample_length - 1) / (train_sample_length + 1)
        mu = 0
        mu_list = []

        for idx in range(input.shape[-1]):
            if idx < train_sample_length:
                alp = torch.min(torch.tensor([(idx - 1) / (idx + 1), alpha]))
                mu = alp * mu + (1 - alp) * torch.mean(input[:, :, idx], dim=1).reshape(batch_size, 1)  # [B, 1]
            else:
                mu = alpha * mu + (1 - alpha) * input[:, (n_freqs // 2 - 1), idx].reshape(batch_size, 1)

            mu_list.append(mu)

            # print("input", input[:, :, idx].min(), input[:, :, idx].max(), input[:, :, idx].mean())
            # print(f"alp {idx}: ", alp)
            # print(f"mu {idx}: {mu[128, 0]}")

        mu = torch.stack(mu_list, dim=-1)  # [B, 1, T]
        input = input / (mu + eps)
        return input

    @staticmethod
    def forgetting_norm(input, sample_length_in_training):
        """
        输入为三维，通过不断估计邻近的均值来作为当前 norm 时的均值

        Args:
            input: [B, F, T]
            sample_length_in_training: 训练时的长度，用于计算平滑因子

        Returns:

        """
        assert input.ndim == 3
        batch_size, n_freqs, n_frames = input.size()
        eps = 1e-10
        mu = 0
        alpha = (sample_length_in_training - 1) / (sample_length_in_training + 1)

        mu_list = []
        for idx in range(input.shape[-1]):
            if idx < sample_length_in_training:
                alp = torch.min(torch.tensor([(idx - 1) / (idx + 1), alpha]))
                mu = alp * mu + (1 - alp) * torch.mean(input[:, :, idx], dim=1).reshape(batch_size, 1)  # [B, 1]
            else:
                current_frame_mu = torch.mean(input[:, :, idx], dim=1).reshape(batch_size, 1)  # [B, 1]
                mu = alpha * mu + (1 - alpha) * current_frame_mu

            mu_list.append(mu)

            # print("input", input[:, :, idx].min(), input[:, :, idx].max(), input[:, :, idx].mean())
            # print(f"alp {idx}: ", alp)
            # print(f"mu {idx}: {mu[128, 0]}")

        mu = torch.stack(mu_list, dim=-1)  # [B, 1, T]
        input = input / (mu + eps)
        return input

    @staticmethod
    def hybrid_norm(input, sample_length_in_training=192):
        """
        Args:
            input: [B, F, T]
            sample_length_in_training:

        Returns:
            [B, F, T]
        """
        assert input.ndim == 3
        device = input.device
        data_type = input.dtype
        batch_size, n_freqs, n_frames = input.size()
        eps = 1e-10

        mu = 0
        alpha = (sample_length_in_training - 1) / (sample_length_in_training + 1)
        mu_list = []
        for idx in range(input.shape[-1]):
            if idx < sample_length_in_training:
                alp = torch.min(torch.tensor([(idx - 1) / (idx + 1), alpha]))
                mu = alp * mu + (1 - alp) * torch.mean(input[:, :, idx], dim=1).reshape(batch_size, 1)  # [B, 1]
                mu_list.append(mu)
            else:
                break
        initial_mu = torch.stack(mu_list, dim=-1)  # [B, 1, T]

        step_sum = torch.sum(input, dim=1)  # [B, T]
        cumulative_sum = torch.cumsum(step_sum, dim=-1)  # [B, T]

        entry_count = torch.arange(n_freqs, n_freqs * n_frames + 1, n_freqs, dtype=data_type, device=device)
        entry_count = entry_count.reshape(1, n_frames)  # [1, T]
        entry_count = entry_count.expand_as(cumulative_sum)  # [1, T] => [B, T]

        cum_mean = cumulative_sum / entry_count  # B, T

        cum_mean = cum_mean.reshape(batch_size, 1, n_frames)  # [B, 1, T]

        # print(initial_mu[0, 0, :50])
        # print("-"*60)
        # print(cum_mean[0, 0, :50])
        cum_mean[:, :, :sample_length_in_training] = initial_mu

        return input / (cum_mean + eps)

    @staticmethod
    def offline_laplace_norm(input):
        """

        Args:
            input: [B, C, F, T]

        Returns:
            [B, C, F, T]
        """
        # utterance-level mu
        mu = torch.mean(input, dim=(1, 2, 3), keepdim=True)

        normed = input / (mu + 1e-5)

        return normed

    @staticmethod
    def cumulative_laplace_norm(input):
        """

        Args:
            input: [B, C, F, T]

        Returns:

        """
        batch_size, num_channels, num_freqs, num_frames = input.size()
        input = input.reshape(batch_size * num_channels, num_freqs, num_frames)

        step_sum = torch.sum(input, dim=1)  # [B * C, F, T] => [B, T]
        cumulative_sum = torch.cumsum(step_sum, dim=-1)  # [B, T]

        entry_count = torch.arange(
            num_freqs,
            num_freqs * num_frames + 1,
            num_freqs,
            dtype=input.dtype,
            device=input.device
        )
        entry_count = entry_count.reshape(1, num_frames)  # [1, T]
        entry_count = entry_count.expand_as(cumulative_sum)  # [1, T] => [B, T]

        cumulative_mean = cumulative_sum / entry_count  # B, T
        cumulative_mean = cumulative_mean.reshape(batch_size * num_channels, 1, num_frames)

        normed = input / (cumulative_mean + EPSILON)

        return normed.reshape(batch_size, num_channels, num_freqs, num_frames)

    @staticmethod
    def offline_gaussian_norm(input):
        """
        Zero-Norm
        Args:
            input: [B, C, F, T]

        Returns:
            [B, C, F, T]
        """
        mu = torch.mean(input, dim=(1, 2, 3), keepdim=True)
        std = torch.std(input, dim=(1, 2, 3), keepdim=True)

        normed = (input - mu) / (std + 1e-5)

        return normed

    @staticmethod
    def cumulative_layer_norm(input):
        """
        Online zero-norm

        Args:
            input: [B, C, F, T]

        Returns:
            [B, C, F, T]
        """
        batch_size, num_channels, num_freqs, num_frames = input.size()
        input = input.reshape(batch_size * num_channels, num_freqs, num_frames)

        step_sum = torch.sum(input, dim=1)  # [B * C, F, T] => [B, T]
        step_pow_sum = torch.sum(torch.square(input), dim=1)

        cumulative_sum = torch.cumsum(step_sum, dim=-1)  # [B, T]
        cumulative_pow_sum = torch.cumsum(step_pow_sum, dim=-1)  # [B, T]

        entry_count = torch.arange(
            num_freqs,
            num_freqs * num_frames + 1,
            num_freqs,
            dtype=input.dtype,
            device=input.device
        )
        entry_count = entry_count.reshape(1, num_frames)  # [1, T]
        entry_count = entry_count.expand_as(cumulative_sum)  # [1, T] => [B, T]

        cumulative_mean = cumulative_sum / entry_count  # [B, T]
        cumulative_var = (cumulative_pow_sum - 2 * cumulative_mean * cumulative_sum) / entry_count + cumulative_mean.pow(2)  # [B, T]
        cumulative_std = torch.sqrt(cumulative_var + EPSILON)  # [B, T]

        cumulative_mean = cumulative_mean.reshape(batch_size * num_channels, 1, num_frames)
        cumulative_std = cumulative_std.reshape(batch_size * num_channels, 1, num_frames)

        normed = (input - cumulative_mean) / cumulative_std

        return normed.reshape(batch_size, num_channels, num_freqs, num_frames)

    def norm_wrapper(self, norm_type: str):
        if norm_type == "offline_laplace_norm":
            norm = self.offline_laplace_norm
        elif norm_type == "cumulative_laplace_norm":
            norm = self.cumulative_laplace_norm
        elif norm_type == "offline_gaussian_norm":
            norm = self.offline_gaussian_norm
        elif norm_type == "cumulative_layer_norm":
            norm = self.cumulative_layer_norm
        else:
            raise NotImplementedError("You must set up a type of Norm. "
                                      "e.g. offline_laplace_norm, cumulative_laplace_norm, forgetting_norm, etc.")
        return norm

    def weight_init(self, m):
        """
        Usage:
            model = Model()
            model.apply(weight_init)
        """
        if isinstance(m, nn.Conv1d):
            init.normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.Conv3d):
            init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.ConvTranspose1d):
            init.normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.ConvTranspose2d):
            init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.ConvTranspose3d):
            init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.BatchNorm1d):
            init.normal_(m.weight.data, mean=1, std=0.02)
            init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.normal_(m.weight.data, mean=1, std=0.02)
            init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm3d):
            init.normal_(m.weight.data, mean=1, std=0.02)
            init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.Linear):
            init.xavier_normal_(m.weight.data)
            init.normal_(m.bias.data)
        elif isinstance(m, nn.LSTM):
            for param in m.parameters():
                if len(param.shape) >= 2:
                    init.orthogonal_(param.data)
                else:
                    init.normal_(param.data)
        elif isinstance(m, nn.LSTMCell):
            for param in m.parameters():
                if len(param.shape) >= 2:
                    init.orthogonal_(param.data)
                else:
                    init.normal_(param.data)
        elif isinstance(m, nn.GRU):
            for param in m.parameters():
                if len(param.shape) >= 2:
                    init.orthogonal_(param.data)
                else:
                    init.normal_(param.data)
        elif isinstance(m, nn.GRUCell):
            for param in m.parameters():
                if len(param.shape) >= 2:
                    init.orthogonal_(param.data)
                else:
                    init.normal_(param.data)

# from audio_zen.model.module.sequence_model import SequenceModel
# from speech_enhance.audio_zen.model.module.causal_conv import TCNBlock

class TCNBlock(nn.Module):
    def __init__(self, in_channels=257, hidden_channel=512, out_channels=257, kernel_size=3, dilation=1,
                 use_skip_connection=True, causal=False):
        super().__init__()
        self.conv1x1 = nn.Conv1d(in_channels, hidden_channel, 1)
        self.prelu1 = nn.PReLU()
        self.norm1 = nn.GroupNorm(1, hidden_channel, eps=1e-8)
        padding = (dilation * (kernel_size - 1)) // 2 if not causal else (
                dilation * (kernel_size - 1))
        self.depthwise_conv = nn.Conv1d(hidden_channel, hidden_channel, kernel_size=kernel_size, stride=1,
                                        groups=hidden_channel, padding=padding, dilation=dilation)
        self.prelu2 = nn.PReLU()
        self.norm2 = nn.GroupNorm(1, hidden_channel, eps=1e-8)
        self.sconv = nn.Conv1d(hidden_channel, out_channels, 1)
        # self.tcn_block = nn.Sequential(
        #     nn.Conv1d(in_channels, hidden_channel, 1),
        #     nn.PReLU(),
        #     nn.GroupNorm(1, hidden_channel, eps=1e-8),
        #     nn.Conv1d(hidden_channel, hidden_channel, kernel_size=kernel_size, stride=1,
        #               groups=hidden_channel, padding=padding, dilation=dilation, bias=True),
        #     nn.PReLU(),
        #     nn.GroupNorm(1, hidden_channel, eps=1e-8),
        #     nn.Conv1d(hidden_channel, out_channels, 1)
        # )

        self.causal = causal
        self.padding = padding
        self.use_skip_connection = use_skip_connection

    def forward(self, x):
        """
            x: [channels, T]
        """
        if self.use_skip_connection:
            y = self.conv1x1(x)
            y = self.norm1(self.prelu1(y))
            y = self.depthwise_conv(y)
            if self.causal:
                y = y[:, :, :-self.padding]
            y = self.norm2(self.prelu2(y))
            output = self.sconv(y)
            return x + output
        else:
            y = self.conv1x1(x)
            y = self.norm1(self.prelu1(y))
            y = self.depthwise_conv(y)
            if self.causal:
                y = y[:, :, :-self.padding]
            y = self.norm2(self.prelu2(y))
            output = self.sconv(y)
            return output



class SequenceModel(nn.Module):
    def __init__(
            self,
            input_size,
            output_size,
            hidden_size,
            num_layers,
            bidirectional,
            sequence_model="GRU",
            output_activate_function="Tanh"
    ):
        """
        序列模型，可选 LSTM 或 CRN，支持子带输入

        Args:
            input_size: 每帧输入特征大小
            output_size: 每帧输出特征大小
            hidden_size: 序列模型隐层单元数量
            num_layers:  层数
            bidirectional: 是否为双向
            sequence_model: LSTM | GRU
            output_activate_function: Tanh | ReLU
        """
        super().__init__()
        # Sequence layer
        self.sequence_model_type = sequence_model
        if self.sequence_model_type == "LSTM":
            self.sequence_model = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
            )
        elif self.sequence_model_type == "GRU":
            self.sequence_model = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
            )
        elif self.sequence_model_type == "TCN":
            self.sequence_model = nn.Sequential(
                TCNBlock(in_channels=input_size, out_channels=input_size, dilation=1),
                TCNBlock(in_channels=input_size, out_channels=input_size, dilation=2),
                TCNBlock(in_channels=input_size, out_channels=input_size, dilation=5),
                TCNBlock(in_channels=input_size, out_channels=input_size, dilation=9),
                TCNBlock(in_channels=input_size, out_channels=input_size, dilation=1),
                TCNBlock(in_channels=input_size, out_channels=input_size, dilation=2),
                TCNBlock(in_channels=input_size, out_channels=input_size, dilation=5),
                TCNBlock(in_channels=input_size, out_channels=input_size, dilation=9),
                nn.ReLU()
            )
        elif self.sequence_model_type == "TCN-subband":
            self.sequence_model = nn.Sequential(
                TCNBlock(in_channels=input_size, hidden_channel=hidden_size, out_channels=input_size, dilation=1),
                TCNBlock(in_channels=input_size, hidden_channel=hidden_size, out_channels=input_size, dilation=2),
                TCNBlock(in_channels=input_size, hidden_channel=hidden_size, out_channels=input_size, dilation=5),
                TCNBlock(in_channels=input_size, hidden_channel=hidden_size, out_channels=input_size, dilation=9),
                TCNBlock(in_channels=input_size, hidden_channel=hidden_size, out_channels=input_size, dilation=1),
                TCNBlock(in_channels=input_size, hidden_channel=hidden_size, out_channels=input_size, dilation=2),
                TCNBlock(in_channels=input_size, hidden_channel=hidden_size, out_channels=input_size, dilation=5),
                TCNBlock(in_channels=input_size, hidden_channel=384, out_channels=input_size, dilation=9),
                nn.ReLU()
            )
        else:
            raise NotImplementedError(f"Not implemented {sequence_model}")

        if self.sequence_model_type == "LSTM" or self.sequence_model_type == "GRU":
            # Fully connected layer
            if bidirectional:
                self.fc_output_layer = nn.Linear(hidden_size * 2, output_size)
            else:
                self.fc_output_layer = nn.Linear(hidden_size, output_size)
        elif self.sequence_model_type == "TCN":
            self.fc_output_layer = nn.Linear(input_size, output_size)
        else:
            self.fc_output_layer = nn.Linear(input_size, output_size)

        # Activation function layer
        if output_activate_function:
            if output_activate_function == "Tanh":
                self.activate_function = nn.Tanh()
            elif output_activate_function == "ReLU":
                self.activate_function = nn.ReLU()
            elif output_activate_function == "ReLU6":
                self.activate_function = nn.ReLU6()
            else:
                raise NotImplementedError(f"Not implemented activation function {self.activate_function}")

        self.output_activate_function = output_activate_function

    def forward(self, x):
        """
        Args:
            x: [B, F, T]
        Returns:
            [B, F, T]
        """
        assert x.dim() == 3
        if self.sequence_model_type == "TCN" or self.sequence_model_type == "TCN-subband":
            x = self.sequence_model(x)  # [B, F, T]
            o = self.fc_output_layer(x.permute(0, 2, 1))    # [B, F, T] => [B, T, F]
            if self.output_activate_function:
                o = self.activate_function(o)
            o = o.permute(0, 2, 1)  # [B, T, F] => [B, F, T]
            return o
        else:
            self.sequence_model.flatten_parameters()
            # contiguous 使元素在内存中连续，有利于模型优化，但分配了新的空间
            # 建议在网络开始大量计算前使用一下
            x = x.permute(0, 2, 1).contiguous()  # [B, F, T] => [B, T, F]
            o, _ = self.sequence_model(x)
            o = self.fc_output_layer(o)
            if self.output_activate_function:
                o = self.activate_function(o)
            o = o.permute(0, 2, 1).contiguous()  # [B, T, F] => [B, F, T]
        return o


# from audio_zen.model.module.attention_model import ChannelSELayer, ChannelECAlayer, ChannelCBAMLayer, \
# ChannelTimeSenseSELayer, ChannelTimeSenseAttentionSELayer, ChannelTimeSenseSEWeightLayer



class ChannelSELayer(nn.Module):
    """
    Re-implementation of Squeeze-and-Excitation (SE) block described in:
        *Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507*
    """

    def __init__(self, num_channels, reduction_ratio=2):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSELayer, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, T)
        :return: output tensor
        """
        # batch_size, num_channels, T = input_tensor.size()
        # Average pooling along each channel
        squeeze_tensor = input_tensor.mean(dim=2)

        # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        a, b = squeeze_tensor.size()
        output_tensor = torch.mul(input_tensor, fc_out_2.view(a, b, 1))
        return output_tensor


class ChannelTimeSenseSELayer(nn.Module):
    """
    Re-implementation of Squeeze-and-Excitation (SE) block described in:
        *Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507*
    """

    def __init__(self, num_channels, reduction_ratio=2, kersize=[3, 5, 10], subband_num=1):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelTimeSenseSELayer, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.smallConv1d = nn.Sequential(
            nn.Conv1d(num_channels, num_channels, kernel_size=kersize[0], groups=num_channels // subband_num),  # [B, num_channels, T]
            nn.AdaptiveAvgPool1d(1),  # [B, num_channels, 1]
            nn.ReLU(inplace=True)
        )
        self.middleConv1d = nn.Sequential(
            nn.Conv1d(num_channels, num_channels, kernel_size=kersize[1], groups=num_channels // subband_num),  # [B, num_channels, T]
            nn.AdaptiveAvgPool1d(1),  # [B, num_channels, 1]
            nn.ReLU(inplace=True)
        )
        self.largeConv1d = nn.Sequential(
            nn.Conv1d(num_channels, num_channels, kernel_size=kersize[2], groups=num_channels // subband_num),  # [B, num_channels, T]
            nn.AdaptiveAvgPool1d(1),  # [B, num_channels, 1]
            nn.ReLU(inplace=True)
        )
        self.feature_concate_fc = nn.Linear(3, 1, bias=True)
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, T)
        :return: output tensor
        """
        # batch_size, num_channels, T = input_tensor.size()
        # Extracting multi-scale information in the time dimension
        small_feature = self.smallConv1d(input_tensor)
        middle_feature = self.middleConv1d(input_tensor)
        large_feature = self.largeConv1d(input_tensor)

        feature = torch.cat([small_feature, middle_feature, large_feature], dim=2)  # [B, num_channels, 3]
        squeeze_tensor = self.feature_concate_fc(feature)[..., 0]  # [B, num_channels]

        # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        a, b = squeeze_tensor.size()
        output_tensor = torch.mul(input_tensor, fc_out_2.view(a, b, 1))
        return output_tensor


class ChannelTimeSenseSEWeightLayer(nn.Module):
    """
    Re-implementation of Squeeze-and-Excitation (SE) block described in:
        *Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507*
    """

    def __init__(self, num_channels, reduction_ratio=2, kersize=[3, 5, 10]):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelTimeSenseSEWeightLayer, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.smallConv1d = nn.Sequential(
            nn.Conv1d(num_channels, num_channels, kernel_size=kersize[0], groups=num_channels),  # [B, num_channels, T]
            nn.AdaptiveAvgPool1d(1),  # [B, num_channels, 1]
            nn.ReLU(inplace=True)
        )
        self.middleConv1d = nn.Sequential(
            nn.Conv1d(num_channels, num_channels, kernel_size=kersize[1], groups=num_channels),  # [B, num_channels, T]
            nn.AdaptiveAvgPool1d(1),  # [B, num_channels, 1]
            nn.ReLU(inplace=True)
        )
        self.largeConv1d = nn.Sequential(
            nn.Conv1d(num_channels, num_channels, kernel_size=kersize[2], groups=num_channels),  # [B, num_channels, T]
            nn.AdaptiveAvgPool1d(1),  # [B, num_channels, 1]
            nn.ReLU(inplace=True)
        )
        self.feature_concate_fc = nn.Linear(3, 1, bias=True)
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, T)
        :return: output tensor
        """
        # batch_size, num_channels, T = input_tensor.size()
        # Extracting multi-scale information in the time dimension
        small_feature = self.smallConv1d(input_tensor)
        middle_feature = self.middleConv1d(input_tensor)
        large_feature = self.largeConv1d(input_tensor)

        feature = torch.cat([small_feature, middle_feature, large_feature], dim=2)  # [B, num_channels, 3]
        squeeze_tensor = self.feature_concate_fc(feature)[..., 0]  # [B, num_channels]

        # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        a, b = squeeze_tensor.size()
        output_tensor = torch.mul(input_tensor, fc_out_2.view(a, b, 1))
        return output_tensor, fc_out_2.view(a, b, 1)


class ChannelTimeSenseAttentionSELayer(nn.Module):
    """
    Re-implementation of Squeeze-and-Excitation (SE) block described in:
        *Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507*
    """

    def __init__(self, num_channels, reduction_ratio=2, kersize=[3, 5, 10]):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelTimeSenseAttentionSELayer, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio

        self.smallConv1d = Conv_Attention_Block(num_channels=num_channels, kersize=kersize[0])
        self.middleConv1d = Conv_Attention_Block(num_channels=num_channels, kersize=kersize[1])
        self.largeConv1d = Conv_Attention_Block(num_channels=num_channels, kersize=kersize[2])

        self.feature_concate_fc = nn.Linear(3, 1, bias=True)
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, T)
        :return: output tensor
        """
        # batch_size, num_channels, T = input_tensor.size()
        # Extracting multi-scale information in the time dimension
        small_feature = self.smallConv1d(input_tensor)
        middle_feature = self.middleConv1d(input_tensor)
        large_feature = self.largeConv1d(input_tensor)

        feature = torch.cat([small_feature, middle_feature, large_feature], dim=2)  # [B, num_channels, 3]
        squeeze_tensor = self.feature_concate_fc(feature)[..., 0]  # [B, num_channels]

        # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        a, b = squeeze_tensor.size()
        output_tensor = torch.mul(input_tensor, fc_out_2.view(a, b, 1))
        return output_tensor


class ChannelCBAMLayer(nn.Module):
    """
    Re-implementation of Squeeze-and-Excitation (SE) block described in:
        *Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507*
    """

    def __init__(self, num_channels, reduction_ratio=2):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelCBAMLayer, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, T)
        :return: output tensor
        """
        # batch_size, num_channels, T = input_tensor.size()
        # Average pooling along each channel
        mean_squeeze_tensor = input_tensor.mean(dim=2)
        max_squeeze_tensor, _ = torch.max(input_tensor, dim=2)  # input_tensor.max(dim=2)
        # channel excitation
        mean_fc_out_1 = self.relu(self.fc1(mean_squeeze_tensor))
        max_fc_out_1 = self.relu(self.fc1(max_squeeze_tensor))
        fc_out_1 = mean_fc_out_1 + max_fc_out_1
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        a, b = mean_squeeze_tensor.size()
        output_tensor = torch.mul(input_tensor, fc_out_2.view(a, b, 1))
        return output_tensor


class ChannelECAlayer(nn.Module):
    """
     a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, k_size=3):
        super(ChannelECAlayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.transpose(-1, -2)).transpose(-1, -2)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


