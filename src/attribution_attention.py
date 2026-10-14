"""Declared interventions on the exact N3.1.0 module, at native stride 4."""
import torch
from torch.nn import functional as F
from n3_attention import N3Attention


class AttributionAttention(N3Attention):
    def __init__(self, channels=24, mode='fixed'):
        super().__init__(channels)
        if mode not in ('fixed', 'pooled', 'unsigned'):
            raise ValueError('Undeclared component intervention')
        self.mode = mode
        if mode == 'fixed':
            for value in (self.scale_raw, self.rho_raw, self.epsilon_raw):
                value.requires_grad_(False)

    def smooth(self, feature, kernel):
        # For ordinary N3 responses this is identical to the archived operation.
        # The pooled control applies the same operation to the two statistic maps.
        width = kernel.shape[-1]
        if feature.dtype != torch.float32:
            raise RuntimeError('The frozen training contract requires FP32')
        count = feature.shape[1]
        return F.conv2d(F.pad(feature, (width // 2,) * 4, mode='replicate'),
                        kernel.expand(count, 1, width, width).contiguous(), groups=count)

    def responses(self, refined):
        if self.mode == 'pooled':
            stats = torch.cat((refined.mean(dim=1, keepdim=True),
                               refined.max(dim=1, keepdim=True).values), dim=1)
            on, off = super().responses(stats)
            batch, _, height, width = on.shape

            def broadcast(values):
                # B,2,K,H,W -> mean over the two rectified statistic responses
                # -> broadcast to B,C,K,H,W -> the original grouped mixer layout.
                shared = values.reshape(batch, 2, 3, height, width).mean(dim=1, keepdim=True)
                return shared.expand(batch, self.channels, 3, height, width).reshape(
                    batch, 3 * self.channels, height, width)

            return broadcast(on), broadcast(off)
        on, off = super().responses(refined)
        if self.mode == 'unsigned':
            magnitude = on + off
            # Keep both original parameter banks. Equal inputs make their
            # effective weights redundant; this is explicitly reported.
            return magnitude, magnitude
        return on, off

    def audit(self):
        values = super().audit()
        values.update(intervention=self.mode,
                      trainable_attention_parameters=sum(p.numel() for p in self.parameters() if p.requires_grad),
                      adaptive_coefficients_trainable=self.mode != 'fixed',
                      response_channels_before_broadcast=2 if self.mode == 'pooled' else self.channels,
                      identical_unsigned_mixer_inputs=self.mode == 'unsigned',
                      full_factorial_attribution=False)
        return values
