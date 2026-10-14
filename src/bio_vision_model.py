"""Prospective N3 joint design; original M0R/M1 controls unchanged."""
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import efficientnet_b0
from attention_math import SCALES, gaussian_kernel
from design_space import CONFIGS, VARIANTS, config_for
from n3_attention import N3Attention
from attribution_attention import AttributionAttention
from budget_attention import make_budget_attention


class FeatureAttention(nn.Module):
    def __init__(self, variant, channels=None):
        super().__init__()
        self.variant = variant
        self.configuration = config_for(variant)
        c = self.configuration
        if c['kind'] not in ('M1', 'N2'):
            raise ValueError('No attention block for the plain baseline')
        channels = c['channels'] if channels is None else channels
        if channels != c['channels']:
            raise ValueError('Native stage channels do not match configuration')
        self.channel = nn.Sequential(nn.Conv2d(channels, 10, 1), nn.ReLU(), nn.Conv2d(10, channels, 1))
        nn.init.zeros_(self.channel[-1].weight)
        nn.init.zeros_(self.channel[-1].bias)
        if c['kind'] == 'M1':
            self.spatial = nn.Conv2d(2, 1, 7, padding=3)
            nn.init.zeros_(self.spatial.weight)
            nn.init.zeros_(self.spatial.bias)
        else:
            self.spatial = nn.Sequential(nn.Conv2d(12, 7, 1), nn.ReLU(), nn.Conv2d(7, 1, 1))
            nn.init.zeros_(self.spatial[-1].weight)
            nn.init.zeros_(self.spatial[-1].bias)
            for number, (kc, sc, ks, ss) in enumerate(SCALES):
                self.register_buffer(f'center_{number}', torch.tensor(gaussian_kernel(kc, sc * c['sigma_factor']), dtype=torch.float32)[None, None])
                self.register_buffer(f'surround_{number}', torch.tensor(gaussian_kernel(ks, ss * c['sigma_factor']), dtype=torch.float32)[None, None])

    @staticmethod
    def smooth(image, kernel):
        side = kernel.shape[-1]
        return F.conv2d(F.pad(image, (side // 2,) * 4, mode='replicate'),
                        kernel.to(dtype=image.dtype).expand(2, 1, side, side).contiguous(), groups=2)

    def spatial_features(self, stats):
        c = self.configuration
        if c['kind'] == 'M1':
            return stats
        z = stats - stats.mean(dim=(-2, -1), keepdim=True)
        global_var = (z * z).mean(dim=(-2, -1), keepdim=True)
        responses = []
        for number in (0, 1, 2):
            center = self.smooth(z, getattr(self, f'center_{number}'))
            local_mean = self.smooth(z, getattr(self, f'surround_{number}'))
            numerator = center - local_mean
            local_var = (self.smooth(z * z, getattr(self, f'surround_{number}')) - local_mean.square()).clamp_min(0)
            variance = local_var + c['rho'] ** 2 * global_var
            response = numerator / torch.sqrt(variance + c['epsilon'] ** 2)
            responses.append(torch.cat((F.relu(response), F.relu(-response)), dim=1))
        return torch.cat(responses, dim=1)

    def forward(self, feature):
        channel_logits = self.channel(F.adaptive_avg_pool2d(feature, 1)) + self.channel(F.adaptive_max_pool2d(feature, 1))
        refined = feature * (2 * torch.sigmoid(channel_logits))
        stats = torch.cat((refined.mean(dim=1, keepdim=True), refined.max(dim=1, keepdim=True).values), dim=1)
        return refined * (2 * torch.sigmoid(self.spatial(self.spatial_features(stats))))


class Branch(nn.Module):
    def __init__(self, variant, imagenet_state):
        super().__init__()
        self.configuration = config_for(variant)
        self.encoder = efficientnet_b0(weights=None)
        self.encoder.load_state_dict(imagenet_state, strict=True)
        self.encoder.classifier = nn.Identity()
        self.stage_attention = (nn.Identity() if variant == 'M0R' else
                                N3Attention() if variant == 'N3_adaptive_s4' else
                                make_budget_attention(variant) if variant in ('M1_budget_s4','N3_late_pooled_s4','N3_magnitude_basis_s4') else AttributionAttention(mode=self.configuration['mode']) if self.configuration['kind'] == 'N3' else FeatureAttention(variant))
        self.grade_head = nn.Sequential(nn.Dropout(0.25), nn.Linear(1280, 5))
        self.lesion_head = nn.Sequential(nn.Dropout(0.20), nn.Linear(1280, 7))
        self.projection = nn.Sequential(nn.Linear(1280, 512), nn.GELU(), nn.Dropout(0.10), nn.Linear(512, 256))

    def forward(self, image):
        feature = image
        c = self.configuration
        for number, stage in enumerate(self.encoder.features):
            feature = stage(feature)
            if number == c['stage_index']:
                if feature.shape[1] != c['channels'] or tuple(feature.shape[-2:]) != (image.shape[-2] // c['stride'], image.shape[-1] // c['stride']):
                    raise RuntimeError('Unexpected native stage channel/spatial dimensions')
                feature = self.stage_attention(feature)
        pooled = torch.flatten(self.encoder.avgpool(feature), 1)
        return {'feature': pooled, 'grade_logits': self.grade_head(pooled),
                'lesion_logits': self.lesion_head(pooled),
                'embedding': F.normalize(self.projection(pooled), dim=1)}


class BioVisionModel(nn.Module):
    def __init__(self, variant, imagenet_state):
        super().__init__()
        if variant not in CONFIGS:
            raise ValueError('Unknown design-space variant')
        self.variant = variant
        self.cfp = Branch(variant, imagenet_state)
        self.uwf = Branch(variant, imagenet_state)

    def forward_modality(self, modality, image):
        if modality not in ('CFP', 'UWF'):
            raise ValueError('Unexpected modality')
        return (self.cfp if modality == 'CFP' else self.uwf)(image)
