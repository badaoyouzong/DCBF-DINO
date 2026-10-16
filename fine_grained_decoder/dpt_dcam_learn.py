import torch.nn as nn
import torch.nn.functional as F


from fine_grained_decoder.blocks import FeatureFusionBlock
from new.DCAM_learn import DCAM_Module
from new.BAFF import BAFF_Module


def _make_fusion_block(features, use_bn, size=None):
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )


class DPTHead(nn.Module):
    def __init__(
            self,
            nclass,
            in_channels,
            features=128,
            use_bn=False,
            out_channels=[96, 192, 384, 768],
            use_dcam=True,
            use_baff=True
    ):
        super(DPTHead, self).__init__()

        self.use_dcam = use_dcam
        self.use_baff = use_baff

        self.projects = nn.ModuleList([
            nn.Conv2d(in_channels, oc, kernel_size=1) for oc in out_channels
        ])

        self.resize_layers = nn.ModuleList([
            nn.Sequential(
                nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
                nn.Conv2d(out_channels[0], out_channels[0], kernel_size=3, padding=1, bias=not use_bn),
                nn.BatchNorm2d(out_channels[0]) if use_bn else nn.Identity(),
                nn.GELU()
            ),
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(out_channels[1], out_channels[1], kernel_size=3, padding=1, bias=not use_bn),
                nn.BatchNorm2d(out_channels[1]) if use_bn else nn.Identity(),
                nn.GELU()
            ),
            nn.Identity(),
            nn.Sequential(
                nn.Conv2d(out_channels[3], out_channels[3], kernel_size=3, stride=2, padding=1, bias=not use_bn),
                nn.BatchNorm2d(out_channels[3]) if use_bn else nn.Identity(),
                nn.GELU()
            )
        ])

        self.rn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels[i], features, kernel_size=3, padding=1, bias=not use_bn),
                nn.BatchNorm2d(features) if use_bn else nn.Identity(),
                nn.GELU()
            ) for i in range(4)
        ])

        if self.use_dcam:
            self.dcam_blocks = nn.ModuleList([
                DCAM_Module(features) for _ in range(3)
            ])
        else:
            self.dcam_blocks = None

        if self.use_baff:
            self.baff_blocks = nn.ModuleList([
                BAFF_Module(features) for _ in range(3)
            ])
        else:
            self.baff_blocks = None

        self.output_conv = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(features, nclass, kernel_size=1)
        )

        self.aux_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(features, features // 2, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(features // 2),
                nn.GELU(),
                nn.Conv2d(features // 2, nclass, kernel_size=1)
            ) for _ in range(2)
        ])

        self._is_verified = False

    def forward(self, out_features, patch_h, patch_w):
        processed = []
        for i, x in enumerate(out_features):
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            x = self.rn_layers[i](x)
            processed.append(x)

        l1, l2, l3, l4 = processed

        path_3 = F.interpolate(l4, size=l3.shape[2:], mode='bilinear', align_corners=True)

        if self.use_dcam:
            path_3 = self.dcam_blocks[2](path_3, l3)
        else:
            path_3 = path_3

        aux_out1 = self.aux_heads[0](path_3)

        if self.use_baff:
            path_3_baff = self.baff_blocks[2](path_3, aux_out1)
        else:
            path_3_baff = path_3

        path_2 = F.interpolate(path_3_baff, size=l2.shape[2:], mode='bilinear', align_corners=True)

        if self.use_dcam:
            path_2 = self.dcam_blocks[1](path_2, l2)
        else:
            path_2 = path_2

        aux_out2 = self.aux_heads[1](path_2)

        if self.use_baff:
            path_2_baff = self.baff_blocks[1](path_2, aux_out2)
        else:
            path_2_baff = path_2

        path_1 = F.interpolate(path_2_baff, size=l1.shape[2:], mode='bilinear', align_corners=True)

        if self.use_dcam:
            path_1 = self.dcam_blocks[0](path_1, l1)
        else:
            path_1 = path_1

        temp_main_out = self.output_conv(path_1)

        if self.use_baff:
            path_1_baff = self.baff_blocks[0](path_1, temp_main_out)
        else:
            path_1_baff = path_1

        main_out = self.output_conv[-1](path_1_baff)

        main_out = F.interpolate(main_out, scale_factor=2, mode='bilinear', align_corners=True)

        if self.training:
            return main_out, aux_out1, aux_out2

        return main_out


class DPT(nn.Module):
    def __init__(
            self,
            encoder_size='base',
            nclass=1,
            features=128,
            out_channels=[96, 192, 384, 768],
            use_bn=False,
            backbone=None,
            use_dcam=True,
            use_baff=True
    ):
        super(DPT, self).__init__()

        self.intermediate_layer_idx = {
            'small': [2, 5, 8, 11],
            'base': [2, 5, 8, 11],
            'large': [4, 11, 17, 23],
            'giant': [9, 19, 29, 39]
        }

        self.encoder_size = encoder_size
        self.backbone = backbone

        self.head = DPTHead(nclass, self.backbone.embed_dim, features, use_bn, out_channels=out_channels,
                            use_dcam=use_dcam, use_baff=use_baff)

    def lock_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def forward(self, x):
        patch_h, patch_w = x.shape[-2] // 16, x.shape[-1] // 16

        if hasattr(self.backbone, 'base_model'):
            features = self.backbone.base_model.model.get_intermediate_layers(
                x, n=self.intermediate_layer_idx[self.encoder_size]
            )
        else:
            features = self.backbone.get_intermediate_layers(
                x, n=self.intermediate_layer_idx[self.encoder_size]
            )

        head_out = self.head(features, patch_h, patch_w)

        target_size = (patch_h * 16, patch_w * 16)

        if self.training:
            main_out, aux1, aux2 = head_out
            main_out = F.interpolate(main_out, size=target_size, mode='bilinear', align_corners=True)
            aux1 = F.interpolate(aux1, size=target_size, mode='bilinear', align_corners=True)
            aux2 = F.interpolate(aux2, size=target_size, mode='bilinear', align_corners=True)
            return main_out, aux1, aux2
        else:
            out = F.interpolate(head_out, size=target_size, mode='bilinear', align_corners=True)
            return out
