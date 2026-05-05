import torch
import torch.nn as nn
import torch.nn.functional as F

from models.SwinTransformers import SwinTransformer


def conv3x3_bn_gelu(in_ch, out_ch, k=3, s=1, p=1, b=False):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, k, s, p, bias=b),
        nn.BatchNorm2d(out_ch),
        nn.GELU(),
    )


def conv1x1_bn_gelu(in_ch, out_ch, b=False):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=b),
        nn.BatchNorm2d(out_ch),
        nn.GELU(),
    )


class ConvMlp(nn.Module):
    def __init__(self, in_ch, hidden_ch=None):
        super().__init__()
        hidden_ch = hidden_ch or in_ch
        self.fc1 = nn.Conv2d(in_ch, hidden_ch, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_ch, in_ch, 1)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class GaussianHead(nn.Module):
    def __init__(self, dim, logvar_clamp=(-10.0, 10.0)):
        super().__init__()
        self.logvar_clamp = logvar_clamp
        self.attn = nn.Sequential(
            nn.BatchNorm2d(dim),
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
            nn.GELU(),
        )
        self.norm_mu = nn.BatchNorm2d(dim)
        self.norm_lv = nn.BatchNorm2d(dim)
        self.head_mu = ConvMlp(dim)
        self.head_lv = ConvMlp(dim)

    def forward(self, x):
        h = x + self.attn(x)
        mu = self.head_mu(self.norm_mu(h))
        logvar = self.head_lv(self.norm_lv(h))
        logvar = torch.clamp(logvar, self.logvar_clamp[0], self.logvar_clamp[1])
        std = torch.exp(0.5 * logvar)
        return mu, logvar, std


class CrossModalGaussianHead(nn.Module):
    def __init__(self, dim, logvar_clamp=(-10.0, 10.0)):
        super().__init__()
        self.logvar_clamp = logvar_clamp
        self.cross_attn = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1),
            nn.BatchNorm2d(dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
        )
        self.norm_mu = nn.BatchNorm2d(dim)
        self.norm_lv = nn.BatchNorm2d(dim)
        self.head_mu = ConvMlp(dim)
        self.head_lv = ConvMlp(dim)

    def forward(self, z_v, z_t):
        h = self.cross_attn(torch.cat([z_v, z_t], dim=1))
        mu = self.head_mu(self.norm_mu(h))
        logvar = self.head_lv(self.norm_lv(h))
        logvar = torch.clamp(logvar, self.logvar_clamp[0], self.logvar_clamp[1])
        std = torch.exp(0.5 * logvar)
        return mu, logvar, std


def kl_to_standard_normal(mu, logvar):
    with torch.cuda.amp.autocast(enabled=False):
        mu = mu.float()
        logvar = logvar.float()
        kl = 0.5 * (torch.exp(logvar) + mu.pow(2) - 1.0 - logvar)
        return kl.mean()


class UAM(nn.Module):
    def __init__(self, in_ch, dim=None):
        super().__init__()
        dim = dim or max(32, in_ch // 4)
        self.proj_v = nn.Conv2d(in_ch, dim, 1)
        self.proj_t = nn.Conv2d(in_ch, dim, 1)
        self.head_v = GaussianHead(dim)
        self.head_t = GaussianHead(dim)
        self.head_vt = CrossModalGaussianHead(dim)
        self.mapping = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1),
            nn.BatchNorm2d(dim),
            nn.GELU(),
            nn.Conv2d(dim, in_ch, 1),
        )
        self.proj_v_out = nn.Sequential(
            nn.Conv2d(dim, in_ch, 1),
            nn.BatchNorm2d(in_ch),
            nn.GELU(),
        )

    @staticmethod
    def reparameterize(mu, std, training):
        if training:
            eps = torch.randn_like(std)
            return mu + std * eps
        return mu

    def forward(self, F_v, F_t):
        f_v = self.proj_v(F_v)
        f_t = self.proj_t(F_t)

        mu_v, logvar_v, std_v = self.head_v(f_v)
        mu_t, logvar_t, std_t = self.head_t(f_t)

        z_v = self.reparameterize(mu_v, std_v, self.training)
        z_t = self.reparameterize(mu_t, std_t, self.training)

        kl_v = kl_to_standard_normal(mu_v, logvar_v)
        kl_t = kl_to_standard_normal(mu_t, logvar_t)

        mu_vt, logvar_vt, std_vt = self.head_vt(mu_v, mu_t)
        z_vt = self.reparameterize(mu_vt, std_vt, self.training)

        F_t_tilde = self.mapping(torch.cat([z_t, z_vt], dim=1))
        F_v_out = self.proj_v_out(z_v) + F_v
        return F_v_out, F_t_tilde, logvar_v, logvar_t, kl_v, kl_t


class ConfidenceGenerator(nn.Module):
    def __init__(self, T_init=1.0, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.log_T = nn.Parameter(torch.tensor(float(torch.log(torch.tensor(T_init)))))
        self.h = nn.Sequential(
            nn.Conv2d(2, 8, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, 1),
        )

    def _inv_uncertainty(self, logvar):
        var = torch.exp(F.softplus(logvar))
        denom = var.mean(dim=1, keepdim=True) + self.eps
        return 1.0 / denom

    def forward(self, logvar_v, logvar_t):
        invu_v = self._inv_uncertainty(logvar_v)
        invu_t = self._inv_uncertainty(logvar_t)
        x = torch.cat([invu_v, invu_t], dim=1)
        T = torch.exp(self.log_T).clamp(min=1e-3)
        conf = torch.sigmoid(self.h(x) / T)
        return conf


class CGM(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.in_ch = in_ch
        self.conf_gen = ConfidenceGenerator()
        hid = max(in_ch // 4, 16)
        self.channel_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, hid, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hid, in_ch * 2, 1),
        )
        self.spatial_branch = nn.Sequential(
            nn.Conv2d(in_ch, hid, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hid, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, F_v, F_t_tilde, logvar_v, logvar_t):
        conf = self.conf_gen(logvar_v, logvar_t)
        gb = self.channel_branch(F_t_tilde)
        gamma_t, beta_t = torch.chunk(gb, 2, dim=1)
        F_m = (gamma_t * F_v + beta_t) * conf
        P_t = self.spatial_branch(F_t_tilde)
        P_t_tilde = P_t * conf
        F_fused = F_v + P_t_tilde * (F_m - F_v)
        return F_fused


class UpsamplingBlock(nn.Module):
    def __init__(self, in_ch_prev, in_ch_skip, out_ch):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.fuse = conv3x3_bn_gelu(in_ch_prev + in_ch_skip, out_ch)
        self.refine = conv3x3_bn_gelu(out_ch, out_ch)

    def forward(self, d_prev, skip):
        d_prev = self.upsample(d_prev)
        if d_prev.shape[-2:] != skip.shape[-2:]:
            d_prev = F.interpolate(d_prev, size=skip.shape[-2:], mode='bilinear', align_corners=True)
        x = torch.cat([d_prev, skip], dim=1)
        x = self.fuse(x)
        x = self.refine(x)
        return x


class Predictor(nn.Module):
    def __init__(self, in_ch, out_size=384):
        super().__init__()
        self.out_size = out_size
        self.sal_head = nn.Sequential(
            conv3x3_bn_gelu(in_ch, max(in_ch // 2, 16)),
            nn.Conv2d(max(in_ch // 2, 16), 1, 3, padding=1),
        )
        self.bd_head = nn.Sequential(
            conv3x3_bn_gelu(in_ch, max(in_ch // 2, 16)),
            nn.Conv2d(max(in_ch // 2, 16), 1, 3, padding=1),
        )

    def forward(self, x):
        sal = F.interpolate(self.sal_head(x), size=self.out_size, mode='bilinear', align_corners=True)
        bd = F.interpolate(self.bd_head(x), size=self.out_size, mode='bilinear', align_corners=True)
        return sal, bd


class UMFNet(nn.Module):
    SWIN_CHANNELS = (128, 256, 512, 1024)

    def __init__(self, swin_embed_dim=128, swin_depths=(2, 2, 18, 2), swin_heads=(4, 8, 16, 32), fuse_layers=(True, True, True, True)):
        super().__init__()
        assert len(fuse_layers) == 4
        self.fuse_layers = fuse_layers
        self.rgb_swin = SwinTransformer(embed_dim=swin_embed_dim, depths=list(swin_depths), num_heads=list(swin_heads))
        self.t_swin = SwinTransformer(embed_dim=swin_embed_dim, depths=list(swin_depths), num_heads=list(swin_heads))

        c1, c2, c3, c4 = self.SWIN_CHANNELS
        self.uam1 = UAM(c1) if fuse_layers[0] else None
        self.uam2 = UAM(c2) if fuse_layers[1] else None
        self.uam3 = UAM(c3) if fuse_layers[2] else None
        self.uam4 = UAM(c4) if fuse_layers[3] else None

        self.cgm1 = CGM(c1) if fuse_layers[0] else None
        self.cgm2 = CGM(c2) if fuse_layers[1] else None
        self.cgm3 = CGM(c3) if fuse_layers[2] else None
        self.cgm4 = CGM(c4) if fuse_layers[3] else None

        self.up1 = UpsamplingBlock(in_ch_prev=c4, in_ch_skip=c3, out_ch=c3)
        self.up2 = UpsamplingBlock(in_ch_prev=c3, in_ch_skip=c2, out_ch=c2)
        self.up3 = UpsamplingBlock(in_ch_prev=c2, in_ch_skip=c1, out_ch=c1)
        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            conv3x3_bn_gelu(c1, c1 // 2),
            conv3x3_bn_gelu(c1 // 2, c1 // 2),
        )

        self.predictor1 = Predictor(c1 // 2)
        self.predictor2 = Predictor(c1)
        self.predictor3 = Predictor(c2)
        self.predictor4 = Predictor(c3)

    def _fuse_one_scale(self, F_v, F_t, uam, cgm):
        if uam is None:
            return F_v + F_t, F_v.new_zeros(()), F_v.new_zeros(())
        F_v_out, F_t_tilde, logvar_v, logvar_t, kl_v, kl_t = uam(F_v, F_t)
        F_fused = cgm(F_v_out, F_t_tilde, logvar_v, logvar_t)
        return F_fused, kl_v, kl_t

    def forward(self, rgb, t):
        rgb_feats = self.rgb_swin(rgb)
        t_feats = self.t_swin(t)

        if len(rgb_feats) < 4 or len(t_feats) < 4:
            raise ValueError(f'Expected at least 4 Swin features, got rgb={len(rgb_feats)}, thermal={len(t_feats)}')

        Fv_1, Fv_2, Fv_3, Fv_4 = rgb_feats[:4]
        Ft_1, Ft_2, Ft_3, Ft_4 = t_feats[:4]

        F_fused_1, klv1, klt1 = self._fuse_one_scale(Fv_1, Ft_1, self.uam1, self.cgm1)
        F_fused_2, klv2, klt2 = self._fuse_one_scale(Fv_2, Ft_2, self.uam2, self.cgm2)
        F_fused_3, klv3, klt3 = self._fuse_one_scale(Fv_3, Ft_3, self.uam3, self.cgm3)
        F_fused_4, klv4, klt4 = self._fuse_one_scale(Fv_4, Ft_4, self.uam4, self.cgm4)

        d3 = self.up1(F_fused_4, F_fused_3)
        d2 = self.up2(d3, F_fused_2)
        d1 = self.up3(d2, F_fused_1)
        d0 = self.up4(d1)

        sal1, bd1 = self.predictor1(d0)
        sal2, bd2 = self.predictor2(d1)
        sal3, bd3 = self.predictor3(d2)
        sal4, bd4 = self.predictor4(d3)

        kl_v = klv1 + klv2 + klv3 + klv4
        kl_t = klt1 + klt2 + klt3 + klt4

        return {
            'sal': [sal1, sal2, sal3, sal4],
            'bd': [bd1, bd2, bd3, bd4],
            'kl_v': kl_v,
            'kl_t': kl_t,
        }

    def load_pre(self, pre_model):
        sd = torch.load(pre_model, map_location='cpu')
        sd = sd.get('model', sd)
        self.rgb_swin.load_state_dict(sd, strict=False)
        self.t_swin.load_state_dict(sd, strict=False)
        print(f'[UMFNet] loaded pretrained Swin-B from {pre_model} into both visible & thermal streams.')
