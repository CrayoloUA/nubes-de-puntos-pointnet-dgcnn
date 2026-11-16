"""PointNet y DGCNN para clasificación de nubes de puntos (PyTorch)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TNet(nn.Module):
    """Red de transformación espacial de PointNet (k x k)."""
    def __init__(self, k):
        super().__init__()
        self.k = k
        self.conv = nn.Sequential(
            nn.Conv1d(k, 64, 1, bias=False), nn.BatchNorm1d(64), nn.ReLU(True),
            nn.Conv1d(64, 128, 1, bias=False), nn.BatchNorm1d(128), nn.ReLU(True),
            nn.Conv1d(128, 1024, 1, bias=False), nn.BatchNorm1d(1024), nn.ReLU(True))
        self.fc = nn.Sequential(
            nn.Linear(1024, 512, bias=False), nn.BatchNorm1d(512), nn.ReLU(True),
            nn.Linear(512, 256, bias=False), nn.BatchNorm1d(256), nn.ReLU(True),
            nn.Linear(256, k * k))
        nn.init.zeros_(self.fc[-1].weight); nn.init.zeros_(self.fc[-1].bias)

    def forward(self, x):                       # x: B x k x N
        f = self.conv(x).max(dim=2)[0]
        m = self.fc(f).view(-1, self.k, self.k)
        return m + torch.eye(self.k, device=x.device)


class PointNet(nn.Module):
    def __init__(self, num_classes, dropout=0.3):
        super().__init__()
        self.t1, self.t2 = TNet(3), TNet(64)
        self.mlp1 = nn.Sequential(nn.Conv1d(3, 64, 1, bias=False), nn.BatchNorm1d(64), nn.ReLU(True),
                                  nn.Conv1d(64, 64, 1, bias=False), nn.BatchNorm1d(64), nn.ReLU(True))
        self.mlp2 = nn.Sequential(nn.Conv1d(64, 64, 1, bias=False), nn.BatchNorm1d(64), nn.ReLU(True),
                                  nn.Conv1d(64, 128, 1, bias=False), nn.BatchNorm1d(128), nn.ReLU(True),
                                  nn.Conv1d(128, 1024, 1, bias=False), nn.BatchNorm1d(1024), nn.ReLU(True))
        self.head = nn.Sequential(
            nn.Linear(1024, 512, bias=False), nn.BatchNorm1d(512), nn.ReLU(True), nn.Dropout(dropout),
            nn.Linear(512, 256, bias=False), nn.BatchNorm1d(256), nn.ReLU(True), nn.Dropout(dropout),
            nn.Linear(256, num_classes))
        self.feat_mat = None

    def forward(self, x):                       # x: B x N x 3
        x = x.transpose(1, 2)
        x = torch.bmm(self.t1(x), x)
        x = self.mlp1(x)
        m = self.t2(x); self.feat_mat = m
        x = torch.bmm(m, x)
        x = self.mlp2(x).max(dim=2)[0]
        return self.head(x)

    def reg_loss(self):
        m = self.feat_mat
        I = torch.eye(m.size(1), device=m.device)
        return torch.mean(torch.norm(torch.bmm(m, m.transpose(1, 2)) - I, dim=(1, 2)))


def knn(x, k):                                   # x: B x C x N
    # Los vecinos se calculan siempre en float32, aunque el resto corra en precisión mixta:
    # con fp16 las distancias pierden resolución y los vecinos cambian.
    with torch.autocast(device_type=x.device.type, enabled=False):
        x = x.float()
        inner = -2 * torch.matmul(x.transpose(2, 1), x)
        xx = (x ** 2).sum(dim=1, keepdim=True)
        return (-xx - inner - xx.transpose(2, 1)).topk(k=k, dim=-1)[1]


def graph_feature(x, k, idx=None):
    B, C, N = x.shape
    if idx is None:
        idx = knn(x, k)
    idx = (idx + torch.arange(B, device=x.device).view(-1, 1, 1) * N).view(-1)
    xt = x.transpose(2, 1).contiguous()
    nb = xt.view(B * N, -1)[idx].view(B, N, k, C)
    xt = xt.view(B, N, 1, C).expand(-1, -1, k, -1)
    return torch.cat((nb - xt, xt), dim=3).permute(0, 3, 1, 2).contiguous()   # B x 2C x N x k


class DGCNN(nn.Module):
    def __init__(self, num_classes, k=20, emb=1024, dropout=0.5):
        super().__init__()
        self.k = k
        def blk(i, o): return nn.Sequential(nn.Conv2d(i, o, 1, bias=False), nn.BatchNorm2d(o), nn.LeakyReLU(0.2))
        self.e1, self.e2, self.e3, self.e4 = blk(6, 64), blk(128, 64), blk(128, 128), blk(256, 256)
        self.conv5 = nn.Sequential(nn.Conv1d(512, emb, 1, bias=False), nn.BatchNorm1d(emb), nn.LeakyReLU(0.2))
        self.head = nn.Sequential(
            nn.Linear(emb * 2, 512, bias=False), nn.BatchNorm1d(512), nn.LeakyReLU(0.2), nn.Dropout(dropout),
            nn.Linear(512, 256, bias=False), nn.BatchNorm1d(256), nn.LeakyReLU(0.2), nn.Dropout(dropout),
            nn.Linear(256, num_classes))

    def forward(self, x):                       # x: B x N x 3
        x = x.transpose(1, 2)
        feats = []
        for blk in (self.e1, self.e2, self.e3, self.e4):
            x = blk(graph_feature(x, self.k)).max(dim=-1)[0]
            feats.append(x)
        x = self.conv5(torch.cat(feats, dim=1))
        x = torch.cat((x.max(dim=-1)[0], x.mean(dim=-1)), dim=1)
        return self.head(x)


def build(name, num_classes):
    return {"pointnet": PointNet, "dgcnn": DGCNN}[name](num_classes)
