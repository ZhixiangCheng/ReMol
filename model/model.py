import torchvision
import torch
from torch import nn


class ReMol(nn.Module):
    def __init__(self, dim=1024, depth=2, heads=2, emb_dropout=0.1, pretrained=False):
        super(ReMol, self).__init__()
        self.resnet = torchvision.models.resnet18(pretrained=pretrained)
        self.resnet.fc = torch.nn.Linear(self.resnet.fc.in_features, 1024)

        encoder_layers = nn.TransformerEncoderLayer(d_model=dim, nhead=heads)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=depth)

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.pool = "cls"
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Linear(dim, 2)

    def forward(self, img, mask, device, reactant, mlp):

        if reactant:
            b, m, C, H, W = img.shape
            reactant_imgs = img.view(-1, C, H, W)
            features = self.resnet(reactant_imgs)  # [batch_size * max_reactants, feature_dim]
            x = features.view(b, m, -1)

            cls_tokens = self.cls_token.expand(b, -1, -1)
            cls_mask = torch.zeros(b, 1, dtype=torch.bool).to(device)
            padding_mask = torch.cat((cls_mask, mask), dim=1)
            x = torch.cat((cls_tokens, x), dim=1)

            x = self.dropout(x) # [batch_size, max_reactants+1, feature_dim]
            x = x.permute(1, 0, 2)  # [max_reactants+1, batch_size, feature_dim]
            x = self.transformer_encoder(x, src_key_padding_mask=padding_mask)
        else:
            b, C, H, W = img.shape
            x = self.resnet(img)
            x = x.unsqueeze(1)
            cls_tokens = self.cls_token.expand(b, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1).to(device)
            x = self.dropout(x)
            x = x.permute(1, 0, 2)
            x = self.transformer_encoder(x)

        x = x.mean(dim=1) if self.pool == 'mean' else x[0, :, :]   # [batch_size, feature_dim]
        x = self.to_latent(x)

        if mlp:
            x = self.mlp_head(x)

        return x


