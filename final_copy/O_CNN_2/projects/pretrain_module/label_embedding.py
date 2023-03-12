from torch import nn
import torch
import random


class RandomLabelEmbeddings(nn.Module):

    def __init__(self, output_dim, embed_path, return_projection=False):
        super().__init__()
        self.candidates_embedding = torch.load(embed_path)  # [200, 20, 768]
        # self.embedding = nn.Embedding.from_pretrained(, freeze=True)
        self.inp_dim = self.candidates_embedding.shape[-1]
        self.return_projection = return_projection
        if return_projection:
            self.proj = nn.Sequential(nn.LeakyReLU(inplace=True),
                                    nn.Linear(self.inp_dim, output_dim))
        self.random_set_candidate_embedding()

    def random_set_candidate_embedding(self):
        device = next(self.proj.parameters()).device

        embeds = [torch.zeros(self.inp_dim).to(device)]
        for cands in self.candidates_embedding:
            embeds.append(random.choice(cands).to(device))
        embeds = torch.stack(embeds)
        self.embedding = nn.Embedding.from_pretrained(embeds,
                                                      freeze=True,
                                                      padding_idx=0).to(device)

    def forward(self, x: torch.LongTensor, *args, **kwargs):
        with torch.no_grad():
            x = self.embedding(x)
        return self.proj(x)


class AttentionLabelEmbedding(nn.Module):

    def __init__(self, output_dim, embed_path, return_projection=False):
        super().__init__()
        candidates_embedding: torch.Tensor = torch.load(
            embed_path, map_location='cpu')  # [num_cls - 1, S, E] -> [num_cls - 1, E]
        self.num_cls = candidates_embedding.shape[0] + 1
        self.embed_dim = candidates_embedding.shape[-1]
        self.num_expr = candidates_embedding.shape[1]
        self.return_projection = return_projection
        if return_projection:
            self.proj = nn.Sequential(nn.LeakyReLU(inplace=True),
                                    nn.Linear(self.embed_dim, output_dim))

        self.num_cls = candidates_embedding.shape[0] + 1
        self.embed_dim = candidates_embedding.shape[-1]
        self.num_expr = candidates_embedding.shape[1]
        embeds = torch.cat((torch.zeros(1, self.num_expr, self.embed_dim),
                           candidates_embedding),
                           dim=0)
        self.register_buffer('candidates_embedding', embeds)
        self.attn_pooling = nn.Sequential(
                                nn.Linear(embeds.shape[-1], 1, bias=False),
                                nn.Tanh()
        )

    def forward(self, x: torch.LongTensor,*args, **kwargs):
        """
        x: [batch_size]
        """
        cands_embeds = self.candidates_embedding[x]
        attn_weight = torch.softmax(self.attn_pooling(cands_embeds), dim=1)  # [num_cls, num_cands, 1]
        embeds = (cands_embeds.permute(0, 2, 1) @ attn_weight).squeeze(-1)
        if self.return_projection:
            embeds = self.proj(embeds)
        # print(embeds.shape)
        return embeds



class WeightBy3DLabelEmbedding(nn.Module):

    def __init__(self, output_dim, embed_path, return_projection=False):
        super().__init__()
        candidates_embedding: torch.Tensor = torch.load(
            embed_path, map_location='cpu')  # [num_cls - 1, S, E] -> [num_cls - 1, E]
        self.num_cls = candidates_embedding.shape[0] + 1
        self.embed_dim = candidates_embedding.shape[-1]
        self.num_expr = candidates_embedding.shape[1]
        self.return_projection = return_projection
        if return_projection:
            self.proj = nn.Sequential(nn.LeakyReLU(inplace=True),
                                    nn.Linear(self.embed_dim, output_dim))

        self.num_cls = candidates_embedding.shape[0] + 1
        self.embed_dim = candidates_embedding.shape[-1]
        self.num_expr = candidates_embedding.shape[1]
        embeds = torch.cat((torch.zeros(1, self.num_expr, self.embed_dim),
                           candidates_embedding),
                           dim=0)
        self.register_buffer('candidates_embedding', embeds)
        self.proj3d2weight = nn.Sequential(
                            nn.Linear(output_dim, self.num_expr, bias=False),
                            nn.Tanh(),
                        )

    def forward(self, label: torch.LongTensor, feature3d: torch.Tensor=None):
        """
        x: [batch_size]
        feature_3d [batch_size, dim]
        """
        cands_embeds = self.candidates_embedding[label] # [B, num_expr,E]
        if feature3d != None:
            attn_weight = torch.softmax(self.proj3d2weight(feature3d), dim=-1) # [B, num_expr]
        else:
            attn_weight = torch.ones(label.shape[0], self.num_expr, device=label.device)
            attn_weight = torch.softmax(attn_weight, dim = -1)
        embeds = (cands_embeds.permute(0, 2, 1) @ attn_weight.unsqueeze(-1)).squeeze(-1) 
        if self.return_projection:
            embeds = self.proj(embeds)
        # print(embeds.shape)
        return embeds
    

if __name__ == '__main__':
    embedings = WeightBy3DLabelEmbedding(
        96, embed_path='../label_text/explaination_embeddings.pth', return_projection=True)
    labels = torch.randint(0, 10, (5, ))
    feature3d = torch.rand((5, 96))
    embeds = embedings(labels,  )
    print(embeds)
    print(embeds.shape)
