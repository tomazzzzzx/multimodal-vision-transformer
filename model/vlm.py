import torch.nn as nn

class MultimodalVLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision = None
        self.proj = nn.Sequential(nn.Linear(1152, 4096), nn.GELU(), nn.Linear(4096, 4096))
        self.decoder = None
    def forward(self, images, input_ids):
        return self.decoder(inputs_embeds=self.proj(self.vision(images)))
