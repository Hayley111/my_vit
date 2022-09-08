from .core.cnn import CNN
from .core.transformer import ViT

def build_model(type:str):
    if type == 'cnn':
        model = CNN()
    if type == 'vit':
        model = ViT()
    return model
