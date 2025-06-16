import torch
import torch.nn as nn


class BaseModel(nn.Module):
    def __init__(self, ):
        super().__init__()

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def from_pretrain(pretrained_model_conf_or_path, *args, **kwargs):
        from . import get

        conf = torch.load(
            pretrained_model_conf_or_path, map_location="cpu"
        )  # Attempt to find the model and instantiate it.
        
        model_class = get(conf["model_name"])     
        model = model_class(*args, **kwargs)
        
        model.load_state_dict(conf["state_dict"])         

        return model

    def serialize(self):
        import pytorch_lightning as pl  # Not used in torch.hub

        model_conf = dict(
            model_name=self.__class__.__name__,
            state_dict=self.get_state_dict(),
        )
        # Additional infos
        infos = dict()
        infos["software_versions"] = dict(
            torch_version=torch.__version__, pytorch_lightning_version=pl.__version__,
        )
        model_conf["infos"] = infos
        return model_conf

    def get_state_dict(self):
        """In case the state dict needs to be modified before sharing the model."""
        return self.state_dict()