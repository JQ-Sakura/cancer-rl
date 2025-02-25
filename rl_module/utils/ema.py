import torch
import torch.nn as nn
from copy import deepcopy
from typing import Dict, Optional

class ModelEMA:
    """Model Exponential Moving Average"""
    
    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.999,
        device: Optional[str] = None
    ):
        """
        Initialize EMA
        
        Args:
            model: Model to apply EMA
            decay: EMA decay rate
            device: Device to store EMA model
        """
        self.ema = deepcopy(model)
        self.ema.eval()
        self.decay = decay
        self.device = device if device else next(model.parameters()).device
        self.ema.to(self.device)
        self.parameter_names = {name for name, _ in self.ema.named_parameters()}
        
    def get_model(self) -> nn.Module:
        """Get EMA model"""
        return self.ema
    
    def update(self, model: nn.Module):
        """Update EMA parameters"""
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in self.parameter_names:
                    ema_param = self.ema.state_dict()[name]
                    model_param = param.detach()
                    ema_param.copy_(ema_param * self.decay + model_param * (1 - self.decay))
                    
    def state_dict(self) -> Dict:
        """Get EMA state dict"""
        return self.ema.state_dict()
    
    def load_state_dict(self, state_dict: Dict):
        """Load EMA state dict"""
        self.ema.load_state_dict(state_dict) 