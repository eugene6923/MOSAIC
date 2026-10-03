from torch import nn
import torch.nn.functional as F
import torch
import torchvision.models as models
import sys
sys.path.append('./diffusers/src')
from diffusers import DiffusionPipeline, UNet2DConditionModel

class condition_encoder(nn.Module):
    def __init__(self, input_dim=768, output_dim=64, dropout_rate=0.1, hidden_dims= None, num_tokens=1):
        super(condition_encoder, self).__init__()
        self.config = {
            "input_dim": input_dim,
            "output_dim": output_dim,
            "hidden_dims": hidden_dims,
            "dropout_rate": dropout_rate,
            "num_tokens": num_tokens
        }
        
        if hidden_dims is None or len(hidden_dims) == 0:
            self.fc = nn.Linear(input_dim, output_dim)
        else:
            self.fc = nn.ModuleList()
            for idx, i in enumerate(hidden_dims):
                self.fc.append(nn.Linear(input_dim, i))
                input_dim = i
                if idx == len(hidden_dims) - 1:
                    self.fc.append(nn.Linear(input_dim, output_dim))
                    
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_rate)
        
        self.hidden_dims = hidden_dims
        self.num_tokens = num_tokens

        if self.num_tokens > 1:
                # Token-specific learnable offsets
            self.token_proj = nn.Parameter(torch.randn(num_tokens, output_dim))
            # Positional encodings for each token
            self.pos_emb = nn.Parameter(torch.randn(1, num_tokens, output_dim))
            
            self.scale = nn.Parameter(torch.ones(1)) 
            
    @property
    def dtype(self):
        return self.fc.weight.dtype

    def forward(self, x):

        if self.hidden_dims and len(self.hidden_dims) > 0:
            for layer in self.fc:
                x = layer(x)
                x = self.relu(x)
                x = self.dropout(x)

        else:
            if isinstance(self.fc, nn.Embedding):
                # Handle one-hot vectors properly
                print(f"Input shape: {x.shape}, Input sample: {x.flatten()[:10]}")
                
                # Flatten to 2D if needed
                if x.dim() > 2:
                    x = x.view(-1, x.shape[-1])
                
                # Convert one-hot to indices
                if x.shape[-1] > 1:  # One-hot vector
                    x = torch.argmax(x, dim=-1)
                    print(f"After argmax: {x}")
                
                # Ensure correct dtype for embedding
                x = x.long()
                print(f"Before embedding lookup: {x}")
            
            x = self.fc(x)
            x = self.relu(x)
            x = self.dropout(x)
        

        if self.num_tokens > 1:

            if x.dim() == 2:
                x = x.unsqueeze(1)  
            elif x.dim() == 4:
                x = x.squeeze(1) 
            
            x = x + self.token_proj.unsqueeze(0)  
            x = x + self.pos_emb
            x = x * self.scale

        return x
    
    def save_pretrained(self, save_directory):
        import os, json, torch
        os.makedirs(save_directory, exist_ok=True)
        # Save weights
        torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))
        # Save config
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            json.dump(self.config, f)
    
    def register_to_config(self, **kwargs):
        if not hasattr(self, "config"):
            self.config = {}
        self.config.update(kwargs)

    @classmethod
    def from_pretrained(cls, load_directory, subfolder=None):
        import os, json
        if subfolder is None:
            with open(os.path.join(load_directory, "config.json"), "r") as f:
                config = json.load(f)
            state_dict = torch.load(os.path.join(load_directory, "pytorch_model.bin"), map_location="cpu")
            
        else:
            with open(os.path.join(load_directory, subfolder, "config.json"), "r") as f:
                config = json.load(f)
            state_dict = torch.load(os.path.join(load_directory, subfolder, "pytorch_model.bin"), map_location="cpu")
        model = cls(**config)
        model.load_state_dict(state_dict)
        return model


class two_condition_encoder(nn.Module):
    def __init__(self, input_dim=768, output_dim=64, dropout_rate=0.1, hidden_dims= None, num_tokens=1):
        super(two_condition_encoder, self).__init__()
        self.config = {
            "input_dim": input_dim,
            "output_dim": output_dim,
            "hidden_dims": hidden_dims,
            "dropout_rate": dropout_rate,
        }
        if hidden_dims is None or len(hidden_dims) == 0:
            self.fc_color = nn.Linear(input_dim, output_dim)
            self.fc_counting = nn.Linear(input_dim, output_dim)
            self.use_hidden = False
        else:
            self.use_hidden = True

            self.fc_color = nn.ModuleList()
            self.fc_counting = nn.ModuleList()

            for idx, i in enumerate(hidden_dims):
                self.fc_color.append(nn.Linear(input_dim, i))
                self.fc_counting.append(nn.Linear(input_dim, i))
                input_dim = i
                if idx == len(hidden_dims) - 1:
                    self.fc_color.append(nn.Linear(input_dim, output_dim))
                    self.fc_counting.append(nn.Linear(input_dim, output_dim))
                    
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_rate)
        
        self.hidden_dims = hidden_dims
            
    @property
    def dtype(self):
        if self.use_hidden:
            return self.fc_color[0].weight.dtype
        else:
            return self.fc_color.weight.dtype

    def forward(self, x):
        if self.hidden_dims and len(self.hidden_dims) > 0:
            x_color = x[:,0,:]
            x_count = x[:,1,:]
            for layer_color, layer_count in zip(self.fc_color, self.fc_counting):
                x_color = layer_color(x_color)
                x_color = self.relu(x_color)
                x_color = self.dropout(x_color)

                x_count = layer_count(x_count)
                x_count = self.relu(x_count)
                x_count = self.dropout(x_count)
                
            color = x_color
            count = x_count
        else:
            color = self.fc_color(x[:,0,:])      # (B,C)
            count = self.fc_counting(x[:,1,:])   # (B,C)

        color = self.relu(color)
        count = self.relu(count)

        color = self.dropout(color)
        count = self.dropout(count)

        out = torch.stack((color, count), dim=1)   # (B,2,C)
        return out
    
    def save_pretrained(self, save_directory):
        import os, json, torch
        os.makedirs(save_directory, exist_ok=True)
        # Save weights
        torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))
        # Save config
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            json.dump(self.config, f)
    
    def register_to_config(self, **kwargs):
        if not hasattr(self, "config"):
            self.config = {}
        self.config.update(kwargs)

    @classmethod
    def from_pretrained(cls, load_directory, subfolder=None):
        import os, json
        if subfolder is None:
            with open(os.path.join(load_directory, "config.json"), "r") as f:
                config = json.load(f)
            state_dict = torch.load(os.path.join(load_directory, "pytorch_model.bin"), map_location="cpu")
            
        else:
            with open(os.path.join(load_directory, subfolder, "config.json"), "r") as f:
                config = json.load(f)
            state_dict = torch.load(os.path.join(load_directory, subfolder, "pytorch_model.bin"), map_location="cpu")
        model = cls(**config)
        model.load_state_dict(state_dict)
        return model


class ClassificationHead(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.fc(x)

class Classifier(nn.Module):
    def __init__(self, num_classes=10, num_colors=None):
        super().__init__()
        
        self.features = nn.Sequential(
            # Block 1: 128x128 -> 64x64
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1),
            
            # Block 2: 64x64 -> 32x32
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),
            
            # Block 3: 32x32 -> 16x16
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.3),
            
            # Block 4: 16x16 -> 8x8
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))  # Global pooling to 4x4
        )
        if num_colors is None:
            self.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(512 * 4 * 4, 1024),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(1024, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(256, num_classes)
            )
        else:
            self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True)
        )
            self.count_heads = nn.ModuleList([
            nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(512, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, num_classes)  # 0-10 counts = 11 classes
            ) for _ in range(num_colors)
        ])
        
        self.num_colors = num_colors

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        if self.num_colors is not None:
            color_counts = []
            for head in self.count_heads:
                color_counts.append(head(x)) 
            return torch.stack(color_counts, dim=1) # Each is (batch_size, 11) for counts 0-10
        return x

class PretrainedClassifier(nn.Module):
    def __init__(self, num_classes=10, num_classes2 = None):
        super().__init__()
        
        # Use pretrained ResNet18
        self.backbone = models.resnet18(pretrained=True)
        
        # Replace the final layer
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

        if num_classes2 is not None:
            self.backbone.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(512, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(256, num_classes2)
            )

    def forward(self, x):
        return self.backbone(x)

