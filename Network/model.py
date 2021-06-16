from torch.nn.modules.activation import LeakyReLU
from torch.nn.modules.dropout import Dropout
from transformers import BertConfig, BertModel
import torch
import torch.nn as nn
from ResNetAE.ResNetAE import ResNetAE

class Reduce(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self,x):
        x = torch.flatten(x, start_dim=1)
        x = x.mean(dim=1)
        return x

class Inception(nn.Module):
    def __init__(self, emb_dim, in_channels=1, out_channels_per_conv=16):
        super().__init__()
        
        self.five       = nn.Conv2d(in_channels=in_channels, out_channels=out_channels_per_conv, kernel_size=(5,emb_dim), stride=(1,1), padding=(2,0), bias=False)
        self.three      = nn.Conv2d(in_channels=in_channels, out_channels=out_channels_per_conv, kernel_size=(3,emb_dim), stride=(1,1), padding=(1,0), bias=False)
        self.stride2    = nn.Conv2d(in_channels=in_channels, out_channels=out_channels_per_conv, kernel_size=(3,emb_dim), stride=(1,2), padding=(1,0), bias=False)
    
    def forward(self,x):
        out_five    = self.five(x)
        out_three   = self.three(x)
        out_stride2 = self.stride2(x)
        
        output = torch.cat((out_five, out_three, out_stride2),dim=1).permute(0,3,2,1) # Cat on channel dim
        return output
        
# New Multi Branch Auto Encoding Transformer
class UVT(nn.Module):
    def __init__(self, AE, Transformer, frames_per_video, embedding_dim, SDmode, rm_branch=None, num_hidden_layers=None):
        super(UVT, self).__init__()
        
        if SDmode == 'cla':
            last_features = 3
            # last_layer = nn.Softmax(dim=-1)
        elif SDmode == 'reg':
            last_features = 1
            self.activation = nn.Tanh()
        else:
            raise ValueError(SDmode, "should be 'reg' or 'cla'")
        
        self.AE = AE
        self.T  = Transformer
        self.rm_branch = rm_branch
        self.SDmode = SDmode
        
        if num_hidden_layers is not None:
            if not self.rm_branch == 'SD':
                self.extremas = nn.Sequential(
                    Inception(emb_dim=embedding_dim, in_channels=1, out_channels_per_conv=embedding_dim//4),
                    Inception(emb_dim=(embedding_dim//4)*3, in_channels=1, out_channels_per_conv=embedding_dim//16),
                    Inception(emb_dim=(embedding_dim//16)*3, in_channels=1, out_channels_per_conv=embedding_dim//64),
                    Inception(emb_dim=(embedding_dim//64)*3, in_channels=1, out_channels_per_conv=1),
                    nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1,3), bias=False)
                )
                
            if not self.rm_branch == 'EF':
                self.ejection = nn.Sequential(
                    Inception(emb_dim=embedding_dim, in_channels=1, out_channels_per_conv=embedding_dim//8),
                    Inception(emb_dim=(embedding_dim//8)*3, in_channels=1, out_channels_per_conv=embedding_dim//64),
                    Inception(emb_dim=(embedding_dim//64)*3, in_channels=1, out_channels_per_conv=1),
                    nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1,3), bias=False),
                    Reduce(),
                    nn.Sigmoid()
                )
        else:
            if not self.rm_branch == 'SD' :
                self.extremas = nn.Sequential(
                    nn.Linear(in_features=embedding_dim, out_features=embedding_dim//2, bias=True),
                    nn.LayerNorm(embedding_dim//2),
                    nn.LeakyReLU(negative_slope=0.05, inplace=True),
                    nn.Linear(in_features=embedding_dim//2, out_features=embedding_dim//4, bias=True),
                    nn.LayerNorm(embedding_dim//4),
                    nn.LeakyReLU(negative_slope=0.05, inplace=True),
                    nn.Linear(in_features=embedding_dim//4, out_features=last_features, bias=True)
                )
                
            if not self.rm_branch == 'EF':
                self.ejection = nn.Sequential(
                    nn.Linear(in_features=embedding_dim, out_features=embedding_dim//2, bias=True),
                    nn.LayerNorm(embedding_dim//2),
                    nn.LeakyReLU(negative_slope=0.05, inplace=True),
                    nn.Linear(in_features=embedding_dim//2, out_features=1, bias=True),
                    Reduce(),
                    nn.Sigmoid()
                )
        
        if num_hidden_layers is not None:
            self.averaging = nn.Conv2d(in_channels=(num_hidden_layers+1), out_channels=1, kernel_size=1, bias=False)
            self.use_conv  = True
        else:
            self.use_conv = False
        
        self.vf = frames_per_video
        self.em = embedding_dim
        
    def forward(self, x, nB, nF):        
        # (BxF) x C x H x W => (BxF) x Emb
        embeddings = self.AE.encode(x).squeeze()
        
        # B x F x Emb => AttHeads+1 x B x F x Emb
        outputs = self.T(embeddings.view(-1, nF, self.em), output_hidden_states=True)
        
        if self.use_conv:
            # AttHeads+1 x (B x F x Emb) => B x AttHeads+1 x F x Emb
            outputs = torch.stack(outputs.hidden_states, dim=1)
            # B x AttHeads+1 x F x Emb => B x 1 x F x Emb => B x F x Emb
            outputs = self.averaging(outputs)
        else:
            # AttHeads+1 x B x F x Emb => B x F x Emb
            outputs = torch.stack(outputs.hidden_states).mean(dim=0)
        
        
        if not self.rm_branch == 'SD':
            # B x F x Emb => B x F x 1
            classes_vec = self.extremas(outputs)
            if self.SDmode == 'reg':
                classes_vec = self.activation(classes_vec)
        else:
            classes_vec = None
        
        if not self.rm_branch == 'EF':
            # B x F x Emb => B x 1 x 1
            ef_prediction = self.ejection(outputs)
        else:
            ef_prediction = None
        
        return classes_vec, ef_prediction

def get_model(emb_dim, img_per_video, SDmode,
              num_hidden_layers = 16,
              intermediate_size = 8192,
              rm_branch = None,
              use_conv = False,
              attention_heads=16
              ):
    # https://huggingface.co/transformers/model_doc/bert.html#bertconfig
    # https://huggingface.co/transformers/model_doc/bert.html#bertfortokenclassification
    # Setup model
    configuration = BertConfig(
        vocab_size=1, # Set to 0/None ?
        hidden_size=emb_dim, # Length of embeddings
        num_hidden_layers=num_hidden_layers, # 16
        num_attention_heads=attention_heads, 
        intermediate_size=intermediate_size, # 8192
        hidden_act='gelu', 
        hidden_dropout_prob=0.1, 
        attention_probs_dropout_prob=0.1, 
        max_position_embeddings=1024, # 64 ?
        type_vocab_size=1, 
        initializer_range=0.02, 
        layer_norm_eps=1e-12, 
        pad_token_id=0, 
        gradient_checkpointing=False, 
        position_embedding_type='absolute', 
        use_cache=True)
    configuration.num_labels = 3 # 4
    
    model_T  = BertModel(configuration).encoder
    
    model_AE = ResNetAE(input_shape=(128, 128, 3), n_ResidualBlock=8, n_levels=4, bottleneck_dim=emb_dim)
    model_AE.decoder = None
    model_AE.fc2 = None
    
    num_berts = num_hidden_layers if use_conv else None
        
    model    = UVT(model_AE, model_T, img_per_video, emb_dim, SDmode, rm_branch=rm_branch, num_hidden_layers=num_berts) 
    
    return model