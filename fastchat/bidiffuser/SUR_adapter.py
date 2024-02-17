import torch
import torch.nn as nn

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, hidden_size=768):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.query_layer = nn.Linear(hidden_size, hidden_size)
        self.key_layer = nn.Linear(hidden_size, hidden_size)
        self.value_layer = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, query, key, value):
        Q = self.query_layer(query)
        K = self.key_layer(key)
        V = value

        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self.hidden_size, dtype=torch.float32)
        )
        attention_weights = nn.functional.softmax(scores, dim=-1)
        weighted_values = torch.matmul(attention_weights, V) + value
        result = self.output_layer(weighted_values) + weighted_values

        return result


class Adapter(nn.Module):
    def __init__(self, depth=2, adapter_weight=0.2, sd_text_size=768):
        super(Adapter, self).__init__()

        self.adapter_weight = adapter_weight
        
        # Transformer Encoder
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=sd_text_size, nhead=8, dim_feedforward=2048
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=depth
        )

        # Attension
        self.attention = Attention(hidden_size=sd_text_size)

        self.mlp = Mlp(in_features=4096, hidden_features=768 * 4, out_features=768, act_layer=nn.GELU,
                                 drop=0.)

        # LLM layer
        self.fc = nn.Linear(sd_text_size, sd_text_size)
        nn.init.zeros_(self.fc.weight)

    def forward(self, x, llm_x):
        llm_x = self.mlp(llm_x)
        out_transformer_encoder = self.transformer_encoder(llm_x)
        out_attention = self.attention(query=x, key=out_transformer_encoder, value=out_transformer_encoder)
        out_llm = self.fc(out_attention)
        out = self.adapter_weight * out_llm + (1 - self.adapter_weight) * x

        return out, out_transformer_encoder, out_llm