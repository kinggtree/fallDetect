import torch
import torch.nn as nn


class TimeDistributedEncoder(nn.Module):
    def __init__(self, module):
        super(TimeDistributedEncoder, self).__init__()
        self.module = module

    def forward(self, x):
        batch_size, time_steps, seq_len, n_features = x.size()

        x_reshape = x.contiguous().view(batch_size * time_steps, seq_len, n_features)
        hidden, _ = self.module(x_reshape)
        output_features = hidden[-1]
        y = output_features.view(batch_size, time_steps, -1)
        
        return y
    
class CrossAttention(nn.Module):
    def __init__(self, query_dim, key_dim, hidden_dim):
        super(CrossAttention, self).__init__()
        self.query_layer = nn.Linear(query_dim, hidden_dim)
        self.key_layer = nn.Linear(key_dim, hidden_dim)
        self.value_layer = nn.Linear(key_dim, hidden_dim)
        self.scale = hidden_dim ** -0.5

    def forward(self, query, key, value):
        Q = self.query_layer(query)
        K = self.key_layer(key)
        V = self.value_layer(value)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attention_weights = torch.softmax(attention_scores, dim=-1)
        context_vector = torch.matmul(attention_weights, V)
        return context_vector


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, dropout):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        outputs, (hidden, cell) = self.lstm(x)
        # 返回最终的 hidden 和 cell 状态
        return hidden, cell


class ContextualFidelityModel(nn.Module):
    # 参数名修改得更清晰
    def __init__(self, lfs_feature_dim, lstm_hidden_dim, hfs_feature_dim, num_classes=1):
        super(ContextualFidelityModel, self).__init__()

        hfs_encoder = Encoder(input_dim=48, hidden_dim=hfs_feature_dim, n_layers=2, dropout=0.1)
        self.hfs_processor = TimeDistributedEncoder(hfs_encoder)

        self.lfs_processor = nn.LSTM(
            input_size=lfs_feature_dim, # 将接收64维特征
            hidden_size=lstm_hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.5
        )

        self.cross_attention = CrossAttention(
            query_dim=lstm_hidden_dim,
            key_dim=hfs_feature_dim, # 维度变为64
            hidden_dim=lstm_hidden_dim
        )
        
        self.post_fusion_processor = nn.LSTM(
            input_size=lstm_hidden_dim * 2, # Concatenated input
            hidden_size=lstm_hidden_dim,
            num_layers=1,
            batch_first=True
        )

        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, feature_sequence, imputed_raw_sequence):
        lfs_output, _ = self.lfs_processor(feature_sequence) # -> (B, 60, lstm_hidden_dim)
        hfs_output = self.hfs_processor(imputed_raw_sequence) # -> (B, 60, raw_cnn_output_dim)
        attention_context = self.cross_attention(
            query=lfs_output, 
            key=hfs_output, 
            value=hfs_output
        ) # -> (B, 60, lstm_hidden_dim)

        combined_features = torch.cat([lfs_output, attention_context], dim=-1)

        final_sequence, (h_n, _) = self.post_fusion_processor(combined_features)
        
        last_step_output = final_sequence[:, -1, :]
        logits = self.classifier(last_step_output)
        state_feature = h_n.squeeze(0) # -> (B, lstm_hidden_dim)

        return logits, state_feature

