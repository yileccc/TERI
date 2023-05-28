import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import *


class Transformer_tagging(nn.Module):

    def __init__(self, model_dimension, fourier_dimension, time_dimension, vocab_size, number_of_heads, number_of_layers, number_cls,
                 dropout_probability, device, log_attention_weights=False, position_encoding=True):
        super(Transformer_tagging, self).__init__()

        self.src_embedding = nn.Embedding(vocab_size, model_dimension)
        self.device = device

        self.time_embedding = LearnableFourierPositionalEncoding(1, fourier_dimension, time_dimension)
        self.dist_embedding = LearnableFourierPositionalEncoding(2, fourier_dimension, time_dimension)
        self.pos_encoding = position_encoding

        self.gcn = GCN(model_dimension, [model_dimension, model_dimension], model_dimension, dropout_probability)

        if position_encoding is True:
            self.pos_embedding = nn.Embedding(200, model_dimension)

        self.encoder = Encoder(model_dimension, number_of_heads, dropout_probability, number_of_layers, device)
        self.mlp = nn.Linear(model_dimension, number_cls)

        self.projection = nn.Sequential(nn.Linear(model_dimension, model_dimension),
                                        nn.ReLU(),
                                        nn.Linear(model_dimension, 512))

        self.init_params()

    def init_params(self, default_initialization=False):
        if not default_initialization:
            for name, p in self.named_parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)


    def forward(self, src_token_ids_batch, src_time_batch, src_coor_batch, src_mask, adj_graph, type):
        src_representations = self.encode(src_token_ids_batch, src_time_batch, src_coor_batch, src_mask, adj_graph)
        # outputs = self.mlp(src_representations_batch) #[N, L, num_cls]
        if type == 'tagging':
            outputs = self.decode(src_representations)
        elif type == 'contrastive':
            outputs = src_representations

        return outputs


    def encode(self, src_token_ids_batch, src_time_batch, src_dist_batch, src_mask, adj_graph):
        (bs, seq_len) = src_token_ids_batch.shape

        src_cxt_embeddings = self.gcn(self.src_embedding.weight, adj_graph)
        src_embeddings_batch = self.src_embedding(src_token_ids_batch)  # get embedding vectors for src token ids
        src_embeddings_batch_cxt = src_cxt_embeddings[src_token_ids_batch.view(-1)].view(bs, seq_len, -1)
        src_embeddings_batch = src_embeddings_batch_cxt + src_embeddings_batch

        if self.pos_encoding:
            src_pos_batch = torch.arange(src_token_ids_batch.size(1), device=self.device).unsqueeze(0).repeat(src_token_ids_batch.size(0), 1)
            src_pos_embeddings_batch = self.pos_embedding(src_pos_batch)
            src_embeddings_batch = src_embeddings_batch + src_pos_embeddings_batch

        src_time_embeddings_batch = self.time_embedding(src_time_batch)
        src_dist_embeddings_batch = self.dist_embedding(src_dist_batch)

        src_representations_batch = self.encoder(src_embeddings_batch, src_time_embeddings_batch,
                                                 src_dist_embeddings_batch, src_mask)  # forward pass through the encoder

        return src_representations_batch

    def decode(self, src_representations):
        outputs = self.mlp(src_representations)

        return outputs
#
# Encoder architecture
#

class Encoder(nn.Module):

    def __init__(self, d_model, num_heads, dropout_probability, num_layers, device):
        super().__init__()

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, dropout_probability).to(device)])

        for _ in range(num_layers-1):
            self.encoder_layers.append(EncoderLayer(d_model, num_heads, dropout_probability).to(device))

        self.norm = nn.LayerNorm(d_model)

    def forward(self, src_embeddings_batch, src_time_embeddings_batch, src_dist_embeddings_batch, src_mask):
        src_representations_batch = src_embeddings_batch + src_time_embeddings_batch + src_dist_embeddings_batch

        for encoder_layer in self.encoder_layers:
            src_representations_batch = encoder_layer(src_representations_batch, src_mask)

        return self.norm(src_representations_batch)


class EncoderLayer(nn.Module):

    def __init__(self, model_dimension, num_heads, dropout_probability):
        super().__init__()

        self.model_dimension = model_dimension

        self.layernorm1 = nn.LayerNorm(model_dimension)
        self.layernorm2 = nn.LayerNorm(model_dimension)

        self.dropout1 = nn.Dropout(p=dropout_probability)
        self.dropout2 = nn.Dropout(p=dropout_probability)

        self.mha = MultiHeadedAttention(model_dimension, num_heads)
        self.pointwise_net = PositionwiseFeedForwardNet(model_dimension)



    def forward(self, src_representations_batch, src_mask):

        attn_output, attn = self.mha(src_representations_batch, src_representations_batch,
                                     src_representations_batch, src_mask)
        attn_output = self.dropout1(attn_output)
        out1  = self.layernorm1(src_representations_batch + attn_output)

        ffn_output = self.pointwise_net(out1)
        ffn_output = self.dropout2(ffn_output)
        out2 = self.layernorm2(ffn_output+out1)

        return out2




class PositionwiseFeedForwardNet(nn.Module):

    def __init__(self, model_dimension, width_mult=1):
        super().__init__()

        self.linear1 = nn.Linear(model_dimension, width_mult * model_dimension)
        self.linear2 = nn.Linear(width_mult * model_dimension, model_dimension)
        self.relu = nn.ReLU()

    def forward(self, representations_batch):
        return self.linear2(self.relu(self.linear1(representations_batch)))


class MultiHeadedAttention(nn.Module):

    def __init__(self, model_dimension, number_of_heads, log_attention_weights=True):
        super().__init__()
        assert model_dimension % number_of_heads == 0, f'Model dimension must be divisible by the number of heads.'

        self.head_dimension = int(model_dimension / number_of_heads)
        self.number_of_heads = number_of_heads

        self.wq = nn.Linear(model_dimension, model_dimension)
        self.wk = nn.Linear(model_dimension, model_dimension)
        self.wv = nn.Linear(model_dimension, model_dimension)

        self.out_projection_net = nn.Linear(model_dimension, model_dimension)

        self.softmax = nn.Softmax(dim=-1)

        self.log_attention_weights = log_attention_weights


    def attention(self, query, key, value, mask):

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dimension)

        if mask is not None:
            scores.masked_fill_(mask == torch.tensor(False), float("-inf"))

        attention_weights = self.softmax(scores)
        intermediate_token_representations = torch.matmul(attention_weights, value)

        return intermediate_token_representations, attention_weights  # attention weights for visualization purposes

    def forward(self, query, key, value, mask):
        batch_size = query.shape[0]
        query = self.wq(query).view(batch_size, -1, self.number_of_heads, self.head_dimension).transpose(1, 2) #(B, NH, S, HD)
        key = self.wk(key).view(batch_size, -1, self.number_of_heads, self.head_dimension).transpose(1, 2)
        value = self.wv(value).view(batch_size, -1, self.number_of_heads, self.head_dimension).transpose(1, 2)

        intermediate_token_representations, attention_weights = self.attention(query, key, value, mask)

        reshaped = intermediate_token_representations.transpose(1, 2).reshape(batch_size, -1, self.number_of_heads * self.head_dimension)

        token_representations = self.out_projection_net(reshaped)

        if self.log_attention_weights:
            return token_representations, attention_weights
        else:
            return token_representations


#
# Input modules
#

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, ninput, nhid, noutput, dropout):
        super(GCN, self).__init__()

        self.gcn = nn.ModuleList()
        self.dropout = dropout
        self.leaky_relu = nn.LeakyReLU(0.2)

        channels = [ninput] + nhid + [noutput]
        for i in range(len(channels) - 1):
            gcn_layer = GraphConvolution(channels[i], channels[i + 1])
            self.gcn.append(gcn_layer)

    def forward(self, x, adj):
        for i in range(len(self.gcn) - 1):
            x = self.leaky_relu(self.gcn[i](x, adj))

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gcn[-1](x, adj)

        return x


class Embedding(nn.Module):

    def __init__(self, vocab_size, model_dimension):
        super().__init__()
        self.embeddings_table = nn.Embedding(vocab_size, model_dimension)
        self.model_dimension = model_dimension

    def forward(self, token_ids_batch):
        assert token_ids_batch.ndim == 2, f'Expected: (batch size, max token sequence length), got {token_ids_batch.shape}'

        embeddings = self.embeddings_table(token_ids_batch)
        return embeddings * math.sqrt(self.model_dimension)


class LearnableFourierPositionalEncoding(nn.Module):
    def __init__(self, M: int, F_dim: int, D: int, gamma=1.0):
        """
        :param M: each point has a M-dimensional positional values
        :param F_dim: depth of the Fourier feature dimension
        :param H_dim: hidden layer dimension
        :param D: positional encoding dimension
        :param gamma: parameter to initialize Wr
        """
        super().__init__()
        self.M = M
        self.F_dim = F_dim
        self.H_dim = D
        self.D = D
        self.gamma = gamma

        # Projection matrix on learned lines (used in eq. 2)
        self.Wr = nn.Linear(self.M, self.F_dim // 2, bias=False)
        # MLP (GeLU(F @ W1 + B1) @ W2 + B2 (eq. 6)
        self.mlp = nn.Sequential(
            nn.Linear(self.F_dim, self.H_dim, bias=True),
            nn.GELU(),
            nn.Linear(self.H_dim, self.D)
        )

        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.Wr.weight.data, mean=0, std=self.gamma ** -2)

    def forward(self, x):
        """
        Produce positional encodings from x
        :param x: tensor of shape [B, N, M] that represents N positions where each position is in the shape of M with batch size B,
        :return: positional encoding for X [B, N, D]
        """
        B, N, M = x.shape
        # Step 1. Compute Fourier features (eq. 2)
        projected = self.Wr(x)
        cosines = torch.cos(projected)
        sines = torch.sin(projected)
        F = 1 / np.sqrt(self.F_dim) * torch.cat([cosines, sines], dim=-1)
        # Step 2. Compute projected Fourier features (eq. 6)
        Y = self.mlp(F)
        # Step 3. Reshape to x's shape
        pos_enc = Y.reshape((B, N, self.D))
        return pos_enc



class PositionalEncoding(nn.Module):

    def __init__(self, model_dimension, dropout_probability, device, expected_max_sequence_length=500):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_probability)

        position_id = torch.arange(0, expected_max_sequence_length).unsqueeze(1)
        frequencies = torch.pow(10000., -torch.arange(0, model_dimension, 2, dtype=torch.float) / model_dimension)

        positional_encodings_table = torch.zeros(expected_max_sequence_length, model_dimension).to(device)
        positional_encodings_table[:, 0::2] = torch.sin(position_id * frequencies).to(device)  # sine on even positions
        positional_encodings_table[:, 1::2] = torch.cos(position_id * frequencies).to(device)  # cosine on odd positions

        self.register_buffer('positional_encodings_table', positional_encodings_table)

    def forward(self, embeddings_batch):
        assert embeddings_batch.ndim == 3 and embeddings_batch.shape[-1] == self.positional_encodings_table.shape[1], \
            f'Expected (batch size, max token sequence length, model dimension) got {embeddings_batch.shape}'

        positional_encodings = self.positional_encodings_table[:embeddings_batch.shape[1]]

        return self.dropout(embeddings_batch + positional_encodings)



class CL_Loss(nn.Module):
    def __init__(self, temperature, device):
        super(CL_Loss, self).__init__()
        self.device = device
        self.temperature = temperature
        self.cossim = nn.CosineSimilarity(dim=-1).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.type = 'mat'

    def forward(self, model, representations, mask, input_lengths):
        """
        contrastive learning loss given one pair sequences
        inputs: [batch1_data, batch2data], shape: 2B x S x D
        """
        traj_reps = torch.sum(representations * mask.squeeze().unsqueeze(-1).float(), dim=1) / input_lengths.unsqueeze(-1).float()  # 2B x D

        traj_projs = model.projection(traj_reps)


        batch_size = traj_projs.shape[0] // 2
        batch_sample_one, batch_sample_two = torch.split(traj_projs, batch_size)

        if self.type == 'cos':
            sim11 = self.cossim(batch_sample_one.unsqueeze(1), batch_sample_one.unsqueeze(0)) / self.temperature
            sim22 = self.cossim(batch_sample_two.unsqueeze(1), batch_sample_two.unsqueeze(0)) / self.temperature
            sim12 = self.cossim(batch_sample_one.unsqueeze(1), batch_sample_two.unsqueeze(0)) / self.temperature
        else:
            sim11 = torch.matmul(batch_sample_one, batch_sample_one.T) / self.temperature
            sim22 = torch.matmul(batch_sample_two, batch_sample_two.T) / self.temperature
            sim12 = torch.matmul(batch_sample_one, batch_sample_two.T) / self.temperature


        mask_sim = torch.eye(sim12.shape[-1], dtype=torch.long).to(self.device)
        sim11[mask_sim == 1] = float("-inf")
        sim22[mask_sim == 1] = float("-inf")

        raw_scores1 = torch.cat([sim12, sim11], dim=-1)
        raw_scores2 = torch.cat([sim22, sim12.transpose(-1, -2)], dim=-1)
        logits = torch.cat([raw_scores1, raw_scores2], dim=-2)
        labels = torch.arange(2 * sim12.shape[-1], dtype=torch.long, device=logits.device)
        ce_loss = self.criterion(logits, labels)
        return ce_loss


