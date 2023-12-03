import numpy as np
import tensorflow as tf
from layer.common.attention import CrossAttention
from layer.common.feed_forward import point_wise_feed_forward_network
from layer.decoder import DecoderLayer
from layer.encoder import EncoderLayer

from masking import create_look_ahead_mask, create_padding_mask
from model.decoder import Decoder
from model.encoder import Encoder
from model.transformer import Transformer

# sample_ca = CrossAttention(num_heads=2, key_dim=512)

# print(pt_emb.shape)
# print(en_emb.shape)
# print(sample_ca(pt_emb, en_emb, en_emb, None).shape)
