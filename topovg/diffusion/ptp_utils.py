"""Attention capture used by TopoVG.

Retains the working Diffusers processor, conditional-head selection and
streaming averages. The supplied Prompt-to-Prompt utilities informed the
attention-capture interface; Compel, notebook display and editing utilities
are not dependencies of this implementation."""
import math
from ..utils.backend import LazyModule

torch = LazyModule('torch')

class AttentionAccumulator:
    """Head-weighted averaging of conditional attention, streaming by layer.

    The incoming tensor contains BOTH uncond and cond batch members. Its second
    half is selected once.
    """
    def reset(self):
        self.sums = {}
        self.counts = {}

    def __init__(self):
        self.reset()

    @staticmethod
    def key(is_cross, queries):
        if is_cross and queries in (64, 256):
            return ('cross', math.isqrt(queries))
        if not is_cross and queries == 4096:
            return ('self', 64)
        return None

    def start(self, key, queries, tokens, heads):
        if key is None:
            return
        if key not in self.sums:
            self.sums[key] = torch.zeros((queries, tokens), dtype=torch.float32)
            self.counts[key] = 0
        self.counts[key] += heads

    def add(self, key, attention, start, stop):
        if key is not None:
            heads = attention.shape[0] // 2
            # Sum in FP32 BEFORE transfer; no full head stack is retained on CPU.
            self.sums[key][start:stop].add_(attention[heads:].float().sum(0).cpu())

    def result(self, key):
        if key not in self.sums or self.counts[key] == 0:
            raise RuntimeError(f'Missing attention layers for {key}')
        return self.sums[key] / self.counts[key]


class CaptureAttentionProcessor:
    """Standard Diffusers attention with observable probabilities.

    Query slicing bounds GPU memory without slicing the softmax key dimension.
    No xformers processor is used: it hides the probability maps being saved.
    """
    def __init__(self, accumulator, query_chunk):
        self.accumulator, self.query_chunk = accumulator, query_chunk

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, temb=None, *args, **kwargs):
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height*width).transpose(1, 2)
        batch_size = hidden_states.shape[0]
        if batch_size != 2:
            raise ValueError('Expected exactly [uncond, cond], one sample at a time')
        is_cross = encoder_hidden_states is not None
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        query = attn.to_q(hidden_states)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        query, key, value = (attn.head_to_batch_dim(x) for x in (query, key, value))
        q_len, k_len = query.shape[1], key.shape[1]
        attention_mask = attn.prepare_attention_mask(attention_mask, k_len, batch_size)
        capture_key = self.accumulator.key(is_cross, q_len)
        self.accumulator.start(capture_key, q_len, k_len, query.shape[0] // 2)
        chunk = self.query_chunk or q_len
        output = torch.empty_like(query)
        for begin in range(0, q_len, chunk):
            end = min(begin+chunk, q_len)
            mask = attention_mask
            if mask is not None and mask.shape[-2] != 1:
                mask = mask[:, begin:end]
            probabilities = attn.get_attention_scores(query[:, begin:end], key, mask)
            self.accumulator.add(capture_key, probabilities, begin, end)
            output[:, begin:end] = torch.bmm(probabilities, value)
            del probabilities
        hidden_states = attn.batch_to_head_dim(output)
        hidden_states = attn.to_out[1](attn.to_out[0](hidden_states))
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        return hidden_states / attn.rescale_output_factor
