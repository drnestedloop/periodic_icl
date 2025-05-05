import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2Model
from transformers.models.gpt2.modeling_gpt2 import ACT2FN

### Defaults are equivalent to 'standard' in Garg et al.

class GPT2Regressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        hidden_dim: int = 256,
        num_layers: int = 12,
        num_heads: int = 8,
        # vocab_size: int = 1,
        max_seq_len: int = 128,
        dropout: float = 0.0,
        activation_fn: str = None,
    ):
        """
        Args:
            input_dim:    dimensionality D of each real-valued input vector.
            output_dim:   dimensionality of each real-valued output (defaults to a scalar).
            hidden_dim:   GPT embedding size (n_embd).
            num_layers:   number of transformer blocks.
            num_heads:    number of attention heads.
            vocab_size:   dummy vocab size (we bypass wte by using inputs_embeds).
            max_seq_len:  max context length (n_positions / n_ctx).
            dropout:      dropout probability for embeddings & residuals.
        """
        super().__init__()
        # 1) a learnable linear projection from your R^input_dim into GPT's embedding space
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # 2) build a GPT-2 config & model
        config = GPT2Config(
            # vocab_size=vocab_size,
            n_embd=hidden_dim,
            n_layer=num_layers,
            n_head=num_heads,
            n_positions=max_seq_len,
            n_ctx=max_seq_len,
            resid_pdrop=dropout,
            embd_pdrop=dropout,
            attn_pdrop=dropout,
        )
        if activation_fn is not None:
            config.activation_function = activation_fn
        self.transformer = GPT2Model(config)

        # 3) a tiny head to map hidden states back down to your real-valued output
        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            x:               (batch, seq_len, input_dim) real-valued inputs
            attention_mask:  (batch, seq_len) mask (1 = keep, 0 = pad), optional

        Returns:
            preds: (batch, seq_len, output_dim) real-valued outputs
        """
        # project inputs into the GPT embedding space
        # -> (batch, seq_len, hidden_dim)
        embeddings = self.input_proj(x)

        # feed into GPT-2; note: we pass inputs_embeds to bypass the token
        # embedding lookup (wte). GPT2Model will add its positional embeddings.
        outputs = self.transformer(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
        )
        # last_hidden_state: (batch, seq_len, hidden_dim)
        h = outputs.last_hidden_state

        # optional dropout + final linear layer
        h = self.dropout(h)
        preds = self.output_proj(h)  # (batch, seq_len, output_dim)

        return preds

class XPlusSin2(nn.Module): # Sinosoidal activation from Ziyin et al. (2020)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x + ( sin(x) )^2
        return x + torch.sin(x).pow(2)

ACT2FN["x_plus_sin2"] = XPlusSin2