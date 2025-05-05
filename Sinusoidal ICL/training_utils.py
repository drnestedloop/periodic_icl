import torch
import torch.nn.functional as F
import math
import torch.nn as nn
from transformers import GPT2Config, GPT2Model
from transformers.models.gpt2.modeling_gpt2 import ACT2FN

from tqdm.auto import tqdm
import wandb
import numpy as np

import core_utils
import custom_model

def generate_sine_batch(
    batch_size: int,
    grid_len: int,
    rand_len: int,
    num_sines: int = 16,
    domain=(0.0, 1.0), # period
    random_domain_factor: int = 2,
    device=torch.device('cuda'),
):
    """
    Returns a batch of interleaved [t, f(t)] sequences, all generated on `device`.

    - grid_len:               number of evenly spaced t's (torch.linspace over domain)
    - rand_len:               number of additional random t's per sample (uniform over domain)
    - num_sines:              number of sine components (frequencies 1..num_sines)
    - domain:                 (t_min, t_max) should be the length of one period
    - random_domain_factor:   number of periods to sample over
    """
    t_min, t_max = domain

    t_base = torch.linspace(t_min, t_max, steps=grid_len, device=device)
    rand_mat = torch.rand(batch_size, grid_len, device=device)
    perms   = rand_mat.argsort(dim=1)

    # 1) make the grid ts: shape (1, grid_len) -> expand to (batch_size, grid_len)
    t_grid = t_base.unsqueeze(0).expand(batch_size, -1)
    t_grid = t_grid.gather(1, perms)

    # 2) make the random ts: shape (batch_size, rand_len)
    # t_rand = torch.rand(batch_size, rand_len, device=device) * random_domain_factor * (t_max - t_min) + t_min # uniform random sampling
    t_rand = torch.randn(batch_size, rand_len, device=device) * 1 + 1 # normal random sampling with mu = 1, stdev = 1,

    # 3) concat to get all ts: shape (batch_size, total_len)
    t_all = torch.cat([t_grid, t_rand], dim=1)
    total_len = grid_len + rand_len

    # 4) prepare your sine parameters on GPU:
    # freqs = [[1,2,3,...,num_sines]] for each batch
    freqs  = 2 * torch.pi * torch.arange(1, num_sines+1, device=device) \
                   .unsqueeze(0) \
                   .expand(batch_size, num_sines)
    # phases = 0 or π/2 at random
    phases = torch.randint(0, 2, (batch_size, num_sines), device=device) \
                   .to(torch.float) * (math.pi/2)
    # amplitudes still random in [0,1)
    amps   = torch.rand(batch_size, num_sines, device=device)

    # 5) compute each wave over all ts:
    #    expand t_all to (batch_size, 1, total_len) so it broadcasts
    t_exp = t_all.unsqueeze(1)                     # (B, 1, L)
    arg   = freqs.unsqueeze(-1) * t_exp + phases.unsqueeze(-1)  # (B, S, L)
    waves = amps.unsqueeze(-1) * torch.sin(arg)    # (B, S, L)

    # 6) sum over the S sines --> (batch_size, total_len)
    f_all = waves.sum(dim=1)

    # 7) interleave t and f(t): stack-->(B, L, 2) then flatten --> (B, 2*L)
    interleaved = torch.stack([t_all, f_all], dim=-1)  # (B, L, 2)
    return interleaved.view(batch_size, total_len * 2)


def train(cfg: dict, use_wandb: bool = True) -> torch.nn.Module:
    """
    Trains a GPT2-based regressor on sine wave data using various curriculum strategies.

    Configuration Keys in `cfg` (defaults shown where applicable):

    Optional (with defaults):

        - name (str): Name for this run.
            Default: "sinusoidal_fit_max_waves_{max_waves}_{activation_fn}_activation_{curriculum}_chapter"

        - activation_fn (str): Activation function name used in the transformer.
            Default: "x_plus_sin2"

        - batch_size (int): Batch size per training step.
            Default: 32

        - curriculum (str): Curriculum strategy to use. One of:
            - "none"         --> fixed wave count
            - "standard"     --> increase waves from 1 to max_waves
            - "chapter"      --> alternating pure + mixed chapters
            - "interspersed" --> standard + random every N epochs
            - "random"       --> uniformly sample waves from [1, max_waves]
            Default: "chapter"

        - curriculum_config (dict): Additional config for curriculum types.
            For "chapter", uses:
                - chapter_waves (list[list[int]]): list of num waves for each chapter
                    Default: [[1], [2], [1, 2], [3], [1, 2, 3], [4], [1, 2, 3, 4], ...]
            For "interspersed", uses:
                - interspersed_every (int): frequency of random epochs.
                    Default: 10

        - lr (float): Learning rate.
            Default: 1e-4

        - epochs (int): Number of training epochs.
            Default: 55000

        - max_waves (int): Maximum number of sine wave components.
            Default: 5

        - grid_len (int): Number of fixed time points sampled per input.
            Default: 2 * max_waves + 1

        - rand_len (int): Number of random time points sampled per input.
            Default: 2 * max_waves + 1

        - save_every (int): How often to save the model (in epochs).
            Default: 5000

        - seed (int, optional): If provided, seeds Python, NumPy, and PyTorch RNGs.

    Args:
        cfg (dict): Configuration for training. Will be logged to Weights & Biases if enabled.
        use_wandb (bool): If True, initializes and logs to WandB. If False, runs silently.

    Returns:
        model (torch.nn.Module): Trained GPT2Regressor model.
    """
    

    # Required or defaulted fields (with order preserved)
    max_waves = cfg.get("max_waves", 5)
    activation_fn = cfg.get("activation_fn", "x_plus_sin2")
    curriculum = cfg.get("curriculum", "chapter")
    curriculum_config = cfg.get("curriculum_config", {})

    cfg.setdefault("name", f"sinusoidal_fit_max_waves_{max_waves}_{activation_fn}_activation_{curriculum}_curriculum")
    cfg.setdefault("activation_fn", activation_fn)
    cfg.setdefault("batch_size", 32)
    cfg.setdefault("curriculum", curriculum)
    cfg.setdefault("lr", 1e-4)
    cfg.setdefault("epochs", 55000)
    cfg.setdefault("max_waves", max_waves)
    cfg.setdefault("grid_len", max_waves * 2 + 1)
    cfg.setdefault("rand_len", max_waves * 2 + 1)
    cfg.setdefault("save_every", 5000)
    cfg.setdefault("entity", None)

    # Optional seed
    if "seed" in cfg:
        np.random.seed(cfg["seed"])
        torch.manual_seed(cfg["seed"])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg["seed"])
    
    if curriculum == "interspersed":
        curriculum_config.setdefault("interspersed_every", 10)
    elif curriculum == "chapter":
        def build_chapter_waves(max_waves):
            chapters = []
            for w in range(1, max_waves + 1):
                chapters.append([w])  # pure
                if w > 1:
                    chapters.append(list(range(1, w + 1)))  # mixed
            return chapters
        
        curriculum_config.setdefault("chapter_waves", build_chapter_waves(max_waves))
        curriculum_config.setdefault("num_chapters", len(curriculum_config["chapter_waves"]))
    cfg["curriculum_config"] = curriculum_config

    wave_scheduler = get_wave_scheduler(cfg)

    if use_wandb:
        core_utils.wandb_setup(project="sinusoidal_icl", name=cfg["name"], config=cfg, entity=cfg["entity"])
    
    core_utils.save_config(cfg=cfg, base_dir=f"sinusoidal_icl/checkpoints/{cfg['name']}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = custom_model.GPT2Regressor(
        input_dim=1,
        output_dim=1,
        max_seq_len=2 * (cfg["grid_len"] + cfg["rand_len"]),
        activation_fn=activation_fn,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    # Training loop
    global_step = 0  # Total training steps (used for logging)
    for epoch in tqdm(range(1, cfg["epochs"] + 1), desc="Epochs"):
        # 1. Select number of sine waves for this epoch using the curriculum
        cur_waves = wave_scheduler(epoch)

        # 2. Determine number of time points (grid + random) based on wave count
        grid_len = 2 * cur_waves + 1
        rand_len = grid_len
        seq_len = 2 * (grid_len + rand_len)  # Each sine has (t, f(t)) --> 2x points

        # 3. Generate a batch of sine superposition sequences
        x_flat = generate_sine_batch(cfg["batch_size"], grid_len, rand_len, cur_waves, device=device)
        B, _ = x_flat.shape

        # 4. Reshape to (B, seq_len, 1) where each timestep has a scalar (t or f(t))
        x = x_flat.view(B, seq_len, 1)

        # 5. Targets are the same as the inputs — we learn to predict f(t) from partial context
        targets = x.clone().squeeze(-1)  # (B, seq_len)

        # 6. Construct a loss mask:
        #    - idx: positions 0 to seq_len-1
        #    - pair_idx: maps every t/f pair to its pair index
        #    - is_f: whether a token is an f(t)
        #    - is_pred: whether this f(t) is in the random (not grid) region
        idx = torch.arange(seq_len, device=device)
        pair_idx = idx // 2
        is_f = (idx % 2 == 1)  # Every second element is f(t)
        is_pred = (pair_idx >= grid_len)  # Only predict in random portion
        loss_mask = (is_f & is_pred)
        mask_b = loss_mask.unsqueeze(0).expand(B, -1)  # (B, seq_len)

        # 7. Prepare masked input (zero out f(t) values to be predicted)
        inp = x.clone()
        inp[mask_b.unsqueeze(-1)] = 0.0

        # 8. Run forward pass and compute loss on masked prediction targets
        preds = model(inp).squeeze(-1)       # (B, seq_len)
        loss = F.mse_loss(preds[mask_b], targets[mask_b])  # Only penalize masked positions

        # 9. Standard optimizer step
        opt.zero_grad()
        loss.backward()
        opt.step()

        # 10. Log metrics to WandB
        global_step += 1
        wandb.log({
            "loss": loss.item(),
            "train/num_waves": cur_waves
        }, step=global_step)

        # 11. Save model periodically
        if epoch % cfg["save_every"] == 0:
            fname = f"{cfg['name']}_ep{epoch:04d}.pth"
            core_utils.save_model(model, f"sinusoidal_icl/checkpoints/{cfg['name']}", fname)
            
    core_utils.save_model(model, f"sinusoidal_icl/checkpoints/{cfg['name']}", f"{cfg['name']}_final.pth")
    wandb.finish()



def get_wave_scheduler(cfg):
    """
    Returns a function epoch_to_num_waves(epoch: int) -> int,
    which gives the number of sine waves to use for a given epoch
    based on the selected curriculum strategy.
    """
    curriculum = cfg["curriculum"]
    max_waves = cfg["max_waves"]
    epochs = cfg["epochs"]
    config = cfg.get("curriculum_config", {})

    if curriculum == "none":
        return lambda epoch: max_waves

    elif curriculum == "standard":
        # Linearly progress 1 --> max_waves
        def schedule(epoch):
            return min(max_waves, 1 + epoch * max_waves // epochs)
        return schedule

    elif curriculum == "random":
        return lambda epoch: int(np.random.randint(1, max_waves + 1))
    
    elif curriculum == "chapter":
        wave_counts = config["chapter_waves"]
        epochs_per_chapter = epochs // len(config["chapter_waves"])
        return lambda epoch: int(np.random.choice(wave_counts[min(epoch // epochs_per_chapter, len(wave_counts) - 1)]))
    
    elif curriculum == "interspersed":
        every_n = config.get("interspersed_every", 10)
        def schedule(epoch):
            curr_max_waves = min(max_waves, 1 + epoch * max_waves // epochs)
            if epoch % every_n == 0:
                return int(np.random.randint(1, curr_max_waves + 1))
            else:
                return curr_max_waves
        return schedule

    else:
        raise ValueError(f"Unknown curriculum: {curriculum}")