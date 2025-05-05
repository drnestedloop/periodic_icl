#@title 4 wave comparison
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import random
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
import math

# 1) Generate one fixed list of test cases

#@title Inference Tools
def run_inference(
    model: torch.nn.Module,
    context: list[tuple[float, float]],
    query_t: float,
    device: torch.device = None,
) -> float:
    """
    Given:
      - model:     a GPT2Regressor, already .to(device) and in eval() mode
      - context:   list of (t, f(t)) pairs, length = m
      - query_t:   the new t at which to predict f(t)
    Returns:
      - predicted f(query_t) as a Python float
    """
    if device is None:
        device = next(model.parameters()).device

    # 1) flatten your context pairs → [t0,f0, t1,f1, …, t_{m-1},f_{m-1}]
    seq = [v for pair in context for v in pair]

    # 2) append the query t, plus a dummy 0.0 placeholder for f
    seq += [query_t, 0.0]

    # 3) to tensor of shape (1, seq_len, 1)
    x = torch.tensor(seq, dtype=torch.float32, device=device).view(1, -1, 1)

    # 4) attention mask (no padding here, so all ones)
    attn_mask = torch.ones(1, x.size(1), dtype=torch.long, device=device)

    # 5) forward, grab the last position’s output
    with torch.no_grad():
        out = model(x, attention_mask=attn_mask)   # (1, seq_len, 1)
    f_pred = out[0, -1, 0].item()

    return f_pred

def generate_test_context(n_waves):
  required_samplings = 2 * n_waves + 1
  t_vals = np.linspace(0, 1, required_samplings)
  f_vals = np.zeros_like(t_vals)
  for n in range(n_waves):
    f_vals += np.random.rand() * np.sin(2 * np.pi * (n + 1) * t_vals + (np.random.rand() < 0.5) * np.pi/2)
  return list(zip(t_vals, f_vals))

def gen_comparison_graphs(model, num_waves, device=None, query_start=0.0, query_end=3.0):
    """
    Generates a comparison plot of the true sum-of-sines vs. model predictions.

    Args:
        model:      a GPT2Regressor in eval() mode
        num_waves:  number of sine components to superimpose
        device:     torch device (defaults to model.device)
        query_start:  t-value at which to start querying the model
        query_end:  t-value at which to stop querying the model
        NOTE: the range [query_start, query_end] should contain the context range ([0,1]) to avoid errors
    """
    if device is None:
        device = next(model.parameters()).device

    # 1) Sample sine parameters
    freqs = 2 * np.pi * np.arange(1, num_waves + 1)
    phases = np.random.choice([0, np.pi/2], size=num_waves)
    amps = np.random.rand(num_waves)

    # 2) True compound sine function
    def f_true(t):
        res = np.zeros_like(t)
        for a, f, p in zip(amps, freqs, phases):
            res += a * np.sin(f * t + p)
        return res

    # 3) Build context: evenly spaced points in [0,1]
    grid_len = 2 * num_waves + 1
    context_ts = np.linspace(0, 1, grid_len)
    context_fs = f_true(context_ts)
    context = list(zip(context_ts.tolist(), context_fs.tolist()))

    # 4) Query a dense set of points and collect model predictions
    query_ts = np.linspace(query_start, query_end, 200)
    preds = []
    for t in query_ts:
        pred = run_inference(model, context, float(t), device=device)
        preds.append(pred)
    preds = np.array(preds)

    # 5) Plot true function, predictions, and context
    plt.figure()
    plt.plot(query_ts, f_true(query_ts))
    plt.plot(query_ts, preds)
    plt.scatter(context_ts, context_fs)
    plt.title(f"True vs. Predicted (num_waves={num_waves})")
    plt.xlabel("t")
    plt.ylabel("f(t)")
    plt.legend(["True", "Predicted", "Context"])
    plt.show()

def mean_se(x):
    """Return (mean, standard error) of array x."""
    m = x.mean()
    se = x.std(ddof=1) / math.sqrt(len(x))
    return m, se

def evaluate_model(model, trials, num_waves):
    """
    Assumptions: 
        The model is trained on a range from 0 to 1.
    """

    ts = np.linspace(0, 3, 200)
    mask1 = ts <= 1.0                 # 0–1
    mask2 = (ts > 1.0) & (ts <= 2.0)  # 1–2
    mask3 = (ts > 2.0) & (ts <= 3.0)  # 2–3

    L = 2 * num_waves + 1
    t_ctx = np.linspace(0, 1, L).reshape(-1,1)

    test_cases = []
    for _ in range(trials):
        freqs  = 2*np.pi * np.arange(1, num_waves+1)
        phases = np.random.choice([0, np.pi/2], num_waves)
        amps   = np.random.rand(num_waves)
        def make_true(freqs=freqs, phases=phases, amps=amps):
            return lambda t: sum(a*np.sin(f*t+p)
                                for a,f,p in zip(amps,freqs,phases))
        f_true = make_true()
        y_ctx = f_true(t_ctx.flatten())
        test_cases.append((f_true, t_ctx, y_ctx))

    # 2) Prepare error containers
    errs_full_gpt, errs_full_knn = [], []
    errs_1_gpt, errs_2_gpt, errs_3_gpt = [], [], []
    errs_1_knn, errs_2_knn, errs_3_knn = [], [], []

    # for per-point plotting
    errs_pt_gpt, errs_pt_knn = [], []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    for i, (f_true, t_ctx, y_ctx) in enumerate(test_cases):
        # --- GPT predictions ---
        preds_gpt = np.array([
            run_inference(model,
                        list(zip(t_ctx.flatten(), y_ctx)),
                        float(t),
                        device)
            for t in ts
        ])
        # --- KNN-3 predictions ---
        knn = KNeighborsRegressor(n_neighbors=3).fit(t_ctx, y_ctx)
        preds_knn = knn.predict(ts.reshape(-1,1))

        # record full-range MSE
        errs_full_gpt.append( ((preds_gpt  - f_true(ts))**2).mean() )
        errs_full_knn.append(((preds_knn - f_true(ts))**2).mean())

        # record interval MSEs
        for preds, e1, e2, e3 in [
            (preds_gpt,  errs_1_gpt,  errs_2_gpt,  errs_3_gpt),
            (preds_knn,  errs_1_knn,  errs_2_knn,  errs_3_knn),
        ]:
            e1.append( ((preds[mask1] - f_true(ts[mask1]))**2).mean() )
            e2.append( ((preds[mask2] - f_true(ts[mask2]))**2).mean() )
            e3.append( ((preds[mask3] - f_true(ts[mask3]))**2).mean() )

        # record per-point squared errors
        errs_pt_gpt.append((preds_gpt  - f_true(ts))**2)
        errs_pt_knn.append((preds_knn - f_true(ts))**2)

        # optional sample plot (with polynomial) ~10%
        if random.random() < 0.1:
            # fit a full-degree poly just for illustration
            deg = t_ctx.shape[0] - 1
            poly = PolynomialFeatures(degree=deg)
            Xp = poly.fit_transform(t_ctx)
            lin = LinearRegression().fit(Xp, y_ctx)
            preds_poly = lin.predict(poly.transform(ts.reshape(-1,1)))

            plt.figure(figsize=(5,3))
            plt.plot(ts, f_true(ts),      label='True',    linewidth=2)
            plt.plot(ts, preds_gpt,       label='GPT',     linestyle='--')
            plt.plot(ts, preds_poly,      label=f'Poly-{deg}', linestyle=':')
            plt.plot(ts, preds_knn,       label='KNN-3',   linestyle='-.')
            plt.scatter(t_ctx, y_ctx,     label='Context', color='k', s=10)
            plt.title(f"Sample #{i+1}")
            plt.xlabel("t"); plt.ylabel("f(t)")
            plt.legend(fontsize='small')
            plt.ylim(-5, 5)
            plt.tight_layout()
            plt.show()

    # convert lists to arrays
    errs_full_gpt = np.array(errs_full_gpt)
    errs_full_knn = np.array(errs_full_knn)

    errs_1_gpt = np.array(errs_1_gpt); errs_2_gpt = np.array(errs_2_gpt); errs_3_gpt = np.array(errs_3_gpt)
    errs_1_knn = np.array(errs_1_knn); errs_2_knn = np.array(errs_2_knn); errs_3_knn = np.array(errs_3_knn)

    errs_pt_gpt = np.vstack(errs_pt_gpt)
    errs_pt_knn = np.vstack(errs_pt_knn)

    # GETTING ERROR

    # Print MSE ± SE for each interval
    for name, e1, e2, e3 in [
        ('GPT',     errs_1_gpt, errs_2_gpt, errs_3_gpt),
        ('KNN-3',   errs_1_knn, errs_2_knn, errs_3_knn),
    ]:
        m1, se1 = mean_se(e1)
        m2, se2 = mean_se(e2)
        m3, se3 = mean_se(e3)
        print(f"{name} MSE [0-1] = {m1:.4f} ± {se1:.4f}, "
            f"[1-2] = {m2:.4f} ± {se2:.4f}, "
            f"[2-3] = {m3:.4f} ± {se3:.4f}")


    # 3) Print interval MSE means
    for name, e1, e2, e3 in [
        ('GPT', errs_1_gpt, errs_2_gpt, errs_3_gpt),
        ('KNN-3', errs_1_knn, errs_2_knn, errs_3_knn),
    ]:
        print(f"{name} MSE ⎡0–1⎤ = {e1.mean():.4f}, ⎡1–2⎤ = {e2.mean():.4f}, ⎡2–3⎤ = {e3.mean():.4f}")

    # 4) Full-range bar chart
    methods = ['GPT','KNN-3']
    means   = [errs_full_gpt.mean(), errs_full_knn.mean()]
    stds    = [errs_full_gpt.std(),  errs_full_knn.std()]

    plt.figure(figsize=(4,3))
    plt.bar(methods, means, yerr=stds, capsize=5)
    plt.ylabel('MSE (0–3)'); plt.title('Mean ± Std over 150 trials')
    plt.tight_layout()
    plt.show()

    # 5) Full-range histogram
    plt.figure(figsize=(5,3))
    bins = np.linspace(0, max(errs_full_gpt.max(), errs_full_knn.max()), 30)
    plt.hist(errs_full_gpt, bins=bins, alpha=0.6, label='GPT')
    plt.hist(errs_full_knn, bins=bins, alpha=0.6, label='KNN-3')
    plt.xlabel('MSE (0–3)'); plt.ylabel('Frequency')
    plt.title('GPT vs KNN Full-Range MSE')
    plt.legend(fontsize='small')
    plt.tight_layout()
    plt.show()

    # 5b) Empirical CDF of full-range MSE
    plt.figure(figsize=(5,3))
    for data, label in [(np.sort(errs_full_gpt), 'GPT'), (np.sort(errs_full_knn), 'KNN-3')]:
        y = np.arange(1, len(data)+1) / len(data)
        plt.plot(data, y, marker='.', linestyle='none', label=label)
    plt.xlabel('MSE'); plt.ylabel('CDF')
    plt.title('Empirical CDF of Full-Range MSE')
    plt.legend(fontsize='small')
    plt.tight_layout()
    plt.show()

    # 8) Per-point error quartile curves on a log y-scale, zoomed to >1e-4
    # (assumes ts, errs_pt_gpt, errs_pt_knn already defined)

    # Compute quartiles at each t
    q25_gpt, q50_gpt, q75_gpt = np.percentile(errs_pt_gpt, [25,50,75], axis=0)
    q25_knn, q50_knn, q75_knn = np.percentile(errs_pt_knn, [25,50,75], axis=0)

    plt.figure(figsize=(6,4))

    # GPT quartile curves
    plt.plot(ts, q25_gpt, linestyle='-', linewidth=2, label='GPT 25th')
    plt.plot(ts, q50_gpt, linestyle='-',  linewidth=2, label='GPT 50th (median)')
    plt.plot(ts, q75_gpt, linestyle='-', linewidth=2, label='GPT 75th')

    # KNN quartile curves
    plt.plot(ts, q25_knn, linestyle='--', linewidth=2, label='KNN-3 25th')
    plt.plot(ts, q50_knn, linestyle='--',linewidth=2, label='KNN-3 50th (median)')
    plt.plot(ts, q75_knn, linestyle='--', linewidth=2, label='KNN-3 75th')

    # log y-axis, zoom in above 1e-4
    plt.yscale('log')
    plt.ylim(1e-4, 1e1)   # show MSE from 1e-4 to 1e-1

    plt.xlabel('t (distance from context end)')
    plt.ylabel('MSE (log scale)')
    plt.title('Per-Point Error Quartiles vs Exact Distance (Zoomed >1e-4)')
    plt.legend(fontsize='small', ncol=2)
    plt.tight_layout()
    plt.show()
    # 9) Binned mean MSE every Δt=0.1

    # assumes ts (shape=(200,)), trials, errs_pt_gpt, errs_pt_knn defined above

    # Flatten to long arrays
    x_flat     = np.tile(ts, trials)       # shape (trials*200,)
    y_flat_gpt = errs_pt_gpt.flatten()
    y_flat_knn = errs_pt_knn.flatten()

    # Define bins of width 0.1
    bin_edges   = np.arange(0, 3.1, 0.1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Compute mean MSE in each bin
    means_gpt = []
    means_knn = []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (x_flat >= lo) & (x_flat < hi)
        means_gpt.append(y_flat_gpt[mask].mean())
        means_knn.append(y_flat_knn[mask].mean())

    # Plot binned means
    plt.figure(figsize=(6,4))
    plt.plot(bin_centers, means_gpt, '-o', label='GPT mean MSE')
    plt.plot(bin_centers, means_knn, '-s', label='KNN mean MSE')
    plt.xlabel('t (distance from context end)')
    plt.ylabel('Mean MSE')
    plt.title('Binned Mean Error (Δt = 0.1)')
    plt.legend(fontsize='small')
    plt.tight_layout()
    plt.show()