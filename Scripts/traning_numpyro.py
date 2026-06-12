import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import arviz as az
from functools import partial

from Scripts.kernel_numpyro import (
    PERIOD_WEEKS, JITTER
)


# =============================================================================
# Reproducibility / data helpers
# =============================================================================

def get_rng_key(seed: int = 42) -> jax.random.PRNGKey:
    return jax.random.PRNGKey(seed)


def make_Xy(data: pd.DataFrame, X_cols, y_col,train=False,Z=None):
    clean = data.dropna(subset=y_col+X_cols)
    if len(X_cols)==1:
        X = jnp.asarray(clean.loc[:,X_cols].values, dtype=jnp.float64).reshape(-1)
        y = jnp.asarray(clean.loc[:,y_col].values, dtype=jnp.float64).reshape(-1)
    else:
        X = jnp.asarray(clean.loc[:,X_cols].values, dtype=jnp.float64)
        y = jnp.asarray(clean.loc[:,y_col].values, dtype=jnp.float64).reshape(-1)
    if train is True:
        y_mean=y.mean()
        y_std=y.std()
        y=(y-y_mean)/y_std
        Z=jnp.asarray([y_mean,y_std])
        return X,y,Z
    else:
        y=(y-Z[0])/Z[1]
    return X, y


def _as_2d(X):
    X = jnp.asarray(X)
    return X[:, None] if X.ndim == 1 else X


def run_mcmc(model_fn, X, y, *,
             num_warmup=1000, num_samples=1000, num_chains=4,
             target_accept_prob=0.9, max_tree_depth=10,chain_method="vectorized", seed=42, **kwargs):
    X = _as_2d(X)
    nuts = NUTS(model_fn, target_accept_prob=target_accept_prob,
                max_tree_depth=max_tree_depth)
    mcmc = MCMC(nuts, num_warmup=num_warmup,
                num_samples=num_samples, num_chains=num_chains)
    mcmc.run(get_rng_key(seed), X, y=y, **kwargs)
    mcmc.print_summary()
    return mcmc


# =============================================================================
# Internal helpers shared by predict + ratio
# =============================================================================

def _stack_posterior(idata):
    post = idata["posterior"].dataset.stack(sample=("chain", "draw"))
    return {k: jnp.asarray(post[k].transpose("sample", ...).values)
            for k in post.data_vars}


def _load_constants(idata):
    const = idata["constant_data"]
    X_train = _as_2d(jnp.asarray(const["X_train"].values))
    y_train = jnp.asarray(const["y_train"].values).reshape(-1)
    return X_train, y_train


def _chunked_vmap(fn, params_stacked, chunk_size, n_samples):
    if chunk_size is None or chunk_size >= n_samples:
        return jax.vmap(fn)(params_stacked)
    n_chunks = (n_samples + chunk_size - 1) // chunk_size
    pad     = n_chunks * chunk_size - n_samples
    padded  = {k: jnp.pad(v, [(0, pad)] + [(0, 0)] * (v.ndim - 1))
               for k, v in params_stacked.items()}
    reshaped = {k: v.reshape((n_chunks, chunk_size) + v.shape[1:])
                for k, v in padded.items()}
    out = jax.lax.map(jax.vmap(fn), reshaped)
    return jax.tree.map(lambda a: a.reshape(-1, *a.shape[2:])[:n_samples], out)


# =============================================================================
# Prediction (fast: mean only; mean+std when bands are needed)
# =============================================================================

def predict_mean(idata, X_new, kernel_builder, *,
                 jitter=JITTER, chunk_size=64):
    X_new = _as_2d(X_new)
    X_train, y_train = _load_constants(idata)
    eye_N = jnp.eye(X_train.shape[0], dtype=X_train.dtype)
    params = _stack_posterior(idata)
    n = next(iter(params.values())).shape[0]

    def fn(p):
        K_NN  = kernel_builder(X_train, X_train, p)
        K_sN  = kernel_builder(X_new,   X_train, p)
        L     = jnp.linalg.cholesky(K_NN + (p["noise"] ** 2 + jitter) * eye_N)
        alpha = jax.scipy.linalg.cho_solve((L, True), y_train)
        return K_sN @ alpha

    return _chunked_vmap(fn, params, chunk_size, n)


def predict_mean_std(idata, X_new, kernel_builder, *,
                     jitter=JITTER, chunk_size=64):
    X_new = _as_2d(X_new)
    X_train, y_train = _load_constants(idata)
    eye_N = jnp.eye(X_train.shape[0], dtype=X_train.dtype)
    params = _stack_posterior(idata)
    n = next(iter(params.values())).shape[0]
    y_mu=jnp.asarray(idata.attrs["mean"], dtype=jnp.float64)
    y_sigma=jnp.asarray(idata.attrs["std"], dtype=jnp.float64)
    def fn(p):
        K_NN  = kernel_builder(X_train, X_train, p)
        K_sN  = kernel_builder(X_new,   X_train, p)
        K_ss  = kernel_builder(X_new,   X_new,   p)
        L     = jnp.linalg.cholesky(K_NN + (p["noise"] ** 2 + jitter) * eye_N)
        alpha = jax.scipy.linalg.cho_solve((L, True), y_train)
        mu    = K_sN @ alpha
        v     = jax.scipy.linalg.cho_solve((L, True), K_sN.T)
        var   = jnp.diag(K_ss - K_sN @ v)
        mean= y_mu + y_sigma * mu
        return mean, jnp.sqrt(jnp.clip(var, a_min=1e-10))

    mu, sd = _chunked_vmap(fn, params, chunk_size, n)
    return {"mean": mu, "std": sd}


# =============================================================================
# Ratio adjustment — one function, every builder
# =============================================================================

def ratio_adjustment(idata, X_new, kernel_builder, period, *,
                     time_col=0, jitter=JITTER, n_quad=200, chunk_size=64):
    """W_n = (1/T) · ∫_{x_n}^{x_n + T} f dt / f(x_n), per posterior sample.

    Parameters
    ----------
    X_new : (N,) or (N, d)
        Base points. 1-D is promoted to (N, 1). For multi-feature models,
        pass (N, d) with the aux columns (flow, source one-hot, ...) set to
        the values you want held fixed across the integration window.
    kernel_builder : any builder using the (n, d) / time-in-col-0 convention.
    period : integration window length (also the period for the kernels here).
    time_col : which column gets slid by quadrature offsets. Default 0.

    Returns
    -------
    {"f_base": (S, N), "integral": (S, N), "ratio": (S, N)}
    """
    X_new = _as_2d(X_new)
    N, d = X_new.shape

    offsets = jnp.linspace(0.0, period, n_quad)
    h_step  = period / (n_quad - 1)
    weights = jnp.full((n_quad,), h_step).at[0].mul(0.5).at[-1].mul(0.5)

    # Shift ONLY the time column; all other columns held fixed per base point.
    delta  = jnp.zeros((n_quad, d)).at[:, time_col].set(offsets)
    X_quad = (X_new[:, None, :] + delta[None, :, :]).reshape(N * n_quad, d)
    X_eval = jnp.concatenate([X_new, X_quad], axis=0)        # (N + N*Q, d)

    X_train, y_train = _load_constants(idata)
    eye_N  = jnp.eye(X_train.shape[0], dtype=X_train.dtype)
    params = _stack_posterior(idata)
    n_samples = next(iter(params.values())).shape[0]

    def fn(p):
        K_NN  = kernel_builder(X_train, X_train, p)
        K_eN  = kernel_builder(X_eval,  X_train, p)
        L     = jnp.linalg.cholesky(K_NN + (p["noise"] ** 2 + jitter) * eye_N)
        alpha = jax.scipy.linalg.cho_solve((L, True), y_train)
        mean  = K_eN @ alpha

        f_base   = mean[:N]
        f_quad   = mean[N:].reshape(N, n_quad)
        integral = jnp.sum(f_quad * weights, axis=-1)        # ∫_0^T f dt
        ratio    = (integral / weights.sum()) / f_base       # mean-over-T / f_base
        return f_base, integral, ratio

    f_base, integral, ratio = _chunked_vmap(fn, params, chunk_size, n_samples)
    return {"f_base": f_base, "integral": integral, "ratio": ratio}




def ratio_adjustment_flow(
    idata,
    X_new,
    kernel_builder,
    period,
    *,
    time_col=0,
    flow_col=None,           # None  -> no flow override
    flow_value=None,         # required if flow_col is not None
    jitter=JITTER,
    n_quad=200,
    chunk_size=64,
):
    if (flow_col is None) != (flow_value is None):
        raise ValueError(
            "flow_col and flow_value must be supplied together "
            "(or both omitted)."
        )

    X_new = _as_2d(X_new).astype(jnp.float64)
    N, d = X_new.shape

    if flow_col is not None and not (0 <= flow_col < d):
        raise ValueError(
            f"flow_col={flow_col} out of range for X_new with {d} columns."
        )

    # --- Quadrature nodes / weights ---------------------------------------
    offsets = jnp.linspace(0.0, period, n_quad)
    h_step  = period / (n_quad - 1)
    weights = jnp.full((n_quad,), h_step).at[0].mul(0.5).at[-1].mul(0.5)
    T_total = weights.sum()                              # == period
    y_mu=jnp.asarray(idata.attrs["mean"], dtype=jnp.float64)
    y_sigma=jnp.asarray(idata.attrs["std"], dtype=jnp.float64)
    # --- Build quadrature design matrix -----------------------------------
    # Start by replicating X_new for every quadrature node so every aux
    # column is carried over automatically.
    X_quad = jnp.broadcast_to(X_new[:, None, :], (N, n_quad, d)).copy()
    X_quad = X_quad.at[:, :, time_col].add(offsets[None, :])
    if flow_col is not None:
        X_quad = X_quad.at[:, :, flow_col].set(flow_value)
    X_quad = X_quad.reshape(N * n_quad, d)

    X_eval = jnp.concatenate([X_new, X_quad], axis=0)    # (N + N*Q, d)

    # --- GP posterior mean per sample -------------------------------------
    X_train, y_train = _load_constants(idata)
    eye_N  = jnp.eye(X_train.shape[0], dtype=X_train.dtype)
    params = _stack_posterior(idata)
    n_samples = next(iter(params.values())).shape[0]

    def fn(p):
        K_NN  = kernel_builder(X_train, X_train, p)
        K_eN  = kernel_builder(X_eval,  X_train, p)
        L     = jnp.linalg.cholesky(K_NN + (p["noise"] ** 2 + jitter) * eye_N)
        alpha = jax.scipy.linalg.cho_solve((L, True), y_train)
        mean_z  = K_eN @ alpha
        mean = y_mu + y_sigma * mean_z
        f_base   = mean[:N]
        f_quad   = mean[N:].reshape(N, n_quad)
        integral = jnp.sum(f_quad * weights, axis=-1)
        ratio    = (integral / T_total) / f_base
        return f_base, integral, ratio

    f_base, integral, ratio = _chunked_vmap(fn, params, chunk_size, n_samples)
    return {"f_base": f_base, "integral": integral, "ratio": ratio}
  
# =============================================================================
# idata bookkeeping (unchanged — kept for the predict+save flow)
# =============================================================================

def _to_np_tree(d):
    return {k: np.asarray(v) for k, v in d.items()}

def _downcast_tree(d, dtype=np.float32):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = _downcast_tree(v, dtype)
        else:
            arr = np.asarray(v)
            out[k] = arr.astype(dtype) if np.issubdtype(arr.dtype, np.floating) else arr
    return out


def save_idata(data, path, *, attrs=None, dtype=np.float32, complevel=5):
    if dtype is not None:
        data = _downcast_tree(data, dtype)
    from_dict_attrs = None
    if attrs:
        clean = {k: ("" if v is None else v) for k, v in attrs.items()}
        from_dict_attrs = {"/": clean}
    idata = az.from_dict(data, attrs=from_dict_attrs)
    if complevel and complevel > 0:
        comp = {"zlib": True, "complevel": int(complevel)}
        for node in idata.subtree:
            ds = node.dataset
            for var in ds.data_vars:
                ds[var].encoding.update(comp)
    idata.to_netcdf(path + ".nc")
    return idata


def _compute_metrics(means, y_test):
    y    = jnp.asarray(y_test)
    err  = means - y[None, :]
    rmse_per_draw = jnp.sqrt(jnp.mean(err ** 2, axis=1))
    ss_res = jnp.sum(err ** 2, axis=1)
    ss_tot = jnp.sum((y - y.mean()) ** 2)
    r2_per_draw = 1.0 - ss_res / ss_tot
    mean_pred  = means.mean(axis=0)
    rmse_point = jnp.sqrt(jnp.mean((mean_pred - y) ** 2))
    r2_point   = 1.0 - jnp.sum((mean_pred - y) ** 2) / ss_tot
    return {
        "rmse_per_draw": np.asarray(rmse_per_draw),
        "r2_per_draw":   np.asarray(r2_per_draw),
        "rmse_point":    float(rmse_point),
        "r2_point":      float(r2_point),
        "rmse_mean":     float(rmse_per_draw.mean()),
        "rmse_q025":     float(jnp.quantile(rmse_per_draw, 0.025)),
        "rmse_q975":     float(jnp.quantile(rmse_per_draw, 0.975)),
        "r2_mean":       float(r2_per_draw.mean()),
        "r2_q025":       float(jnp.quantile(r2_per_draw, 0.025)),
        "r2_q975":       float(jnp.quantile(r2_per_draw, 0.975)),
    }


def predict_and_save(mcmc, X_train1, y_train, X_test1, y_test, kernel_builder,
                     name, *, idata_attrs=None, obs_name="y",
                     jitter=JITTER, chunk_size=64):
    """Predict on X_test with mean+std and write an idata netcdf for later reuse."""
    X_train = _as_2d(X_train1); X_test = _as_2d(X_test1)
    y_train = jnp.asarray(y_train).reshape(-1)
    y_test  = jnp.asarray(y_test).reshape(-1)

    samples_chain = mcmc.get_samples(group_by_chain=True)
    samples_flat  = mcmc.get_samples()
    n_chains      = mcmc.num_chains

    # Predict directly off the MCMC samples (no idata round-trip needed).
    eye_N = jnp.eye(X_train.shape[0], dtype=X_train.dtype)
    params = {k: jnp.asarray(v) for k, v in samples_flat.items()}
    n = next(iter(params.values())).shape[0]

    def fn(p):
        K_NN  = kernel_builder(X_train, X_train, p)
        K_sN  = kernel_builder(X_test,  X_train, p)
        K_ss  = kernel_builder(X_test,  X_test,  p)
        L     = jnp.linalg.cholesky(K_NN + (p["noise"] ** 2 + jitter) * eye_N)
        alpha = jax.scipy.linalg.cho_solve((L, True), y_train)
        mu    = K_sN @ alpha
        v     = jax.scipy.linalg.cho_solve((L, True), K_sN.T)
        var   = jnp.diag(K_ss - K_sN @ v)
        return mu, jnp.sqrt(jnp.clip(var, a_min=1e-10))

    means, stds = _chunked_vmap(fn, params, chunk_size, n)

    noise_s = params["noise"][:, None]
    total_std = jnp.sqrt(stds ** 2 + noise_s ** 2)
    log_liks  = dist.Normal(means, total_std).log_prob(y_test[None, :])

    metrics  = _compute_metrics(means, y_test)
    n_draws  = means.shape[0] // n_chains
    reshape  = lambda x: np.asarray(x).reshape(n_chains, n_draws, *np.asarray(x).shape[1:])

    data = {
        "posterior":            _to_np_tree(samples_chain),
        "posterior_predictive": {obs_name: reshape(means)},
        "log_likelihood":       {obs_name: reshape(log_liks)},
        "observed_data":        {obs_name: np.asarray(y_test)},
        "constant_data": {
            "X_train":       np.asarray(X_train1),
            "y_train":       np.asarray(y_train),
            "X_test":        np.asarray(X_test1),
            "y_test":        np.asarray(y_test),
            "pred_std":      reshape(stds),
            "rmse_per_draw": reshape(metrics["rmse_per_draw"]),
            "r2_per_draw":   reshape(metrics["r2_per_draw"]),
        },
    }
    scalar_keys = ["rmse_point", "r2_point", "rmse_mean", "rmse_q025", "rmse_q975",
                   "r2_mean", "r2_q025", "r2_q975"]
    attrs = {k: metrics[k] for k in scalar_keys} | {"name": name}
    if idata_attrs:
        attrs.update(idata_attrs)
    return save_idata(data, name, attrs=attrs)