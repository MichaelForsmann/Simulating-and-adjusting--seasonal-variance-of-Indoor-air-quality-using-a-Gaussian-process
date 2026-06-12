import jax
import jax.numpy as jnp
import numpy as np
import arviz as az
import numpyro.distributions as dist
import numpyro.distributions as dist
import numpyro.handlers as handlers
from numpyro.infer import MCMC, NUTS, init_to_median, init_to_value,Predictive
from numpyro.infer.util import log_density
import jax
import jax.numpy as jnp
import numpy as np
import arviz as az
from arviz_base import from_dict     # <-- ArviZ 1.x location
import numpyro.distributions as dist
from numpyro.infer import MCMC, Predictive
import pandas as pd
import numpyro
def _to_numpy(x):
    if isinstance(x, dict):
        return {k: _to_numpy(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(_to_numpy(v) for v in x)
    if isinstance(x, (jnp.ndarray, jax.Array)):
        return np.asarray(x)
    return x


def get_rng_key(seed: int = 0):
    return jax.random.PRNGKey(seed)


def to_arviz(
    mcmc: MCMC,
    model_fn,
    X: jnp.ndarray,
    y: jnp.ndarray,
    num_prior_samples: int = 500,
    seed: int = 0,
    obs_site: str = "obs",
    noise_site: str = "noise",
    **model_kwargs,
):
    """Build an ArviZ 1.x InferenceData from a NumPyro MCMC run."""
    k1, k2 = jax.random.split(get_rng_key(seed))

    # --- Posterior samples ---
    samples_cd   = mcmc.get_samples(group_by_chain=True)   # (chain, draw, ...)
    samples_flat = mcmc.get_samples(group_by_chain=False)  # (chain*draw, ...)
    num_chains = mcmc.num_chains
    num_draws  = next(iter(samples_cd.values())).shape[1]

    # --- Predictive draws ---
    posterior_pred = Predictive(model_fn, samples_flat)(k1, X, **model_kwargs)
    prior_pred     = Predictive(model_fn, num_samples=num_prior_samples)(k2, X, **model_kwargs)

    # --- Log-likelihood (n_samples, n_obs) ---
    mean_pp = posterior_pred[obs_site]
    if noise_site in samples_flat:
        noise = samples_flat[noise_site][:, None]
        log_lik_flat = dist.Normal(mean_pp, noise).log_prob(y[None, :])
    else:
        log_lik_flat = dist.Normal(
            mean_pp, mean_pp.std(axis=0, keepdims=True)
        ).log_prob(y[None, :])

    n_obs = log_lik_flat.shape[-1]

    # --- Reshape flat -> (chain, draw, ...) and convert to NumPy ---
    def _reshape_cd(d):
        out = {}
        for k, v in d.items():
            v = np.asarray(v)
            out[k] = v.reshape(num_chains, num_draws, *v.shape[1:])
        return out

    posterior          = _to_numpy(samples_cd)
    posterior_pred_cd  = _reshape_cd(posterior_pred)
    log_lik_cd         = np.asarray(log_lik_flat).reshape(num_chains, num_draws, n_obs)

    # Prior predictive: arviz expects (chain, draw, ...). Give it chain=1.
    prior_pred_cd = {
        k: np.asarray(v)[None, ...] for k, v in prior_pred.items()
    }

    idata = from_dict(
        {
            "posterior":            posterior,
            "posterior_predictive": posterior_pred_cd,
            "log_likelihood":       {obs_site: log_lik_cd},
            "prior_predictive":     prior_pred_cd,
            "observed_data":        {obs_site: np.asarray(y)},
        },
        coords={"obs_id": np.arange(n_obs)},
        dims={obs_site: ["obs_id"]},
    )
    return idata



# ---------- Metrics ----------

def rmse(y_true, y_pred):
    return jnp.sqrt(jnp.mean((y_true - y_pred) ** 2))


def r2_score(y_true, y_pred):
    ss_res = jnp.sum((y_true - y_pred) ** 2)
    ss_tot = jnp.sum((y_true - jnp.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)


def sampled_rmse(y_true, mean_samples):
    err = mean_samples - y_true[None, :]
    return jnp.sqrt(jnp.mean(err ** 2, axis=1))


def sampled_r2(y_true, mean_samples):
    ss_res = jnp.sum((mean_samples - y_true[None, :]) ** 2, axis=1)
    ss_tot = jnp.sum((y_true - jnp.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)


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


def fit_corr(
    X,
    num_warmup=500,
    num_samples=500,
    num_chains=4,
    seed=0,
    transform="auto",        # "auto" | "log" | "rank" | "none"
):
    """
    transform:
      "log"  -> np.log1p(X), then standardize  (good for positive skewed data)
      "rank" -> rank-transform, then standardize (Spearman-style; most robust)
      "none" -> just standardize
      "auto" -> "log" if all columns are non-negative, else "rank"
    """
    # 1) DataFrame -> ndarray
    if isinstance(X, pd.DataFrame):
        col_names = X.columns.tolist()
        X = X.to_numpy(dtype=np.float64)
    else:
        X = np.asarray(X, dtype=np.float64)
        col_names = [f"x{i}" for i in range(X.shape[1])]

    # 2) Drop NaN rows
    mask = ~np.isnan(X).any(axis=1)
    if mask.sum() < len(X):
        print(f"Dropping {len(X) - mask.sum()} rows with NaN")
    X = X[mask]

    # 3) Choose transform
    if transform == "auto":
        transform = "log" if (X >= 0).all() else "rank"

    if transform == "log":
        if (X < 0).any():
            raise ValueError("log transform requires non-negative data.")
        X = np.log1p(X)                             # handles zeros safely
    elif transform == "rank":
        X = pd.DataFrame(X).rank().to_numpy(dtype=np.float64)
    elif transform != "none":
        raise ValueError(f"Unknown transform: {transform}")

    # 4) Guard constant columns
    sd = X.std(0)
    bad = np.where(sd == 0)[0]
    if bad.size:
        raise ValueError(
            f"Constant columns after transform: "
            f"{[col_names[i] for i in bad]} — drop them."
        )

    # 5) Standardize
    X = (X - X.mean(0)) / sd

    # 6) Final sanity check — fail loudly if still pathological
    mx = float(np.abs(X).max())
    if mx > 15:
        print(f"WARNING: max |X| = {mx:.1f} after transform+standardize. "
              "Consider transform='rank' for heavier-tailed data.")

    # 7) Fit
    kernel = NUTS(
        corr_model,
        target_accept_prob=0.95,
        init_strategy=init_to_median(num_samples=30),
    )
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=True,
    )
    mcmc.run(jax.random.PRNGKey(seed), X=jnp.asarray(X))
    return az.from_numpyro(mcmc)

def corr_model(X):
    """
    Multivariate-normal model on standardized data.
    Returns posterior over the full DxD correlation matrix.

    X : (N, D) array, already standardized (mean 0, sd 1) per column
        Use ranks (then standardize) for a Spearman-style version.
    """
    N, D = X.shape

    # Means (should be ~0 after standardization, but let the model see)
    mu = numpyro.sample("mu", dist.Normal(0.0, 1.0).expand([D]))

    # Per-variable scale (should be ~1 after standardization)
    sigma = numpyro.sample("sigma", dist.HalfNormal(1.0).expand([D]))

    # LKJ prior on correlation matrix via its Cholesky factor.
    # concentration=1 -> uniform over correlation matrices.
    # >1 favours identity (weaker correlations); <1 favours stronger.
    L_corr = numpyro.sample(
        "L_corr", dist.LKJCholesky(D, concentration=1.0)
    )

    # Reconstruct full correlation & covariance matrices as deterministics
    corr = numpyro.deterministic("corr", L_corr @ L_corr.T)
    scale_tril = sigma[:, None] * L_corr
    numpyro.deterministic("cov", scale_tril @ scale_tril.T)

    numpyro.sample(
        "obs",
        dist.MultivariateNormal(loc=mu, scale_tril=scale_tril),
        obs=X,
    )
def corr(Data,varibles):
    corr_data=Data.loc[:,varibles]
    idata_b = fit_corr(corr_data,transform="rank")
    corr_data=idata_b["posterior"]["corr"]
    R_mean = corr_data.mean(axis=(0,1))
    R_lo   = np.quantile(corr_data, 0.05, axis=(0,1))
    R_hi   = np.quantile(corr_data, 0.95, axis=(0,1)) 
    df_corr=pd.DataFrame(R_mean.values,index=varibles,columns=varibles)
    hdi=pd.DataFrame(((np.sign(R_hi)-np.sign(R_lo))==0),index=varibles,columns=varibles)
    return idata_b,R_mean,df_corr,hdi