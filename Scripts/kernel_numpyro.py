

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist


PERIOD_WEEKS = 52.143  # one year in weeks
JITTER       = 1e-4    # Cholesky stability


# =============================================================================
# Kernel primitives
# =============================================================================

def periodic_kernel(x1, x2, variance, lengthscale, period=PERIOD_WEEKS):
    diffs = x1[:, None] - x2[None, :]
    return variance * jnp.exp(
        -2.0 * jnp.sin(jnp.pi * diffs / period) ** 2 / lengthscale ** 2
    )


def rbf_kernel(x1, x2, variance, lengthscale):
    sq = (x1[:, None] - x2[None, :]) ** 2
    return variance * jnp.exp(-0.5 * sq / lengthscale ** 2)


def linear_kernel(x1, x2, variance):
    x1 = jnp.atleast_2d(x1).reshape(-1, 1) if x1.ndim == 1 else x1
    x2 = jnp.atleast_2d(x2).reshape(-1, 1) if x2.ndim == 1 else x2
    return variance * (x1 @ x2.T)


def constant_kernel(n1, n2, variance):
    return variance * jnp.ones((n1, n2))


# =============================================================================
# Helpers
# =============================================================================

def _as_2d(X):
    X = jnp.asarray(X)
    return X[:, None] if X.ndim == 1 else X


def _gp_obs(K, noise, y, n):
    K = K + (noise ** 2 + JITTER) * jnp.eye(n)
    numpyro.sample("obs",
        dist.MultivariateNormal(jnp.zeros(n), covariance_matrix=K),
        obs=y)


# =============================================================================
# Builders — pure-time models read X[:, 0]; multi-feature read additional cols
# =============================================================================

def make_periodic_builder():
    def builder(X1, X2, p):
        X1 = _as_2d(X1); X2 = _as_2d(X2)
        return periodic_kernel(X1[:, 0], X2[:, 0],
                               p["kernel_variance"], p["lengthscale"])
    return builder


def make_rbf_builder():
    def builder(X1, X2, p):
        X1 = _as_2d(X1); X2 = _as_2d(X2)
        return rbf_kernel(X1[:, 0], X2[:, 0], p["kernel_variance"], p["lengthscale"])
    return builder


def make_harmonic_builder(k, period, harmonic_decay=2.0, harmonic_strength=10.0):
    """Sum_h linear_kernel([cos(h w t), sin(h w t)]) + constant. 1-D-friendly."""
    omega = 2.0 * jnp.pi / period

    def builder(X1, X2, p):
        X1 = _as_2d(X1); X2 = _as_2d(X2)
        t1, t2 = X1[:, 0], X2[:, 0]
        n1, n2 = t1.shape[0], t2.shape[0]
        K = constant_kernel(n1, n2, p["constant_variance"])
        for h in range(k):
            hv = p[f"harmonic_var"][h]
            a1 = (h + 1) * omega * t1
            a2 = (h + 1) * omega * t2
            Xh1 = jnp.stack([jnp.cos(a1), jnp.sin(a1)], axis=1)
            Xh2 = jnp.stack([jnp.cos(a2), jnp.sin(a2)], axis=1)
            K = K + linear_kernel(Xh1, Xh2, hv)
        return K
    return builder


def make_periodic_flow_builder():
    def builder(X1, X2, p):
        n1, n2 = X1.shape[0], X2.shape[0]
        K_per = periodic_kernel(X1[:, 0], X2[:, 0],
                                p["per_variance"], p["per_lengthscale"])
        K_rbf = rbf_kernel(X1[:, 1], X2[:, 1], 1.0, p["rbf_lengthscale"])
        K_const = constant_kernel(n1, n2, p["constant_variance"])
        return K_per * K_rbf+K_const
    return builder


def make_harmonic_flow_builder(k, period):
    omega = 2.0 * jnp.pi / period

    def builder(X1, X2, p):
        t1, t2 = X1[:, 0], X2[:, 0]
        f1, f2 = X1[:, 1], X2[:, 1]
        n1, n2 = t1.shape[0], t2.shape[0]
        K = constant_kernel(n1, n2, p["constant_variance"])
        for h in range(k):
            hv = p[f"harmonic_var"][h]
            a1 = (h + 1) * omega * t1
            a2 = (h + 1) * omega * t2
            Xh1 = jnp.stack([jnp.cos(a1), jnp.sin(a1)], axis=1)
            Xh2 = jnp.stack([jnp.cos(a2), jnp.sin(a2)], axis=1)
            K = K + linear_kernel(Xh1, Xh2, hv)
        return K * rbf_kernel(f1, f2, 1.0, p["rbf_lengthscale"])
    return builder


def make_flow_combined_builder():
    def builder(X1, X2, p):
        X1 = _as_2d(X1); X2 = _as_2d(X2)
        S1, S2 = X1[:, 2:], X2[:, 2:]
        n1, n2 = X1.shape[0], X2.shape[0]
        K_per = periodic_kernel(X1[:, 0], X2[:, 0], 1.0, p["per_lengthscale"])
        K_rbf = rbf_kernel(X1[:, 1], X2[:, 1], 1.0, p["rbf_lengthscale"])
        K_const = constant_kernel(n1, n2, p["constant_variance"])
        K_source = (S1 * p["source_var"]) @ S2.T
        return K_const + K_per * K_rbf * K_source
    return builder


def make_combined_temp_builder():
    def builder(X1, X2, p):
        X1 = _as_2d(X1); X2 = _as_2d(X2)
        S1, S2 = X1[:, 3:], X2[:, 3:]
        n1, n2 = X1.shape[0], X2.shape[0]
        K_per = periodic_kernel(X1[:, 0], X2[:, 0], 1.0, p["per_lengthscale"])
        K_temp = rbf_kernel(X1[:, 2], X2[:, 2], 1.0, p["temp_lengthscale"])
        K_const = constant_kernel(n1, n2, p["constant_variance"])
        K_source = (S1 * p["source_var"]) @ S2.T
        return K_const + (
            K_per * p["comp_weights"][0]
            + K_temp * p["comp_weights"][1]
            + K_per * K_temp * p["comp_weights"][2]
        ) * K_source
    return builder


def make_flow_combined_temp_builder():
    def builder(X1, X2, p):
        X1 = _as_2d(X1); X2 = _as_2d(X2)
        S1, S2 = X1[:, 3:], X2[:, 3:]
        n1, n2 = X1.shape[0], X2.shape[0]
        K_per = periodic_kernel(X1[:, 0], X2[:, 0], 1.0, p["per_lengthscale"])
        K_temp = rbf_kernel(X1[:, 2], X2[:, 2], 1.0, p["temp_lengthscale"])
        K_rbf = rbf_kernel(X1[:, 1], X2[:, 1], 1.0, p["rbf_lengthscale"])
        K_const = constant_kernel(n1, n2, p["constant_variance"])
        K_source = (S1 * p["source_var"]) @ S2.T
        return K_const + (
            K_per * p["comp_weights"][0]
            + K_temp * p["comp_weights"][1]
            + K_per * K_temp * p["comp_weights"][2]
        ) * K_source * K_rbf
    return builder


def make_harmonic_combined_builder(k, period):
    omega = 2.0 * jnp.pi / period

    def builder(X1, X2, p):
        X1 = _as_2d(X1); X2 = _as_2d(X2)
        S1 = X1[:, 2:]
        S2 = X2[:, 2:]
        n1, n2 = X1.shape[0], X2.shape[0]
        K_harm = jnp.zeros((n1, n2))
        for h in range(k):
            hv = p[f"harmonic_var"][h]
            a1 = (h + 1) * omega * X1[:, 0]
            a2 = (h + 1) * omega * X2[:, 0]
            Xh1 = jnp.stack([jnp.cos(a1), jnp.sin(a1)], axis=1)
            Xh2 = jnp.stack([jnp.cos(a2), jnp.sin(a2)], axis=1)
            K_harm = K_harm + linear_kernel(Xh1, Xh2, hv)
        K_rbf = rbf_kernel(X1[:, 1], X2[:, 1], 1.0, p["rbf_lengthscale"])
        K_const = constant_kernel(n1, n2, p["constant_variance"])
        sv = p["source_var"]                       # shape (n_sources,)
        K_source = (S1 * sv) @ S2.T
        return K_const + K_harm * K_rbf * K_source
    return builder


# =============================================================================
# Model factories — each returns (model_fn, builder_fn) using the SAME kernel
# =============================================================================

def make_periodic(period=PERIOD_WEEKS):
    builder = make_periodic_builder()

    def model(X, y=None):
        X = _as_2d(X); n = X.shape[0]
        p = {
            "kernel_variance": numpyro.sample("kernel_variance", dist.HalfNormal(1)),
            "lengthscale":     numpyro.sample("lengthscale",     dist.LogNormal(0.0, 0.5)),
        }
        noise = numpyro.sample("noise", dist.HalfNormal(1))
        _gp_obs(builder(X, X, p), noise, y, n)

    return model, builder


def make_rbf():
    builder = make_rbf_builder()

    def model(X, y=None):
        X = _as_2d(X); n = X.shape[0]
        p = {"kernel_variance": numpyro.sample("kernel_variance", dist.HalfNormal(1)),
            "lengthscale":     numpyro.sample("lengthscale",     dist.LogNormal(0.8, 0.8)),
        }
        noise = numpyro.sample("noise", dist.HalfNormal(1))
        _gp_obs(builder(X, X, p), noise, y, n)

    return model, builder


def make_harmonic(k, period=PERIOD_WEEKS):
    builder = make_harmonic_builder(k, period)

    def model(X, y=None):
        X = _as_2d(X); n = X.shape[0]
        p = {"constant_variance":
             numpyro.sample("constant_variance", dist.InverseGamma(2.0, 1.0))}
        with numpyro.plate("harmonics", k):
            p["harmonic_var"] = numpyro.sample("harmonic_var", dist.HalfCauchy(0.75))
        noise = numpyro.sample("noise", dist.HalfNormal(1))
        _gp_obs(builder(X, X, p), noise, y, n)

    return model, builder


def make_periodic_flow(period=PERIOD_WEEKS):
    builder = make_periodic_flow_builder()

    def model(X, y=None):
        X = _as_2d(X); n = X.shape[0]
        p = {
            "per_variance":    numpyro.sample("per_variance",    dist.HalfNormal(1)),
            "per_lengthscale": numpyro.sample("per_lengthscale", dist.LogNormal(0.0, 0.5)),
            "rbf_lengthscale": numpyro.sample("rbf_lengthscale", dist.LogNormal(0.8, 0.8)),
            "constant_variance": numpyro.sample("constant_variance", dist.HalfNormal(1))
        }
        noise = numpyro.sample("noise", dist.HalfNormal(1))
        _gp_obs(builder(X, X, p), noise, y, n)

    return model, builder


def make_harmonic_flow(k, period=PERIOD_WEEKS):
    builder = make_harmonic_flow_builder(k, period)

    def model(X, y=None):
        X = _as_2d(X); n = X.shape[0]
        p = {
            "constant_variance": numpyro.sample("constant_variance", dist.HalfNormal(1)),
            "rbf_lengthscale":   numpyro.sample("rbf_lengthscale",   dist.LogNormal(0.8, 0.8)),
        }
        with numpyro.plate("harmonics", k):
            p["harmonic_var"] = numpyro.sample("harmonic_var", dist.HalfCauchy(1))
        noise = numpyro.sample("noise", dist.HalfNormal(1))
        _gp_obs(builder(X, X, p), noise, y, n)

    return model, builder


def make_flow_combined(period=PERIOD_WEEKS):
    builder = make_flow_combined_builder()

    def model(X, y=None):
        X = _as_2d(X)
        n = X.shape[0]
        n_sources = X.shape[1] - 2
        p = {
            "per_lengthscale":   numpyro.sample("per_lengthscale",   dist.LogNormal(0.0, 0.5)),
            "rbf_lengthscale":   numpyro.sample("rbf_lengthscale",   dist.LogNormal(0.8, 0.8)),
            "constant_variance": numpyro.sample("constant_variance", dist.HalfNormal(1)),
        }
        with numpyro.plate("sources", n_sources):
            p["source_var"] = numpyro.sample("source_var", dist.HalfCauchy(0.75))
        noise = numpyro.sample("noise", dist.HalfNormal(1))
        _gp_obs(builder(X, X, p), noise, y, n)

    return model, builder


def make_harmonic_combined(k, period=PERIOD_WEEKS):
    builder = make_harmonic_combined_builder(k, period)
    def model(X, y=None):
        X = _as_2d(X); n = X.shape[0]
        n_sources = X.shape[1] - 2
        w = jnp.arange(1, k + 1, dtype=jnp.float32) ** (-1)
        w = w / w.sum()
        p = { "constant_variance": numpyro.sample("constant_variance", dist.HalfNormal(1)),
            "rbf_lengthscale":   numpyro.sample("rbf_lengthscale",   dist.LogNormal(0.8, 0.8)),
            "harmonic_var": numpyro.sample("harmonic_var",      dist.Dirichlet(w))}        
            
        if n_sources > 0:
            with numpyro.plate("sources", n_sources):
                p["source_var"] = numpyro.sample("source_var", dist.HalfCauchy(0.75))
        else:
            p["source_var"] = jnp.zeros(0)
        noise = numpyro.sample("noise", dist.HalfNormal(1))
        _gp_obs(builder(X, X, p), noise, y, n)
    return model, builder


def make_flow_temp_combined(period=PERIOD_WEEKS):
    builder = make_flow_combined_temp_builder()

    def model(X, y=None):
        X = _as_2d(X)
        n = X.shape[0]
        n_sources = X.shape[1] - 3
        p = {
            "per_lengthscale":   numpyro.sample("per_lengthscale",   dist.LogNormal(0.0, 0.5)),
            "rbf_lengthscale":   numpyro.sample("rbf_lengthscale",   dist.LogNormal(0.8, 0.8)),
            "constant_variance": numpyro.sample("constant_variance", dist.HalfNormal(1.)),
            "temp_lengthscale":  numpyro.sample("temp_lengthscale",  dist.LogNormal(jnp.log(8.0), 0.5)),
            "comp_weights":      numpyro.sample("comp_weights",      dist.Dirichlet(jnp.ones(3)*0.33)),
        }
        with numpyro.plate("sources", n_sources):
            p["source_var"] = numpyro.sample("source_var", dist.HalfCauchy(0.75))
        noise = numpyro.sample("noise", dist.HalfNormal(1))
        _gp_obs(builder(X, X, p), noise, y, n)

    return model, builder


def make_temp_combined(period=PERIOD_WEEKS):
    builder = make_combined_temp_builder()

    def model(X, y=None):
        X = _as_2d(X)
        n = X.shape[0]
        n_sources = X.shape[1] - 3
        p = {
            "per_lengthscale":   numpyro.sample("per_lengthscale",   dist.LogNormal(0.0, 0.5)),
            "constant_variance": numpyro.sample("constant_variance", dist.HalfNormal(1)),
            "temp_lengthscale":  numpyro.sample("temp_lengthscale",  dist.LogNormal(jnp.log(8.0), 0.5)),
            "comp_weights":      numpyro.sample("comp_weights",      dist.Dirichlet(jnp.ones(3)*0.33)),
        }
        with numpyro.plate("sources", n_sources):
            p["source_var"] = numpyro.sample("source_var",dist.HalfCauchy(0.75))
        noise = numpyro.sample("noise", dist.HalfNormal(1))
        _gp_obs(builder(X, X, p), noise, y, n)

    return model, builder