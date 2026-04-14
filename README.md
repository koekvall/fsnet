# fsnet (Finite-Support Elastic Net)

An R package for elastic net-regularized regression with finite-support
response variables. Handles interval-censored, categorical, and other
discrete responses using a concave likelihood. Fits the full solution
path efficiently using proximal Newton or accelerated proximal gradient
descent (FISTA).

## Installation

```r
install.packages("devtools")
devtools::install_github("koekvall/fsnet")
```

## Example

### Interval-censored regression

```r
library(fsnet)

# Simulate data
set.seed(1)
n <- 200
p <- 50
b <- c(1, -0.5, 0.8, rep(0, p - 3))  # sparse coefficients
X <- matrix(rnorm(n * p), n, p)

# Generate interval-censored responses (integer-valued)
Y <- generate_norm(X, b, d = 1, sigma = 1)

# Fit lasso-penalized model over a grid of lambda values
lam <- exp(seq(log(1), log(1e-4), length.out = 50))
fit <- fsnet_ic(Y[, 1:2], X, lam = lam, alpha = 1)

# Coefficient path (first 5 predictors)
matplot(log(fit$lam), t(fit$beta[1:5, ]), type = "l",
        xlab = "log(lambda)", ylab = "Coefficient",
        main = "Lasso path")

# Select lambda by 5-fold cross-validation
fit_cv <- fsnet_ic(Y[, 1:2], X, lam = lam, alpha = 1, nfold = 5)
fit_cv$b_star[1:5]  # coefficients at selected lambda
```

### Categorical regression

```r
# Simulate ordinal response with 4 categories
set.seed(2)
n <- 300
p <- 20
b <- c(1, -1, rep(0, p - 2))
X <- matrix(rnorm(n * p), n, p)
eta <- X %*% b
probs <- cbind(pnorm(-1 - eta), pnorm(0 - eta) - pnorm(-1 - eta),
               pnorm(1 - eta) - pnorm(0 - eta), 1 - pnorm(1 - eta))
Y <- factor(apply(probs, 1, function(p) sample(1:4, 1, prob = p)))

# Fit ridge-penalized ordinal regression
lam <- exp(seq(log(1), log(1e-4), length.out = 30))
fit_cat <- fsnet_cat(Y, X, lam = lam, alpha = 0)

# Estimated coefficients at smallest penalty
fit_cat$beta[, ncol(fit_cat$beta)]
```

## Main functions

- `fsnet_ic(Y, X, ...)` -- interval-censored regression (e.g., rounded or binned continuous data)
- `fsnet_cat(Y, X, ...)` -- categorical (ordinal) regression
- `fsnet_flex(M, Z, ...)` -- flexible low-level interface for general finite-support likelihoods

Common arguments:
- `lam` -- vector of penalty parameters (solution path)
- `alpha` -- elastic net mixing (1 = lasso, 0 = ridge)
- `method` -- `"prox_newt"` (default, proximal Newton) or `"fista"`
- `nfold` -- number of cross-validation folds (1 = no CV)

See `?fsnet_ic`, `?fsnet_cat`, and `?fsnet_flex` for details.

## Citation

The corresponding paper is published in [Biometrics](https://doi.org/10.1111/biom.13760) (also available on [arXiv](http://arxiv.org/abs/2203.04582)).

```bibtex
@article{ekvall2023concave,
  title={Concave likelihood-based regression with finite-support response variables},
  author={Ekvall, Karl Oskar and Bottai, Matteo},
  journal={Biometrics},
  volume={79},
  number={3},
  pages={1823--1834},
  year={2023},
  publisher={Wiley},
  doi={10.1111/biom.13760}
}
```
