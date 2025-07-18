import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import weibull_min
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import jax
from flask import Flask, render_template_string

# H2O AutoML
import h2o
from h2o.automl import H2OAutoML

warnings.filterwarnings("ignore")

# === 📁 Load and Clean Dataset ===
df = pd.read_csv("construction_machine_final_data.csv")
df = df.replace([np.inf, -np.inf], np.nan).dropna()
df['Lifespan'] = df['Operating_Hours'] + df['Remaining_Useful_Life']

# === ⚙ Features and Target ===
target = 'Remaining_Useful_Life'
features = df.select_dtypes(include=[np.number]).columns.tolist()
features = [f for f in features if f not in ['Remaining_Useful_Life', 'Lifespan']]

X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === 📈 Weibull Classic ===
shape, loc, scale = weibull_min.fit(df['Lifespan'], floc=0)
def model1_weibull(t):
    return float(weibull_min.cdf(t, c=shape, scale=scale))

# === 📊 Bayesian Weibull ===
def numpyro_model(data):
    beta = numpyro.sample("beta", dist.HalfNormal(5.0))
    eta = numpyro.sample("eta", dist.HalfNormal(5000.0))
    with numpyro.plate("data", len(data)):
        numpyro.sample("obs", dist.Weibull(concentration=beta, scale=eta), obs=data)

lifespan_data = df['Lifespan'].values
nuts_kernel = NUTS(numpyro_model)
mcmc = MCMC(nuts_kernel, num_warmup=500, num_samples=1000, progress_bar=False)
mcmc.run(jax.random.PRNGKey(0), data=lifespan_data)
posterior = mcmc.get_samples()
beta_samples = posterior['beta']
eta_samples = posterior['eta']

def model_bayes_weibull(t):
    probs = 1 - jnp.exp(-(t / eta_samples) ** beta_samples)
    probs = jnp.clip(probs, 0, 1)
    mean = float(jnp.mean(probs))
    lower, upper = np.percentile(np.array(probs), [2.5, 97.5])
    plt.figure(figsize=(8, 4))
    sns.histplot(probs, kde=True, color='skyblue')
    plt.axvline(mean, color='red', linestyle='--', label=f'Mean: {mean:.4f}')
    plt.axvline(lower, color='green', linestyle='--', label=f'2.5%: {lower:.4f}')
    plt.axvline(upper, color='orange', linestyle='--', label=f'97.5%: {upper:.4f}')
    plt.title(f"Bayesian Failure Probability at t = {t}")
    plt.xlabel("Failure Probability")
    plt.legend(); plt.grid(True)
    buf = BytesIO(); plt.tight_layout(); plt.savefig(buf, format='png'); buf.seek(0); plt.close()
    return mean, float(lower), float(upper), 'data:image/png;base64,' + base64.b64encode(buf.read()).decode('utf-8')

# === 📉 Linear Regression ===
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
mae_lr = mean_absolute_error(y_test, y_pred_lr)
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

def plot_lr_actual_vs_pred():
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=y_test, y=y_pred_lr, color='tomato', edgecolor='k')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'gray', linestyle='--')
    plt.xlabel("Actual RUL"); plt.ylabel("Predicted RUL"); plt.title("Linear Regression")
    buf = BytesIO(); plt.tight_layout(); plt.savefig(buf, format='png'); buf.seek(0); plt.close()
    return 'data:image/png;base64,' + base64.b64encode(buf.read()).decode('utf-8')

# === 🌳 AdaBoost Regressor ===
param_grid = {'n_estimators': [100], 'learning_rate': [0.3], 'estimator__max_depth': [4]}
ada = AdaBoostRegressor(estimator=DecisionTreeRegressor(), random_state=42)
grid = GridSearchCV(ada, param_grid, cv=3, n_jobs=-1)
grid.fit(X_train, y_train)
best_ada = grid.best_estimator_
y_pred_ada = best_ada.predict(X_test)
rmse_ada = np.sqrt(mean_squared_error(y_test, y_pred_ada))
mae_ada = mean_absolute_error(y_test, y_pred_ada)
mse_ada = mean_squared_error(y_test, y_pred_ada)
r2_ada = r2_score(y_test, y_pred_ada)

def plot_ada_actual_vs_pred():
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=y_test, y=y_pred_ada, color='seagreen')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'gray', linestyle='--')
    plt.xlabel("Actual RUL"); plt.ylabel("Predicted RUL"); plt.title("AdaBoost")
    buf = BytesIO(); plt.tight_layout(); plt.savefig(buf, format='png'); buf.seek(0); plt.close()
    return 'data:image/png;base64,' + base64.b64encode(buf.read()).decode('utf-8')

# === ⚡ H2O AutoML ===
h2o.init(max_mem_size_GB=1, nthreads=-1)
h2o.no_progress()
df_h2o = h2o.H2OFrame(df[features + [target]])
train_h2o, test_h2o = df_h2o.split_frame(ratios=[0.8], seed=1)
aml = H2OAutoML(max_models=5, max_runtime_secs=60, exclude_algos=["DeepLearning"], seed=1)
aml.train(x=features, y=target, training_frame=train_h2o)
preds_h2o = aml.leader.predict(test_h2o).as_data_frame().values.flatten()
y_true_h2o = test_h2o[target].as_data_frame().values.flatten()
rmse_h2o = np.sqrt(mean_squared_error(y_true_h2o, preds_h2o))
mse_h2o = mean_squared_error(y_true_h2o, preds_h2o)
mae_h2o = mean_absolute_error(y_true_h2o, preds_h2o)
r2_h2o = r2_score(y_true_h2o, preds_h2o)

def plot_h2o_actual_vs_pred():
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=y_true_h2o, y=preds_h2o, color='mediumslateblue')
    plt.plot([min(y_true_h2o), max(y_true_h2o)], [min(y_true_h2o), max(y_true_h2o)], 'gray', linestyle='--')
    plt.xlabel("Actual RUL"); plt.ylabel("Predicted RUL"); plt.title("H2O AutoML")
    buf = BytesIO(); plt.tight_layout(); plt.savefig(buf, format='png'); buf.seek(0); plt.close()
    return 'data:image/png;base64,' + base64.b64encode(buf.read()).decode('utf-8')

# === 📊 Visualization ===
def plot_weibull_curve():
    t_range = np.linspace(0, df['Lifespan'].max(), 200)
    plt.figure(figsize=(6, 4))
    plt.plot(t_range, weibull_min.cdf(t_range, c=shape, scale=scale), color='dodgerblue')
    plt.title('Weibull CDF'); plt.grid(True)
    buf = BytesIO(); plt.tight_layout(); plt.savefig(buf, format='png'); buf.seek(0); plt.close()
    return 'data:image/png;base64,' + base64.b64encode(buf.read()).decode('utf-8')

def plot_weibull_summary():
    summary = df['Lifespan'].describe()
    fig, axes = plt.subplots(2, 1, figsize=(8, 5))
    t_range = np.linspace(0, df['Lifespan'].max(), 200)
    axes[0].plot(t_range, weibull_min.cdf(t_range, c=shape, scale=scale), color='teal')
    for label, val in zip(['Min', '25%', '50%', '75%', 'Max'],
                          [summary['min'], summary['25%'], summary['50%'], summary['75%'], summary['max']]):
        axes[0].axvline(val, linestyle='--', label=f'{label}: {int(val)}')
    axes[0].legend(); axes[0].grid(True)
    axes[1].boxplot(df['Lifespan'], vert=False)
    axes[1].set_title('Lifespan Boxplot')
    buf = BytesIO(); plt.tight_layout(); plt.savefig(buf, format='png'); buf.seek(0); plt.close()
    return 'data:image/png;base64,' + base64.b64encode(buf.read()).decode('utf-8')

def correlation_heatmap():
    plt.figure(figsize=(10, 6))
    sns.heatmap(df[features + [target]].corr(), annot=True, cmap='coolwarm')
    plt.title("Correlation Heatmap")
    buf = BytesIO(); plt.tight_layout(); plt.savefig(buf, format='png'); buf.seek(0); plt.close()
    return 'data:image/png;base64,' + base64.b64encode(buf.read()).decode('utf-8')

def scatter_plots():
    num_cols = [col for col in features if col != target]
    n_cols = 3
    n_rows = int(np.ceil(len(num_cols) / n_cols))
    plt.figure(figsize=(n_cols * 5, n_rows * 4))
    for idx, col in enumerate(num_cols):
        plt.subplot(n_rows, n_cols, idx + 1)
        sns.scatterplot(data=df, x=col, y=target, alpha=0.6)
        plt.title(f"{col} vs RUL"); plt.grid(True)
    buf = BytesIO(); plt.tight_layout(); plt.savefig(buf, format='png'); buf.seek(0); plt.close()
    return 'data:image/png;base64,' + base64.b64encode(buf.read()).decode('utf-8')

# Flask App 
app = Flask(__name__)

@app.route('/')
def index():
    t = 1000
    weibull_prob = model1_weibull(t)
    mean_bayes, lower_bayes, upper_bayes, bayes_plot = model_bayes_weibull(t)
    return render_template_string(TEMPLATE,
        t=t,
        weibull_prob=weibull_prob,
        mean_bayes=mean_bayes,
        lower_bayes=lower_bayes,
        upper_bayes=upper_bayes,
        bayes_plot=bayes_plot,
        weibull_curve=plot_weibull_curve(),
        summary_plot=plot_weibull_summary(),
        lr_plot=plot_lr_actual_vs_pred(),
        rmse_lr=rmse_lr, mae_lr=mae_lr, mse_lr=mse_lr, r2_lr=r2_lr,
        ada_plot=plot_ada_actual_vs_pred(),
        rmse_ada=rmse_ada, mae_ada=mae_ada, mse_ada=mse_ada, r2_ada=r2_ada,
        h2o_plot=plot_h2o_actual_vs_pred(),
        rmse_h2o=rmse_h2o, r2_h2o=r2_h2o, mae_h2o=mae_h2o, mse_h2o=mse_h2o,
        heatmap_plot=correlation_heatmap(),
        scatter_plot=scatter_plots()
    )

TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
  <title>Failure Prediction Dashboard</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body style="background-color:#121212; color:white;">
<div class="container mt-4">
  <h2 class="text-center mb-4">Failure Prediction at t = {{ t }} hrs</h2>
  <div class="row">
    <div class="col-md-6">
      <h5>Weibull Classic Probability: {{ weibull_prob }}</h5>
      <img src="{{ weibull_curve }}" class="img-fluid" />
    </div>
    <div class="col-md-6">
      <h5>Bayesian Weibull</h5>
      <p>Mean: {{ mean_bayes }} <br> 95% CI: [{{ lower_bayes }}, {{ upper_bayes }}]</p>
      <img src="{{ bayes_plot }}" class="img-fluid" />
    </div>
    <div class="col-md-6">
      <h5>Linear Regression</h5>
      <ul>
        <li>RMSE: {{ rmse_lr }}</li>
        <li>MAE: {{ mae_lr }}</li>
        <li>MSE: {{ mse_lr }}</li>
        <li>R²: {{ r2_lr }}</li>
      </ul>
      <img src="{{ lr_plot }}" class="img-fluid" />
    </div>
    <div class="col-md-6">
      <h5>AdaBoost</h5>
      <ul>
        <li>RMSE: {{ rmse_ada }}</li>
        <li>MAE: {{ mae_ada }}</li>
        <li>MSE: {{ mse_ada }}</li>
        <li>R²: {{ r2_ada }}</li>
      </ul>
      <img src="{{ ada_plot }}" class="img-fluid" />
    </div>
    <div class="col-md-6">
      <h5>H2O AutoML</h5>
      <ul>
        <li>RMSE: {{ rmse_h2o }}</li>
        <li>MAE: {{ mae_h2o }}</li>
        <li>MSE: {{ mse_h2o }}</li>
        <li>R²: {{ r2_h2o }}</li>
      </ul>
      <img src="{{ h2o_plot }}" class="img-fluid" />
    </div>
    <div class="col-md-6">
      <h5>📦 Lifespan 5-Point Summary</h5>
      <img src="{{ summary_plot }}" class="img-fluid" />
    </div>
    <div class="col-md-6">
      <h5>🔗 Feature Correlation Heatmap</h5>
      <img src="{{ heatmap_plot }}" class="img-fluid" />
    </div>
    <div class="col-md-12">
      <h5>🔷 Scatter Plots: Feature vs RUL</h5>
      <img src="{{ scatter_plot }}" class="img-fluid" />
    </div>
  </div>
</div>
</body>
</html>
"""

if __name__ == '__main__':
    app.run(debug=False)