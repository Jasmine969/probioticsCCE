import optuna
from calc_X_s import calc_x_s
from plotly import offline
import pandas as pd

path = 'optuna_res_dry/'


def objective(trial):
    ta = trial.suggest_float('T_a', 60, 110)
    va = trial.suggest_float('v_a', 0.45, 1.05)
    ws = trial.suggest_float('w_s', 0.055, 0.2, log=True)
    vd = trial.suggest_float('V_d', 0.5, 2.5)
    dur = trial.suggest_float('dur', 50, 300)
    x_t, s_t = calc_x_s({
        'Ta': ta + 273.15, 'va': va,
        'ws': ws, 'vd': vd * 1e-9, 'dur': dur
    }, ops='linux')
    return x_t, s_t


study = optuna.create_study(
    study_name='test',
    storage='sqlite:///' + path + 'dry-opt1.sqlite3',
    directions=['minimize', 'maximize']
)
study.optimize(objective, n_trials=10000)
pareto = optuna.visualization.plot_pareto_front(
    study, target_names=['X (kg/kg)', 'survival rate'])
offline.plot(pareto)
writer = pd.ExcelWriter(path + 'opt1.xlsx')
study.trials_dataframe().to_excel(writer, index=None)
writer.save()
