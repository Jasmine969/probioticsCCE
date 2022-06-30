import optuna
from calc_X_s import calc_x_s
from plotly import offline
import pandas as pd

path = 'optuna_res_dry/'


def objective(trial):
    ta = trial.suggest_float('T_a', 60, 73)
    va = trial.suggest_float('v_a', 0.6, 1.2)
    ws = trial.suggest_float('w_s', 0.1, 0.2)
    vd = trial.suggest_float('V_d', 0.5, 1.0)
    dur = trial.suggest_float('dur', 125, 200)
    x_t, s_t = calc_x_s({
        'Ta': ta + 273.15, 'va': va,
        'ws': ws, 'vd': vd * 1e-9, 'dur': dur
    }, ops='win')
    return x_t, s_t


study = optuna.create_study(
    study_name='test',
    storage='sqlite:///' + path + 'dry-opt2.sqlite3',
    directions=['minimize', 'maximize']
)
study.optimize(objective, n_trials=1000)
pareto = optuna.visualization.plot_pareto_front(
    study, target_names=['X (kg/kg)', 'survival rate'])
offline.plot(pareto)
writer = pd.ExcelWriter(path + 'opt2.xlsx')
study.trials_dataframe().to_excel(writer, index=None)
writer.save()
