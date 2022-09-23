import optuna
from plotly import offline
import pandas as pd

study = optuna.load_study(
    'test',
    'sqlite:///' + 'dry-opt1.sqlite3'
)
hyperparameters = [
    'Ta', 'va', 'ws', 'Vd', 'dur'
]


def target(t: optuna.trial.FrozenTrial) -> float:
    return t.values[1]


pareto = optuna.visualization.plot_pareto_front(study, target_names=['X (kg/kg)', 'survival rate'])
offline.plot(pareto)

writer = pd.ExcelWriter('opt1.xlsx')
study.trials_dataframe().to_excel(writer, index=None)
writer.save()
