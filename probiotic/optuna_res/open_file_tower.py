import optuna
from plotly import offline
from matplotlib import pyplot as plt

path = 'optuna_res/test35/vali4/tower/'

study = optuna.load_study(
    'attn_prob2',
    'sqlite:///' + path + 's_non-tilde_tower_trial.sqlite3'
)
hyperparameters = [
    'decay',
    'lr',
    'flood',
    'dk',
    'dv',
    'drop',
    'head',
    'bs',
    'itv',
    # 'threshold',
    'itp_weight',
    # 'last_weight',
]
pareto = optuna.visualization.plot_pareto_front(study, target_names=['normal', 'lg'])
offline.plot(pareto)
# writer = pd.ExcelWriter('optuna_res/s_tilde.xlsx')
# study.trials_dataframe().to_excel(writer, index=None)
# writer.save()
