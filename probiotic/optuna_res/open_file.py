import optuna
from plotly import offline
from matplotlib import pyplot as plt

study = optuna.load_study('attn_prob', 'sqlite:///test35/vali2/s_non-tilde_pureacc.sqlite3')
importance = optuna.visualization.plot_param_importances(study)
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
parallel = optuna.visualization.plot_parallel_coordinate(
    study, hyperparameters)
slices = optuna.visualization.plot_slice(
    study, hyperparameters)

# offline.plot(importance)
offline.plot(parallel)
# offline.plot(slices)
# writer = pd.ExcelWriter('optuna_res/s_tilde.xlsx')
# study.trials_dataframe().to_excel(writer, index=None)
# writer.save()
