import src.disclination_ed as disc_ed
import src.utils as utils
from importlib import reload

# %% reloading
reload(disc_ed)
reload(utils)

# %% Trivial q vs z
data_fname = 'from_cluster/model_half_False_other_False_mass_4.0'
results, params = utils.load_results(data_fname)

fig_fname = 'publishing/full_model_q_vs_z_trivial'
disc_ed.plot_charge_per_layer(data_fname=data_fname, save=True, ylim=0.4, fig_fname=fig_fname)

# %% Trivial q vs z
data_fname = 'from_cluster/model_half_False_other_False_mass_2.0'
results, params = utils.load_results(data_fname)

fig_fname = 'publishing/full_model_q_vs_z_topological'
disc_ed.plot_charge_per_layer(data_fname=data_fname, save=True, ylim=0.4, fig_fname=fig_fname)
