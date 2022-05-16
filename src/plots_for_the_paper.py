import src.plotting as plt

# # %% Trivial q vs z
# data_fname = 'from_cluster/model_half_False_other_False_mass_4.0'
# fig_fname = 'publishing/full_model_q_vs_z_trivial'
# plt.plot_charge_per_layer(data_fname=data_fname, save=True, ylim=0.4, fig_fname=fig_fname)
#
# # %% Trivial q vs z
# data_fname = 'from_cluster/model_half_False_other_False_mass_2.0'
# fig_fname = 'publishing/full_model_q_vs_z_topological'
# plt.plot_charge_per_layer(data_fname=data_fname, save=True, ylim=0.4, fig_fname=fig_fname)
#
# # %% Topological Charge Density
# data_fname = 'from_cluster/model_half_False_other_False_mass_2.0'
# fig_fname = 'publishing/full_model_charge_density_topological'
# plt.plot_disclination_rho('bottom', data_fname=data_fname, save=True, fig_fname=fig_fname)

# %% Full model charge vs mass
data_folder = ''
fig_fname = 'publishing/full_model_charge_vs_mass'
plt.plot_q_vs_mass(data_folder, True, False, False, mod=True, save=False, fig_fname=fig_fname)
