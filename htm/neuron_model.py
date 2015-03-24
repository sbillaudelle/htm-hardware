import pyNN.nest as pynn

pynn.nest.Install('brainscales_hw_module')
pynn.nest.Install('hmf_hw_module')

iaf_4_cond_exp = pynn.native_cell_type('iaf_4_cond_exp')
aeif_4_cond_exp = pynn.native_cell_type('aeif_4_cond_exp')
