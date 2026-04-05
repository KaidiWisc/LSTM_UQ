import logging
from typing import Dict

import pandas as pd
import torch
import torch.nn as nn

from neuralhydrology.datautils.utils import get_frequency_factor, sort_frequencies
from neuralhydrology.modelzoo.head import get_head
from neuralhydrology.modelzoo.basemodel import BaseModel
from neuralhydrology.utils.config import Config

import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule
from pyro.nn import PyroSample

pyro.enable_validation(True)  

assert issubclass(PyroModule[nn.Linear], nn.Linear)
assert issubclass(PyroModule[nn.Linear], PyroModule)
assert issubclass(PyroModule[nn.LSTM], nn.LSTM)
assert issubclass(PyroModule[nn.LSTM], PyroModule)
assert issubclass(PyroModule[nn.ModuleDict], nn.ModuleDict)
assert issubclass(PyroModule[nn.ModuleDict], PyroModule)


LOGGER = logging.getLogger(__name__)


class MTSLSTM(BaseModel):
    """Multi-Timescale LSTM (MTS-LSTM) from Gauch et al. [#]_.

    An LSTM architecture that allows simultaneous prediction at multiple timescales within one model.
    There are two flavors of this model: MTS-LTSM and sMTS-LSTM (shared MTS-LSTM). The MTS-LSTM processes inputs at
    low temporal resolutions up to a point in time. Then, the LSTM splits into one branch for each target timescale.
    Each branch processes the inputs at its respective timescale. Finally, one prediction head per timescale generates
    the predictions for that timescale based on the LSTM output.
    Optionally, one can specify:
    - a different hidden size for each LSTM branch (use a dict in the ``hidden_size`` config argument)
    - different dynamic input variables for each timescale (use a dict in the ``dynamic_inputs`` config argument)
    - the strategy to transfer states from the initial shared low-resolution LSTM to the per-timescale
    higher-resolution LSTMs. By default, this is a linear transfer layer, but you can specify 'identity' to use an
    identity operation or 'None' to turn off any transfer (via the ``transfer_mtlstm_states`` config argument).


    The sMTS-LSTM variant has the same overall architecture, but the weights of the per-timescale branches (including
    the output heads) are shared.
    Thus, unlike MTS-LSTM, the sMTS-LSTM cannot use per-timescale hidden sizes or dynamic input variables.
    
    Parameters
    ----------
    cfg : Config
        The run configuration.

    References
    ----------
    .. [#] Gauch, M., Kratzert, F., Klotz, D., Grey, N., Lin, J., and Hochreiter, S.: Rainfall-Runoff Prediction at
        Multiple Timescales with a Single Long Short-Term Memory Network, arXiv Preprint,
        https://arxiv.org/abs/2010.07921, 2020.
    """
    # specify submodules of the model that can later be used for finetuning. Names must match class attributes
    module_parts = ['lstms', 'transfer_fcs', 'heads']

    def __init__(self, cfg: Config):
        super(MTSLSTM, self).__init__(cfg=cfg)
        self.lstms = None
        self.transfer_fcs = None
        self.heads = None
        self.dropout = None
        
        self.bvi=cfg.bvi
        self.headname=cfg.head.lower()
        self.batch_size=cfg.batch_size
        self.last_freq=cfg.predict_last_n["1H"]
        
        if self.headname=="umal":
            self.ntau=cfg.n_taus
        
        self._slice_timestep = {}
        self._frequency_factors = []

        self._seq_lengths = cfg.seq_length
        self._is_shared_mtslstm = self.cfg.shared_mtslstm  # default: a distinct LSTM per timescale
        self._transfer_mtslstm_states = self.cfg.transfer_mtslstm_states  # default: linear transfer layer
        transfer_modes = [None, "None", "identity", "linear"]
        if self._transfer_mtslstm_states["h"] not in transfer_modes \
                or self._transfer_mtslstm_states["c"] not in transfer_modes:
            raise ValueError(f"MTS-LSTM supports state transfer modes {transfer_modes}")

        if len(cfg.use_frequencies) < 2:
            raise ValueError("MTS-LSTM expects more than one input frequency")
        self._frequencies = sort_frequencies(cfg.use_frequencies)

        # start to count the number of inputs
        input_sizes = len(cfg.static_attributes + cfg.hydroatlas_attributes + cfg.evolving_attributes)

        # if is_shared_mtslstm, the LSTM gets an additional frequency flag as input.
        if self._is_shared_mtslstm:
            input_sizes += len(self._frequencies)

        if cfg.use_basin_id_encoding:
            input_sizes += cfg.number_of_basins
        if cfg.head.lower() == "umal":
            input_sizes += 1

        if isinstance(cfg.dynamic_inputs, list):
            input_sizes = {freq: input_sizes + len(cfg.dynamic_inputs) for freq in self._frequencies}
        else:
            if self._is_shared_mtslstm:
                raise ValueError(f'Different inputs not allowed if shared_mtslstm is used.')
            input_sizes = {freq: input_sizes + len(cfg.dynamic_inputs[freq]) for freq in self._frequencies}

        if not isinstance(cfg.hidden_size, dict):
            LOGGER.info("No specific hidden size for frequencies are specified. Same hidden size is used for all.")
            self._hidden_size = {freq: cfg.hidden_size for freq in self._frequencies}
        else:
            self._hidden_size = cfg.hidden_size

        if (self._is_shared_mtslstm
            or self._transfer_mtslstm_states["h"] == "identity"
            or self._transfer_mtslstm_states["c"] == "identity") \
                and any(size != self._hidden_size[self._frequencies[0]] for size in self._hidden_size.values()):
            raise ValueError("All hidden sizes must be equal if shared_mtslstm is used or state transfer=identity.")

        # create layer depending on selected frequencies
        
        if cfg.bvi==0:
            self._init_modules(input_sizes)
            self._reset_parameters()
        else:
            self._init_modules_BVI(input_sizes) #!!!

        # frequency factors are needed to determine the time step of information transfer
        self._init_frequency_factors_and_slice_timesteps()

    def _init_modules_BVI(self, input_sizes: Dict[str, int]):
        
        self.lstms = PyroModule[nn.ModuleDict]()
        self.transfer_fcs = PyroModule[nn.ModuleDict]()
        self.heads = PyroModule[nn.ModuleDict]()
        self.dropout = nn.Dropout(p=self.cfg.output_dropout)
        
        for idx, freq in enumerate(self._frequencies):
            freq_input_size = input_sizes[freq]


            if self._is_shared_mtslstm and idx > 0:
                self.lstms[freq] = self.lstms[self._frequencies[idx - 1]]  # same LSTM for all frequencies.
                self.heads[freq] = self.heads[self._frequencies[idx - 1]]  # same head for all frequencies.
            else:
                self.lstms[freq] = PyroModule[nn.LSTM](input_size=freq_input_size, hidden_size=self._hidden_size[freq])  #!!!
                # Define Prior Distributions for LSTM Parameters               
                
                self.lstms[freq].weight_ih_l0 = PyroSample(
                        dist.Normal(0., 0.1).expand([4 * self._hidden_size[freq], freq_input_size]).to_event(2))
                
                self.lstms[freq].weight_hh_l0 = PyroSample(dist.Normal(0., 0.1).
                                                           expand([4 * self._hidden_size[freq], self._hidden_size[freq]])
                                                           .to_event(2)) #,name=f"weight_hh_l0_{freq}")
                self.lstms[freq].bias_ih_l0 = PyroSample(dist.Normal(0., 0.1).expand([4 * self._hidden_size[freq]]).
                                                         to_event(1)) #,name=f"bias_ih_l0_{freq}")
                self.lstms[freq].bias_hh_l0 = PyroSample(dist.Normal(0., 1.).expand([4 * self._hidden_size[freq]])
                                                         .to_event(1)) #,name=f"bias_hh_l0_{freq}")      
                
                self.heads[freq] = get_head(self.cfg, n_in=self._hidden_size[freq], n_out=self.output_size, freq=freq)
                
                
            if idx < len(self._frequencies) - 1:
                for state in ["c", "h"]:
                    if self._transfer_mtslstm_states[state] == "linear":
                        self.transfer_fcs[f"{state}_{freq}"] = PyroModule[nn.Linear](self._hidden_size[freq],
                                                                         self._hidden_size[self._frequencies[idx + 1]]).to('cuda')
                        self.transfer_fcs[f"{state}_{freq}"].bias = PyroSample(dist.Normal(0., 0.1)
                                                .expand([self._hidden_size[self._frequencies[idx + 1]]])
                                                .to_event(1)) #,name=f"tsf_{state}_{freq}_bias")
                        self.transfer_fcs[f"{state}_{freq}"].weight=PyroSample(dist.Normal(0., 0.1)
                                            .expand([self._hidden_size[self._frequencies[idx + 1]], self._hidden_size[freq]])
                                            .to_event(2)) #,name=f"tsf_{state}_{freq}_weight")
                        
                    elif self._transfer_mtslstm_states[state] == "identity":
                        self.transfer_fcs[f"{state}_{freq}"] = nn.Identity().to('cuda')
                    else:
                        pass
                    
                    
    def _init_modules(self, input_sizes: Dict[str, int]):
        self.lstms = nn.ModuleDict()
        self.transfer_fcs = nn.ModuleDict()
        self.heads = nn.ModuleDict()
        self.dropout = nn.Dropout(p=self.cfg.output_dropout)
        for idx, freq in enumerate(self._frequencies):
            freq_input_size = input_sizes[freq]

            if self._is_shared_mtslstm and idx > 0:
                self.lstms[freq] = self.lstms[self._frequencies[idx - 1]]  # same LSTM for all frequencies.
                self.heads[freq] = self.heads[self._frequencies[idx - 1]]  # same head for all frequencies.
            else:
                self.lstms[freq] = nn.LSTM(input_size=freq_input_size, hidden_size=self._hidden_size[freq])
                self.heads[freq] = get_head(self.cfg, n_in=self._hidden_size[freq], n_out=self.output_size)

            if idx < len(self._frequencies) - 1:
                for state in ["c", "h"]:
                    if self._transfer_mtslstm_states[state] == "linear":
                        self.transfer_fcs[f"{state}_{freq}"] = nn.Linear(self._hidden_size[freq],
                                                                         self._hidden_size[self._frequencies[idx + 1]])
                    elif self._transfer_mtslstm_states[state] == "identity":
                        self.transfer_fcs[f"{state}_{freq}"] = nn.Identity()
                    else:
                        pass

    def _init_frequency_factors_and_slice_timesteps(self):
        for idx, freq in enumerate(self._frequencies):
            if idx < len(self._frequencies) - 1:
                frequency_factor = get_frequency_factor(freq, self._frequencies[idx + 1])
                if frequency_factor != int(frequency_factor):
                    raise ValueError('Adjacent frequencies must be multiples of each other.')
                self._frequency_factors.append(int(frequency_factor))
                # we want to pass the state of the day _before_ the next higher frequency starts,
                # because e.g. the mean of a day is stored at the same date at 00:00 in the morning.
                slice_timestep = int(self._seq_lengths[self._frequencies[idx + 1]] / self._frequency_factors[idx])
                self._slice_timestep[freq] = slice_timestep

    def _reset_parameters(self):
        if self.cfg.initial_forget_bias is not None:
            for freq in self._frequencies:
                hidden_size = self._hidden_size[freq]
                self.lstms[freq].bias_hh_l0.data[hidden_size:2 * hidden_size] = self.cfg.initial_forget_bias

    def _prepare_inputs(self, data: Dict[str, torch.Tensor], freq: str) -> torch.Tensor:
        """Concat all different inputs to the time series input"""
        suffix = f"_{freq}"
        # transpose to [seq_length, batch_size, n_features]
        x_d = data[f'x_d{suffix}'].transpose(0, 1)

        # concat all inputs
        if f'x_s{suffix}' in data and 'x_one_hot' in data:
            x_s = data[f'x_s{suffix}'].unsqueeze(0).repeat(x_d.shape[0], 1, 1)
            x_one_hot = data['x_one_hot'].unsqueeze(0).repeat(x_d.shape[0], 1, 1)
            x_d = torch.cat([x_d, x_s, x_one_hot], dim=-1)
        elif f'x_s{suffix}' in data:
            x_s = data[f'x_s{suffix}'].unsqueeze(0).repeat(x_d.shape[0], 1, 1)
            x_d = torch.cat([x_d, x_s], dim=-1)
        elif 'x_one_hot' in data:
            x_one_hot = data['x_one_hot'].unsqueeze(0).repeat(x_d.shape[0], 1, 1)
            x_d = torch.cat([x_d, x_one_hot], dim=-1)
        else:
            pass

        if self._is_shared_mtslstm:
            # add frequency one-hot encoding
            idx = self._frequencies.index(freq)
            one_hot_freq = torch.zeros(x_d.shape[0], x_d.shape[1], len(self._frequencies)).to(x_d)
            one_hot_freq[:, :, idx] = 1
            x_d = torch.cat([x_d, one_hot_freq], dim=2)

        return x_d

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Perform a forward pass on the MTS-LSTM model.
        
        Parameters
        ----------
        data : Dict[str, torch.Tensor]
            Input data for the forward pass. See the documentation overview of all models for details on the dict keys.

        Returns
        -------
        Dict[str, torch.Tensor]
            Model predictions for each target timescale.
        """
        x_d = {freq: self._prepare_inputs(data, freq) for freq in self._frequencies}

        # initial states for lowest frequencies are set to zeros
        batch_size = x_d[self._frequencies[0]].shape[1]
        lowest_freq_hidden_size = self._hidden_size[self._frequencies[0]]
        h_0_transfer = x_d[self._frequencies[0]].new_zeros((1, batch_size, lowest_freq_hidden_size))
        c_0_transfer = torch.zeros_like(h_0_transfer)

        outputs = {}
        for idx, freq in enumerate(self._frequencies):
            if idx < len(self._frequencies) - 1:
                # get predictions and state up to the time step of information transfer
                slice_timestep = self._slice_timestep[freq]
                
                # lstm_output_slice1, (h_n_slice1, c_n_slice1) = self.lstms[freq](x_d[freq][:-slice_timestep],
                #                                                                 (h_0_transfer, c_0_transfer))
                # import copy
                # lstm_copy = copy.deepcopy(self.lstms[freq])
                
                lstm_copy = nn.LSTM(input_size=self.lstms[freq].input_size, hidden_size=self._hidden_size[freq]).to('cuda')
                lstm_copy.weight_ih_l0.data.copy_(self.lstms[freq].weight_ih_l0.to('cuda'))
                lstm_copy.weight_hh_l0.data.copy_(self.lstms[freq].weight_hh_l0.to('cuda'))
                lstm_copy.bias_ih_l0.data.copy_(self.lstms[freq].bias_ih_l0.to('cuda'))
                lstm_copy.bias_hh_l0.data.copy_(self.lstms[freq].bias_hh_l0.to('cuda'))
                

                lstm_output_slice1, (h_n_slice1, c_n_slice1) =lstm_copy(x_d[freq][:-slice_timestep],
                                                                                (h_0_transfer, c_0_transfer))       

                # project the states through a hidden layer to the dimensions of the next LSTM
                if self._transfer_mtslstm_states["h"] is not None:
                    self.transfer_fcs[f"h_{freq}"]=self.transfer_fcs[f"h_{freq}"].to('cuda')
                    self.transfer_fcs[f"h_{freq}"].bias=self.transfer_fcs[f"h_{freq}"].bias.to('cuda')
                    self.transfer_fcs[f"h_{freq}"].weight=self.transfer_fcs[f"h_{freq}"].weight.to('cuda')
                    h_0_transfer = self.transfer_fcs[f"h_{freq}"](h_n_slice1.to('cuda'))

                if self._transfer_mtslstm_states["c"] is not None:
                    self.transfer_fcs[f"c_{freq}"]=self.transfer_fcs[f"c_{freq}"].to('cuda')
                    self.transfer_fcs[f"c_{freq}"].bias=self.transfer_fcs[f"c_{freq}"].bias.to('cuda')
                    self.transfer_fcs[f"c_{freq}"].weight=self.transfer_fcs[f"c_{freq}"].weight.to('cuda')
                    c_0_transfer = self.transfer_fcs[f"c_{freq}"](c_n_slice1.to('cuda'))

                # get predictions of remaining part and concat results

                self.lstms[freq] = self.lstms[freq].to('cuda')
                self.lstms[freq].weight_ih_l0=self.lstms[freq].weight_ih_l0.to('cuda')
                self.lstms[freq].weight_hh_l0=self.lstms[freq].weight_hh_l0.to('cuda')
                self.lstms[freq].bias_ih_l0=self.lstms[freq].bias_ih_l0.to('cuda')
                self.lstms[freq].bias_hh_l0=self.lstms[freq].bias_hh_l0.to('cuda')

                lstm_output_slice2, _ = self.lstms[freq](x_d[freq][-slice_timestep:], (h_n_slice1, c_n_slice1))
                # lstm_output_slice2, _ = lstm_copy(x_d[freq][-slice_timestep:], (h_n_slice1, c_n_slice1))
                lstm_output = torch.cat([lstm_output_slice1, lstm_output_slice2], dim=0)

            else:
                # for highest frequency, we can pass the entire sequence at once
                self.lstms[freq] = self.lstms[freq].to('cuda')
                self.lstms[freq].weight_ih_l0=self.lstms[freq].weight_ih_l0.to('cuda')
                self.lstms[freq].weight_hh_l0=self.lstms[freq].weight_hh_l0.to('cuda')
                self.lstms[freq].bias_ih_l0=self.lstms[freq].bias_ih_l0.to('cuda')
                self.lstms[freq].bias_hh_l0=self.lstms[freq].bias_hh_l0.to('cuda')
                lstm_output, _ = self.lstms[freq](x_d[freq], (h_0_transfer, c_0_transfer))
            
            if self.headname=="regression":
                self.heads[freq]=self.heads[freq].to('cuda')
                self.heads[freq].net[f"l0_{freq}"].weight=self.heads[freq].net[f"l0_{freq}"].weight.to('cuda')
                self.heads[freq].net[f"l0_{freq}"].bias=self.heads[freq].net[f"l0_{freq}"].bias.to('cuda')
            else:
                self.heads[freq]=self.heads[freq].to('cuda')
                self.heads[freq].fcs[f"fc1_{freq}"]=self.heads[freq].fcs[f"fc1_{freq}"].to('cuda')
                self.heads[freq].fcs[f"fc1_{freq}"].weight=self.heads[freq].fcs[f"fc1_{freq}"].weight.to('cuda')
                self.heads[freq].fcs[f"fc1_{freq}"].bias=self.heads[freq].fcs[f"fc1_{freq}"].bias.to('cuda')
                self.heads[freq].fcs[f"fc2_{freq}"]=self.heads[freq].fcs[f"fc2_{freq}"].to('cuda')
                self.heads[freq].fcs[f"fc2_{freq}"].weight=self.heads[freq].fcs[f"fc2_{freq}"].weight.to('cuda')
                self.heads[freq].fcs[f"fc2_{freq}"].bias=self.heads[freq].fcs[f"fc2_{freq}"].bias.to('cuda')

            head_out = self.heads[freq](self.dropout(lstm_output.transpose(0, 1)))
            outputs.update({f'{key}_{freq}': value for key, value in head_out.items()})
            
        if self.bvi==1:
            
            if self.headname=="umal":
                # {'mu': m, 'b': b}
                mu=outputs["mu_1H"]
                b=outputs["b_1H"]
                t=data["tau_1H"]
                
                org_batch_size = int(mu.shape[0] / self.ntau)
                
                #torch.Size([32, 24, 3])
                mu_tgt = torch.cat(mu[:, -self.last_freq:, :].split(org_batch_size, 0), 2)
                b_tgt = torch.cat(b[:, -self.last_freq:, :].split(org_batch_size, 0), 2)
                t_tgt = torch.cat(t[:, -self.last_freq:, :].split(org_batch_size, 0), 2)
                y_tgt= data['y_1H'][:,-self.last_freq:, 0]  # ([32, 36, 1])->([32, 24, 1])
                
                #torch.Size([32, 24, 3])
                loc=mu_tgt
                scales=b_tgt/torch.sqrt((1-t_tgt)*t_tgt)
                asy=torch.sqrt(t_tgt/(1-t_tgt))
                
                num_components=loc.shape  #torch.Size([32, 24, 3])
                mixture_weights = torch.ones((num_components))*1/loc.shape[2]
                # # component = pyro.sample('component', dist.Categorical(mixture_weights)) ##torch.Size([32, 24])
                # categorical_dist = torch.distributions.Categorical(logits=mixture_weights )
                # component = categorical_dist.sample()
                
                # # Select the mean and scale for the chosen components
                # sel_loc = torch.gather(loc, dim=-1, index=component.unsqueeze(-1)).squeeze(-1)  # torch.Size([32, 24])
                # sel_scales = torch.gather(scales, dim=-1, index=component.unsqueeze(-1)).squeeze(-1)  # Shape:[32, 24]
                # sel_asy = torch.gather(asy, dim=-1, index=component.unsqueeze(-1)).squeeze(-1) 
                
                # with pyro.plate("data"):
                #     obs = pyro.sample('obs', dist.AsymmetricLaplace(sel_loc, sel_scales, sel_asy).to_event(2), obs=y_tgt)
                mask = ~torch.isnan(y_tgt)
                
                with pyro.plate("data"):
                    # distri=dist.MixtureSameFamily(dist.Categorical(mixture_weights),dist.AsymmetricLaplace(loc, scales, asy))
                    # obs = pyro.sample('obs', distri.to_event(2), obs=y_tgt, obs_mask=mask)
                    distri=dist.MixtureSameFamily(dist.Categorical(mixture_weights[mask]),
                                                  dist.AsymmetricLaplace(loc[mask], scales[mask], asy[mask]))
                    obs = pyro.sample('obs', distri.to_event(1), obs=y_tgt[mask])
                    
                
            if self.headname=="cmal":
                # return {'mu': m, 'b': b, 'tau': t, 'pi': p}
                pis=outputs["pi_1H"]
                mu=outputs["mu_1H"]
                b=outputs["b_1H"]
                t=outputs["tau_1H"]

                mu_tgt = mu[:, -self.last_freq:, :]#torch.cat(.split(self.batch_size, 0), 2)
                b_tgt = b[:, -self.last_freq:, :]#.split(self.batch_size, 0), 2)
                t_tgt = t[:, -self.last_freq:, :]#.split(self.batch_size, 0), 2)
                pi_tgt= pis[:, -self.last_freq:, :]#.split(self.batch_size, 0), 2)
                y_tgt= data['y_1H'][:,-self.last_freq:, 0]  # ([32, 36, 1])->([32, 24, 1])
                            
                #torch.Size([32, 24, 3])
                loc=mu_tgt
                scales=b_tgt/torch.sqrt((1-t_tgt)*t_tgt)
                asy=torch.sqrt(t_tgt/(1-t_tgt))
                
                # mixture_weights = torch.tensor(pi_tgt)
                # component = pyro.sample('component', dist.Categorical([ mixture_weights])) ##torch.Size([32, 24])
                # categorical_dist = torch.distributions.Categorical(logits=pi_tgt )
                # component = categorical_dist.sample()

                # # Select the mean and scale for the chosen components
                # sel_loc = torch.gather(loc, dim=-1, index=component.unsqueeze(-1)).squeeze(-1)  # torch.Size([32, 24])
                # sel_scales = torch.gather(scales, dim=-1, index=component.unsqueeze(-1)).squeeze(-1)  # Shape:[32, 24]
                # sel_asy = torch.gather(asy, dim=-1, index=component.unsqueeze(-1)).squeeze(-1) 
                
                # with pyro.plate("data"):
                #     obs = pyro.sample('obs', dist.AsymmetricLaplace(sel_loc, sel_scales, sel_asy).to_event(2), obs=y_tgt)
                mask = ~torch.isnan(y_tgt)
                with pyro.plate("data"):
                    distri=dist.MixtureSameFamily(dist.Categorical(pi_tgt[mask]),dist.AsymmetricLaplace(loc[mask], scales[mask], asy[mask]))
                    obs = pyro.sample('obs', distri.to_event(1), obs=y_tgt[mask])
                
                    
            if self.headname=="gmm":
                #  {'mu': mu, 'sigma': torch.exp(sigma) + self._eps, 'pi': torch.softmax(pi, dim=-1)}
                pis=outputs["pi_1H"]
                means=outputs["mu_1H"]
                sigmas=outputs["sigma_1H"]
        
                means_tgt = means[:, -self.last_freq:, :]#torch.cat(.split(self.batch_size, 0), 2)
                sigmas_tgt = sigmas[:, -self.last_freq:, :]#.split(self.batch_size, 0), 2)
                pis_tgt= pis[:, -self.last_freq:, :]#.split(self.batch_size, 0), 2)
                y_tgt= data['y_1H'][:,-self.last_freq:, 0]  # ([32, 36, 1])->([32, 24, 1])
                
                # categorical_dist = torch.distributions.Categorical(logits=pis_tgt )
                # component = categorical_dist.sample()
                mask = ~torch.isnan(y_tgt)
                with pyro.plate("data"):
                    # component = pyro.sample('component', dist.Categorical(pis_tgt))
                    # sel_means= torch.gather(means_tgt, dim=-1, index=component.unsqueeze(-1)).squeeze(-1)  # torch.Size([32, 24])
                    # sel_sigmas = torch.gather(sigmas_tgt, dim=-1, index=component.unsqueeze(-1)).squeeze(-1)  # Shape:[32, 24]                    
                    # obs = pyro.sample('obs', dist.Normal(sel_means, sel_sigmas).to_event(2), obs=y_tgt)

                    distri=dist.MixtureSameFamily(dist.Categorical(pis_tgt[mask]),dist.Normal(means_tgt[mask],sigmas_tgt[mask]))
                    obs = pyro.sample('obs', distri.to_event(1), obs=y_tgt[mask])

            if self.headname=="regression":
                # {'y_hat': self.net(x)}
                mean =outputs['y_hat_1H']  #([32, 36, 1])
                mean_tgt= mean [:,-self.last_freq:, 0] 
                y_tgt= data['y_1H'][:,-self.last_freq:, 0] 
                
                sigma = pyro.sample("sigma", dist.Uniform(0., 1.)).expand(mean_tgt.shape).to('cuda')
                mask = ~torch.isnan(y_tgt)
                with pyro.plate("data"):
                    obs = pyro.sample("obs", dist.Normal(mean_tgt[mask], sigma[mask]).to_event(1), obs=y_tgt[mask])                

        return outputs
