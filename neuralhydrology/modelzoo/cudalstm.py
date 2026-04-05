from typing import Dict

import torch
import torch.nn as nn

from neuralhydrology.modelzoo.inputlayer import InputLayer
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

class CudaLSTM(BaseModel):
    """LSTM model class, which relies on PyTorch's CUDA LSTM class.

    This class implements the standard LSTM combined with a model head, as specified in the config. Depending on the
    embedding settings, static and/or dynamic features may or may not be fed through embedding networks before being
    concatenated and passed through the LSTM.
    To control the initial forget gate bias, use the config argument `initial_forget_bias`. Often it is useful to set
    this value to a positive value at the start of the model training, to keep the forget gate closed and to facilitate
    the gradient flow.
    The `CudaLSTM` class only supports single-timescale predictions. Use `MTSLSTM` to train a model and get
    predictions on multiple temporal resolutions at the same time.

    Parameters
    ----------
    cfg : Config
        The run configuration.
    """
    # specify submodules of the model that can later be used for finetuning. Names must match class attributes
    module_parts = ['embedding_net', 'lstm', 'head']

    def __init__(self, cfg: Config):
        super(CudaLSTM, self).__init__(cfg=cfg)
        
        self.last_freq=cfg.predict_last_n
        self.embedding_net = InputLayer(cfg)
        self.bvi=cfg.bvi
        self.headname=cfg.head.lower()
        
        self.bviwithini=cfg.bviwithini
        self.inipath=cfg.inipath
        self.device=cfg.device
        self.bvisigma=cfg.bvisigma
        
        self.batch_size=cfg.batch_size
        self.lstmini=cfg.lstmini
        self.headini=cfg.headini

        if cfg.head=="umal":
            self.ntau=cfg.n_taus
        
        if cfg.bvi==0 :
            self.lstm = nn.LSTM(input_size=self.embedding_net.output_size, hidden_size=cfg.hidden_size)
        
                    
        if ((cfg.bvi==1) & (cfg.bviwithini==0)) | ((cfg.bvi==1) & (cfg.bviwithini==1) & (self.lstmini==0)) :
            self.lstm = PyroModule[nn.LSTM](input_size=self.embedding_net.output_size, hidden_size=cfg.hidden_size).to(self.device) 

            self.lstm.weight_ih_l0 = PyroSample(
                    			dist.Normal(torch.tensor(0. , device=cfg.device), torch.tensor( cfg.bvisigma, device=cfg.device))
									.expand([4 * cfg.hidden_size, self.embedding_net.output_size]).to_event(2))
            
            self.lstm.weight_hh_l0 = PyroSample(
                                dist.Normal(torch.tensor(0. , device=cfg.device), torch.tensor(cfg.bvisigma, device=cfg.device))
									. expand([4 * cfg.hidden_size, cfg.hidden_size]).to_event(2))

            self.lstm.bias_ih_l0 = PyroSample(
                                dist.Normal(torch.tensor(0. , device=cfg.device), torch.tensor( cfg.bvisigma, device=cfg.device))
										.expand([4 * cfg.hidden_size]).to_event(1)) 

            self.lstm.bias_hh_l0 = PyroSample(dist.Normal(
                                torch.tensor(0. , device=cfg.device), torch.tensor( cfg.bvisigma, device=cfg.device))
									.expand([4 * cfg.hidden_size]).to_event(1))    
           
            
        if (cfg.bvi==1) & (cfg.bviwithini==1) & (self.lstmini==1):
            
            allwei=torch.load(cfg.inipath, map_location=cfg.device)
            
            # allwei=torch.load("D:\!WISC_Res\LSTM\VariationalV\Daily\cmal\model_epoch200.pt", map_location="cpu")
            self.lstm = PyroModule[nn.LSTM](input_size=self.embedding_net.output_size, hidden_size=cfg.hidden_size).to(self.device) 


            self.lstm.weight_ih_l0 = PyroSample(
                   			dist.Normal(torch.tensor(allwei['lstm.weight_ih_l0'] , device=cfg.device), torch.tensor( cfg.bvisigma, device=cfg.device))
   									.to_event(2))
           
            self.lstm.weight_hh_l0 = PyroSample(
                               dist.Normal(torch.tensor(allwei['lstm.weight_hh_l0'] , device=cfg.device), torch.tensor(cfg.bvisigma, device=cfg.device))
   									.to_event(2))
   
            self.lstm.bias_ih_l0 = PyroSample(
                               dist.Normal(torch.tensor(allwei['lstm.bias_ih_l0'] , device=cfg.device), torch.tensor( cfg.bvisigma, device=cfg.device))
   										.to_event(1)) 
   
            self.lstm.bias_hh_l0 = PyroSample(
                               dist.Normal(torch.tensor(allwei['lstm.bias_hh_l0'], device=cfg.device), torch.tensor( cfg.bvisigma, device=cfg.device))
   									    .to_event(1))  
           
                
            
        self.dropout = nn.Dropout(p=cfg.output_dropout)

        self.head = get_head(cfg=cfg, n_in=cfg.hidden_size, n_out=self.output_size,freq='1D')

        self._reset_parameters()

    def _reset_parameters(self):
        """Special initialization of certain model weights."""
        if self.cfg.initial_forget_bias is not None:
            self.lstm.bias_hh_l0.data[self.cfg.hidden_size:2 * self.cfg.hidden_size] = self.cfg.initial_forget_bias

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Perform a forward pass on the CudaLSTM model.

        Parameters
        ----------
        data : Dict[str, torch.Tensor]
            Dictionary, containing input features as key-value pairs.

        Returns
        -------
        Dict[str, torch.Tensor]
            Model outputs and intermediate states as a dictionary.
                - `y_hat`: model predictions of shape [batch size, sequence length, number of target variables].
                - `h_n`: hidden state at the last time step of the sequence of shape [batch size, 1, hidden size].
                - `c_n`: cell state at the last time step of the sequence of shape [batch size, 1, hidden size].
        """
        # possibly pass dynamic and static inputs through embedding layers, then concatenate them
        x_d = self.embedding_net(data)
        lstm_output, (h_n, c_n) = self.lstm(input=x_d)

        # reshape to [batch_size, seq, n_hiddens]
        lstm_output = lstm_output.transpose(0, 1)
        h_n = h_n.transpose(0, 1)
        c_n = c_n.transpose(0, 1)

        pred = {'lstm_output': lstm_output, 'h_n': h_n, 'c_n': c_n}
        pred.update(self.head(self.dropout(lstm_output)))
        
        
        if self.bvi==1:
            
            if self.headname=="umal":
                # {'mu': m, 'b': b}
                mu=pred["mu"]
                b=pred["b"]
                t=data["tau"]
                
                org_batch_size = int(mu.shape[0] / self.ntau)
                
                #torch.Size([32, 24, 3])
                mu_tgt = torch.cat(mu[:, -self.last_freq:, :].split(org_batch_size, 0), 2)
                b_tgt = torch.cat(b[:, -self.last_freq:, :].split(org_batch_size, 0), 2)
                t_tgt = torch.cat(t[:, -self.last_freq:, :].split(org_batch_size, 0), 2)
                y_tgt= data['y'][:,-self.last_freq:, 0]  # ([32, 36, 1])->([32, 24, 1])
                
                #torch.Size([32, 24, 3])
                loc=mu_tgt
                scales=b_tgt/torch.sqrt((1-t_tgt)*t_tgt)
                asy=torch.sqrt(t_tgt/(1-t_tgt))

                
                num_components=loc.shape  #torch.Size([32, 24, 3])
                mixture_weights = torch.ones((num_components))*1/loc.shape[2]
                mask = ~torch.isnan(y_tgt)
                mixture_weights=mixture_weights.to(self.device)
           
                
                with pyro.plate("data"):

                    distri=dist.MixtureSameFamily(dist.Categorical(mixture_weights[mask]),
                                                  dist.AsymmetricLaplace(loc[mask], scales[mask], asy[mask]))
                    obs = pyro.sample('obs', distri.to_event(1), obs=y_tgt[mask])
                
                    
            
            if self.headname=="cmal":
                # return {'mu': m, 'b': b, 'tau': t, 'pi': p}
                pis=pred["pi"]
                mu=pred["mu"]
                b=pred["b"]
                t=pred["tau"]
                
                mu_tgt = mu[:, -self.last_freq:, :]
                b_tgt = b[:, -self.last_freq:, :]
                t_tgt = t[:, -self.last_freq:, :]
                pi_tgt= pis[:, -self.last_freq:, :]
                y_tgt= data['y'][:,-self.last_freq:, 0]  # ([32, 36, 1])->([32, 24, 1])
                            
                #torch.Size([32, 24, 3])
                loc=mu_tgt
                scales=b_tgt/torch.sqrt((1-t_tgt)*t_tgt)
                asy=torch.sqrt(t_tgt/(1-t_tgt))
                mask = ~torch.isnan(y_tgt)                

                with pyro.plate("data"):
                    distri=dist.MixtureSameFamily(dist.Categorical(pi_tgt[mask]),dist.AsymmetricLaplace(loc[mask], scales[mask], asy[mask]))
                    obs = pyro.sample('obs', distri.to_event(1), obs=y_tgt[mask])

            
            if self.headname=="gmm":
                pis=pred["pi"]
                means=pred["mu"]
                sigmas=pred["sigma"]
        
                means_tgt = means[:, -self.last_freq:, :]
                sigmas_tgt = sigmas[:, -self.last_freq:, :]
                pis_tgt= pis[:, -self.last_freq:, :]
                y_tgt= data['y'][:,-self.last_freq:, 0]  # ([32, 36, 1])->([32, 24, 1])
                mask = ~torch.isnan(y_tgt)

                with pyro.plate("data"):

                    distri=dist.MixtureSameFamily(dist.Categorical(pis_tgt[mask]),dist.Normal(means_tgt[mask],sigmas_tgt[mask]))
                    obs = pyro.sample('obs', distri.to_event(1), obs=y_tgt[mask])
                    
            if self.headname=="regression":
                
                y_tgt=data['y'][:,-self.last_freq,0]
                mask = ~torch.isnan(y_tgt)
                
                mean = pred['lstm_output'][:,-self.last_freq,0]
                sigma = pyro.sample("sigma", dist.Uniform(0., 1.)).to(self.device) 
                
                with pyro.plate("data"):
                    obs = pyro.sample("obs", dist.Normal(mean[mask], sigma).to_event(1), 
                                      obs=y_tgt[mask])
            
        return pred
