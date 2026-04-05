import logging
from typing import Dict

import torch
import torch.nn as nn

from neuralhydrology.utils.config import Config

import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule
from pyro.nn import PyroSample

LOGGER = logging.getLogger(__name__)


def get_head(cfg: Config, n_in: int, n_out: int, freq: str) -> nn.Module:
    """Get specific head module, depending on the run configuration.

    Parameters
    ----------
    cfg : Config
        The run configuration.
    n_in : int
        Number of input features.
    n_out : int
        Number of output features.

    Returns
    -------
    nn.Module
        The model head, as specified in the run configuration.
    """
    if cfg.head.lower() == "regression":
        head = Regression(n_in=n_in, n_out=n_out, activation=cfg.output_activation, bvi=cfg.bvi, freq=freq, 
                          inipath=cfg.inipath, bviwithini=cfg.bviwithini,devi=cfg.device,bvisigma=cfg.bvisigma,
                          modeltype=cfg.model,inityp=cfg.headini,headfix=cfg.headfix)
        
    elif cfg.head.lower() == "gmm":
        head = GMM(n_in=n_in, n_out=n_out, bvi=cfg.bvi, freq=freq, 
                   inipath=cfg.inipath, bviwithini=cfg.bviwithini, devi=cfg.device,bvisigma=cfg.bvisigma,
                   modeltype=cfg.model,inityp=cfg.headini,headfix=cfg.headfix)
        
    elif cfg.head.lower() == "umal":
        head = UMAL(n_in=n_in, n_out=n_out, bvi=cfg.bvi, freq=freq, inipath=cfg.inipath, 
                    bviwithini=cfg.bviwithini, devi=cfg.device,bvisigma=cfg.bvisigma,
                    modeltype=cfg.model,inityp=cfg.headini,headfix=cfg.headfix)
        
    elif cfg.head.lower() == "cmal":
        head = CMAL(n_in=n_in, n_out=n_out, bvi=cfg.bvi, freq=freq, inipath=cfg.inipath, 
                    bviwithini=cfg.bviwithini, devi=cfg.device,bvisigma=cfg.bvisigma,
                    modeltype=cfg.model,inityp=cfg.headini,headfix=cfg.headfix)
        
    elif cfg.head.lower() == "":
        raise ValueError(f"No 'head' specified in the config but is required for {cfg.model}")
    else:
        raise NotImplementedError(f"{cfg.head} not implemented or not linked in `get_head()`")
        
    return head


class Regression(nn.Module):
    """Single-layer regression head with different output activations.
    
    Parameters
    ----------
    n_in : int
        Number of input neurons.
    n_out : int
        Number of output neurons.
    activation : str, optional
        Output activation function. Can be specified in the config using the `output_activation` argument. Supported
        are {'linear', 'relu', 'softplus'}. If not specified (or an unsupported activation function is specified), will
        default to 'linear' activation.
    """

    def __init__(self, n_in: int, n_out: int, activation: str = "linear", bvi: bool = 0, freq: str="1H",
                 inipath: str='', bviwithini: bool=0, devi: str="cpu", bvisigma: float=0.1,modeltype: str='None',
                 inityp: bool=True, headfix: bool=False):
        
        super(Regression, self).__init__()
        
        if bvi ==0:
            # TODO: Add multi-layer support
            layers = [nn.Linear(n_in, n_out)]
            if activation != "linear":
                if activation.lower() == "relu":
                    layers.append(nn.ReLU())
                elif activation.lower() == "softplus":
                    layers.append(nn.Softplus())
                else:
                    LOGGER.warning(f"## WARNING: Ignored output activation {activation} and used 'linear' instead.")
            self.net = nn.Sequential(*layers)
            self.bvi=0

        if ((bvi==1) & (bviwithini==0)) | ((bvi==1) & (bviwithini==1) & (inityp==0)): 
            
            # TODO: Add multi-layer support
            layer0=PyroModule[nn.Linear](n_in, n_out).to(devi)
            layer0.bias = PyroSample(dist.Normal(torch.tensor(0. , device=devi), torch.tensor(bvisigma, device=devi))
													.expand([ n_out]).to_event(1)) 
            layer0.weight = PyroSample(dist.Normal(torch.tensor(0. , device=devi), torch.tensor(bvisigma, device=devi))
													.expand([ n_out, n_in]).to_event(2)) 

            layers={}
            layers[f"l0_{freq}"]= layer0
            if activation != "linear":
                if activation.lower() == "relu":
                    layers["l1"]=nn.ReLU()
                elif activation.lower() == "softplus":
                    layers["l2"]=nn.Softplus()
                else:
                    LOGGER.warning(f"## WARNING: Ignored output activation {activation} and used 'linear' instead.")
            self.net = PyroModule[nn.ModuleDict](layers)
            self.bvi=1

        if (bvi==1) & (bviwithini==1) & (inityp==1): 
            
            allwei=torch.load(inipath, map_location=devi)
            
            # TODO: Add multi-layer support
            
            if headfix==1:
                
                layer0=nn.Linear(n_in, n_out)
                
                if modeltype=="mtslstm":
                    layer0.bias.data  = allwei[f'heads.{freq}.net.0.bias']
                    
                    layer0.weight.data  = allwei[f'heads.{freq}.net.0.weight']
    
                else:
                    
                    layer0.bias.data  = allwei[f'head.net.0.bias'] 
                    layer0.weight.data  = allwei[f'head.net.0.weight']
            else:

                layer0=PyroModule[nn.Linear](n_in, n_out)
                
                if modeltype=="mtslstm":
                    layer0.bias = PyroSample(dist.Normal(allwei[f'heads.{freq}.net.0.bias'], bvisigma)
                                             .to_event(1)) 
                    layer0.weight = PyroSample(dist.Normal(allwei[f'heads.{freq}.net.0.weight'], bvisigma)
                                               .to_event(2)) 
    
                else:
                    
                    layer0.bias = PyroSample(dist.Normal(allwei[f'head.net.0.bias'], bvisigma)
                                             .to_event(1)) 
                    layer0.weight = PyroSample(dist.Normal(allwei[f'head.net.0.weight'], bvisigma)
                                               .to_event(2)) 
                    
        
            layers={}
            layers[f"l0_{freq}"]= layer0
            if activation != "linear":
                if activation.lower() == "relu":
                    layers["l1"]=nn.ReLU()
                elif activation.lower() == "softplus":
                    layers["l2"]=nn.Softplus()
                else:
                    LOGGER.warning(f"## WARNING: Ignored output activation {activation} and used 'linear' instead.")
            
            if headfix==1:
                self.net = nn.ModuleDict(layers)
            else:
                self.net = PyroModule[nn.ModuleDict](layers) 
                
            self.bvi=1

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Perform a forward pass on the Regression head.
        
        Parameters
        ----------
        x : torch.Tensor

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing the model predictions in the 'y_hat' key.
        """
        # return {'y_hat': self.net(x)}
        
        for layer in self.net.values():
            x = layer(x)  # Pass through each layer
        return {'y_hat': x}


class GMM(nn.Module):
    """Gaussian Mixture Density Network

    A mixture density network with Gaussian distribution as components. Good references are [#]_ and [#]_. The latter 
    one forms the basis for our implementation. As such, we also use two layers in the head to provide it with 
    additional flexibility, and exponential activation for the variance estimates and a softmax for weights.  

    Parameters
    ----------
    n_in : int
        Number of input neurons.
    n_out : int
        Number of output neurons. Corresponds to 3 times the number of components.
    n_hidden : int
        Size of the hidden layer.
    
    References
    ----------
    .. [#] C. M. Bishop: Mixture density networks. 1994.
    .. [#] D. Ha: Mixture density networks with tensorflow. blog.otoro.net, 
           URL: http://blog.otoro.net/2015/11/24/mixture-density-networks-with-tensorflow, 2015.
    """

    def __init__(self, n_in: int, n_out: int, n_hidden: int = 100, bvi: bool = 0, freq: str="1H",
                 inipath: str='', bviwithini: bool=0, devi: str="cpu", bvisigma: float=0.1,modeltype: str='None'
                 ,inityp: bool=True, headfix: bool=False):
        super(GMM, self).__init__()
        
        if bvi==0:
            self.fc1 = nn.Linear(n_in, n_hidden)
            self.fc2 = nn.Linear(n_hidden, n_out)
            self._eps = 1e-5
            self.bvi=0
            
        if ((bvi==1) & (bviwithini==0)) | ((bvi==1) & (bviwithini==1) & (inityp==0)): 
            
            self.fcs=PyroModule[nn.ModuleDict]().to(devi)
            
            self.fcs[f"fc1_{freq}"] =PyroModule[nn.Linear](n_in, n_hidden)
            self.fcs[f"fc1_{freq}"].bias = PyroSample(dist.Normal(torch.tensor(0. , device=devi), torch.tensor(bvisigma, device=devi))
										.expand([ n_hidden]).to_event(1)) 
            self.fcs[f"fc1_{freq}"].weight = PyroSample(dist.Normal(torch.tensor(0. , device=devi), torch.tensor(bvisigma, device=devi))
										.expand([n_hidden, n_in]).to_event(2))     
            
            self.fcs[f"fc2_{freq}"] = PyroModule[nn.Linear](n_hidden, n_out).to(devi)
            self.fcs[f"fc2_{freq}"].bias = PyroSample(dist.Normal(torch.tensor(0. , device=devi), torch.tensor(bvisigma, device=devi))
											.expand([n_out]).to_event(1)) 
            self.fcs[f"fc2_{freq}"].weight = PyroSample(dist.Normal(torch.tensor(0. , device=devi), torch.tensor(bvisigma, device=devi))
											.expand([n_out, n_hidden]).to_event(2)) 

            self._eps = 1e-5            
            self.bvi=1
            
        if (bvi==1) & (bviwithini==1) & (inityp==1): 
            
            allwei=torch.load(inipath, map_location=devi)
            
            if headfix==1:
                self.fcs=nn.ModuleDict()
                
                if modeltype=="mtslstm":
                    self.fcs[f"fc1_{freq}"] =nn.Linear(n_in, n_hidden)
                    
                    self.fcs[f"fc1_{freq}"].bias.data  = allwei[f'heads.{freq}.fc1.bias']
                    
                    self.fcs[f"fc1_{freq}"].weight.data  = allwei[f'heads.{freq}.fc1.weight']   
                    
                    self.fcs[f"fc2_{freq}"] = nn.Linear(n_hidden, n_out)
                    
                    self.fcs[f"fc2_{freq}"].bias.data  = allwei[f'heads.{freq}.fc2.bias']
                                                                          
                    self.fcs[f"fc2_{freq}"].weight.data  = allwei[f'heads.{freq}.fc2.weight']
                                                                            
                else:
                    self.fcs[f"fc1_{freq}"] = nn.Linear(n_in, n_hidden)
                    
                    self.fcs[f"fc1_{freq}"].bias.data  = allwei[f'head.fc1.bias']
                                                                          
                    self.fcs[f"fc1_{freq}"].weight.data  = allwei[f'head.fc1.weight'] 
                    
                    self.fcs[f"fc2_{freq}"] = nn.Linear(n_hidden, n_out)
                    
                    self.fcs[f"fc2_{freq}"].bias.data  = allwei[f'head.fc2.bias']
                                                                          
                    self.fcs[f"fc2_{freq}"].weight.data  = allwei[f'head.fc2.weight']
            else:

                self.fcs=PyroModule[nn.ModuleDict]()
                
                if modeltype=="mtslstm":
                    self.fcs[f"fc1_{freq}"] =PyroModule[nn.Linear](n_in, n_hidden)
                    self.fcs[f"fc1_{freq}"].bias = PyroSample(dist.Normal(allwei[f'heads.{freq}.fc1.bias'], bvisigma)
                                                             .to_event(1)) 
                    self.fcs[f"fc1_{freq}"].weight = PyroSample(dist.Normal(allwei[f'heads.{freq}.fc1.weight'], bvisigma)
                                                            .to_event(2))   
                    
                    self.fcs[f"fc2_{freq}"] = PyroModule[nn.Linear](n_hidden, n_out)
                    self.fcs[f"fc2_{freq}"].bias = PyroSample(dist.Normal(allwei[f'heads.{freq}.fc2.bias'], bvisigma)
                                                              .to_event(1)) 
                    self.fcs[f"fc2_{freq}"].weight = PyroSample(dist.Normal(allwei[f'heads.{freq}.fc2.weight'], bvisigma)
                                                            .to_event(2))
                else:
                    self.fcs[f"fc1_{freq}"] =PyroModule[nn.Linear](n_in, n_hidden)
                    self.fcs[f"fc1_{freq}"].bias = PyroSample(dist.Normal(allwei[f'head.fc1.bias'], bvisigma)
                                                             .to_event(1)) 
                    self.fcs[f"fc1_{freq}"].weight = PyroSample(dist.Normal(allwei[f'head.fc1.weight'], bvisigma)
                                                            .to_event(2))   
                    
                    self.fcs[f"fc2_{freq}"] = PyroModule[nn.Linear](n_hidden, n_out)
                    self.fcs[f"fc2_{freq}"].bias = PyroSample(dist.Normal(allwei[f'head.fc2.bias'], bvisigma)
                                                              .to_event(1)) 
                    self.fcs[f"fc2_{freq}"].weight = PyroSample(dist.Normal(allwei[f'head.fc2.weight'], bvisigma)
                                                            .to_event(2))
                    

            self._eps = 1e-5            
            self.bvi=1
            
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Perform a GMM head forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Output of the previous model part. It provides the basic latent variables to compute the GMM components.

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing mixture parameters and weights; where the key 'mu' stores the means, the key
            'sigma' the variances, and the key 'pi' the weights.
        """

        if self.bvi==0:
            h = torch.relu(self.fc1(x))
            h = self.fc2(h)
        else:
            # do this because not sure freq
            keyfc1 = [key for key in self.fcs if key.startswith("fc1")][0]
            keysfc2 = [key for key in self.fcs if key.startswith("fc2")][0]
            h = torch.relu(self.fcs[keyfc1](x))
            h = self.fcs[keysfc2](h)         
            

        # split output into mu, sigma and weights
        mu, sigma, pi = h.chunk(3, dim=-1)

        return {'mu': mu, 'sigma': torch.exp(sigma) + self._eps, 'pi': torch.softmax(pi, dim=-1)}


class CMAL(nn.Module):
    """Countable Mixture of Asymmetric Laplacians.

    An mixture density network with Laplace distributions as components.

    The CMAL-head uses an additional hidden layer to give it more expressiveness (same as the GMM-head).
    CMAL is better suited for many hydrological settings as it handles asymmetries with more ease. However, it is also
    more brittle than GMM and can more often throw exceptions. Details for CMAL can be found in [#]_.

    Parameters
    ----------
    n_in : int
        Number of input neurons.
    n_out : int
        Number of output neurons. Corresponds to 4 times the number of components.
    n_hidden : int
        Size of the hidden layer.
        
    References
    ----------
    .. [#] D.Klotz, F. Kratzert, M. Gauch, A. K. Sampson, G. Klambauer, S. Hochreiter, and G. Nearing: 
        Uncertainty Estimation with Deep Learning for Rainfall-Runoff Modelling. arXiv preprint arXiv:2012.14295, 2020.
    """

    def __init__(self, n_in: int, n_out: int, n_hidden: int = 100, bvi: bool = 0, freq: str="1H",
                 inipath: str='', bviwithini: bool=0, devi: str="cpu", bvisigma: float=0.1,modeltype: str='None',
                 inityp: bool=True, headfix: bool=False):
        super(CMAL, self).__init__()
        
        if bvi==0:
            self.fc1 = nn.Linear(n_in, n_hidden)
            self.fc2 = nn.Linear(n_hidden, n_out)
    
            self._softplus = torch.nn.Softplus(2)
            self._eps = 1e-5
            self.bvi=0
            
        if ((bvi==1) & (bviwithini==0)) | ((bvi==1) & (bviwithini==1) & (inityp==0)): 

            self.fcs=PyroModule[nn.ModuleDict]().to(devi)
            
            self.fcs[f"fc1_{freq}"] =PyroModule[nn.Linear](n_in, n_hidden).to(devi)
            self.fcs[f"fc1_{freq}"].bias = PyroSample(dist.Normal(torch.tensor(0. , device=devi), torch.tensor(bvisigma, device=devi))
																					.expand([ n_hidden]).to_event(1)) 
            self.fcs[f"fc1_{freq}"].weight = PyroSample(dist.Normal(torch.tensor(0. , device=devi), torch.tensor(bvisigma, device=devi))
																					.expand([n_hidden, n_in]).to_event(2))     
            
            self.fcs[f"fc2_{freq}"] = PyroModule[nn.Linear](n_hidden, n_out).to(devi)
            self.fcs[f"fc2_{freq}"].bias = PyroSample(dist.Normal(torch.tensor(0. , device=devi), torch.tensor(bvisigma, device=devi))
																						.expand([n_out]).to_event(1)) 
            self.fcs[f"fc2_{freq}"].weight = PyroSample(dist.Normal(torch.tensor(0. , device=devi), torch.tensor(bvisigma, device=devi))
																						.expand([n_out, n_hidden]).to_event(2))

            self.bvi=1
               
            self._softplus = torch.nn.Softplus(2)
            self._eps = 1e-5          
        
        
        if (bvi==1) & (bviwithini==1) & (inityp==1): 
            
            
            allwei=torch.load(inipath, map_location=devi)
            
            if headfix==1:
                
                self.fcs=nn.ModuleDict()
                
                if modeltype=="mtslstm":
                        
                    self.fcs[f"fc1_{freq}"] =PyroModule[nn.Linear](n_in, n_hidden).to(devi)
                    self.fcs[f"fc1_{freq}"].bias.data  = allwei[f'heads.{freq}.fc1.bias']
                    
                    self.fcs[f"fc1_{freq}"].weight.data  = allwei[f'heads.{freq}.fc1.weight']
                    
                    self.fcs[f"fc2_{freq}"] = nn.Linear(n_hidden, n_out)
                    
                    self.fcs[f"fc2_{freq}"].bias.data  = allwei[f'heads.{freq}.fc2.bias']
                                                                          
                    self.fcs[f"fc2_{freq}"].weight.data  = allwei[f'heads.{freq}.fc2.weight']
                                                                            
                else:
                    self.fcs[f"fc1_{freq}"] =nn.Linear(n_in, n_hidden)
                    
                    self.fcs[f"fc1_{freq}"].bias.data  = allwei[f'head.fc1.bias']
                                                                          
                    self.fcs[f"fc1_{freq}"].weight.data  = allwei[f'head.fc1.weight']
                    
                    self.fcs[f"fc2_{freq}"] = nn.Linear(n_hidden, n_out)
                    
                    self.fcs[f"fc2_{freq}"].bias.data  = allwei[f'head.fc2.bias']
                                                                          
                    self.fcs[f"fc2_{freq}"].weight.data  = allwei[f'head.fc2.weight']
                                                                            
            else:

                self.fcs=PyroModule[nn.ModuleDict]()
                
                if modeltype=="mtslstm":
                        
                    self.fcs[f"fc1_{freq}"] =PyroModule[nn.Linear](n_in, n_hidden).to(devi)
                    self.fcs[f"fc1_{freq}"].bias = PyroSample(dist.Normal(allwei[f'heads.{freq}.fc1.bias'], bvisigma)
                                                             .to_event(1)) 
                    self.fcs[f"fc1_{freq}"].weight = PyroSample(dist.Normal(allwei[f'heads.{freq}.fc1.weight'],bvisigma)
                                                            .to_event(2))   
                    
                    self.fcs[f"fc2_{freq}"] = PyroModule[nn.Linear](n_hidden, n_out)
                    self.fcs[f"fc2_{freq}"].bias = PyroSample(dist.Normal(allwei[f'heads.{freq}.fc2.bias'], bvisigma)
                                                              .to_event(1)) 
                    self.fcs[f"fc2_{freq}"].weight = PyroSample(dist.Normal(allwei[f'heads.{freq}.fc2.weight'], bvisigma)
                                                            .to_event(2))
                else:
                    self.fcs[f"fc1_{freq}"] =PyroModule[nn.Linear](n_in, n_hidden)
                    self.fcs[f"fc1_{freq}"].bias = PyroSample(dist.Normal(allwei[f'head.fc1.bias'], bvisigma)
                                                             .to_event(1)) 
                    self.fcs[f"fc1_{freq}"].weight = PyroSample(dist.Normal(allwei[f'head.fc1.weight'],bvisigma)
                                                            .to_event(2))   
                    
                    self.fcs[f"fc2_{freq}"] = PyroModule[nn.Linear](n_hidden, n_out)
                    self.fcs[f"fc2_{freq}"].bias = PyroSample(dist.Normal(allwei[f'head.fc2.bias'], bvisigma)
                                                              .to_event(1)) 
                    self.fcs[f"fc2_{freq}"].weight = PyroSample(dist.Normal(allwei[f'head.fc2.weight'], bvisigma)
                                                        .to_event(2))
                    
            self._softplus = torch.nn.Softplus(2)
            self._eps = 1e-5            
            self.bvi=1
            
            
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Perform a CMAL head forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Output of the previous model part. It provides the basic latent variables to compute the CMAL components.

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary, containing the mixture component parameters and weights; where the key 'mu'stores the means,
            the key 'b' the scale parameters, the key 'tau' the skewness parameters, and the key 'pi' the weights).
        """
        
        if self.bvi==0:
            h = torch.relu(self.fc1(x))
            h = self.fc2(h)
        else:
            # do this because not sure freq
            keyfc1 = [key for key in self.fcs if key.startswith("fc1")][0]
            keysfc2 = [key for key in self.fcs if key.startswith("fc2")][0]
            h = torch.relu(self.fcs[keyfc1](x))
            h = self.fcs[keysfc2](h)         
            
            
        m_latent, b_latent, t_latent, p_latent = h.chunk(4, dim=-1)

        # enforce properties on component parameters and weights:
        m = m_latent  # no restrictions (depending on setting m>0 might be useful)
        b = self._softplus(b_latent) + self._eps  # scale > 0 (softplus was working good in tests)
        t = (1 - self._eps) * torch.sigmoid(t_latent) + self._eps  # 0 > tau > 1
        p = (1 - self._eps) * torch.softmax(p_latent, dim=-1) + self._eps  # sum(pi) = 1 & pi > 0
        t[t>=0.9999]=0.9999
        return {'mu': m, 'b': b, 'tau': t, 'pi': p}


class UMAL(nn.Module):
    """Uncountable Mixture of Asymmetric Laplacians.

    An implicit approximation to the mixture density network with Laplace distributions which does not require to
    pre-specify the number of components. An additional hidden layer is used to provide the head more expressiveness.
    General details about UMAL can be found in [#]_. A major difference between their implementation 
    and ours is the binding-function for the scale-parameter (b). The scale needs to be lower-bound. The original UMAL 
    implementation uses an elu-based binding. In our experiment however, this produced under-confident predictions
    (too large variances). We therefore opted for a tailor-made binding-function that limits the scale from below and 
    above using a sigmoid. It is very likely that this needs to be adapted for non-normalized outputs.   

    Parameters
    ----------
    n_in : int
        Number of input neurons.
    n_out : int
        Number of output neurons. Corresponds to 2 times the output-size, since the scale parameters are also predicted.
    n_hidden : int
        Size of the hidden layer.

    References
    ----------
    .. [#] A. Brando, J. A. Rodriguez, J. Vitria, and A. R. Munoz: Modelling heterogeneous distributions 
        with an Uncountable Mixture of Asymmetric Laplacians. Advances in Neural Information Processing Systems, 
        pp. 8838-8848, 2019.
    """

    def __init__(self, n_in: int, n_out: int, n_hidden: int = 100, bvi: bool = 0, freq: str="1H",
                 inipath: str='', bviwithini: bool=0, devi: str="cpu", bvisigma: float=0.1,modeltype: str='None',
                 inityp: bool=True, headfix: bool=False):
        super(UMAL, self).__init__()
        
        if bvi==0:
            self.fc1 = nn.Linear(n_in, n_hidden)
            self.fc2 = nn.Linear(n_hidden, n_out)
            self._upper_bound_scale = 0.5  # this parameter found empirical by testing UMAL for a limited set of basins
            self._eps = 1e-5
            self.bvi=0
            
        if ((bvi==1) & (bviwithini==0)) | ((bvi==1) & (bviwithini==1) & (inityp==0)): 

            self.fcs=PyroModule[nn.ModuleDict]()
            
            self.fcs[f"fc1_{freq}"] =PyroModule[nn.Linear](n_in, n_hidden).to(devi)
            self.fcs[f"fc1_{freq}"].bias = PyroSample(dist.Normal(torch.tensor(0. , device=devi), torch.tensor(bvisigma, device=devi))
												.expand([ n_hidden]).to_event(1)) 
            self.fcs[f"fc1_{freq}"].weight = PyroSample(dist.Normal(torch.tensor(0. , device=devi), torch.tensor(bvisigma, device=devi))
																						.expand([n_hidden, n_in]).to_event(2))     
            
            self.fcs[f"fc2_{freq}"] = PyroModule[nn.Linear](n_hidden, n_out).to(devi)
            self.fcs[f"fc2_{freq}"].bias = PyroSample(dist.Normal(torch.tensor(0. , device=devi), torch.tensor(bvisigma, device=devi))
																						.expand([n_out]).to_event(1)) 
            self.fcs[f"fc2_{freq}"].weight = PyroSample(dist.Normal(torch.tensor(0. , device=devi), torch.tensor(bvisigma, device=devi))
																						.expand([n_out, n_hidden]).to_event(2)) 

            self._upper_bound_scale = 0.5  # this parameter found empirical by testing UMAL for a limited set of basins
            self._eps = 1e-5  
            self.bvi=1

        if (bvi==1) & (bviwithini==1) & (inityp==1): 
            
            allwei=torch.load(inipath, map_location=devi)
            
            if headfix==1:
                
                self.fcs=nn.ModuleDict()
                
                if modeltype=="mtslstm":
                    self.fcs[f"fc1_{freq}"] =nn.Linear(n_in, n_hidden)
                    
                    self.fcs[f"fc1_{freq}"].bias.data  = allwei[f'heads.{freq}.fc1.bias']
                                                                          
                    self.fcs[f"fc1_{freq}"].weight.data  = allwei[f'heads.{freq}.fc1.weight']   
                    
                    self.fcs[f"fc2_{freq}"] = PyroModule[nn.Linear](n_hidden, n_out)
                    
                    self.fcs[f"fc2_{freq}"].bias.data  = allwei[f'heads.{freq}.fc2.bias']
                                                                          
                    self.fcs[f"fc2_{freq}"].weight.data  = allwei[f'heads.{freq}.fc2.weight']
                else:
                    self.fcs[f"fc1_{freq}"] =nn.Linear(n_in, n_hidden)
                    
                    self.fcs[f"fc1_{freq}"].bias.data  = allwei[f'head.fc1.bias']
                                                                          
                    self.fcs[f"fc1_{freq}"].weight.data  = allwei[f'head.fc1.weight']                                                                           
                    
                    self.fcs[f"fc2_{freq}"] = nn.Linear(n_hidden, n_out)
                    
                    self.fcs[f"fc2_{freq}"].bias.data  =allwei[f'head.fc2.bias']
                                                                          
                    self.fcs[f"fc2_{freq}"].weight.data  = allwei[f'head.fc2.weight']
            else:
                
                self.fcs=PyroModule[nn.ModuleDict]()

                if modeltype=="mtslstm":
                    self.fcs[f"fc1_{freq}"] =PyroModule[nn.Linear](n_in, n_hidden)
                    self.fcs[f"fc1_{freq}"].bias = PyroSample(dist.Normal(allwei[f'heads.{freq}.fc1.bias'], bvisigma)
                                                             .to_event(1)) 
                    self.fcs[f"fc1_{freq}"].weight = PyroSample(dist.Normal(allwei[f'heads.{freq}.fc1.weight'], bvisigma)
                                                            .to_event(2))   
                    
                    self.fcs[f"fc2_{freq}"] = PyroModule[nn.Linear](n_hidden, n_out)
                    self.fcs[f"fc2_{freq}"].bias = PyroSample(dist.Normal(allwei[f'heads.{freq}.fc2.bias'], bvisigma)
                                                              .to_event(1)) 
                    self.fcs[f"fc2_{freq}"].weight = PyroSample(dist.Normal(allwei[f'heads.{freq}.fc2.weight'], bvisigma)
                                                            .to_event(2))
                else:
                    self.fcs[f"fc1_{freq}"] =PyroModule[nn.Linear](n_in, n_hidden)
                    self.fcs[f"fc1_{freq}"].bias = PyroSample(dist.Normal(allwei[f'head.fc1.bias'], bvisigma)
                                                             .to_event(1)) 
                    self.fcs[f"fc1_{freq}"].weight = PyroSample(dist.Normal(allwei[f'head.fc1.weight'], bvisigma)
                                                            .to_event(2))   
                    
                    self.fcs[f"fc2_{freq}"] = PyroModule[nn.Linear](n_hidden, n_out)
                    self.fcs[f"fc2_{freq}"].bias = PyroSample(dist.Normal(allwei[f'head.fc2.bias'], bvisigma)
                                                              .to_event(1)) 
                    self.fcs[f"fc2_{freq}"].weight = PyroSample(dist.Normal(allwei[f'head.fc2.weight'], bvisigma)
                                                            .to_event(2))
                
                    
                    
            self._upper_bound_scale = 0.5 
            self._eps = 1e-5            
            self.bvi=1
            

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Perform a UMAL head forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Output of the previous model part. It provides the basic latent variables to compute the UMAL components.

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing the means ('mu') and scale parameters ('b') to parametrize the asymmetric Laplacians.
        """
        
        if self.bvi==0:
            h = torch.relu(self.fc1(x))
            h = self.fc2(h)
        else:
            # do this because not sure freq
            keyfc1 = [key for key in self.fcs if key.startswith("fc1")][0]
            keysfc2 = [key for key in self.fcs if key.startswith("fc2")][0]
            h = torch.relu(self.fcs[keyfc1](x))
            h = self.fcs[keysfc2](h)            
            
            

        m_latent, b_latent = h.chunk(2, dim=-1)

        # enforce properties on component parameters and weights:
        m = m_latent  # no restrictions (depending on setting m>0 might be useful)
        b = self._upper_bound_scale * torch.sigmoid(b_latent) + self._eps  # bind scale from two sides.
        return {'mu': m, 'b': b}
