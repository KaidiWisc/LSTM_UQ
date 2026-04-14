import functools
from contextlib import ExitStack  # python 3
import torch
from torch.distributions import biject_to, constraints
import pyro
import pyro.distributions as dist
from pyro.distributions.util import sum_rightmost
from pyro.infer.autoguide.initialization import InitMessenger, init_to_feasible
from pyro.nn import PyroModule, PyroParam
from pyro.ops.tensor_utils import periodic_repeat
from pyro.infer.autoguide import AutoGuide
#code adapted from https://pyro.ai/examples/

def _deep_setattr(obj, key, val):

    def _getattr(obj, attr):
        obj_next = getattr(obj, attr, None)
        if obj_next is not None:
            return obj_next
        setattr(obj, attr, PyroModule())
        return getattr(obj, attr)

    lpart, _, rpart = key.rpartition(".")

    if lpart:
        obj = functools.reduce(_getattr, [obj] + lpart.split('.'))
    setattr(obj, rpart, val)

def _deep_getattr(obj, key):
    for part in key.split("."):
        obj = getattr(obj, part)
    return obj


class var_dist_define(AutoGuide):

    def __init__(self, model, *,
                 init_loc_fn=init_to_feasible,
                 init_scale=0.1,
                 create_plates=None,
                 cfg=None):
        
        self.cfg=cfg
        self.init_loc_fn = init_loc_fn

        if not isinstance(init_scale, float) or not (init_scale > 0):
            raise ValueError("Expected init_scale > 0. but got {}".format(init_scale))
        self._init_scale = init_scale

        model = InitMessenger(self.init_loc_fn)(model)
        super().__init__(model, create_plates=create_plates)

    def _setup_prototype(self, *args, **kwargs):
        super()._setup_prototype(*args, **kwargs)

        self._event_dims = {}
        self._cond_indep_stacks = {}
        self.locs = PyroModule()
        self.scales = PyroModule()


        for name, site in self.prototype_trace.iter_stochastic_nodes():
            # Collect unconstrained event_dims, which may differ from constrained event_dims.
            init_loc = biject_to(site["fn"].support).inv(site["value"].detach()).detach()
            event_dim = site["fn"].event_dim + init_loc.dim() - site["value"].dim()
            self._event_dims[name] = event_dim

            # Collect independence contexts.
            self._cond_indep_stacks[name] = site["cond_indep_stack"]

            for frame in site["cond_indep_stack"]:
                full_size = getattr(frame, "full_size", frame.size)
                if full_size != frame.size:
                    dim = frame.dim - event_dim
                    init_loc = periodic_repeat(init_loc, full_size, dim).contiguous()
            init_scale = torch.full_like(init_loc, self._init_scale)

            if self.cfg.guideini==1:
                allwei=torch.load(self.cfg.guideinipath, map_location=self.cfg.device)
                
                weinmdc = [key for key in allwei if name in key]
                if len(weinmdc)==1:
                    weinm=weinmdc[0]
                    init_loc=allwei[weinm]
                
                if (len(weinmdc)==0) & (name!="sigma") & (name!="sigmaD")  & (name!="sigmaH"):
                    # name:weinm
                    lut={"l0_1D.weight":'heads.1D.net.0.weight',"l0_1D.bias":'heads.1D.net.0.bias',
 "l0_1H.weight":'heads.1H.net.0.weight',"l0_1H.bias":'heads.1H.net.0.bias',"fc1_1D.weight":'heads.1D.fc1.weight',
 "fc1_1D.bias":'heads.1D.fc1.bias',"fc2_1D.weight":'heads.1D.fc2.weight',"fc2_1D.bias":'heads.1D.fc2.bias',
 "fc1_1H.weight":'heads.1H.fc1.weight',"fc1_1H.bias": 'heads.1H.fc1.bias',"fc2_1H.weight":'heads.1H.fc2.weight',
 "fc2_1H.bias":'heads.1H.fc2.bias'}
                    weinm=lut[name]
                    init_loc=allwei[weinm]

    
            _deep_setattr(self.locs, name, PyroParam(init_loc, constraints.real, event_dim))  #!!!
            _deep_setattr(self.scales, name, PyroParam(init_scale, constraints.positive, event_dim))

    def _get_loc_and_scale(self, name):
        site_loc = _deep_getattr(self.locs, name)
        site_scale = _deep_getattr(self.scales, name)
        return site_loc, site_scale

    def forward(self, *args, **kwargs):

        if self.prototype_trace is None:
            self._setup_prototype(*args, **kwargs)

        plates = self._create_plates(*args, **kwargs)
        result = {}
        for name, site in self.prototype_trace.iter_stochastic_nodes():
            transform = biject_to(site["fn"].support)

            with ExitStack() as stack:
                for frame in site["cond_indep_stack"]:
                    if frame.vectorized:
                        stack.enter_context(plates[frame.name])

                site_loc, site_scale = self._get_loc_and_scale(name)
                unconstrained_latent = pyro.sample(
                    name + "_unconstrained",
                    dist.Normal(
                        site_loc, site_scale,
                    ).to_event(self._event_dims[name]),
                    infer={"is_auxiliary": True}
                )

                value = transform(unconstrained_latent)
                log_density = transform.inv.log_abs_det_jacobian(value, unconstrained_latent)
                log_density = sum_rightmost(log_density, log_density.dim() - value.dim() + site["fn"].event_dim)
                delta_dist = dist.Delta(value, log_density=log_density,
                                        event_dim=site["fn"].event_dim)

                result[name] = pyro.sample(name, delta_dist)

        return result


    def median(self, *args, **kwargs):

        medians = {}
        for name, site in self.prototype_trace.iter_stochastic_nodes():
            site_loc, _ = self._get_loc_and_scale(name)
            median = biject_to(site["fn"].support)(site_loc)
            if median is site_loc:
                median = median.clone()
            medians[name] = median

        return medians

    def quantiles(self, quantiles, *args, **kwargs):

        results = {}

        for name, site in self.prototype_trace.iter_stochastic_nodes():
            site_loc, site_scale = self._get_loc_and_scale(name)

            site_quantiles = torch.tensor(quantiles, dtype=site_loc.dtype, device=site_loc.device)
            site_quantiles_values = dist.Normal(site_loc, site_scale).icdf(site_quantiles)
            constrained_site_quantiles = biject_to(site["fn"].support)(site_quantiles_values)
            results[name] = constrained_site_quantiles

        return results