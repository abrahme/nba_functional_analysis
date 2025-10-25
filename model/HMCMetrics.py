from dataclasses import dataclass
from numpyro.infer.hmc import HMCState
from numpyro.infer import NUTS
from .inference_utils import single_log_terms



class NUTSWithMetrics(NUTS):
    def sample(self, rng_key, state, *args, **kwargs):
        state = super().sample(rng_key, state, *args, **kwargs)
        model_kwargs = kwargs.get("model_kwargs", {})
        lp, ll = single_log_terms(state.z, self.model, model_kwargs)
        return state, {"log_prior": lp, "log_likelihood": ll}



