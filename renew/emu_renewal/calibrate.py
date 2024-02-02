import numpy as np
import pymc as pm
import pytensor.tensor as pt

def get_wrapped_ll(priors, llfunc):
    prior_types = []
    for prior in priors:
        if prior.size > 1:
            prior_types.append(pt.dvector)
        else:
            prior_types.append(pt.dscalar)

    # define a pytensor Op for our likelihood function
    class CustomLogLike(pt.Op):
        """
        Specify what type of object will be passed and returned to the Op when it is
        called. In our case we will be passing it a vector of values (the parameters
        that define our model) and returning a single "scalar" value (the
        log-likelihood)
        """

        itypes = prior_types
        otypes = [pt.dscalar]

        def __init__(self):
            self.llfunc = llfunc

        def perform(self, node, inputs, outputs):
            params = inputs
            logl = self.llfunc(*params)
            outputs[0][0] = np.array(logl)  # output the log-likelihood

    return CustomLogLike


def use_model(priors, llfunc) -> list:
    logl = get_wrapped_ll(priors, llfunc)()

    pymc_priors = [p.to_pymc() for p in priors]
    invars = [pt.as_tensor_variable(v) for v in pymc_priors]

    # use a Potential to "call" the Op and include it in the logp computation
    ll = logl(*invars)
    pm.Potential('loglikelihood', ll)

    return pymc_priors
