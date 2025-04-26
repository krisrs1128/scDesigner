from copy import deepcopy
from ..simulators import CompositeGLMSimulator


def split_glm(simulator, submodel, keys=["group1", "group2"]):
    initial_model = {
        "formula": simulator.formula,
        "simulator": simulator,
        "var_names": None,
        "_fitted": True,
    }

    # default to the formula in original model
    if not "formula" in submodel:
        submodel["formula"] = simulator.formula

    # default to the original simulator type, but remove existing parameters
    if not "simulator" in submodel:
        submodel["simulator"] = deepcopy(simulator)
        submodel["simulator"].params = None

    specification = {keys[0]: initial_model, keys[1]: submodel}
    return CompositeGLMSimulator(specification)
