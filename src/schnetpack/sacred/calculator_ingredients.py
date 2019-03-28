from sacred import Ingredient
import os
import torch

from schnetpack.md.calculators import SchnetPackCalculator, SGDMLCalculator
from schnetpack.md.utils import MDUnits

calculator_ingradient = Ingredient('calculator')


@calculator_ingradient.config
def config():
    """configuration for the calculator ingredient"""
    calculator = 'schnet_calculator'
    required_properties = ['y', 'dydx']
    force_handle = 'dydx'
    position_conversion = 1.0 / MDUnits.angs2bohr
    force_conversion = 1.0 / MDUnits.auforces2aseforces
    property_conversion = {}
    model_path = 'eth_ens_01.model'


@calculator_ingradient.named_config
def sgdml():
    """
    Configuration for the sGDML calculator, which uses the sGDML model published in [#sgdml1]_and [#sgdml2]_.
    Available properties are energies and forces. In its current state, sGDML is only available for molecules of the
    same size. In order to use the calculator, the sgdml code package available online at
    https://github.com/stefanch/sGDML and described in [#sgdml3]_ is required.

    References
    ----------
    .. [#sgdml1] Chmiela, Tkatchenko, Sauceda, Poltavsky, Schütt, Müller:
       Energy-conserving Molecular Force Fields.
       Science Advances, 3 (5), e1603015. 2017.
    .. [#sgdml2] Chmiela, Sauceda, Müller, Tkatchenko:
       Towards Exact Molecular Dynamics Simulations with Machine-Learned Force Fields.
       Nature Communications, 9 (1), 3887. 2018.
    .. [#sgdml3] Chmiela, Sauceda, Poltavsky, Müller, Tkatchenko:
       sGDML: Constructing accurate and data efficient molecular force fields using machine learning.
       Computer Physics Communications (in press). https://doi.org/10.1016/j.cpc.2019.02.007
    """
    calculator = 'sgdml_calculator'
    required_properties = ['energy', 'forces']
    force_handle = 'forces'
    position_conversion = 1.0 / MDUnits.angs2bohr
    force_conversion = 1.0 / MDUnits.Ha2kcalpmol / MDUnits.angs2bohr
    property_conversion = {}
    model_path = 'sgdml_model.npz'


@calculator_ingradient.capture
def load_model_schnet(_log, model_path, device):
    # If model is a directory, search for best_model file
    if os.path.isdir(model_path):
        model_path = os.path.join(model_path, 'best_model')
    _log.info('Loaded model from {:s}'.format(model_path))
    model = torch.load(model_path).to(device)
    return model


@calculator_ingradient.capture
def load_model_sgdml(_log, model_path, device):
    import numpy as np
    try:
        parameters = np.load(model_path)
    except:
        raise ValueError("Could not read sGDML model from {:s}".format(model_path))

    from sgdml.torchtools import GDMLTorchPredict
    model = GDMLTorchPredict(parameters).to(device)
    _log.info('Loaded sGDML model from {:s}'.format(model_path))

    return model


@calculator_ingradient.capture
def build_calculator(_log, required_properties, force_handle,
                     position_conversion, force_conversion,
                     property_conversion, calculator, device):
    """
    Build the calculator object from the provided settings.

    Args:
        model (torch.nn.module): the model which is used for property calculation
        required_properties (list): list of properties that are calculated by the model
        force_handle (str): name of the forces property in the model output
        position_conversion (float): conversion factor for positions
        force_conversion (float): conversion factor for forces
        property_conversion (dict): dictionary with conversion factors for other properties
        calculator (src.schnetpack.md.calculator.Calculator): calculator object

    Returns:
        the calculator object
    """
    _log.info(f'Using {calculator}')

    position_conversion = MDUnits.parse_mdunit(position_conversion)
    force_conversion = MDUnits.parse_mdunit(force_conversion)

    if calculator == 'schnet_calculator':

        model = load_model_schnet(device=device)
        return SchnetPackCalculator(model,
                                    required_properties=required_properties,
                                    force_handle=force_handle,
                                    position_conversion=position_conversion,
                                    force_conversion=force_conversion,
                                    property_conversion=property_conversion)
    elif calculator == 'sgdml_calculator':
        model = load_model_sgdml(device=device)
        return SGDMLCalculator(model,
                               required_properties=required_properties,
                               force_handle=force_handle,
                               position_conversion=position_conversion,
                               force_conversion=force_conversion,
                               property_conversion=property_conversion)
    else:
        raise NotImplementedError
