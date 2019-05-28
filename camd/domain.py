"""
Preliminary module for determining search spaces
"""
import os
import pandas as pd
import abc
import warnings

from camd import S3_CACHE
from camd.utils.s3 import cache_s3_objs
from protosearch.build_bulk.oqmd_interface import OqmdInterface

from pymatgen.io.ase import AseAtomsAdaptor
from matminer.featurizers.base import MultipleFeaturizer
from matminer.featurizers.composition import ElementProperty, Stoichiometry, ValenceOrbital, IonProperty
from matminer.featurizers.structure import (SiteStatsFingerprint, StructuralHeterogeneity,
                                            ChemicalOrdering, StructureComposition, MaximumPackingEfficiency)

from pymatgen import Composition, Element
import itertools
import numpy as np



# Just an initial idea, will need some fleshing out
class Domain(abc.ABC):
    @abc.abstractmethod
    def sample(self, num_samples):
        """
        Abstract method for sampling from a domain

        Args:
            num_samples:

        Returns:

        """
        pass

    @abc.abstractmethod
    def candidates(self):
        pass



class StructureDomain(Domain):
    def __init__(self, formulas):
        """
        Creates an ML-ready search domain with given chemical formulas.

        Args:
            formulas(list): List of chemical formulas to create new material candidates.

        """
        self.formulas = formulas
        self.structures = None
        self.features = None
        self.features_df = None

    @classmethod
    def from_bounds(cls, bounds, charge_balanced=True, create_subsystems=False, **kwargs):
        """
        Convenience constructor that delivers an ML-ready domain from defined chemical boundaries.
        Args:
            bounds:
            charge_balanced:
            frequency_threshold:
            create_subsystems:
            kwargs: arguments to pass to formula creator
        Returns:

        """
        formulas = create_formulas(bounds, charge_balanced=charge_balanced,
                                   create_subsystems=create_subsystems, **kwargs)
        print("Generated the following chemical formulas for the given system: {}".format(formulas))
        return cls(formulas)

    @property
    def bounds(self):
        bounds = set()
        for formula in self.formulas:
            bounds = bounds.union(Composition(formula).as_dict().keys())
        return bounds

    @property
    def bounds_string(self):
        return '-'.join(self.bounds)

    def get_structures(self):
        if self.formulas:
            self.structures = get_structures_from_protosearch(self.formulas)
            print("Generated {} hypothetical structures".format(len(self.structures)))
        else:
            raise("Need formulas to create structures")

    @property
    def compositions(self):
        return [s.composition for s in self.structures]

    @property
    def reduced_formulas(self):
        if self.structures:
            return [s.composition.reduced_formula for s in self.structures["pmg_structures"]]
        else:
            warnings.warn("No structures available yet.")
            return []

    def featurize_structures(self, featurizer=None, **kwargs):
        if not self.structures:
            warnings.warn("No structures available. Attemtting to generate structures first.")
            self.get_structures()

        print("Generating features")

        # Defaults to  PRB Ward descriptors
        featurizer = featurizer if featurizer else MultipleFeaturizer([
            SiteStatsFingerprint.from_preset("CoordinationNumber_ward-prb-2017"),
            StructuralHeterogeneity(),
            ChemicalOrdering(),
            MaximumPackingEfficiency(),
            SiteStatsFingerprint.from_preset("LocalPropertyDifference_ward-prb-2017"),
            StructureComposition(Stoichiometry()),
            StructureComposition(ElementProperty.from_preset("magpie")),
            StructureComposition(ValenceOrbital(props=['frac'])),
            StructureComposition(IonProperty(fast=True))
        ])

        features = featurizer.featurize_many(self.structures['pmg_structures'], **kwargs)
        self.features_df = pd.DataFrame.from_records(features, columns=featurizer.feature_labels())
        self.features_df.index = self.structures.index
        return self.features_df

    @property
    def candidates(self, include_formula=True):
        if not self.features_df:
            self.featurize_structures()
        if include_formula:
            _features_df = self.features_df.copy()
            _features_df['formula'] = self.reduced_formulas
            return _features_df
        return self.features_df

    def sample(self, num_samples):
        self.candidates.sample(num_samples)


def get_structures_from_protosearch(formulas, source='icsd', db_interface=None):
    """
       Function to create a dataframe of structures corresponding
       to formulas from OQMD prototypes

       Args:
           formulas ([str]): list of chemical formulas from which
               to generate candidate structures
           db_interface (DbInterface): interface to OQMD database
               by default uses the one stored in s3

       Returns:
            pandas.DataFrame of structures generated and their unique ids.

       """
    if db_interface is None:
        obj = "camd/shared-data/protosearch-data/materials-db/oqmd/oqmd_ver3.db"
        cache_s3_objs([obj])
        oqmd_db_path = os.path.join(S3_CACHE, obj)
        db_interface = OqmdInterface(oqmd_db_path)
    dataframes = [
        db_interface.create_proto_data_set(
            source=source, chemical_formula=formula)
        for formula in formulas
    ]
    _structures = pd.concat(dataframes)

    ase_adap = AseAtomsAdaptor()
    pmg_structures = [ase_adap.get_structure(_structures.iloc[i]['atoms']) for i in range(len(_structures))]
    _structures['pmg_structures'] = pmg_structures
    structure_uids = [_structures.iloc[i]['proto_name'] +
                           '_' + '_'.join(pmg_structures[i].symbol_set) for i in range(len(_structures))]
    _structures.index = structure_uids
    return  _structures


def get_stoichiometric_formulas(n_components, grid=None):
    """
    Returns unique
    """
    grid = grid if grid else list(range(1,8))
    args = [grid for _ in range(n_components)]
    stoics = np.array(list(itertools.product(*args)))
    fracs = stoics.astype(float)/np.sum(stoics,axis=1)[:,None]
    _, indices, counts = np.unique(fracs,axis=0, return_index=True, return_counts = True)
    return stoics[ indices ]


def create_formulas(bounds, charge_balanced=True, oxi_states_extend=None, oxi_states_override=None,
                    all_oxi_states=False, create_subsystems=False, grid=None):
    """
    Creates a list of formulas given the bounds of a chemical space.
    """
    stoichs = get_stoichiometric_formulas(len(bounds), grid=grid)

    formulas = []
    for f in stoichs:
        f_ = ''
        for i in range(len(f)):
            f_ += bounds[i] + f.astype(str).tolist()[i]
        formulas.append(f_)

    if charge_balanced:

        charge_balanced_formulas=[]

        if oxi_states_extend:
            oxi_states_override = oxi_states_override if oxi_states_override else {}
            for k, v in oxi_states_extend.items():
                v = v if type(v) == list else [v]
                _states = v + list(Element[k].common_oxidation_states)
                if k in oxi_states_override:
                    oxi_states_override[k] += v
                else:
                    oxi_states_override[k] = _states

        for formula in formulas:
            c = Composition(formula)
            if c.oxi_state_guesses(oxi_states_override=oxi_states_override, all_oxi_states=all_oxi_states):
                charge_balanced_formulas.append(formula)
        return charge_balanced_formulas
    else:
        return formulas