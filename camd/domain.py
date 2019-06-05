"""
Preliminary module for determining search spaces
"""
import os
import pandas as pd
import abc
import warnings
import itertools
import numpy as np

from camd import S3_CACHE
from camd.utils.s3 import cache_s3_objs
from protosearch.build_bulk.oqmd_interface import OqmdInterface

from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen import Composition, Element
from matminer.featurizers.base import MultipleFeaturizer
from matminer.featurizers.composition import ElementProperty, Stoichiometry, ValenceOrbital, IonProperty
from matminer.featurizers.structure import (SiteStatsFingerprint, StructuralHeterogeneity,
                                            ChemicalOrdering, StructureComposition, MaximumPackingEfficiency)


class DomainBase(abc.ABC):
    """
    Domains combine geberation and featurization and prepare the search space for CAMD Loop.
    """
    @abc.abstractmethod
    def candidates(self):
        """
        Primary method for every Domain to provide candidates.
        Returns:
            pandas.DataFrame: features for generated hypothetical structures. The Index of dataframe should be
                the unique ids for the structures.
        """
        pass

    @property
    @abc.abstractmethod
    def bounds(self):
        """
        Returns:
            list: names of dimensions of the search space.
        """
        pass

    @abc.abstractmethod
    def sample(self, num_samples):
        """
        Abstract method for sampling from created domain
        Args:
            num_samples:
        Returns:
        """
        pass

    @property
    def bounds_string(self):
        """
        Returns: a string representation of search space bounds: e.g. "Ir-Fe-O" or "x1-x2-x3"
        """
        return '-'.join(self.bounds)


class StructureDomain(DomainBase):
    """
    Provides machine learning ready candidate domains (search spaces) for hypothetical structures.
    If scanning an entire system, use the StructureDomain.from_bounds method.
    Args:
        formulas ([str]): list of chemical formulas to create new material candidates.
    """
    def __init__(self, formulas, n_max_atoms=None):
        self.formulas = formulas
        self.n_max_atoms = n_max_atoms
        self.features = None
        self._hypo_structures = None

    @classmethod
    def from_bounds(cls, bounds, n_max_atoms=None, charge_balanced=True, create_subsystems=False, **kwargs):
        """
        Convenience constructor that delivers an ML-ready domain from defined chemical boundaries.
        Args:
            bounds:
            charge_balanced:
            frequency_threshold:
            create_subsystems:
            kwargs: arguments to pass to formula creator
        """
        formulas = create_formulas(bounds, charge_balanced=charge_balanced,
                                   create_subsystems=create_subsystems, **kwargs)
        print("Generated chemical formulas: {}".format(formulas))
        return cls(formulas, n_max_atoms)

    @property
    def bounds(self):
        bounds = set()
        for formula in self.formulas:
            bounds = bounds.union(Composition(formula).as_dict().keys())
        return bounds

    def get_structures(self):
        """
        Method to call the external structure generator.
        """
        if self.formulas:
            print("Generating hypothetical structures...")
            self._hypo_structures = get_structures_from_protosearch(self.formulas)
            print("Generated {} hypothetical structures".format(len(self.hypo_structures)))
        else:
            raise("Need formulas to create structures")

    @property
    def hypo_structures(self):
        if self._hypo_structures is None:
            self.get_structures()
        if self.n_max_atoms:
            n_max_filter = [i.num_sites <= self.n_max_atoms for i in self._hypo_structures['pmg_structures']]
            if self._hypo_structures is not None:
                return self._hypo_structures[n_max_filter]
            else:
                return None
        else:
            return self._hypo_structures

    @property
    def hypo_structures_dict(self):
        return self.hypo_structures["pmg_structures"].to_dict()

    @property
    def compositions(self):
        """
        Returns:
            list: Compositions of hypothetical structures generated.
        """
        if self.hypo_structures is not None:
            return [s.composition for s in self.hypo_structures]
        else:
            warnings.warn("No stuctures available.")
            return []

    @property
    def reduced_formulas(self):
        if self.valid_structures is not None: # Note the redundancy here is for pandas to work
            return [s.composition.reduced_formula for s in self.valid_structures["pmg_structures"]]
        else:
            warnings.warn("No structures available yet.")
            return []

    def featurize_structures(self, featurizer=None, **kwargs):
        """
        Featurizes the hypothetical structures available from hypo_structures method. Hypothetical structures for
            which featurization fails is removed and valid structures are made available as valid_structures
        Args:
            featurizer (Featurizer): A MatMiner Featurizer. Defaults to MultipleFeaturizer with
                PRB Ward Voronoi descriptors.
            **kwargs (dict): kwargs passed to featurize_many method of featurizer.
        Returns:
            pandas.DataFrame: features
        """
        if self.hypo_structures is None: # Note the redundancy here is for pandas to work
            warnings.warn("No structures available. Generating structures.")
            self.get_structures()

        print("Generating features")
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

        features = featurizer.featurize_many(self.hypo_structures['pmg_structures'], ignore_errors=True, **kwargs)

        n_species, formula = [], []
        for s in self.hypo_structures['pmg_structures']:
            n_species.append(len(s.composition.elements))
            formula.append(s.composition.reduced_formula)

        self._features_df = pd.DataFrame.from_records(features, columns=featurizer.feature_labels())
        self._features_df.index = self.hypo_structures.index
        self._features_df['N_species'] = n_species
        self._features_df['Composition'] = formula
        self.features = self._features_df.dropna(axis=0, how='any')

        self._valid_structure_labels = list(self.features.index)
        self.valid_structures = self.hypo_structures.loc[self._valid_structure_labels]

        print("{} out of {} structures were successfully featurized.".format(self.features.shape[0],
                                                                             self._features_df.shape[0]))
        return self.features

    def candidates(self, include_formula=True):
        """
        Args:
            include_formula (bool): Adds a column named "formula" to the dataframe.
        Returns:
            feature vectors of valid hypothetical structures.
        """
        if self._hypo_structures is None:
            self.get_structures()

        if self.features is None:
            self.featurize_structures()
        if include_formula:
            _features = self.features.copy()
            _features['formula'] = self.reduced_formulas
            return _features
        return self.features

    def sample(self, num_samples):
        self.candidates().sample(num_samples)


def get_structures_from_protosearch(formulas, source='icsd', db_interface=None):
    """
    Calls protosearch to get the hypothetical structures.
    Args:
        formulas ([str]): list of chemical formulas from which to generate candidate structures
        source (str): project name in OQMD to be used as source. defaults to ICSD.
        db_interface (DbInterface): interface to OQMD database by default uses the one stored in s3
    Returns:
        pandas.DataFrame of hypothetical pymatgen structures generated and their unique ids from protosearch

    TODO:
        - For efficiency, n_max_atoms can be handled within OqmdInterface
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

    # Drop bad structures
    _structures.dropna(axis=0, how='any', inplace=True)

    # conversion to pymatgen structures
    ase_adap = AseAtomsAdaptor()
    pmg_structures = [ase_adap.get_structure(_structures.iloc[i]['atoms'])
                      for i in range(len(_structures))]
    _structures['pmg_structures'] = pmg_structures

    structure_uids = [_structures.iloc[i]['proto_name'].replace('_','-') +
                           '-' + '-'.join(pmg_structures[i].symbol_set) for i in range(len(_structures))]
    _structures.index = structure_uids
    return _structures


def get_stoichiometric_formulas(n_components, grid=None):
    """
    Args:
        n_components (int): number of components (dimensions)
        grid (list): a range of integers
    Returns:
        list: unique stoichiometric formula from an allowed grid of integers.
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
    TODO:
        - implement create_subsystems
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