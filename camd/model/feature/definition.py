"""
Core feature definitions.

Helper methods.

"""


from matminer.featurizers.structure import SiteStatsFingerprint, \
    StructuralHeterogeneity, ChemicalOrdering, MaximumPackingEfficiency, \
    StructureComposition
from matminer.featurizers.composition import ElementProperty, Stoichiometry, \
    ValenceOrbital, IonProperty


def feature_definition(feature_id):
    """
    Looks up a feature definition and returns the method to compute features
    from the material structure, the feature labels and feature types.

    In many cases, features are computed in a bundle together with other
    features. Hence, a fourth value that is returned is the sub_index of the
    feature requested within the list of features returned by the computation
    method.

    Args:
        feature_id: int
            primary key value in camd.feature table


    Returns: tuple
        featurizer: method
        labels: list of str of feature labels
        types: list of str on the type of feature ('numeric', 'categorical')
        sub_index: int

    """
    index, sub_index = index_sub_index(feature_id)
    featurizer = feature_directory[index]['featurizer']
    labels = feature_directory[index]['labels']
    types = feature_directory[index]['types']
    return featurizer, labels, types, sub_index


def index_sub_index(feature_id):
    """
    Helper method to determine block index and sub_index of a feature.

    Args:
        feature_id: int
            primary key value in camd.feature table

    Returns: int, int
        index, sub_index

    """
    index_keys = feature_index_blocks
    index = index_keys[0]
    for i in index_keys:
        if i > feature_id:
            break
        else:
            index = i
    sub_index = feature_id - index
    return index, sub_index


def number_of_features():
    """
    Provides the number of features in the feature directory.

    Returns: int
        Number of features
    """
    max_index = max(list(feature_directory.keys()))
    return max_index + len(feature_directory[max_index]['labels']) - 1


def directory_integrity_check():
    """
    Method to check integrity of feature definition enumeration.

    Checks:
    - number of labels matches number types in feature sub sets
    - feature indices not overlapping; no gaps in feature enumeration

    Returns: None

    Raises:
        ValueError in case of detected inconsistencies

    """

    # number of labels matches number types
    for index in feature_index_blocks:
        sub_set = feature_directory[index]
        if len(sub_set['labels']) != len(sub_set['types']):
            raise ValueError(f'Failed feature definition integrity check. ' +
                             f'Number of labels mismatches the number of ' +
                             f' types in sub set {index}')

    # feature indices not overlapping; no gaps in feature enumeration
    for i in range(len(feature_index_blocks) - 1):
        index = feature_index_blocks[i]
        sub_set = feature_directory[index]
        if len(sub_set['labels']) + index != feature_index_blocks[i + 1]:
            if len(sub_set['labels']) + index < feature_index_blocks[i + 1]:
                raise ValueError('Failed feature definition integrity check. ' +
                                 'GAP in feature enumeration in sub_set ' +
                                 f' {index}')
            else:
                raise ValueError('Failed feature definition integrity check. ' +
                                 'OVERLAP in feature enumeration in sub_set ' +
                                 f' {index}')

    return


feature_index_blocks = (1, 6, 15, 18, 19, 129, 135, 267, 271, 272)

feature_directory = {
    1: {
        'featurizer': lambda x: [[xx] for xx in SiteStatsFingerprint.from_preset("CoordinationNumber_ward-prb-2017").featurize(x.structure())],
        'labels': SiteStatsFingerprint.from_preset("CoordinationNumber_ward-prb-2017").feature_labels(),
        'types': ('numerical', 'numerical', 'numerical', 'numerical', 'numerical',)
        },
    6: {
        'featurizer': lambda x: [[xx] for xx in StructuralHeterogeneity()\
            .featurize(x.structure())],
        'labels': StructuralHeterogeneity().feature_labels(),
        'types': ['numerical'] * len(StructuralHeterogeneity().feature_labels())
    },
    15: {
        'featurizer': lambda x: [[xx] for xx in
                                 ChemicalOrdering().featurize(x.structure())],
        'labels': ChemicalOrdering().feature_labels(),
        'types': ['numerical'] * len(ChemicalOrdering().feature_labels())
    },
    18: {
        'featurizer': lambda x: [[xx] for xx in MaximumPackingEfficiency()\
            .featurize(x.structure())],
        'labels': MaximumPackingEfficiency().feature_labels(),
        'types': ['numerical'] *
                 len(MaximumPackingEfficiency().feature_labels())
    },
    19: {
        'featurizer': lambda x: [[xx] for xx in SiteStatsFingerprint\
            .from_preset("LocalPropertyDifference_ward-prb-2017")\
            .featurize(x.structure())],
        'labels': SiteStatsFingerprint\
            .from_preset("LocalPropertyDifference_ward-prb-2017")\
            .feature_labels(),
        'types': ['numerical'] *
                 len(SiteStatsFingerprint\
                     .from_preset("LocalPropertyDifference_ward-prb-2017")\
                     .feature_labels())
    },
    129: {
        'featurizer': lambda x: [[xx] for xx in \
                                 StructureComposition(Stoichiometry())\
                                     .featurize(x.structure())],
        'labels': StructureComposition(Stoichiometry()).feature_labels(),
        'types': ['numerical'] *
                 len(StructureComposition(Stoichiometry()).feature_labels())
    },
    135: {
        'featurizer': lambda x: [[xx] for xx in \
                                 StructureComposition(ElementProperty\
                                                      .from_preset("magpie"))\
                                     .featurize(x.structure())],
        'labels': StructureComposition(ElementProperty.from_preset("magpie"))\
            .feature_labels(),
        'types': ['numerical'] *
                 len(StructureComposition(ElementProperty\
                                          .from_preset("magpie"))\
                     .feature_labels())
    },
    267: {
        'featurizer': lambda x: [[xx] for xx in \
                                 StructureComposition(
                                     ValenceOrbital(props=['frac'])) \
                                     .featurize(x.structure())],
        'labels': StructureComposition(ValenceOrbital(props=['frac'])) \
            .feature_labels(),
        'types': ['numerical'] *
                 len(StructureComposition(ValenceOrbital(props=['frac'])) \
                     .feature_labels())
    },
    271: {
        'featurizer': lambda x: [[float(StructureComposition(\
            IonProperty(fast=True)).featurize(x.structure())[0])]],
        'labels': [StructureComposition(IonProperty(fast=True)) \
            .feature_labels()[0]],
        'types': ['numerical']
    },
    272: {
        'featurizer': lambda x: [[xx] for xx in \
                                 StructureComposition(IonProperty(fast=True)) \
                                     .featurize(x.structure())[1:]],
        'labels': StructureComposition(IonProperty(fast=True)) \
            .feature_labels()[1:],
        'types': ['numerical'] *
                 len(StructureComposition(IonProperty(fast=True)) \
                     .feature_labels()[1:])
    },
}
