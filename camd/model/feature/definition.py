"""
Core feature definitions.

Helper methods.

"""


from matminer.featurizers.structure import SiteStatsFingerprint, \
    StructuralHeterogeneity, ChemicalOrdering


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


feature_index_blocks = (1, 6, 15, )

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
    }
}

