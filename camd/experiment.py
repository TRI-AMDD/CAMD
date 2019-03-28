# Copyright Toyota Research Institute 2019

def get_dft_calcs_aft(uids, df):
    """
    Mock function that mimics fetching DFT calculations
    """
    uids = [uids] if type(uids) != list else uids
    return df.loc[uids]


def get_dft_calcs_from_northwestern(uids):
    """
    Placeholder function for fetching DFT calculations from Northwestern
    """
    pass


def get_dft_calcs_from_MC1(uids):
    """
    Placeholder function for fetching DFT calculations from MC1
    """
    pass