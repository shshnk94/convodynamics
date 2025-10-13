import pandas as pd

def adaptability(
    speaker_a: pd.Series, 
    speaker_b: pd.Series
):

    """
    Calculate the adaptibility between two speakers using Spearman correlation.
    """

    return speaker_a.reset_index(drop=True).corr(speaker_b.reset_index(drop=True), method='spearman')

def predictability(
    speaker: pd.Series
):

    """
    Calculate the predictability of a speaker's turn lengths using lag-1 autocorrelation.
    """

    return speaker.autocorr(lag=1)