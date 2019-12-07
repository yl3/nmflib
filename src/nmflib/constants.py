"""Constant values."""

import pandas as pd
from pandas.api.types import CategoricalDtype

COMPLEMENT = pd.Series({'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'})

TRINUC_96 = []
"""96 possible trinucleotide contexts on the pyrimidine strand."""
for mid in ['C', 'T']:
    for left in ['A', 'C', 'G', 'T']:
        for right in ['A', 'C', 'G', 'T']:
            TRINUC_96.append(left + mid + right)

TRINUC_96_CAT = CategoricalDtype(categories=TRINUC_96, ordered=True)

MUTS_6 = ['C>A', 'C>G', 'C>T', 'T>A', 'T>C', 'T>G']
"""Six possible mutation types on the pyrimidine strand."""

MUTS_6_CAT = CategoricalDtype(categories=MUTS_6, ordered=True)
