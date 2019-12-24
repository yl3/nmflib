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

HUMAN_GENOME_TRINUCS = pd.Series({
    'ACA': 119319665,
    'ACC':  67716581,
    'ACG':  15261799,
    'ACT':  95152441,
    'CCA': 107948518,
    'CCC':  76345540,
    'CCG':  16293649,
    'CCT': 103832469,
    'GCA':  84864803,
    'GCC':  69111393,
    'GCG':  14133840,
    'GCT':  81939019,
    'TCA': 115916350,
    'TCC':  91246648,
    'TCG':  13284813,
    'TCT': 131394898,
    'ATA': 120839286,
    'ATC':  78873566,
    'ATG': 107787971,
    'ATT': 146385078,
    'CTA':  75820589,
    'CTC':  99277618,
    'CTG': 119124939,
    'CTT': 118095574,
    'GTA':  66308902,
    'GTC':  55342053,
    'GTG':  88821965,
    'GTT':  86977562,
    'TTA': 120794051,
    'TTC': 118349660,
    'TTG': 112314468,
    'TTT': 226970912,
})
"""Human genome trinucleotide counts.

Computed from Homo_sapiens.GRCh38.dna.primary_assembly.fa.
"""

HUMAN_EXOME_TRINUCS = pd.Series({
    'ACA': 1159213,
    'ACC':  970837,
    'ACG':  433771,
    'ACT':  936982,
    'CCA': 1705905,
    'CCC': 1346224,
    'CCG':  745156,
    'CCT': 1457160,
    'GCA': 1352759,
    'GCC': 1481551,
    'GCG':  642552,
    'GCT': 1420847,
    'TCA': 1327242,
    'TCC': 1452785,
    'TCG':  461051,
    'TCT': 1418413,
    'ATA':  603716,
    'ATC':  943959,
    'ATG': 1185633,
    'ATT':  874926,
    'CTA':  517988,
    'CTC': 1446981,
    'CTG': 2016669,
    'CTT': 1353882,
    'GTA':  574746,
    'GTC':  912014,
    'GTG': 1220269,
    'GTT':  817985,
    'TTA':  561067,
    'TTC': 1358863,
    'TTG': 1138305,
    'TTT': 1263716,
})
"""Human exome trinucleotide counts.

Computed from Homo_sapiens.GRCh38.dna.primary_assembly.fa using Ensembl v92
protein coding genes' coding regions as the exonic region.
"""

MUTS_6 = ['C>A', 'C>G', 'C>T', 'T>A', 'T>C', 'T>G']
"""Six possible mutation types on the pyrimidine strand."""

MUTS_6_CAT = CategoricalDtype(categories=MUTS_6, ordered=True)
