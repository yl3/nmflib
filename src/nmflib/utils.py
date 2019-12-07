"""General purpose functions for point mutations."""

import nmflib


def revcomp(seq):
    """Reverse complement a DNA string sequence."""
    seq_rev = seq[::-1]
    seq_revcomp = "".join([nmflib.COMPLEMENT[nt] for nt in seq_rev])
    return seq_revcomp
