"""General purpose functions for point mutations."""

import mutlib


def revcomp(seq):
    """Reverse complement a DNA string sequence."""
    seq_rev = seq[::-1]
    seq_revcomp = "".join([mutlib.COMPLEMENT[nt] for nt in seq_rev])
    return seq_revcomp
