"""General purpose functions for point mutations."""

COMPLEMENT = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}


def revcomp(seq):
    """Reverse complement a DNA string sequence."""
    seq_rev = seq[::-1]
    seq_revcomp = "".join([COMPLEMENT[nt] for nt in seq_rev])
    return seq_revcomp
