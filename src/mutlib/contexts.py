"""Code for handling mutation contexts."""

import mutlib
import pysam


class ContextFinder:
    """Class for finding and manipulating mutation contexts."""

    def __init__(self, fasta):
        self.fasta = pysam.FastaFile(fasta)

    def get_unnormalised_context(self, chrom, pos, context_len):
        """Get the unnormalised context of a chromosomal position."""
        # pysam considers start position 0-based.
        if not context_len % 2 == 1:
            raise ValueError("'context_len' must be odd.")
        expand = context_len // 2
        start = pos - 1 - expand
        end = pos + expand
        context = self.fasta.fetch(chrom, start, end)
        centre = context[expand]
        return context, centre

    def get_context(self, chrom, pos, context_len=3):
        """Return the context flipped into the pyrimidine strand."""
        context, centre = self.get_unnormalised_context(chrom, pos, context_len)
        if centre in ('A', 'G'):
            context = mutlib.revcomp(context)
            centre = mutlib.revcomp(centre)
        return context, centre
