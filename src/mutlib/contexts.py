"""Code for handling mutation contexts."""

import logging
import numpy as np
import pandas as pd
import progressbar
import pysam

import mutlib
import mutlib.constants


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
        context = self.fasta.fetch(chrom, start, end).upper()
        return context

    def get_context(self, chrom, pos, context_len=3):
        """Return the context flipped into the pyrimidine strand."""
        context = self.get_unnormalised_context(chrom, pos, context_len)
        centre = context[context_len // 2]
        if centre in ('A', 'G'):
            context = mutlib.revcomp(context)
            flipped = True
        else:
            flipped = False
        return context, flipped

    def get_context_array(self, chroms, pos, context_len=3):
        """Get contexts for an array of values.

        Returns:
            pandas.DataFrame: A data frame with columns for context and whether
                it was flipped to the pyrimidine strand.
        """
        outputs = []
        chroms_pos = list(zip(chroms, pos))
        for c, p in progressbar.progressbar(chroms_pos):
            outputs.append(self.get_context(c, p, context_len))
        out_df = pd.DataFrame(outputs, columns=['context', 'flipped'])
        if isinstance(chroms, pd.Series):
            out_df.index = chroms.index
        else:
            out_df.index = [c + ":" + str(p) for c, p in chroms_pos]
        return out_df


def count_snv_types(contexts, ref, alt, samples, context_len=3):
    """Compute single base substitution type contexts by sample.

    Args:
        contexts (pandas.DataFrame): A data frame with two columns: contexts
            and whether the context is reverse complemented with respect to the
            reference genome.
        ref (array-like): A list of reference bases.
        alt (array-like): A list of alternate bases mutated into.
        samples (array-like): A list of samples of each mutation.
        context_len (array-like): Length of the mutation context.

    Returns:
        pandas.DataFrame: A data frame of shape (<mutation types>,
            <samples>) with counts for each mutation type in each sample.
    """
    # Sanity check.
    if any(pd.Series(ref) == pd.Series(alt)):
        raise ValueError("Some ref bases are the same as alt.")

    central_bases = contexts['context'].str[context_len // 2]
    central_bases = np.where(
        contexts['flipped'],
        mutlib.COMPLEMENT[central_bases.values],
        central_bases)
    df = pd.DataFrame(
        {'ref': ref,
         'alt': alt,
         'samples': samples,
         'context': contexts['context'],
         'flipped': contexts['flipped'],
         'queried_ref': central_bases}
    )

    flipped_mut = np.where(
        df['flipped'],
        (mutlib.COMPLEMENT[df['ref'].values].values + '>'
            + mutlib.COMPLEMENT[df['alt'].values].values),
        df['ref'] + '>' + df['alt'])
    df['snv'] = flipped_mut

    # Identify and mask variants with problems.
    ref_mismatch = df['ref'] != df['queried_ref']
    msg = "{}/{} reference bases mismatch".format(sum(ref_mismatch),
                                                  len(ref_mismatch))
    logging.info(msg)
    bad_context = df['context'].str.contains(r'[^ACGT]')
    msg = "{}/{} contexts contain a non-ACGT".format(sum(bad_context),
                                                     len(ref_mismatch))
    logging.info(msg)
    df.loc[ref_mismatch | bad_context, 'context'] = np.nan
    df.loc[ref_mismatch | bad_context, 'snv'] = np.nan

    mut_counts = df.groupby(['samples', 'snv', 'context']).size().unstack(0)
    mut_counts = mut_counts.fillna(0).astype(int)
    return mut_counts


def compute_and_count_snv_types(fasta, chroms, pos, ref, alt, samples,
                                context_len=3):
    """First compute mutation context types, then count them by sample.

    Args:
        fasta (str): Path to a reference genome FASTA file.
        chroms (array-like): A list of chromosomes as strings.
        pos (array-like): A list of chromosomal positions.
        ref (array-like): A list of reference bases.
        alt (array-like): A list of alternate bases mutated into.
        samples (array-like): A list of samples of each mutation.
        context_len (array-like): Length of the mutation context.

    Returns:
        pandas.DataFrame: A data frame of shape (<mutation types>,
            <samples>) with counts for each mutation type in each sample.
    """
    context_finder = ContextFinder(fasta)
    contexts = context_finder.get_context_array(chroms, pos, context_len)
    mut_counts = count_snv_types(contexts, ref, alt, samples, context_len)
    return mut_counts
