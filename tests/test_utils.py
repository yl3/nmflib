"""Test mutlib.utils."""

import pytest

import mutlib
import mutlib.contexts


def test_revcomp():
    assert mutlib.utils.revcomp('ACGT') == 'ACGT'
    assert mutlib.utils.revcomp('AATTCCGG') == 'CCGGAATT'
    assert mutlib.utils.revcomp('A') == 'T'


@pytest.mark.datafiles('test_data/hs37d5.1:1-100000.fa.gz')
def test_get_context(datafiles):
    fasta_file = str(datafiles) + '/hs37d5.1:1-100000.fa.gz'
    context_finder = mutlib.contexts.ContextFinder(fasta_file)

    # Flipped context
    assert context_finder.get_unnormalised_context('1', 10002, 3) \
        == ('TAA', 'A')
    assert context_finder.get_context('1', 10002, 3) == ('TTA', 'T')

    # Unflipped context
    assert context_finder.get_unnormalised_context('1', 10004, 5) \
        == ('AACCC', 'C')
    assert context_finder.get_context('1', 10004, 5) == ('AACCC', 'C')

    # Context length must be an odd number.
    with pytest.raises(ValueError):
        context_finder.get_context('1', 10004, 2)
    with pytest.raises(ValueError):
        context_finder.get_context('1', 10004, 4)
