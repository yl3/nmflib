"""Test nmflib.utils."""

import pytest
import pandas as pd

import nmflib
import nmflib.contexts


def test_revcomp():
    assert nmflib.utils.revcomp('ACGT') == 'ACGT'
    assert nmflib.utils.revcomp('AATTCCGG') == 'CCGGAATT'
    assert nmflib.utils.revcomp('A') == 'T'


@pytest.mark.datafiles('test_data/hs37d5.1:1-100000.fa.gz')
def test_get_context(datafiles):
    fasta_file = str(datafiles) + '/hs37d5.1:1-100000.fa.gz'
    context_finder = nmflib.contexts.ContextFinder(fasta_file)

    # Flipped context
    assert context_finder.get_unnormalised_context('1', 10002, 3) == 'TAA'
    assert context_finder.get_context('1', 10002, 3) == ('TTA', True)

    # Unflipped context
    assert context_finder.get_unnormalised_context('1', 10004, 5) == 'AACCC'
    assert context_finder.get_context('1', 10004, 5) \
        == ('AACCC', False)

    # Context length must be an odd number.
    with pytest.raises(ValueError):
        context_finder.get_context('1', 10004, 2)
    with pytest.raises(ValueError):
        context_finder.get_context('1', 10004, 4)

    # Test the context array
    context_df = context_finder.get_context_array(['1', '1'], [10002, 10004])
    assert isinstance(context_df, pd.DataFrame)
    assert (list(context_df.itertuples(index=False, name=None))
            == [('TTA', True), ('ACC', False)])

    # Test if the Pandas index is passed over properly.
    context_df = context_finder.get_context_array(['1', '1'], [10002, 10004])
    assert (context_df.index == ['1:10002', '1:10004']).all()

    context_df = context_finder.get_context_array(
        pd.Series(['1', '1'], index=['test1', 'test2']),
        [10002, 10004])
    assert (context_df.index == ['test1', 'test2']).all()


def test_count_snv_types(datafiles):
    contexts = pd.DataFrame(
        [('ATT', True),
         ('ATT', True),
         ('GCC', False)],
        columns=['context', 'flipped']
    )
    ref = ['A', 'A', 'C']
    alt = ['T', 'T', 'T']
    samples = ['sample_1', 'sample_1', 'sample_2']

    # Check that ref must be different to alt.
    with pytest.raises(ValueError):
        bad_alt = ['A', 'T', 'C']  # First alt is the same as first ref.
        nmflib.contexts.count_snv_types(contexts, ref, bad_alt, samples, 3)

    # Correct ref, no problem
    mut_counts = nmflib.contexts.count_snv_types(contexts, ref, alt, samples, 3)
    expected_index = pd.DataFrame([('C>T', 'GCC'),
                                   ('T>A', 'ATT')])
    assert (mut_counts.index.to_frame() == expected_index.values).all(None)
    assert (mut_counts.columns == ['sample_1', 'sample_2']).all()
    expected_values = pd.DataFrame(
        [(0, 1),
         (2, 0)]
    )
    assert (mut_counts == expected_values.values).all(None)

    # If the (first) reference doesn't match that of the context, the respective
    # mutations should be ignored.
    expected_values = pd.DataFrame(
        [(0, 1),
         (1, 0)]
    )
    bad_ref = ['C', 'A', 'C']  # ref mismatch with the "reference"
    mut_counts = nmflib.contexts.count_snv_types(contexts, bad_ref, alt,
                                                 samples, 3)
    assert (mut_counts == expected_values.values).all(None)

    # A context that contains N's should be ignored.
    bad_contexts = pd.DataFrame(
        [('NTT', True),
         ('ATT', True),
         ('GCC', False)],
        columns=['context', 'flipped']
    )
    mut_counts = nmflib.contexts.count_snv_types(bad_contexts, ref, alt,
                                                 samples, 3)
    expected_index = pd.DataFrame([('C>T', 'GCC'),
                                   ('T>A', 'ATT')])
    assert (mut_counts.index.to_frame() == expected_index.values).all(None)
    assert (mut_counts.columns == ['sample_1', 'sample_2']).all()
    expected_values = pd.DataFrame([(0, 1),
                                    (1, 0)])
    assert (mut_counts == expected_values.values).all(None)


@pytest.mark.datafiles('test_data/hs37d5.1:1-100000.fa.gz')
def test_compute_and_count_snv_types(datafiles):
    fasta_file = str(datafiles) + '/hs37d5.1:1-100000.fa.gz'
    chroms = ['1', '1', '1']
    pos = [10002, 10004, 10002]
    ref = ['A', 'C', 'A']
    alt = ['T', 'G', 'T']
    samples = ['sample_1', 'sample_1', 'sample_2']
    mut_counts = nmflib.contexts.compute_and_count_snv_types(
        fasta_file, chroms, pos, ref, alt, samples, 3)
    expected_index = pd.DataFrame([('C>G', 'ACC'),
                                   ('T>A', 'TTA')])
    assert (mut_counts.index.to_frame() == expected_index.values).all(None)
    assert (mut_counts.columns == ['sample_1', 'sample_2']).all()
    expected_values = pd.DataFrame([(1, 0),
                                    (1, 1)])
    assert (mut_counts == expected_values.values).all(None)
