import numpy as np, csv
from collections import defaultdict


"""
# Vocab Utils
"""

PAD_SYMBOL = '###_PAD_###'  #  -> 0
START_SYMBOL = '###_START_###'  #  -> 1
END_SYMBOL = '###_END_###'  #  -> 2
OOV_SYMBOL = '###_OOV_###'  #  -> 3


def make_vocab(filepaths=None, token_fn=None, token_lists=None,
    include_start_symbol=False, include_end_symbol=False,
    include_oov_symbol=False, include_pad_symbol=False,
    standard_special_symbols=False, verbose=False):
    """ Create a vocabulary dict for a dataset.
    Accepts either a list of filepaths together with a `token_fn` to read and
    tokenize the files, or a list of token_lists that have already been
    processed. Optionally includes special symbols.

    In order to make it easy to adding padding, start/end, or OOV tokens
    at other times, it's helpful to give special entries standard values, which
    can be set by setting standard_special_symbols=True.

    '###_PAD_###' --> 0
    '###_START_###' --> 1
    '###_END_###' --> 2
    '###_OOV_###' --> 3
    """

    # Validate args
    if bool(filepaths) and bool(token_lists):
            raise Exception("You should only pass one of `filepaths` and `token_lists`")

    if bool(filepaths) ^ bool(token_fn):
            raise Exception("Can't use only one of `filepaths` and `token_fn`")

    if standard_special_symbols and not (include_start_symbol and \
        include_end_symbol and include_oov_symbol and include_pad_symbol):
        raise Exception("standard_special_symbols needs to include all 4 symbol.")

    # Initialize special symbols
    special_symbols = []
    if include_pad_symbol:
        special_symbols.append(PAD_SYMBOL)
    if include_start_symbol:
        special_symbols.append(START_SYMBOL)
    if include_end_symbol:
        special_symbols.append(END_SYMBOL)
    if include_oov_symbol:
        special_symbols.append(OOV_SYMBOL)

    counter = 0

    # Make vocab dict and initialize with special symbols
    vocab = {}
    for sym in special_symbols:
        vocab[sym] = counter
        counter += 1

    if token_lists is None: # Get tokens from filepaths and put in token_lists
        if verbose:
            token_lists = [token_fn(f) for f in tqdm(filepaths)]
        else:
            token_lists = [token_fn(f) for f in filepaths]

    # Loop through tokens and add to vocab
    if verbose: token_lists = tqdm(token_lists)

    for sequence in token_lists:
        for token in sequence:
            if token not in vocab:
                vocab[token] = counter
                counter += 1

    return vocab

def make_reverse_vocab(vocab, default_type=str, merge_fn=None):
    # Flip the keys and values in a dict.
    """ Straightforward function unless the values of the vocab are 'unhashable'
        i.e. a list. For example, a phoneme dictionary maps 'SAY' to
        ['S', 'EY1']. In this case, pass in a function merge_fn, which specifies
        how to combine the list items into a hashable key. This could be a
        lambda fn, e.g merge_fn = lambda x: '_'.join(x).

        It's also possible that there could be collisions - e.g. with
        homophones. If default_type is list, collisions will be combined into
        a list. If not, they'll be overwritten.

        Args:
            merge_fn: a function to combine lists into hashable keys

    """
    rv = defaultdict(default_type)
    for k in vocab.keys():
        if merge_fn is not None:
            if default_type is list:
                rv[merge_fn(vocab[k])].append(k)
            else:
                rv[merge_fn(vocab[k])] = k
        else:
            if default_type is list:
                rv[vocab[k]].append(k)
            else:
                rv[vocab[k]] = k
    return rv

def filter_vocab(vocab, word_list):
    # Filters a vocab dict to only words in the given word_list
    v = {}
    for key, value in tqdm(vocab.items()):
        if key in word_list:
            v[key] = value
    return v