import re
from collections import deque, Counter
from functools import reduce
from itertools import chain, product
from operator import add

import numpy as np

BASE_MAP = {'A': (1, 0), 'T': (0, 1), 'C': (0, 0), 'G': (1, 1)}
REV_BASE_MAP = {(1, 0): 'A', (0, 1): 'T', (0, 0): 'C', (1, 1): 'G'}
REV_BASE_ARRAY = np.array([['C', 'T'], ['A', 'G']])
RC_MAP = dict(zip("ATCG\n", "TAGC\n"))
VERBOSE = False


def string_to_ints(string):
    """
    converts a base dna string into upper + lower bit integers
    """
    upper, lower = *zip(*map(BASE_MAP.get, string)),
    return int(reduce(add, map(str, upper)), base=2), int(reduce(add, map(str, lower)), base=2)


def ints_to_string_numpy(upper, lower, length):
    """
    >>> ints_to_string_numpy(np.array([211, 280], dtype=np.int64), np.array([81, 665], dtype=np.int64), 10)
    array(['CCAGCGCCAG', 'TATCCGGCCT'], dtype='<U10')
    """
    return ints_to_string_numpy_helper(upper, lower, length)


def ints_to_string_numpy_helper(upper, lower, length):
    if length == 1:
        return REV_BASE_ARRAY[(upper & 1, lower & 1)]
    return np.char.add(ints_to_string_numpy_helper(upper >> 1, lower >> 1, length - 1),
                       REV_BASE_ARRAY[(upper & 1, lower & 1)])


def ints_to_string(upper, lower, length):
    """
    >>> ints_to_string(211, 81, 10)
    'CCAGCGCCAG'
    >>> ints_to_string(280, 665, 10)
    'TATCCGGCCT'
    """
    return "".join(ints_to_string_helper(upper, lower, length))


def ints_to_string_helper(upper, lower, length):
    if length == 0:
        return []
    return ints_to_string_helper(upper >> 1, lower >> 1, length - 1) + [REV_BASE_MAP[(upper & 1, lower & 1)]]


def make_table(length, score_function):
    """
    for all possible comparisons between length "length" strands, compute and store
    the associated score
    """
    table = np.zeros(2 ** length, dtype=np.int64)
    for h, bits in enumerate(product([0, 1], repeat=length)):
        table[h] = score_function(bits)
    return table


def reverse_complement(dna_string):
    """
    return the reverse complement of a string representing a DNA sequence
    >>> reverse_complement('CCAGCGCCAG')
    'CTGGCGCTGG'
    """
    return "".join(map(RC_MAP.get, dna_string))[::-1]


def frame_iteration(sequence, length):
    """
    given an iterable sequence, efficiently generate "length" long
    overlapping subsequences
    """
    head = deque(sequence[:length])
    tail = sequence[length:]
    total = len(sequence) - length + 1
    d = total // 100
    yield string_to_ints(head)
    for i, element in enumerate(tail):
        head.popleft()
        head.append(element)
        if '\n' in head:
            continue
        yield string_to_ints(head)
        if VERBOSE and i % d == 0:
            print(f'{i}, {i // d:.2f}% complete')


def transcribe(genome, length, do_reverse_complement=True):
    gen = frame_iteration(genome, length)
    if do_reverse_complement:
        rgen = reverse_complement(genome)
        genrc = frame_iteration(rgen, length)
        all_subs = np.array(list(chain(gen, genrc)), dtype=np.int32)
    else:
        all_subs = np.array(list(gen), dtype=np.int32)
    all_subs.flags.writeable = False
    return all_subs


def most_common(genome, length, n):
    seqs = Counter([])
    for segment in genome.substrings_iter(length):
        seqs.update(zip(*segment.transcribe(length)))
    return np.array([a for a, _ in seqs.most_common(n)], dtype=np.int32).T


INVALID_REGEX = re.compile(r'''
(?P<invalid>[^ACGTRYMKSWHBVDN])
''', re.X)
UNCERTAIN_REGEX = re.compile(r'''
(?P<uncertain>[RYMKSWHBVDN])
''', re.X)
replacement_dict = {
    'R': 'GA',
    'Y': 'TC',
    'M': 'AC',
    'K': 'GT',
    'S': 'GC',
    'W': 'AT',
    'H': 'ACT',
    'B': 'GTC',
    'V': 'GCA',
    'D': 'GAT',
    'N': 'GATC',
}


def get_substrings(genome_str: str, length: int):
    """
    iterate over lists of independent substrings of genome_str, with all possible substitutions for uncertain bases
    """
    matches = list(UNCERTAIN_REGEX.finditer(genome_str))
    if not matches:
        yield genome_str
        return
    f = [match.start() for match in matches]
    for s in get_ranges(f, length, size=len(genome_str)):
        if len(genome_str[s]) < length:
            continue
        yield list(expand(genome_str[s]))
    return


def get_ranges(indices, length, size):
    """
    iterate over slices of a genome whose subscores can be computed independently
    """
    if indices[0] > length:
        yield slice(0, indices[0])
    r, indices = indices[0:1], indices[1:]
    while indices:
        j, indices = indices[0], indices[1:]
        if j - r[-1] > length:
            yield slice(max(0, r[0] - length + 1), min(r[-1] + length, size))
            yield slice(max(0, r[-1] + 1), min(j, size))
            r = [j]
        else:
            r.append(j)
    yield slice(max(0, r[0] - length), min(r[-1] + length, size))
    yield slice(max(0, r[-1] + 1), size)
    return


def expand(genome_str: str):
    """
    take a string and return an iterator over all replacements of uncertain bases
    """
    if not (matches := list(UNCERTAIN_REGEX.finditer(genome_str))):
        yield genome_str
        return
    uncertain_bases = [(match.group(), match.start()) for match in matches]
    old_bs, poss = zip(*uncertain_bases)
    product(replacement_dict[b] for b in old_bs)
    for new_bs in product(*[replacement_dict[b] for b in old_bs]):
        new_string = genome_str
        for pos, old_b, new_b in zip(poss, old_bs, new_bs):
            new_string = new_string.replace(old_b, new_b, 1)
        yield new_string
    return
