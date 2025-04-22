from collections import abc
from functools import reduce
from itertools import product
from operator import add

import numpy as np
import pandas as pd

from genometargeting._compute import compute_full, compute_only
from .genome import Genome
from .utils import ints_to_string_numpy, reverse_complement


class Computer:
    def __init__(self, genomes: [Genome], length, score_function, do_reverse_complement=False):
        if not isinstance(genomes, abc.Iterable):
            genomes = [genomes]
        self.original_genomes = genomes
        self.genomes = reduce(add, [list(genome.substrings_iter(length)) for genome in genomes])
        self.length = length
        self.table = self.make_table(length, score_function)
        self.reverse_complement = do_reverse_complement

    @staticmethod
    def make_table(length, score_function):
        """
        for all possible comparisons between length "length" strands, compute and store
        the associated score
        """
        table = np.zeros(2 ** length, dtype=np.int64)
        for h, bits in enumerate(product([0, 1], repeat=length)):
            table[h] = score_function(bits)
        return table

    def __call__(self, threads, n=None, compare=False):
        if n is not None:
            computers = self.get_computers_top(threads, n)
        else:
            computers = self.get_computers_full(threads)
        return self.compute(computers)

    def get_computers_full(self, threads):
        length = self.length
        transcribed = [g.transcribe(length, self.reverse_complement) for g in self.genomes]
        return [compute_full(length, *gs, self.table, self.reverse_complement, threads) for gs in transcribed]

    def get_computers_top(self, threads, n):
        length = self.length
        transcribed = [g.transcribe(length, self.reverse_complement) for g in self.genomes]
        top = *map(
            np.copy,
            np.unique(
                np.hstack([g.most_common(length, n) for g in self.original_genomes]), axis=1
            )
        ),
        return [compute_only(length, *gs, *top, self.table, self.reverse_complement, threads) for gs in transcribed]

    def compute(self, computers):
        df = pd.DataFrame([])
        for genome, computer in zip(self.genomes, computers):
            upper, lower, int_scores = np.array(list(computer), dtype=np.int64).T
            probes = list(ints_to_string_numpy(upper, lower, length=self.length))
            revp = list(map(reverse_complement, probes))
            d = {
                "name": genome.name,
                "segment": genome.segment,
                "probe": revp,
                "substitution": genome.substitution,
                "score": int_scores,
            }
            df = pd.concat([df, pd.DataFrame(d)])
            if self.reverse_complement:
                d["probe"] = probes
                df = pd.concat([df, pd.DataFrame(d)])

        max_val = df.groupby(["name", "segment", "probe"]).max().reset_index().groupby(["name", "probe"]).sum()
        min_val = df.groupby(["name", "segment", "probe"]).min().reset_index().groupby(["name", "probe"]).sum()
        index = df["probe"].unique()
        columns = pd.MultiIndex.from_tuples(product(df["name"].unique(), ("min", "max")))
        return pd.DataFrame(
            np.log([min_val["score"], max_val["score"]])
            .reshape(2, len(self.original_genomes), len(index))
            .transpose(2, 1, 0)
            .reshape(len(index), -1), index=index, columns=columns)
