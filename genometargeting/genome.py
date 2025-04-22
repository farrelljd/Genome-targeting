import dataclasses
import re

import numpy as np
from joblib import Memory

from .utils import transcribe, most_common, get_substrings

location = './cachedir'
memory = Memory(location, verbose=0)

INVALID_REGEX = re.compile(r'''
(?P<invalid>[^ACGTRYMKSWHBVDN])
''', re.X)
UNCERTAIN_REGEX = re.compile(r'''
(?P<uncertain>[RYMKSWHBVDN])
''', re.X)


@dataclasses.dataclass
class Genome:
    string: str = dataclasses.field(repr=False)
    name: str
    description: str = ""
    segment: int = 0
    substitution: int = 0

    @classmethod
    def read(cls, gen_file, filetype=None):
        readers = {'gen': cls.read_gen,
                   'gb': cls.read_genbank,
                   'fasta': cls.read_fasta,
                   'fna': cls.read_fasta}
        if filetype is None:
            filetype = gen_file.strip().split('.')[-1]
        try:
            return readers[filetype](gen_file)
        except IndexError:
            raise ValueError("only genbank records (.gb) and single-line files (.gen) are supported")

    @classmethod
    def read_gen(cls, gen_file, name=None):
        genome_str = open(gen_file, 'r').readline().strip().upper()
        if name is None:
            name = gen_file.split('/')[-1].split('.')[0]
        return Genome(genome_str, name)

    @classmethod
    def read_genbank(cls, genbank_file, name=None):
        with open(genbank_file, 'r') as f:
            line = "\n"
            genome_str = ""
            while f.readline():
                while line.strip().upper() != "ORIGIN":
                    line = f.readline()
                line = f.readline()
                while line.strip() != "//":
                    genome_str += "".join(line.strip().split()[1:]).upper()
                    line = f.readline()
                f.readline()
                genome_str += '\n'
        genome_str = genome_str.strip()
        if name is None:
            name = genbank_file.split('/')[-1].split('.')[0]
        return Genome(genome_str, name)

    @classmethod
    def read_fasta(cls, gen_file, name=None):
        blocks = "".join(open(gen_file, 'r').readlines()).split(">")[1:]
        if len(blocks) > 1:
            print("multiple entries found. choosing the first entry.")
        lines = blocks[0].splitlines()
        genome_str = "".join(line.strip().upper() for line in lines[1:])
        description = f'{lines[0].strip()}'
        if name is None:
            name = description.split(" ")[0]
        print(f"processed entry >{description}")
        return Genome(genome_str, name, description)

    def __len__(self):
        return len(self.string.replace('\n', ''))

    def shape(self):
        return tuple(map(len, self.string.split('\n')))

    def __hash__(self):
        return hash(self.string)

    def __getitem__(self, item):
        return self.string[item]

    def __eq__(self, other):
        return self.string == other.string

    def __format__(self, format_spec):
        return format(self.name, format_spec)

    def transcribe(self, length, do_reverse_complement=True):
        _transcribe = memory.cache(transcribe)
        gu, gl = _transcribe(self, length, do_reverse_complement).T
        return np.copy(gu), np.copy(gl)

    def most_common(self, length, n):
        _most_common = memory.cache(most_common)
        return _most_common(self, length, n)

    def substrings_iter(self, length: int):
        for i, substring in enumerate(get_substrings(self.string, length)):
            if isinstance(substring, str):
                yield Genome(substring, self.name, self.description, segment=i, substitution=0)
            else:
                for j, ss in enumerate(substring):
                    yield Genome(ss, self.name, self.description, segment=i, substitution=j)
