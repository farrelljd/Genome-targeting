import argparse
import pathlib

import pandas as pd


def get_parser():
    parser = argparse.ArgumentParser(description="calculate the overlap score between a genome and all length l probes")
    parser.add_argument('length', type=int, help='probe length')
    parser.add_argument('threads', type=int, help='number of openmp threads to use', default=1)
    parser.add_argument('-n', '--number-of-probes', type=int,
                        help='only consider the n most common subsequences in the genomes')
    parser.add_argument('--compare', action='store_true',
                        help='compare the scores of all pairs of genomes')
    parser.add_argument('--file', help='output file name', type=pathlib.Path)
    parser.add_argument('--write-excel', help='write an excel spreadsheet', action='store_true')
    parser.add_argument('-c', '--do-reverse-complement', action='store_true',
                        help='include the reverse complement of the genome in the score calculation')
    parser.add_argument('genomes', nargs="+",
                        help='one or more genbank records (.gb), fasta files (.fasta, .fna) or one-line files containing genomes (.gen)')
    return parser


def main():
    from genometargeting import power_four_score_factory, Genome, Computer

    parser = get_parser()
    args = parser.parse_args()

    gen_files = args.genomes
    length = args.length
    threads = args.threads
    n = args.number_of_probes
    compare_flag = args.compare
    file = args.file
    score_function = power_four_score_factory(length)
    do_rc = args.do_reverse_complement

    genomes = *map(Genome.read, gen_files),

    probe_info = f"{n} most common" if n is not None else f"{4**length} (all length {length} probes)"
    score_function_info = (
        f'{score_function.__name__} (i.e., the one used in Curk et al, PNAS (2020))'
        if score_function.__name__ == "power_four_score"
        else score_function.__name__
    )
    if file is None:
        output_file = "stdout"
    elif args.write_excel:
        output_file = file.with_suffix(".xlsx")
    else:
        output_file = file.with_suffix(".pkl")

    print(rf'''JOB INFO:
    number of threads: {threads}
    probe length: {length}
    number of probes: {probe_info}
    genomes: {gen_files}
    score function: {score_function_info}
    reverse complement?: {"yes" if do_rc else "no"}
    output file: {output_file}''', flush=True)

    compute = Computer(genomes, length, score_function, do_reverse_complement=do_rc)
    output = compute(threads, n, compare_flag)
    if file is None:
        print(output)
    elif args.write_excel:
        output.to_excel(output_file)
    else:
        output.to_pickle(output_file)


if __name__ == '__main__':
    main()
