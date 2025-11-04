from datasets import load_dataset
import random

class MSADataset:
    def __init__(self, split="train", path="dotan1111/MSA-nuc-3-seq", max_len=150):
        """
        Loads the MSA dataset and filters out examples where the aligned
        sequence (MSA) length exceeds max_len.

        Args:
            split (str): dataset split to load ('train', 'validation', 'test')
            path (str): Hugging Face dataset path
            max_len (int): maximum allowed alignment length (in columns)
        """
        print(f"Loading {split} split from {path} ...")
        dataset = load_dataset(path, split=split)

        # Filter based on the length of the aligned MSA
        print(f"Filtering MSAs longer than {max_len} bases ...")
        dataset = dataset.filter(lambda ex: len(ex["MSA"]) <= max_len)

        self.dataset = dataset
        self.split = split
        self.size = len(dataset)
        self.max_len = max_len

        print(f"✅ Loaded {self.size} examples (max MSA length = {max_len})")

    def __len__(self):
        return self.size

    def get_row(self, i):
        """
        Returns a single example as (unaligned_sequences, reference_alignment)
        """
        row = self.dataset[i]
        seqs = list(row["unaligned_seqs"].values())  # list of sequences
        msa = row["MSA"]
        return seqs, msa

    def sample_row(self):
        """
        Returns a random example as (unaligned_sequences, reference_alignment)
        """
        i = random.randint(0, self.size - 1)
        return self.get_row(i)
