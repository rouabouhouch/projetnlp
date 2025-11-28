import torch
from torch.utils.data import Dataset
from data_processing import tokenizer, one_hot, Vocab


class EmotionDataset(Dataset):
    """
    Dataset PyTorch pour générer le codage one-hot d'une phrase pour un RNN.
    Chaque élément : (X, y)
      - X : matrice [sentence_length, vocab_size]
      - y : entier correspondant à la classe d'émotion
    """

    def __init__(self, texts, emotions, vocab, classes, max_len=30):
        """
        texts : liste des phrases
        emotions : liste des étiquettes
        vocab : Vocab (mots -> id)
        classes : Vocab (émotions -> id)
        max_len : longueur fixe des séquences (padding / tronquage)
        """
        self.texts = texts
        self.emotions = emotions
        self.vocab = vocab
        self.classes = classes
        self.vocab_size = len(vocab)
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def _pad_or_truncate(self, encoded):
        """Ajuste la longueur de la phrase à max_len avec <pad>."""
        if len(encoded) > self.max_len:
            return encoded[:self.max_len]
        pad_id = self.vocab["<pad>"]
        return encoded + [pad_id] * (self.max_len - len(encoded))

    def __getitem__(self, idx):
        # 1. Récupérer phrase + label
        sentence = self.texts[idx]
        label = self.emotions[idx]

        # 2. Tokeniser + encoder
        tokens = tokenizer(sentence)
        encoded = self.vocab(tokens)
        encoded = self._pad_or_truncate(encoded)

        # 3. Encodage one-hot (shape [sentence_len, vocab_size])
        one_hot_matrix = one_hot(encoded, num_classes=self.vocab_size)

        # 4. Conversion en tenseurs
        X = torch.tensor(one_hot_matrix, dtype=torch.float32)
        y = torch.tensor(self.classes[label], dtype=torch.long)

        return X, y
