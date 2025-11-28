# data_processing.py


# -------------------------
# 1. Load file
# -------------------------
def load_file(file_path, sep=";"):
    texts = []
    emotions = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  # on saute les lignes vides
            parts = line.split(sep)
            if len(parts) >= 2:
                text = parts[0].strip()
                emotion = parts[1].strip()
                texts.append(text)
                emotions.append(emotion)
    return texts, emotions


# -------------------------
# 2. Tokenizer
# -------------------------
def tokenizer(sentence):
    """
    Découpe une phrase en tokens simples (sans regex, sans libs).
    """
    tokens = []
    current = ""

    for ch in sentence:
        if ch.isalnum() or ch == "'":
            current += ch.lower()
        else:
            if current:
                tokens.append(current)
                current = ""
            if ch in [".", ",", "!", "?", ";", ":"]:
                tokens.append(ch)

    if current:
        tokens.append(current)

    return tokens


# -------------------------
# 3. Yield tokens (flat generator)
# -------------------------
def yield_tokens(texts):
    """
    Génère tous les tokens du corpus, un par un.
    list(yield_tokens(texts)) -> ['i','feel','good',...]
    """
    for s in texts:
        for tok in tokenizer(s):
            yield tok


# -------------------------
# 4. Vocab class
# -------------------------
class Vocab:
    """
    Petit wrapper pour rendre le vocabulaire 'callable' comme dans le notebook du prof.
    - vocab(tokens) -> [ids]
    - vocab['word'] -> id
    - len(vocab) -> nombre total de tokens
    """
    def __init__(self, mapping):
        self.mapping = mapping

    def __call__(self, tokens):
        """Transforme une liste de mots en liste d'IDs."""
        return [self.mapping.get(tok, self.mapping["<unk>"]) for tok in tokens]

    def __getitem__(self, key):
        """Permet d'accéder à l'index d'un mot."""
        return self.mapping[key]

    def __len__(self):
        return len(self.mapping)


# -------------------------
# 5. Build vocab from iterator
# -------------------------
def build_vocab_from_iterator(data, specials=None):
    """
    Construit un vocabulaire {token: id} adaptable :
    - si data est une liste de tokens => construit un vocabulaire directement
    - si data est un itérateur de listes (ex: yield_tokens) => aplatit et construit
    - specials : tokens spéciaux facultatifs (["<pad>", "<unk>"])
    """

    if specials is None:
        specials = []

    mapping = {}
    idx = 0

    # Ajouter d'abord les tokens spéciaux
    for sp in specials:
        mapping[sp] = idx
        idx += 1

    # Gérer les différents types d'entrée
    for item in data:
        if isinstance(item, (list, tuple)):
            for tok in item:
                if tok not in mapping:
                    mapping[tok] = idx
                    idx += 1
        else:
            if item not in mapping:
                mapping[item] = idx
                idx += 1

    # Retourner un vocabulaire "callable"
    return Vocab(mapping)

# -------------------------
# 6. One-hot encoding (sans torch)
# -------------------------
def one_hot(indices, num_classes):
    """
    Crée une représentation one-hot sans utiliser PyTorch.
    indices : liste ou séquence d'entiers (IDs des mots)
    num_classes : taille du vocabulaire (nombre total de classes)
    Retourne : liste de listes (une par mot)
    Exemple :
        one_hot([1, 3], 5) ->
        [[0,1,0,0,0],
         [0,0,0,1,0]]
    """
    one_hot_vectors = []

    for idx in indices:
        # créer un vecteur plein de zéros
        vector = [0] * num_classes
        # placer un 1 à la bonne position
        if 0 <= idx < num_classes:
            vector[idx] = 1
        one_hot_vectors.append(vector)

    return one_hot_vectors