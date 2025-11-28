import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleRNN(nn.Module):
    def __init__(self, input_size, emb_size, hidden_size, class_size):
        """
        input_size  : taille du vocabulaire (vocab_size)
        emb_size    : taille de l'embedding
        hidden_size : dimension de l'état caché
        class_size  : nombre de classes (émotions)
        """
        super().__init__()
        self.hidden_size = hidden_size

        # 1. Embedding linéaire (de one-hot -> espace dense)
        self.i2e = nn.Linear(input_size, emb_size)

        # 2. Combine input embedding + hidden state
        self.i2h = nn.Linear(emb_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(emb_size + hidden_size, class_size)

        # Optionnel : normalisation pour stabilité
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_vec, hidden):
        """
        input_vec : [batch_size, vocab_size] (one-hot ou float)
        hidden    : [batch_size, hidden_size]
        """
        # Embedding du mot courant
        embedded = torch.tanh(self.i2e(input_vec))  # [batch, emb_size]

        # Concaténation input + hidden
        combined = torch.cat((embedded, hidden), dim=1)

        # Mise à jour de l’état caché
        hidden = torch.tanh(self.i2h(combined))

        # Calcul de la sortie
        output = self.softmax(self.i2o(combined))
        return output, hidden

    def init_hidden(self, batch_size=1, device="cpu"):
        return torch.zeros(batch_size, self.hidden_size, device=device)
