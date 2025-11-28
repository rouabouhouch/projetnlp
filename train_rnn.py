import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data_processing import load_file, yield_tokens, build_vocab_from_iterator
from dataset_nlp import EmotionDataset
from rnn_model import SimpleRNN


class RNNTrainer:
    """
    Classe utilitaire pour entraîner et évaluer un réseau RNN sur le dataset d'émotions.
    """

    def __init__(self,
                 train_path="./dataset/train.txt",
                 test_path=None,
                 emb_size=64,
                 hidden_size=128,
                 max_len=15,
                 batch_size=32,
                 lr=0.002,
                 epochs=10):
        self.train_path = train_path
        self.test_path = test_path
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.max_len = max_len
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # placeholders
        self.vocab = None
        self.classes = None
        self.rnn = None
        self.criterion = None
        self.optimizer = None
        self.train_loader = None

    # ---------------------
    # Étape 1 — Préparer les données
    # ---------------------
    def prepare_data(self):
        text, emotion = load_file(self.train_path)
        self.vocab = build_vocab_from_iterator(yield_tokens(text), specials=["<pad>", "<unk>"])
        self.classes = build_vocab_from_iterator(yield_tokens(emotion))
        dataset = EmotionDataset(text, emotion, self.vocab, self.classes, max_len=self.max_len)
        self.train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        print(f" Data prepared: {len(dataset)} samples, vocab={len(self.vocab)}, classes={len(self.classes)}")

    # ---------------------
    # Étape 2 — Construire le modèle
    # ---------------------
    def build_model(self):
        vocab_size = len(self.vocab)
        class_size = len(self.classes)
        self.rnn = SimpleRNN(vocab_size, self.emb_size, self.hidden_size, class_size).to(self.device)
        self.criterion = nn.NLLLoss()
        self.optimizer = optim.Adam(self.rnn.parameters(), lr=self.lr)
        print(f" Model initialized on {self.device}")

    # ---------------------
    # Étape 3 — Entraîner
    # ---------------------
    def train(self):
        self.prepare_data()
        self.build_model()

        for epoch in range(1, self.epochs + 1):
            self.rnn.train()
            total_loss, correct, total = 0, 0, 0

            for X, y in self.train_loader:
                X, y = X.to(self.device), y.to(self.device)
                batch_size, seq_len, _ = X.shape
                hidden = self.rnn.init_hidden(batch_size, device=self.device)

                # Entrée séquentielle
                for t in range(seq_len):
                    output, hidden = self.rnn(X[:, t, :], hidden)

                loss = self.criterion(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)

            acc = 100 * correct / total
            avg_loss = total_loss / len(self.train_loader)
            print(f"Epoch {epoch:02d} | Loss: {avg_loss:.4f} | Accuracy: {acc:.2f}%")

        print(" Training complete.")

    # ---------------------
    # Étape 4 — Évaluation
    # ---------------------
    def evaluate(self, test_path=None):
        if test_path is None:
            test_path = self.test_path
        if test_path is None:
            print("⚠️ No test file provided.")
            return

        text, emotion = load_file(test_path)
        dataset = EmotionDataset(text, emotion, self.vocab, self.classes, max_len=self.max_len)
        loader = DataLoader(dataset, batch_size=self.batch_size)

        self.rnn.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X, y in loader:
                X, y = X.to(self.device), y.to(self.device)
                hidden = self.rnn.init_hidden(X.size(0), device=self.device)
                for t in range(X.size(1)):
                    output, hidden = self.rnn(X[:, t, :], hidden)
                pred = output.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)

        acc = 100 * correct / total
        print(f" Test accuracy: {acc:.2f}%")
        return acc
