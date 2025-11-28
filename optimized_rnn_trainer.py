# ==========================================================
#  Optimized RNN Trainer — TF-IDF Filtering + Class Balancing
# ==========================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from collections import Counter
from data_processing import load_file, tokenizer, yield_tokens, build_vocab_from_iterator
from dataset_nlp import EmotionDataset
from rnn_model import SimpleRNN


class OptimizedRNNTrainer:
    def __init__(self,
                 train_path="./dataset/train.txt",
                 test_path="./dataset/test.txt",
                 emb_size=64,
                 hidden_size=64,
                 max_len=15,
                 batch_size=32,
                 lr=0.002,
                 epochs=10,
                 min_tf=1,
                 max_df_ratio=0.97):
        self.train_path = train_path
        self.test_path = test_path
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.max_len = max_len
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.min_tf = min_tf
        self.max_df_ratio = max_df_ratio
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----------------------------
    # 1. Softer TF-IDF-like filtering
    # ----------------------------
    def filter_vocab_by_tfidf(self, texts):
        N = len(texts)
        tf, df = Counter(), Counter()
        for s in texts:
            tokens = tokenizer(s)
            unique = set(tokens)
            tf.update(tokens)
            for t in unique:
                df[t] += 1
        return [
            tok for tok, count in tf.items()
            if count >= self.min_tf and (df[tok] / N) <= self.max_df_ratio
        ]

    # ----------------------------
    # 2. Prepare data
    # ----------------------------
    def prepare_data(self):
        train_text, train_emotion = load_file(self.train_path)
        test_text, test_emotion = load_file(self.test_path)

        # Apply TF-IDF filtering
        filtered_tokens = self.filter_vocab_by_tfidf(train_text)
        print(f" Filtered vocab size: {len(filtered_tokens)}")

        self.vocab = build_vocab_from_iterator(filtered_tokens, specials=["<pad>", "<unk>"])
        self.classes = build_vocab_from_iterator(yield_tokens(train_emotion))

        self.train_dataset = EmotionDataset(train_text, train_emotion, self.vocab, self.classes, max_len=self.max_len)
        self.test_dataset = EmotionDataset(test_text, test_emotion, self.vocab, self.classes, max_len=self.max_len)

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

    # ----------------------------
    # 3. Build model and loss
    # ----------------------------
    def build_model(self):
        # Class balancing
        text, emotion = load_file(self.train_path)
        class_counts = Counter(emotion)
        total = sum(class_counts.values())
        weights = [total / class_counts[c] for c in self.classes.mapping.keys()]
        weights = torch.tensor(weights, dtype=torch.float32)

        self.rnn = SimpleRNN(len(self.vocab), self.emb_size, self.hidden_size, len(self.classes)).to(self.device)
        self.criterion = nn.NLLLoss(weight=weights.to(self.device))
        self.optimizer = optim.Adam(self.rnn.parameters(), lr=self.lr)

    # ----------------------------
    # 4. Training loop
    # ----------------------------
    def train(self):
        self.prepare_data()
        self.build_model()
        print("\n Training started...\n")

        for epoch in range(1, self.epochs + 1):
            self.rnn.train()
            total_loss, correct, total = 0, 0, 0

            for X, y in self.train_loader:
                X, y = X.to(self.device), y.to(self.device)
                batch_size, seq_len, _ = X.shape

                hidden = self.rnn.init_hidden(batch_size, device=self.device)
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

            epoch_loss = total_loss / len(self.train_loader)
            epoch_acc = 100 * correct / total
            print(f"Epoch {epoch:02d} | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}%")

        print("\n Training finished.\n")

    # ----------------------------
    # 5. Evaluation
    # ----------------------------
    def evaluate(self):
        self.rnn.eval()
        test_correct, test_total = 0, 0
        with torch.no_grad():
            for X, y in self.test_loader:
                X, y = X.to(self.device), y.to(self.device)
                hidden = self.rnn.init_hidden(X.size(0), device=self.device)
                for t in range(X.size(1)):
                    output, hidden = self.rnn(X[:, t, :], hidden)
                pred = output.argmax(dim=1)
                test_correct += (pred == y).sum().item()
                test_total += y.size(0)

        acc = 100 * test_correct / test_total
        print(f"\ Test accuracy: {acc:.2f}%")
        return acc
