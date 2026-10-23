import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# 1. Generacion de datos
def generar_datos(n_puntos=600, seq_len=50):
    t = np.linspace(0, 50, n_puntos)
    serie = np.sin(t)
    X, y = [], []
    for i in range(len(serie) - seq_len):
        X.append(serie[i:i + seq_len])
        y.append(serie[i + seq_len])
    X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)
    return X, y

# 2. Modelo RNN Simple
class SimpleRNNModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=16, output_size=1):
        super(SimpleRNNModel, self).__init__()
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out_final = out[:, -1, :]
        return self.fc(out_final)

# 3. Modelo LSTM
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=16, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, (hn, cn) = self.lstm(x)
        out_final = out[:, -1, :]
        return self.fc(out_final)

SEQ_LEN = 50
HIDDEN_SIZE = 16
EPOCHS = 100
LR = 0.01

X, y = generar_datos(n_puntos=600, seq_len=SEQ_LEN)
print("=" * 50)
print(f"Dimensiones de X (3D): {X.shape}")
print(f"Dimensiones de y:      {y.shape}")
print("=" * 50)

rnn_model = SimpleRNNModel(1, HIDDEN_SIZE, 1)
lstm_model = LSTMModel(1, HIDDEN_SIZE, 1)

print(f"Parametros RNN:  {sum(p.numel() for p in rnn_model.parameters())}")
print(f"Parametros LSTM: {sum(p.numel() for p in lstm_model.parameters())}")
print("=" * 50)

criterion = nn.MSELoss()
opt_rnn = optim.Adam(rnn_model.parameters(), lr=LR)
opt_lstm = optim.Adam(lstm_model.parameters(), lr=LR)

losses_rnn, losses_lstm = [], []

print("Entrenando modelos...")
for epoch in range(EPOCHS):
    rnn_model.train()
    opt_rnn.zero_grad()
    out_rnn = rnn_model(X)
    l_rnn = criterion(out_rnn, y)
    l_rnn.backward()
    opt_rnn.step()
    losses_rnn.append(l_rnn.item())

    lstm_model.train()
    opt_lstm.zero_grad()
    out_lstm = lstm_model(X)
    l_lstm = criterion(out_lstm, y)
    l_lstm.backward()
    opt_lstm.step()
    losses_lstm.append(l_lstm.item())

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Loss RNN: {l_rnn.item():.6f} | Loss LSTM: {l_lstm.item():.6f}")

plt.figure(figsize=(10, 5))
plt.plot(losses_rnn, label="RNN Simple", color="red", linestyle="--")
plt.plot(losses_lstm, label="LSTM", color="blue")
plt.title(f"Convergencia: RNN Simple vs LSTM (SEQ_LEN={SEQ_LEN})")
plt.xlabel("Epocas")
plt.ylabel("MSE Loss")
plt.legend()
plt.grid(True)
plt.savefig("perdida_comparativa.png")
plt.show()
