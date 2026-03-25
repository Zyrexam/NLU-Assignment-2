import os
import random
import string
import torch
import torch.nn as nn
import torch.optim as optim

# ====================== CONFIG ======================
TRAINING_FILE = 'Problem_2/TrainingNames.txt'
HIDDEN_SIZE = 128
NUM_ITERS = 25000
LEARNING_RATE = 0.001
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_EVAL_SAMPLES = 1000
NUM_SAMPLE_NAMES = 100

all_letters = string.ascii_letters + " '-"
n_letters = len(all_letters) + 1

# ====================== DATA ======================
def read_names():
    with open(TRAINING_FILE, 'r', encoding='utf-8') as f:
        names = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(names)} Indian names.\n")
    return names

def name_to_tensor(name):
    tensor = torch.zeros(len(name) + 1, 1, n_letters)
    for i, ch in enumerate(name):
        if ch in all_letters:
            tensor[i][0][all_letters.find(ch)] = 1
    tensor[-1][0][-1] = 1  # EOS
    return tensor

# ====================== MODELS ======================
class VanillaRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(n_letters, HIDDEN_SIZE)
        self.fc = nn.Linear(HIDDEN_SIZE, n_letters)

    def forward(self, x, h):
        out, h = self.rnn(x, h)
        out = self.fc(out)
        return out, h

    def init_hidden(self):
        return torch.zeros(1, 1, HIDDEN_SIZE, device=DEVICE)


class BLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(n_letters, HIDDEN_SIZE, bidirectional=True)
        self.fc = nn.Linear(HIDDEN_SIZE * 2, n_letters)

    def forward(self, x, h):
        out, h = self.lstm(x, h)
        out = self.fc(out)
        return out, h

    def init_hidden(self):
        return (torch.zeros(2, 1, HIDDEN_SIZE, device=DEVICE),
                torch.zeros(2, 1, HIDDEN_SIZE, device=DEVICE))


class RNNWithAttention(nn.Module):
    def __init__(self):
        super().__init__()

        self.input_proj = nn.Linear(n_letters, HIDDEN_SIZE)  
        self.gru = nn.GRU(HIDDEN_SIZE, HIDDEN_SIZE)

        self.attn_combine = nn.Linear(HIDDEN_SIZE + n_letters, HIDDEN_SIZE)
        self.fc = nn.Linear(HIDDEN_SIZE, n_letters)

    def forward(self, x, h, enc_out):

        if enc_out.size(0) == 0:
            x_proj = self.input_proj(x)   
            out, h = self.gru(x_proj, h)

        else:
            attn_weights = torch.softmax(
                torch.bmm(h.transpose(0,1), enc_out.transpose(0,1).transpose(1,2)), dim=2
            )

            context = torch.bmm(attn_weights, enc_out.transpose(0,1))

            combined = torch.cat((x, context), dim=2)
            combined = self.attn_combine(combined).tanh()

            out, h = self.gru(combined, h)

        out = self.fc(out)
        return out, h

    def init_hidden(self):
        return torch.zeros(1, 1, HIDDEN_SIZE, device=DEVICE)


# ====================== TRAINING ======================
def train(model, names, model_name):
    model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    print(f"Training {model_name} ...")

    for it in range(1, NUM_ITERS + 1):
        name = random.choice(names)
        inp = name_to_tensor(name)[:-1].to(DEVICE)
        tgt = name_to_tensor(name)[1:].to(DEVICE)

        h = model.init_hidden()
        loss = 0.0
        enc_out = torch.zeros(0, 1, HIDDEN_SIZE, device=DEVICE)

        for i in range(inp.size(0)):
            if isinstance(model, RNNWithAttention):
                output, h = model(inp[i].unsqueeze(0), h, enc_out)
                enc_out = torch.cat([enc_out, h.detach()], dim=0)
            else:
                output, h = model(inp[i].unsqueeze(0), h)

            loss += criterion(output.squeeze(0), tgt[i].argmax().unsqueeze(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if it % 5000 == 0:
            print(f"  Iter {it:5d} | Loss: {loss.item()/inp.size(0):.4f}")

    torch.save(model.state_dict(), f"Problem_2/{model_name}_model.pth")
    print(f"{model_name} training finished and saved!\n")


# ====================== SAMPLING ======================
def generate_name(model, max_len=25):
    model.eval()
    with torch.no_grad():
        start = random.choice(string.ascii_uppercase)
        x = torch.zeros(1, 1, n_letters, device=DEVICE)
        x[0, 0, all_letters.find(start)] = 1

        h = model.init_hidden()
        name = start
        enc_out = torch.zeros(0, 1, HIDDEN_SIZE, device=DEVICE)

        for _ in range(max_len):
            if isinstance(model, RNNWithAttention):
                output, h = model(x, h, enc_out)
                enc_out = torch.cat([enc_out, h.detach()], dim=0)
            else:
                output, h = model(x, h)

            idx = output[0, 0].argmax()
            if idx == n_letters - 1:
                break
            ch = all_letters[idx]
            name += ch

            x = torch.zeros(1, 1, n_letters, device=DEVICE)
            x[0, 0, idx] = 1
        return name


# ====================== EVALUATION & SAVING ======================
def evaluate_and_save(model, names, model_name):
    print(f"Evaluating {model_name}...")

    generated = [generate_name(model) for _ in range(NUM_EVAL_SAMPLES)]
    generated = [g for g in generated if len(g) > 2]

    train_set = set(names)
    novelty = sum(1 for g in generated if g not in train_set) / len(generated) * 100 if generated else 0
    diversity = len(set(generated)) / len(generated) * 100 if generated else 0

    print(f"   Novelty Rate : {novelty:.2f}%")
    print(f"   Diversity    : {diversity:.2f}%")
    print(f"   Valid names  : {len(generated)}\n")

    # Save results
    os.makedirs("Problem_2/results", exist_ok=True)

    # Save 100 sample names
    with open(f"Problem_2/results/{model_name}_samples.txt", "w", encoding="utf-8") as f:
        f.write(f"Generated Samples from {model_name} Model\n")
        f.write("="*50 + "\n")
        for name in generated[:NUM_SAMPLE_NAMES]:
            f.write(name + "\n")

    # Save metrics
    with open(f"Problem_2/results/{model_name}_metrics.txt", "w", encoding="utf-8") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Novelty Rate: {novelty:.2f}%\n")
        f.write(f"Diversity: {diversity:.2f}%\n")
        f.write(f"Valid names generated: {len(generated)}\n")

    return novelty, diversity


# ====================== MAIN ======================
if __name__ == "__main__":
    names = read_names()

    models = {
        "vanilla": VanillaRNN(),
        "blstm": BLSTM(),
        "attention": RNNWithAttention()
    }

    results = {}

    for mname, model in models.items():
        params = sum(p.numel() for p in model.parameters())
        print(f"{mname} has {params:,} trainable parameters\n")

        train(model, names, mname)

        print(f"10 Sample names from {mname}:")
        for _ in range(10):
            print("   ", generate_name(model))
        print("-" * 60)

        nov, div = evaluate_and_save(model, names, mname)
        results[mname] = (nov, div)

    # Final Comparison Table
    print("\n" + "="*65)
    print("FINAL COMPARISON TABLE")
    print("="*65)
    print(f"{'Model':<12} {'Novelty Rate (%)':<20} {'Diversity (%)':<15}")
    print("-"*65)
    for m, (n, d) in results.items():
        print(f"{m:<12} {n:<20.2f} {d:<15.2f}")
    print("="*65)

    # Save final comparison
    with open("Problem_2/results/final_comparison.txt", "w", encoding="utf-8") as f:
        f.write("FINAL COMPARISON TABLE\n")
        f.write("="*65 + "\n")
        f.write(f"{'Model':<12} {'Novelty Rate (%)':<20} {'Diversity (%)':<15}\n")
        f.write("-"*65 + "\n")
        for m, (n, d) in results.items():
            f.write(f"{m:<12} {n:<20.2f} {d:<15.2f}\n")
        f.write("="*65 + "\n")