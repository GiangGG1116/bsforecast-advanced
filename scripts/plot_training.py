import json
import argparse
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", required=True)
    ap.add_argument("--out", default="artifacts/training_history.png")
    args = ap.parse_args()

    with open(args.history, "r", encoding="utf-8") as f:
        hist = json.load(f)

    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(hist.get("loss", []), label="Training Loss")
    if "val_loss" in hist: plt.plot(hist["val_loss"], label="Validation Loss")
    plt.title("Loss = Weighted Huber + α·sMAPE(core)"); plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()

    plt.subplot(1,2,2)
    if "mape_core" in hist: plt.plot(hist["mape_core"], label="Train MAPE Core")
    if "val_mape_core" in hist: plt.plot(hist["val_mape_core"], label="Val MAPE Core")
    plt.title("MAPE Core (3 primary cols)"); plt.xlabel("Epoch"); plt.ylabel("MAPE (fraction)"); plt.legend()

    plt.tight_layout(); plt.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"Saved: {args.out}")

if __name__ == "__main__":
    main()
