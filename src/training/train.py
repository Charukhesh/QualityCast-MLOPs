import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import f1_score, precision_score, accuracy_score
import mlflow
import mlflow.pytorch
import matplotlib.pyplot as plt
import logging
import gc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MLP Architecture
class CastingMLP(nn.Module):
    def __init__(self, input_size):
        super(CastingMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1), # Binary output
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

def train_quality_model():
    try:
        mlflow.set_tracking_uri("http://mlflow:5000")
        mlflow.set_experiment("Casting_Quality_PyTorch")

        import os
        print("MLFLOW URI:", os.environ.get("MLFLOW_TRACKING_URI"))
        import socket
        print("Resolving mlflow:", socket.gethostbyname("mlflow"))

        # Hyperparameters
        LR = 0.0005
        BATCH_SIZE = 16
        EPOCHS = 15
        IMG_SIZE = 64
        INPUT_SIZE = IMG_SIZE * IMG_SIZE

        # Data Prep
        transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        full_dataset = datasets.ImageFolder(root="/opt/airflow/data/processed", transform=transform)
        
        # Split 80/20
        train_size = int(0.8 * len(full_dataset))
        test_size = len(full_dataset) - train_size
        train_ds, val_ds = torch.utils.data.random_split(full_dataset, [train_size, test_size])

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

        # Model Setup
        device = "cpu"
        model = CastingMLP(INPUT_SIZE).to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=LR)

        best_val_f1 = 0.0
        best_model_path = "best_model_weights.pth"
        with mlflow.start_run():
            mlflow.log_param("lr", LR)
            mlflow.log_param("batch_size", BATCH_SIZE)
            mlflow.log_param("epochs", EPOCHS)
            mlflow.log_param("img_size", IMG_SIZE)

            for epoch in range(EPOCHS):
                # TRAINING PHASE
                model.train()
                train_preds, train_true = [], []
                total_train_loss = 0

                for images, labels in train_loader:
                    labels = labels.float().unsqueeze(1)
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_train_loss += loss.item()
                    train_preds.extend((outputs > 0.5).int().numpy())
                    train_true.extend(labels.int().numpy())

                # VALIDATION PHASE
                model.eval()
                val_preds, val_true = [], []
                total_val_loss = 0
                with torch.no_grad():
                    for images, labels in val_loader:
                        labels = labels.float().unsqueeze(1)
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                        
                        total_val_loss += loss.item()
                        val_preds.extend((outputs > 0.5).int().numpy())
                        val_true.extend(labels.int().numpy())

                # Calculate Metrics per Epoch
                metrics = {
                    "train_loss": total_train_loss / len(train_loader),
                    "train_acc": accuracy_score(train_true, train_preds),
                    "train_f1": f1_score(train_true, train_preds),
                    "train_prec": precision_score(train_true, train_preds, zero_division=0),
                    "val_loss": total_val_loss / len(val_loader),
                    "val_acc": accuracy_score(val_true, val_preds),
                    "val_f1": f1_score(val_true, val_preds),
                    "val_prec": precision_score(val_true, val_preds, zero_division=0)
                }

                for k, v in metrics.items():
                    mlflow.log_metric(k, v, step=epoch)

                logger.info(f"Epoch {epoch}: Val Acc={metrics['val_acc']:.4f}, Val F1={metrics['val_f1']:.4f}")

                current_val_f1 = f1_score(val_true, val_preds)
                if current_val_f1 > best_val_f1:
                    best_val_f1 = current_val_f1
                    torch.save(model.state_dict(), best_model_path)
                    logger.info(f"New Best Model found at Epoch {epoch} with F1: {best_val_f1:.4f}")

                del train_preds, train_true, val_preds, val_true
                gc.collect()

            # Architecture TXT
            with open("model_architecture.txt", "w") as f:
                f.write(str(model))
            mlflow.log_artifact("model_architecture.txt")

            # Log model weights
            mlflow.log_artifact(best_model_path)

            import json
            with open("metrics.json", "w") as f:
                json.dump(metrics, f)

            # Prediction Visualization Plot
            images, labels = next(iter(val_loader))
            outputs = model(images)
            preds = (outputs > 0.5).int()
            
            fig, axes = plt.subplots(1, 5, figsize=(15, 3))
            classes = {0: "OK", 1: "DEF"}
            for i in range(5):
                img = images[i].permute(1, 2, 0).numpy() # Convert to HWC for plt
                axes[i].imshow(img, cmap='gray')
                axes[i].set_title(f"P:{classes[preds[i].item()]} / T:{classes[labels[i].item()]}")
                axes[i].axis("off")
            plt.savefig("predictions.png")
            mlflow.log_artifact("predictions.png")

            # Full PyTorch Model
            mlflow.pytorch.log_model(model, "quality_inspection_model")

            logger.info("Training complete. All artifacts and epoch metrics logged.")

    except Exception as e:
        logger.error(f"Training Failed: {e}")
        raise e

if __name__ == "__main__":
    train_quality_model()