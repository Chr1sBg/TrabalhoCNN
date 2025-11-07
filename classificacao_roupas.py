#integrantes: Christian, Gabriel Costa, Vitor, Yago

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

# Configuração
dataset_path = "./dataset"
classes = sorted(os.listdir(dataset_path))

print(f"Classes: {classes}")
for cls in classes:
    folder = os.path.join(dataset_path, cls)
    num_images = len([f for f in os.listdir(folder) if f.endswith(('.jpg', '.jpeg', '.png'))])
    print(f"{cls}: {num_images} imagens")

# Transformações
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64, 64)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Dataset
class CustomImageDataset(Dataset):
    def __init__(self, base_dir, classes, transform=None):
        self.transform = transform
        self.images, self.labels = [], []
        
        for idx, cls in enumerate(classes):
            folder = os.path.join(base_dir, cls)
            for img_name in os.listdir(folder):
                if img_name.endswith(('.jpg', '.jpeg', '.png')):
                    self.images.append(os.path.join(folder, img_name))
                    self.labels.append(idx)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = cv2.imread(self.images[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

# Arquitetura CNN
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 16 * 16)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Treinamento
def train_model(model, train_loader, criterion, optimizer, device, epochs):
    model.train()
    train_losses = []
    
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        train_losses.append(epoch_loss)
        print(f"Época [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Acurácia: {epoch_acc:.2f}%")
    
    return train_losses

# Avaliação
def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    return accuracy, all_preds, all_labels, cm

# K-Fold Cross Validation
def k_fold_cross_validation(dataset, k_splits, num_classes, epochs, batch_size, lr, device):
    kfold = KFold(n_splits=k_splits, shuffle=True, random_state=42)
    results = {
        'fold_accuracies': [],
        'predictions': [],
        'true_labels': [],
        'confusion_matrices': [],
        'train_losses': []
    }
    
    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        print(f"\n{'='*50}\nFOLD {fold+1}/{k_splits}\n{'='*50}\n")
        
        train_subsampler = Subset(dataset, train_ids)
        test_subsampler = Subset(dataset, test_ids)
        
        train_loader = DataLoader(train_subsampler, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_subsampler, batch_size=batch_size, shuffle=False)
        
        print(f"Treino: {len(train_ids)}, Teste: {len(test_ids)}\n")
        
        model = SimpleCNN(num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        train_losses = train_model(model, train_loader, criterion, optimizer, device, epochs)
        accuracy, preds, labels, cm = evaluate_model(model, test_loader, device)
        
        results['fold_accuracies'].append(accuracy)
        results['predictions'].append(preds)
        results['true_labels'].append(labels)
        results['confusion_matrices'].append(cm)
        results['train_losses'].append(train_losses)
        
        print(f"\nAcurácia do Fold {fold+1}: {accuracy*100:.2f}%")
        print(f"Loss final: {train_losses[-1]:.4f}")
    
    return results

# Visualização
def plot_results(results, classes):
    accuracies = [acc * 100 for acc in results['fold_accuracies']]
    
    print(f"\n{'='*60}\nRELATÓRIO FINAL\n{'='*60}\n")
    print(f"Acurácias: {[f'{acc:.2f}%' for acc in accuracies]}")
    print(f"Média: {np.mean(accuracies):.2f}% ± {np.std(accuracies):.2f}%")
    print(f"Melhor Fold: {np.argmax(accuracies)+1} ({max(accuracies):.2f}%)")
    
    best_fold = np.argmax(results['fold_accuracies'])
    unique_labels = np.unique(results['true_labels'][best_fold])
    labels_in_fold = [classes[i] for i in unique_labels]
    
    print(f"\nRelatório - Melhor Fold (Fold {best_fold+1}):")
    print("-" * 60)
    print(classification_report(
        results['true_labels'][best_fold],
        results['predictions'][best_fold],
        target_names=labels_in_fold,
        labels=unique_labels,
        digits=4
    ))
    
    # Gráficos
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Acurácia por fold
    axes[0].bar(range(1, len(accuracies)+1), accuracies, color='steelblue')
    axes[0].set_xlabel('Fold')
    axes[0].set_ylabel('Acurácia (%)')
    axes[0].set_title('Acurácia por Fold')
    axes[0].set_ylim([0, 100])
    
    # Loss de treino
    for i, losses in enumerate(results['train_losses']):
        axes[1].plot(losses, label=f'Fold {i+1}')
    axes[1].set_xlabel('Época')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Loss de Treino')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('resultados.png', dpi=150, bbox_inches='tight')
    print(f"\nGráficos salvos em 'resultados.png'")
    
    # Matriz de confusão
    cm = results['confusion_matrices'][best_fold]
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels_in_fold, yticklabels=labels_in_fold)
    plt.title(f'Matriz de Confusão - Fold {best_fold+1}')
    plt.ylabel('Real')
    plt.xlabel('Predito')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
    print(f"Matriz de confusão salva em 'confusion_matrix.png'")

# Main
if __name__ == "__main__":
    print("\n" + "="*60)
    print("CLASSIFICAÇÃO DE ROUPAS - CNN")
    print("="*60 + "\n")
    
    dataset = CustomImageDataset(dataset_path, classes, transform)
    print(f"Total de imagens: {len(dataset)}\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Dispositivo: {device}\n")
    
    results = k_fold_cross_validation(
        dataset=dataset,
        k_splits=5,
        num_classes=len(classes),
        epochs=15,
        batch_size=8,
        lr=0.001,
        device=device
    )
    
    plot_results(results, classes)
    print("\n" + "="*60)
    print("CONCLUÍDO!")
    print("="*60)
