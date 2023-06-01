import argparse
import copy
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from networks import Baseline, CustomInception, ResInception


def parse_option():
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--model_name", type=str, default="Baseline")
    parser.add_argument("--model_path", type=str, default=None)
    return parser.parse_args()


def save_learning_curves(loss, accuracy):
    phases = loss.keys()

    metrics = ["Loss", "Accuracy"]
    figs = {x: plt.figure() for x in metrics}
    axes = {x: figs[x].subplots() for x in metrics}

    for phase in phases:
        axes["Loss"].plot(loss[phase], label=phase)
        axes["Accuracy"].plot(accuracy[phase], label=phase)

    for metric in metrics:
        axes[metric].set_xlabel("Epoch")
        axes[metric].set_ylabel(metric)
        axes[metric].legend()
        figs[metric].savefig(metric + ".png")
        print(f"{metric} curve is saved as {metric}.png")


def train_model(
    model,
    criterion,
    optimizer,
    dataloaders,
    device,
    dataset_sizes,
    num_epochs=25,
    save_path="./models/model.pth",
):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    phases = ["train", "val"]

    losses = {x: [] for x in phases}
    accuracies = {x: [] for x in phases}

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("-" * 10)

        for phase in phases:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            losses[phase].append(epoch_loss)
            accuracies[phase].append(epoch_acc.item())

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # deep copy the model
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(
        f"Training complete in \
            {time_elapsed // 60:.0f}m \
            {time_elapsed % 60:.0f}s"
    )
    print(f"Best val Acc: {best_acc:4f}")

    save_learning_curves(losses, accuracies)

    # load best model weights
    model.load_state_dict(best_model_wts)
    torch.save(best_model_wts, save_path)
    print(f"Best model is saved as '{save_path}'")
    return model


def predict(model, dataloader, device):
    model.eval()

    pred_list = []
    true_list = []
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            pred_list.append(preds.cpu())
            true_list.append(labels.cpu())

    pred_list = torch.cat(pred_list)
    true_list = torch.cat(true_list)
    print(f"Accuracy: {accuracy_score(true_list, pred_list):.3f}")
    print(f"macro-F1: {f1_score(true_list, pred_list, average='macro'):.3f}")


def main():
    args = parse_option()

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    phases = ["train", "val"]

    image_datasets = {
        x: datasets.CIFAR10(
            root="./data",
            train=True if x == "train" else False,
            download=True,
            transform=transform,
        )
        for x in phases
    }
    num_classes = 10

    dataloaders = {
        x: DataLoader(
            image_datasets[x],
            batch_size=args.batch_size,
            shuffle=True if x == "train" else False,
            num_workers=4,
            pin_memory=True,
        )
        for x in phases
    }
    dataset_sizes = {x: len(image_datasets[x]) for x in phases}

    if args.model_name == "Baseline":
        model = Baseline(num_classes=num_classes)
        if args.model_path is None:
            model_path = "./models/baseline.pth"
    elif args.model_name == "Inception":
        model = CustomInception(num_classes=num_classes)
        if args.model_path is None:
            model_path = "./models/inception.pth"
    elif args.model_name == "ResInception":
        model = ResInception(num_classes=num_classes)
        if args.model_path is None:
            model_path = "./models/res_inception.pth"
    else:
        raise NotImplementedError("The model has not implemented: " + args.model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model = train_model(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        dataloaders=dataloaders,
        dataset_sizes=dataset_sizes,
        device=device,
        num_epochs=args.epochs,
        save_path=model_path if args.model_path is None else args.model_path,
    )
    preds = predict(model=model, dataloader=dataloaders["val"], device=device)


if __name__ == "__main__":
    main()
