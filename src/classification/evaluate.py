import logging

import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch import autocast, nn
from torch.utils.data import DataLoader

from src.common.progress_tools import iterate_with_progress
from src.common.utils import DEVICE

logger = logging.getLogger(__name__)


def evaluate_classification_model(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    use_amp: bool,
    class_names: list[str],
    split_name: str = "Test",
) -> tuple[float, float, torch.Tensor]:
    """Evaluate a classification model and log aggregate statistics."""
    model = model.to(DEVICE)
    model.eval()

    total_loss = torch.tensor(0.0, device=DEVICE)
    correct = torch.tensor(0.0, device=DEVICE, dtype=torch.int64)
    total = torch.tensor(0.0, device=DEVICE, dtype=torch.int64)
    predictions: list[int] = []
    targets: list[int] = []
    labels = list(range(len(class_names)))

    with torch.inference_mode():
        for data, target in iterate_with_progress(
            data_loader,
            description=f"Eval · {split_name}",
            transient=True,
        ):
            data, target = data.to(DEVICE), target.to(DEVICE)

            with autocast(device_type=DEVICE, enabled=use_amp):
                output = model(data)
                loss = criterion(output, target)

            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum()
            total += target.size(0)
            total_loss += loss
            predictions.extend(pred.cpu().tolist())
            targets.extend(target.cpu().tolist())

    total_count = int(total.item())
    if total_count == 0:
        raise ValueError(f"{split_name} data loader is empty; cannot compute metrics")

    accuracy = 100.0 * (correct / total).item()
    avg_loss = total_loss.item() / len(data_loader)
    cm = confusion_matrix(targets, predictions, labels=labels)
    report = classification_report(
        targets,
        predictions,
        labels=labels,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )

    logger.info(
        "%s | loss=%.4f acc=%.2f%% correct=%d/%d | cm(shape=%s diagonal=%d/%d)",
        split_name,
        avg_loss,
        accuracy,
        int(correct.item()),
        total_count,
        cm.shape,
        int(cm.trace()),
        int(cm.sum()),
    )
    logger.debug("%s classification report:\n%s", split_name, report)

    return avg_loss, accuracy, data
