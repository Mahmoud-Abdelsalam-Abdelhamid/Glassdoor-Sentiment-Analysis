import pandas as pd
import torch


def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler):
    model = model.train()
    total_loss = 0
    correct_predictions = 0
    total_examples = 0

    for batch in data_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Compute loss manually
        loss = loss_fn(logits, labels)

        total_loss += loss.item()

        # Compute accuracy
        preds = torch.argmax(logits, dim=1)
        correct_predictions += torch.sum(preds == labels).item()
        total_examples += labels.size(0)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    avg_loss = total_loss / len(data_loader)
    accuracy = correct_predictions / total_examples

    return avg_loss, accuracy




def eval_model(model, data_loader, device):
    model = model.eval()
    correct_predictions = 0
    total_examples = 0

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            # logits is already a tensor
            _, preds = torch.max(logits, dim=1)

            correct_predictions += torch.sum(preds == labels).item()
            total_examples += labels.size(0)

    accuracy = correct_predictions / total_examples
    return accuracy



