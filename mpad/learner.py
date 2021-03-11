from models import MPAD
import torch
from torch import optim
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report
import os


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


class Learner:
    def __init__(self, experiment_name, device, multi_label):

        self.experiment_name = experiment_name
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.device = device
        self.writer = None
        self.train_step = 0
        self.multi_label = multi_label
        self.best_score = 0

        self.graph_preprocess_args = None
        self.epoch = -1
        self.model_save_dir = os.path.join(experiment_name, "models")
        self.model_type = None
        self.model_args = None
        self.log_dir = None

        os.makedirs(self.model_save_dir, exist_ok=True)
        self.best_model_path = os.path.join(self.model_save_dir, "model_best.pt")

    def set_graph_preprocessing_args(self, args):

        self.graph_preprocess_args = args

    def get_graph_preprocessing_args(self):
        return self.graph_preprocess_args

    def init_model(self, model_type="mpad", lr=0.1, **kwargs):
        # Store model type
        self.model_type = model_type.lower()
        # Initiate model
        if model_type.lower() == "mpad":
            self.model = MPAD(**kwargs)
        else:
            raise AssertionError("Currently only MPAD is supported as model")

        self.model_args = kwargs

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=50, gamma=0.5
        )

        self.criterion = torch.nn.CrossEntropyLoss()

    def train_epoch(self, dataloader, eval_every):
        self.epoch += 1
        self.model.train()
        total_iters = -1

        with tqdm(initial=0, total=eval_every) as pbar_train:
            for batch_ix, batch in enumerate(dataloader):
                total_iters += 1

                batch = (t.to(self.device) for t in batch)
                A, nodes, y, n_graphs = batch

                preds = self.model(nodes, A, n_graphs)

                loss = self.criterion(preds, y)

                self.optimizer.zero_grad()
                loss.backward()

                # grad norm clipping?
                self.optimizer.step()
                self.scheduler.step()

                pbar_train.update(1)
                pbar_train.set_description(
                    "Training step {} -> loss: {}".format(total_iters + 1, loss.item())
                )

                if (total_iters + 1) % eval_every == 0:
                    # Stop training
                    break

    def compute_metrics(self, y_pred, y_true):

        if self.multi_label:
            raise NotImplementedError()
        else:
            # Compute weighted average of F1 score
            y_pred = np.argmax(y_pred, axis=1)
            class_report = classification_report(y_true, y_pred, output_dict=True)
            return class_report["weighted avg"]["f1-score"]

    def save_model(self, is_best):

        to_save = {
            "experiment_name": self.experiment_name,
            "model_type": self.model_type,
            "graph_preprocess_args":self.graph_preprocess_args,
            "epoch": self.epoch,
            "model_args": self.model_args,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        # Save model indexed by epoch nr
        save_path = os.path.join(
            self.model_save_dir, self.experiment_name + "_{}.pt".format(self.epoch)
        )
        torch.save(to_save, save_path)

        if is_best:
            # Save best model separately
            torch.save(to_save, self.best_model_path)

    def load_model(self, path, lr=0.1):

        to_load = torch.load(path)

        self.epoch = to_load["epoch"]
        # Set up architecture of model
        self.init_model(
            model_type=to_load["model_type"],
            lr=lr,
            **to_load["model_args"]  # pass as kwargs
        )
        # Store kwargs for
        self.set_graph_preprocessing_args(to_load["graph_preprocess_args"])

        self.model.load_state_dict(to_load["state_dict"])
        self.optimizer.load_state_dict(to_load["optimizer"])

    def load_best_model(self):
        # Load the best model of the current experiment
        self.load_model(self.best_model_path)

    def evaluate(self, dataloader, save_model=True):

        self.model.eval()
        y_pred = []
        y_true = []
        running_loss = 0

        ######################################
        # Infer the model on the dataset
        ######################################
        with tqdm(initial=0, total=len(dataloader)) as pbar_eval:
            with torch.no_grad():
                for batch_idx, batch in enumerate(dataloader):
                    batch = (t.to(self.device) for t in batch)
                    A, nodes, y, n_graphs = batch

                    preds = self.model(nodes, A, n_graphs)
                    loss = self.criterion(preds, y)
                    running_loss += loss.item()
                    # store predictions and targets
                    y_pred.extend(list(preds.cpu().detach().numpy()))
                    y_true.extend(list(np.round(y.cpu().detach().numpy())))

                    pbar_eval.update(1)
                    pbar_eval.set_description(
                        "Eval step {} -> loss: {}".format(batch_idx + 1, loss.item())
                    )

        ######################################
        # Compute metrics
        ######################################
        f1 = self.compute_metrics(y_pred, y_true)

        if f1 > self.best_score and save_model:
            print("Saving new best model with F1 score {:.03f}".format(f1))
            self.best_score = f1
            self.save_model(is_best=True)
        else:
            print(
                "Current F1-score: {:.03f}, previous best: {:.03f}".format(
                    f1, self.best_score
                )
            )
