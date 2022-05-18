import torch
import matplotlib.pyplot as plt


class Evaluator:
    """
    Evaluator is used to perform evaluation along the training, keep track of it or
    perform standart evaluation when needed.
    """

    SAMPLE_MIN = 5000

    def __init__(self, model, train, test, val, labels_linear=False):
        self.model = model
        self.X_train, self.y_train = train
        self.X_test, self.y_test = test
        self.X_val, self.y_val = val
        self.labels_linear = labels_linear

        self.accuracies = {}
        self.losses = {}

    def to_prediction(self, y):
        """
        Transform the raw model output to actual prediction
        """
        if not self.labels_linear:
            return torch.argmax(y.cpu(), dim=1)
        return torch.round(y.cpu())

    def predict(self, X, to_prediction=False):
        """
        Infer using the model on X values
        """
        output, hidden = self.model(X)
        if to_prediction:
            return self.to_prediction(output)
        return output

    def sample(self, *args):
        """
        create a common sampling idx and sample the same way all values in *args
        """
        idx = torch.randperm(args[0].shape[0])[:self.SAMPLE_MIN]
        return (a[idx] for a in args)

    def sampleIfNeeded(self, *args):
        """
        Check if the args is too large, if so sample it
        """
        if args[0].shape[0] > self.SAMPLE_MIN:
            return self.sample(*args)
        return tuple(args)

    def accuracy(self, X, y, set_name=None, set_train_mode=True):
        """
        Infer and compute the accuracy of the prediction
        Can also keep track of it.
        """
        X, y = self.sampleIfNeeded(X, y)
        self.model.eval()
        y_pred = self.predict(X, to_prediction=True).view(y.shape[0])
        acc = torch.sum(y_pred == self.to_prediction(y)) / y_pred.shape[0]

        # Save accuracy in history
        if set_name is not None:
            if set_name not in self.accuracies:
                self.accuracies[set_name] = []
            self.accuracies[set_name].append(acc)
        if set_train_mode:
            self.model.train()
        return acc

    def loss(self, X, y, criterion, set_name=None, set_train_mode=True):
        """
        Infer and compute the loss of the prediction
        Can also keep track of it.
        """
        X, y = self.sampleIfNeeded(X, y)
        self.model.eval()
        y_hat = self.predict(X, to_prediction=False)
        loss = criterion(y_hat.squeeze(), y).item()

        # Save loss in history
        if set_name is not None:
            if set_name not in self.losses:
                self.losses[set_name] = []
            self.losses[set_name].append(loss)
        if set_train_mode:
            self.model.train()
        return loss

    def final_eval(self):
        """
        Perform a final evaluation of the model by computing accuracy for all sets and plotting
        - Accuracy history of val and train sets
        - Loss history of val and train sets
        """
        self.model.eval()
        test_acc = float(self.accuracy(self.X_test, self.y_test))
        train_acc = float(self.accuracy(self.X_train, self.y_train, set_name='train'))
        val_acc = float(self.accuracy(self.X_val, self.y_val, set_name='val'))

        print('Test Accuracy', test_acc)
        print('Train Accuracy', train_acc)
        print('Val Accuracy', val_acc)

        fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(25, 25))

        ax1.plot(self.accuracies['train'], label='training')
        ax1.plot(self.accuracies['val'], label='validation')
        ax1.legend()
        ax1.set_title('Accuracy over epochs')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Accuracy')

        ax2.plot(self.losses['train'], label='training')
        ax2.plot(self.losses['val'], label='validation')
        ax2.legend()
        ax2.set_title('Loss over epochs')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Loss')

        plt.show()

        return test_acc, train_acc, val_acc
