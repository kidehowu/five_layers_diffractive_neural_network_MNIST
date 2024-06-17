import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from copy import deepcopy
from torch.optim.lr_scheduler import LambdaLR
import torch.nn.functional as F


def set_seed(seed=42):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def make_lr_fn(start_lr, end_lr, num_iter, step_mode='exp'):
    if step_mode == 'linear':
        factor = (end_lr / start_lr - 1) / num_iter

        def lr_fn(iteration):
            return 1 + iteration * factor
    else:
        factor = (np.log(end_lr) - np.log(start_lr)) / num_iter

        def lr_fn(iteration):
            return np.exp(factor) ** iteration
    return lr_fn


def draw_detect_regions():
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.set_facecolor('black')

    rect = patches.Rectangle((12, 75), 12, 12, linewidth=2, edgecolor='red', facecolor='none', linestyle='--')
    ax.add_patch(rect)
    rect = patches.Rectangle((43, 75), 12, 12, linewidth=2, edgecolor='red', facecolor='none', linestyle='--')
    ax.add_patch(rect)
    rect = patches.Rectangle((73, 75), 12, 12, linewidth=2, edgecolor='red', facecolor='none', linestyle='--')
    ax.add_patch(rect)
    rect = patches.Rectangle((9, 44), 12, 12, linewidth=2, edgecolor='red', facecolor='none', linestyle='--')
    ax.add_patch(rect)
    rect = patches.Rectangle((32, 44), 12, 12, linewidth=2, edgecolor='red', facecolor='none', linestyle='--')
    ax.add_patch(rect)
    rect = patches.Rectangle((56, 44), 12, 12, linewidth=2, edgecolor='red', facecolor='none', linestyle='--')
    ax.add_patch(rect)
    rect = patches.Rectangle((79, 44), 12, 12, linewidth=2, edgecolor='red', facecolor='none', linestyle='--')
    ax.add_patch(rect)
    rect = patches.Rectangle((12, 13), 12, 12, linewidth=2, edgecolor='red', facecolor='none', linestyle='--')
    ax.add_patch(rect)
    rect = patches.Rectangle((43, 13), 12, 12, linewidth=2, edgecolor='red', facecolor='none', linestyle='--')
    ax.add_patch(rect)
    rect = patches.Rectangle((73, 13), 12, 12, linewidth=2, edgecolor='red', facecolor='none', linestyle='--')
    ax.add_patch(rect)
    ax.text(16, 79, '0', color='white')
    ax.text(47, 79, '1', color='white')
    ax.text(77, 79, '2', color='white')
    ax.text(13, 48, '3', color='white')
    ax.text(36, 48, '4', color='white')
    ax.text(60, 48, '5', color='white')
    ax.text(83, 48, '6', color='white')
    ax.text(16, 17, '7', color='white')
    ax.text(47, 17, '8', color='white')
    ax.text(77, 17, '9', color='white')

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()
    return


class Diffractive4(object):
    def __init__(self, model, loss_fn, optimizer):
        # Here we define the attributes of our class

        # We start by storing the arguments as attributes
        # to use them later
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Let's send the model to the specified device right away
        self.model.to(self.device)

        # These attributes are defined here, but since they are
        # not informed at the moment of creation, we keep them None
        self.train_loader = None
        self.val_loader = None

        # These attributes are going to be computed internally
        self.losses = []
        self.val_losses = []
        self.learning_rates = []
        self.total_epochs = 0
        self.scheduler = None
        self.is_batch_lr_scheduler = False

        self.visualization = {}
        self.handles = {}

        # Creates the train_step function for our model,
        # loss function and optimizer
        # Note: there are NO ARGS there! It makes use of the class
        # attributes directly
        self.train_step_fn = self._make_train_step_fn()
        # Creates the val_step function for our model and loss
        self.val_step_fn = self._make_val_step_fn()

    def to(self, device):
        # This method allows the user to specify a different device
        # It sets the corresponding attribute (to be used later in
        # the mini-batches) and sends the model to the device
        try:
            self.device = device
            self.model.to(self.device)
        except RuntimeError:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"Couldn't send it to {device}, sending it to {self.device} instead.")
            self.model.to(self.device)

    def set_loaders(self, train_loader, val_loader=None):
        # This method allows the user to define which train_loader (and val_loader, optionally) to use
        # Both loaders are then assigned to attributes of the class
        # So they can be referred to later
        self.train_loader = train_loader
        self.val_loader = val_loader

    def _make_train_step_fn(self):
        # This method does not need ARGS... it can refer to
        # the attributes: self.model, self.loss_fn and self.optimizer

        # Builds function that performs a step in the train loop
        def perform_train_step_fn(x, y):
            # Sets model to TRAIN mode
            self.model.train()

            # Step 1 - Computes our model's predicted output - forward pass
            yhat = self.model(x)
            # Step 2 - Computes the loss
            loss = self.loss_fn(yhat, y)
            # Step 3 - Computes gradients
            loss.backward()
            # Step 4 - Updates parameters using gradients and the learning rate
            self.optimizer.step()
            self.optimizer.zero_grad()

            # Returns the loss
            return loss.item()

        # Returns the function that will be called inside the train loop
        return perform_train_step_fn

    def _make_val_step_fn(self):
        # Builds function that performs a step in the validation loop
        def perform_val_step_fn(x, y):
            # Sets model to EVAL mode
            self.model.eval()

            # Step 1 - Computes our model's predicted output - forward pass
            yhat = self.model(x)
            # Step 2 - Computes the loss
            loss = self.loss_fn(yhat, y)
            # There is no need to compute Steps 3 and 4, since we don't update parameters during evaluation
            return loss.item()

        return perform_val_step_fn

    def _run_all_mini_batch(self, validation=False):
        # The mini-batch can be used with both loaders
        # The argument `validation`defines which loader and
        # corresponding step function is going to be used
        if validation:
            data_loader = self.val_loader
            step_fn = self.val_step_fn
        else:
            data_loader = self.train_loader
            step_fn = self.train_step_fn

        if data_loader is None:
            return None

        n_batches = len(data_loader)
        # Once the data loader and step function, this is the same
        # mini-batch loop we had before
        mini_batch_losses = []
        for i, (x_batch, y_batch) in enumerate(data_loader):
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            mini_batch_loss = step_fn(x_batch, y_batch)
            mini_batch_losses.append(mini_batch_loss)

            if not validation:
                self._mini_batch_schedulers(i / n_batches)

        loss = np.mean(mini_batch_losses)
        return loss

    def _mini_batch_schedulers(self, frac_epoch):
        if self.scheduler:
            if self.is_batch_lr_scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.CosineAnnealingWarmRestarts):
                    self.scheduler.step(self.total_epochs + frac_epoch)
                else:
                    self.scheduler.step()

                current_lr = list(map(lambda d: d['lr'], self.scheduler.optimizer.state_dict()['param_groups']))
                self.learning_rates.append(current_lr)

    def _epoch_schedulers(self, val_loss):
        if self.scheduler:
            if not self.is_batch_lr_scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

                current_lr = list(map(lambda d: d['lr'], self.scheduler.optimizer.state_dict()['param_groups']))
                self.learning_rates.append(current_lr)

    def set_seed(self, seed=42):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        try:
            self.train_loader.sampler.generator.manual_seed(seed)
        except AttributeError:
            pass

    def train(self, n_epochs, seed=42):
        # To ensure reproducibility of the training process
        self.set_seed(seed)

        for epoch in range(n_epochs):
            # Keeps track of the numbers of epochs
            # by updating the corresponding attribute
            self.total_epochs += 1

            # inner loop
            # Performs training using mini-batches
            loss = self._run_all_mini_batch(validation=False)
            self.losses.append(loss)

            # VALIDATION
            # no gradients in validation!
            '''with torch.no_grad():
                # Performs evaluation using mini-batches
                val_loss = self._run_all_mini_batch(validation=True)
                self.val_losses.append(val_loss)

            self._epoch_schedulers(val_loss)'''

    def save_checkpoint(self, filename):
        # Builds dictionary with all elements for resuming training
        checkpoint = {'epoch': self.total_epochs,
                      'model_state_dict': self.model.state_dict(),
                      'optimizer_state_dict': self.optimizer.state_dict(),
                      'loss': self.losses,
                      'val_loss': self.val_losses}

        torch.save(checkpoint, filename)

    def load_checkpoint(self, filename, dummy=None):
        # Loads dictionary
        checkpoint = torch.load(filename)
        if dummy is not None:
            checkpoint2 = torch.load(f'saved_model/dummies/{dummy}.pth')
            checkpoint['model_state_dict']['propagation.h'] = checkpoint2['model_state_dict']['propagation.h']
            checkpoint['model_state_dict']['intensity_sum.gaussians'] = checkpoint2['model_state_dict'][
                'intensity_sum.gaussians']

        # Restore state for model and optimizer
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.total_epochs = checkpoint['epoch']
        self.losses = checkpoint['loss']
        self.val_losses = checkpoint['val_loss']

        self.model.train()  # always use TRAIN for resuming training

    def predict(self, x):
        # Set is to evaluation mode for predictions
        self.model.eval()
        # Takes aNumpy input and make it a float tensor
        x_tensor = torch.as_tensor(x).float()
        # Send input to device and uses model for prediction
        y_hat_tensor = self.model(x_tensor.to(self.device))
        # Set it back to train mode
        self.model.train()
        # Detaches it, brings it to CPU and back to Numpy
        return y_hat_tensor.detach().cpu().numpy()

    def plot_losses(self):
        fig = plt.figure(figsize=(10, 4))
        plt.plot(self.losses, label='Training Loss', c='b')
        plt.plot(self.val_losses, label='Validation Loss', c='r')
        plt.yscale('log')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        return fig

    def compute_correct_rate(self, x, y, threshold=.5):
        self.model.eval()
        yhat = self.model(x.to(self.device))
        y = y.to(self.device)
        self.model.train()

        # We get the size of the batch and the number of classes
        # (only 1, if it is binary)
        n_samples, n_dims = yhat.shape
        if n_dims > 1:
            # In a multiclass classification, the biggest logit
            # always wins, so we don't bother getting probabilities

            # This is PyTorch's version of argmax,
            # but it returns a tuple: (max value, index of max value)
            _, predicted = torch.max(yhat, 1)
        else:
            n_dims += 1
            # In binary classification, we NEED to check if the
            # last layer is a sigmoid (and then it produces probs)
            if isinstance(self.model, nn.Sequential) and \
                    isinstance(self.model[-1], nn.Sigmoid):
                predicted = (yhat > threshold).long()
            # or something else (logits), which we need to convert
            # using a sigmoid
            else:
                predicted = (torch.sigmoid(yhat) > threshold).long()

        # How many samples got classified correctly for each class
        result = []
        for c in range(n_dims):
            n_class = (y == c).sum().item()
            n_correct = (predicted[y == c] == c).sum().item()
            result.append((n_correct, n_class))
        return torch.tensor(result)

    @staticmethod
    def loader_apply(loader, func, reduce='sum'):
        results = [func(x, y) for i, (x, y) in enumerate(loader)]
        results = torch.stack(results, axis=0)

        if reduce == 'sum':
            results = results.sum(axis=0)
        elif reduce == 'mean':
            results = results.float().mean(axis=0)

        return results

    def lr_range_test(self, data_loader, end_lr, num_iter=100, step_mode='exp', alpha=0.05, ax=None):
        # Since the test updates both model and optimizer we need to store
        # their initial states to restore them in the end
        previous_states = {'model': deepcopy(self.model.state_dict()),
                           'optimizer': deepcopy(self.optimizer.state_dict())}
        # Retrieves the learning rate set in the optimizer
        start_lr = self.optimizer.state_dict()['param_groups'][0]['lr']

        # Builds a custom function and corresponding scheduler
        lr_fn = make_lr_fn(start_lr, end_lr, num_iter)
        scheduler = LambdaLR(self.optimizer, lr_lambda=lr_fn)

        # Variables for tracking results and iterations
        tracking = {'loss': [], 'lr': []}
        iteration = 0

        # If there are more iterations than mini-batches in the data loader,
        # it will have to loop over it more than once
        while (iteration < num_iter):
            # That's the typical mini-batch inner loop
            for x_batch, y_batch in data_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                # Step 1
                yhat = self.model(x_batch)
                # Step 2
                loss = self.loss_fn(yhat, y_batch)
                # Step 3
                loss.backward()

                # Here we keep track of the losses (smoothed)
                # and the learning rates
                tracking['lr'].append(scheduler.get_last_lr()[0])
                if iteration == 0:
                    tracking['loss'].append(loss.item())
                else:
                    prev_loss = tracking['loss'][-1]
                    smoothed_loss = alpha * loss.item() + (1 - alpha) * prev_loss
                    tracking['loss'].append(smoothed_loss)

                iteration += 1
                # Number of iterations reached
                if iteration == num_iter:
                    break

                # Step 4
                self.optimizer.step()
                scheduler.step()
                self.optimizer.zero_grad()

        # Restores the original states
        self.optimizer.load_state_dict(previous_states['optimizer'])
        self.model.load_state_dict(previous_states['model'])

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        else:
            fig = ax.get_figure()
        ax.plot(tracking['lr'], tracking['loss'])
        if step_mode == 'exp':
            ax.set_xscale('log')
        ax.set_xlabel('Learning Rate')
        ax.set_ylabel('Loss')
        fig.tight_layout()
        return tracking, fig

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    @staticmethod
    def _visualize_tensors(axs, x, y=None, yhat=None, layer_name='', title=None):
        # The number of images is the number of subplots in a row
        n_images = len(axs)
        # Gets max and min values for scaling the grayscale
        minv, maxv = np.min(x[:n_images]), np.max(x[:n_images])
        # For each image
        for j, image in enumerate(x[:n_images]):
            ax = axs[j]
            # Sets title, labels, and removes ticks
            if title is not None:
                ax.set_title('{} #{}'.format(title, j), fontsize=12)
            '''ax.set_ylabel(
                '{}\n{}x{}'.format(layer_name, *np.atleast_2d(image).shape),
                rotation=0, labelpad=40
            )'''
            xlabel1 = '' if y is None else '\nLabel: {}'.format(y[j])
            xlabel2 = '' if yhat is None else '\nPredicted: {}'.format(yhat[j])
            xlabel = '{}{}'.format(xlabel1, xlabel2)
            if len(xlabel):
                ax.set_xlabel(xlabel, fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])

            # Plots weight as an image
            ax.imshow(
                np.atleast_2d(image.squeeze()),
                cmap='gray',
                vmin=minv,
                vmax=maxv
            )
        return

    @staticmethod
    def _visualize_intensity(axs, x, y=None, yhat=None, layer_name='', title=None):
        # The number of images is the number of subplots in a row
        n_images = len(axs)
        # Gets max and min values for scaling the grayscale
        minv, maxv = np.min(x[:n_images]), np.max(x[:n_images]) / 3
        # For each image
        for j, image in enumerate(x[:n_images]):
            ax = axs[j]
            # Sets title, labels, and removes ticks
            if title is not None:
                ax.set_title('{} #{}'.format(title, j), fontsize=12)
            '''ax.set_ylabel(
                '{}\n{}x{}'.format(layer_name, *np.atleast_2d(image).shape),
                rotation=0, labelpad=40
            )'''
            xlabel1 = '' if y is None else '\nLabel: {}'.format(y[j])
            xlabel2 = '' if yhat is None else '\nPredicted: {}'.format(yhat[j])
            xlabel = '{}{}'.format(xlabel1, xlabel2)
            if len(xlabel):
                ax.set_xlabel(xlabel, fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])

            # Plots weight as an image
            ax.imshow(
                np.atleast_2d(image.squeeze()),
                cmap='gray',
                vmin=minv,
                vmax=maxv
            )
        return

    @staticmethod
    def _visualize_phases(fig, axs, x, y=None, yhat=None, layer_name='', title=None):
        # The number of images is the number of subplots in a row
        n_images = len(axs)
        # Gets max and min values for scaling the grayscale
        minv, maxv = 0, np.pi
        # For each image
        for j, image in enumerate(x[:n_images]):
            ax = axs[j]
            # Sets title, labels, and removes ticks
            ax.set_xticks([])
            ax.set_yticks([])

            # Plots weight as an image
            im = ax.imshow(
                np.atleast_2d(image.squeeze()),
                cmap='jet',
                vmin=minv,
                vmax=maxv
            )
        cbar_ax = fig.add_axes([0.92, 0.12, 0.01, 0.75])  # Adjust these values as needed
        cbar = fig.colorbar(im, cax=cbar_ax)
        # Set custom tick positions
        cbar.set_ticks([0, np.pi])
        # Set custom tick labels
        cbar.set_ticklabels(['0', '$\\pi$'])

        return

    def attach_intensity_hooks(self, layer_name, hook_fn=None):
        # Clear any previous values
        self.visualization = {}
        # Creates the dictionary to map layer objects to their names
        modules = list(self.model.named_modules())
        layer_names = {layer: name for name, layer in modules[1:]}

        if hook_fn is None:
            # Hook function to be attached to the forward pass
            def hook_fn(layer, inputs, outputs):
                name = layer_names[layer]
                self.visualization[name] = outputs.detach().cpu().numpy()

        for name, layer in modules:
            # If the layer is in our list
            if name == layer_name:
                # Initializes the corresponding key in the dictionary
                self.visualization[name] = None
                # Register the forward hook and keep the handle in another dict
                self.handles[name] = layer.register_forward_hook(hook_fn)

    def attach_hooks(self, layers_to_hook, hook_fn=None):
        # Clear any previous values
        self.visualization = {}
        # Creates the dictionary to map layer objects to their names
        modules = list(self.model.named_modules())
        layer_names = {layer: name for name, layer in modules[1:]}

        if hook_fn is None:
            # Hook function to be attached to the forward pass
            def hook_fn(layer, inputs, outputs):
                # Gets the layer name
                name = layer_names[layer]
                # Detaches outputs
                values = outputs.detach().cpu().numpy()
                # Since the hook function may be called multiple times
                # for example, if we make predictions for multiple mini-batches
                # it concatenates the results
                if self.visualization[name] is None:
                    self.visualization[name] = values
                else:
                    self.visualization[name] = np.concatenate([self.visualization[name], values])

        for name, layer in modules:
            # If the layer is in our list
            if name in layers_to_hook:
                # Initializes the corresponding key in the dictionary
                self.visualization[name] = None
                # Register the forward hook and keep the handle in another dict
                self.handles[name] = layer.register_forward_hook(hook_fn)

    def remove_hooks(self):
        # Loops through all hooks and removes them
        for handle in self.handles.values():
            handle.remove()
        # Clear the dict, as all hooks have been removed
        self.handles = {}

    def plot_intensity(self, iterat, n_images, size, cut):
        total_rows = 2
        layer = 'intensity_capture'
        data, label = next(iterat)
        data_gpu = data[:n_images].to(self.device)

        self.attach_intensity_hooks(layer)
        with torch.no_grad():
            self.model.eval()
            yhat = self.model(data_gpu)
            self.model.train()
        self.remove_hooks()

        yhat = yhat.argmax(dim=1)
        fig, axes = plt.subplots(total_rows, n_images, figsize=(2 * n_images, 2 * total_rows))
        axes = np.atleast_2d(axes).reshape(total_rows, n_images)

        output = self.visualization[layer]
        output = output[:, cut:size - cut, cut:size - cut]

        Diffractive4._visualize_intensity(
            axes[0, :],
            output,
            label,
            yhat,
            layer_name=layer
        )

        Diffractive4._visualize_tensors(
            axes[1, :],
            data.numpy(),
        )

        '''for ax in axes.flat:
            ax.label_outer()'''

        plt.tight_layout()
        return fig

    def visualize_phasemask(self, layers_name, **kwargs):
        try:
            # Gets the layer object from the model
            n_weights = []
            for name in layers_name:
                layer = getattr(self.model, name)
                weights = layer.mod.data
                weights = torch.pi * torch.sigmoid(weights)
                weights = weights.cpu().numpy()
                n_weights.append(weights)

            # The weights have channels_out (filter), channels_in, H, W shape
            n_channels = len(layers_name)

            # Builds a figure
            fig, axes = plt.subplots(1, n_channels, figsize=(2 * n_channels + 1, 2))
            # For each channel_out (filter)
            Diffractive4._visualize_phases(
                fig,
                axes,
                n_weights,
            )

            return fig
        except AttributeError:
            return

    def visualize_outputs(self, layers, size, n_padd, n_images=8, y=None, yhat=None):
        layers = filter(lambda l: l in self.visualization.keys(), layers)
        layers = list(layers)
        shapes = [self.visualization[layer].shape for layer in layers]
        n_rows = [shape[1] if len(shape) == 4 else 1 for shape in shapes]
        total_rows = np.sum(n_rows)

        fig, axes = plt.subplots(total_rows, n_images,
                                 figsize=(2 * n_images, 2 * total_rows))
        axes = np.atleast_2d(axes).reshape(total_rows, n_images)

        # Loops through the layers, one layer per row of subplots
        row = 0
        for i, layer in enumerate(layers):
            # Takes the produced feature maps for that layer
            output = self.visualization[layer]
            if output.shape[1] > size:
                output = np.abs(output[:n_images, n_padd:size + n_padd, n_padd:size + n_padd])
            is_vector = len(output.shape) == 2

            for j in range(n_rows[i]):
                Diffractive4._visualize_tensors(
                    axes[row, :],
                    output,
                    y,
                    yhat,
                    layer_name=layers[i],
                    title='Image' if (row == 0) else None
                )
                row += 1

        for ax in axes.flat:
            ax.label_outer()

        plt.tight_layout()
        return fig
