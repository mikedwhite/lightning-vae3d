import yaml
from pathlib import Path


class MetadataAlpha():
    def __init__(self):
        super().__init__()
        self.metadata_dict = {}

    def create(self,
               TrainerModule,
               loss_func1,
               loss_func2,
               alpha,
               parallel,
               patch_size,
               n_channels,
               n_epochs,
               learning_rate,
               weight_decay,
               batch_size,
               latent_dim,
               amsgrad):
        """Create an instance of the metadata dictionary for the alpha VAE.

        Parameters
        ----------
        TrainerModule : lightning.LightningModule
        loss_func1 : torch.nn.Module
        loss_func2 : torch.nn.Module
        alpha : float
        parallel : bool
        patch_size : int
        n_channels : int
        n_epochs : int
        learning_rate : float
        weight_decay : float
        batch_size : int
        latent_dim : int
        amsgrad : bool
        """
        self.metadata_dict = {'trainer': TrainerModule.__name__,
                              'loss_func1': loss_func1.__class__.__name__,
                              'loss_func2': loss_func2.__class__.__name__,
                              'alpha': alpha,
                              'parallel': parallel,
                              'patch_size': patch_size,
                              'n_channels': n_channels,
                              'n_epochs': n_epochs,
                              'initial_epoch': 0,
                              'learning_rate': learning_rate,
                              'weight_decay': weight_decay,
                              'batch_size': batch_size,
                              'latent_dim': latent_dim,
                              'amsgrad': amsgrad,
                              'train_loss': [],
                              'val_loss': [],
                              }

    def save(self, path):
        """Save metadata dictionary as yaml file.

        Parameters
        ----------
        path : str
            File name and path to save the metadata dictionary. Should be specified as a .yaml file.
        """
        with open(path, 'w') as f:
            yaml.dump(self.metadata_dict, f, default_flow_style=False, sort_keys=False)

    def load(self, path):
        """Load metadata dictionary from yaml file.

        Parameters
        ----------
        path : str
            File name and path to load the metadata dictionary from. Should be specified as a .yaml file.
        """
        self.metadata_dict = yaml.safe_load(Path(path).read_text())


class MetadataBeta():
    def __init__(self):
        super().__init__()
        self.metadata_dict = {}

    def create(self,
               TrainerModule,
               loss_func,
               beta,
               parallel,
               patch_size,
               n_channels,
               n_epochs,
               learning_rate,
               weight_decay,
               batch_size,
               latent_dim,
               amsgrad):
        """Create an instance of the metadata dictionary for the beta VAE.

        Parameters
        ----------
        TrainerModule : lightning.LightningModule
        loss_func : torch.nn.Module
        beta : float
        parallel : bool
        patch_size : int
        n_channels : int
        n_epochs : int
        learning_rate : float
        weight_decay : float
        batch_size : int
        latent_dim : int
        amsgrad : bool
        """
        self.metadata_dict = {'trainer': TrainerModule.__name__,
                              'loss_func1': loss_func.__class__.__name__,
                              'loss_func2': 'KLDivergence',
                              'beta': beta,
                              'parallel': parallel,
                              'patch_size': patch_size,
                              'n_channels': n_channels,
                              'n_epochs': n_epochs,
                              'initial_epoch': 0,
                              'learning_rate': learning_rate,
                              'weight_decay': weight_decay,
                              'batch_size': batch_size,
                              'latent_dim': latent_dim,
                              'amsgrad': amsgrad,
                              'train_loss': [],
                              'val_loss': [],
                              }

    def save(self, path):
        """Save metadata dictionary as yaml file.

        Parameters
        ----------
        path : str
            File name and path to save the metadata dictionary. Should be specified as a .yaml file.
        """
        with open(path, 'w') as f:
            yaml.dump(self.metadata_dict, f, default_flow_style=False, sort_keys=False)

    def load(self, path):
        """Load metadata dictionary from yaml file.

        Parameters
        ----------
        path : str
            File name and path to load the metadata dictionary from. Should be specified as a .yaml file.
        """
        self.metadata_dict = yaml.safe_load(Path(path).read_text())


class MetadataAlphaBeta():
    def __init__(self):
        super().__init__()
        self.metadata_dict = {}

    def create(self,
               TrainerModule,
               loss_func1,
               loss_func2,
               alpha,
               beta,
               parallel,
               patch_size,
               n_channels,
               n_epochs,
               learning_rate,
               weight_decay,
               batch_size,
               latent_dim,
               amsgrad):
        """Create an instance of the metadata dictionary for the alpha VAE.

        Parameters
        ----------
        TrainerModule : lightning.LightningModule
        loss_func1 : torch.nn.Module
        loss_func2 : torch.nn.Module
        alpha : float
        beta : float
        parallel : bool
        patch_size : int
        n_channels : int
        n_epochs : int
        learning_rate : float
        weight_decay : float
        batch_size : int
        latent_dim : int
        amsgrad : bool
        """
        self.metadata_dict = {'trainer': TrainerModule.__name__,
                              'loss_func1': loss_func1.__class__.__name__,
                              'loss_func2': loss_func2.__class__.__name__,
                              'alpha': alpha,
                              'beta': beta,
                              'parallel': parallel,
                              'patch_size': patch_size,
                              'n_channels': n_channels,
                              'n_epochs': n_epochs,
                              'initial_epoch': 0,
                              'learning_rate': learning_rate,
                              'weight_decay': weight_decay,
                              'batch_size': batch_size,
                              'latent_dim': latent_dim,
                              'amsgrad': amsgrad,
                              'train_loss': [],
                              'val_loss': [],
                              }

    def save(self, path):
        """Save metadata dictionary as yaml file.

        Parameters
        ----------
        path : str
            File name and path to save the metadata dictionary. Should be specified as a .yaml file.
        """
        with open(path, 'w') as f:
            yaml.dump(self.metadata_dict, f, default_flow_style=False, sort_keys=False)

    def load(self, path):
        """Load metadata dictionary from yaml file.

        Parameters
        ----------
        path : str
            File name and path to load the metadata dictionary from. Should be specified as a .yaml file.
        """
        self.metadata_dict = yaml.safe_load(Path(path).read_text())
