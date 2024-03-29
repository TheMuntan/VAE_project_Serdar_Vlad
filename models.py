from typing import Tuple

import torch
import torch.nn as nn


class Print(nn.Module):
    """"
    This model is for debugging purposes (place it in nn.Sequential to see tensor dimensions).
    """

    def __init__(self):
        super(Print, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        print(x.shape)
        return x


class VanillaAutoEncoder(nn.Module):
    """"
    Here, your implementation of the Vanilla (i.e. the basic) Autoencoder (AE) should be made.
    TODO: Implement the forward method.
    """

    def __init__(self, config: dict):
        super(VanillaAutoEncoder, self).__init__()
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def generate(self, latent_vector: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent_vector)

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        return self.encoder(image)


class VariationalAutoEncoder(nn.Module):
    """
    Here, your implementation of the Variational Autoencoder (VAE) should be made.
    TODO: Implement the forward method. Use the correct return type!
    """

    def __init__(self, config: dict):
        super(VariationalAutoEncoder, self).__init__()
        self.encoder = VariationalEncoder(config)
        self.decoder = Decoder(config)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :rtype: tuple consisting of (decoded image, latent_vector, mu, log_var)
        """
        latent_vector,mu,log_var = self.encoder(x)  # Returns 

        x = self.decoder(latent_vector)

        return x,latent_vector,mu,log_var
        pass

    def generate(self, latent_vector: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent_vector)

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        return self.encoder(image)[0]


class Encoder(nn.Module):
    """"
    This class will contain the basic Encoder network which encodes the most important features into the latent space.
    TODO: implement the init and the forward method.
    """

    def __init__(self, config: dict):
        """TODO: define your layers here, as described in the assignment."""
        super(Encoder, self).__init__()

        ### Convolutional block
        self.encoder_cnn = nn.Sequential(
        nn.Conv2d(1,8,3,2,1),
        nn.ReLU(),
        nn.Conv2d(8,16,3,2,1),
        nn.ReLU(),
        nn.Conv2d(16,32,3,2,0),
        nn.ReLU()
        )

        ### Flatten layer
        self.encoder_flatten = nn.Flatten(1)
        ### Linear block
        self.encoder_lin = nn.Sequential(
        nn.Linear(288,128),
        nn.ReLU(),
        nn.Linear(128,config["latent_dim"]),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """TODO: implement the forward method."""
        x = self.encoder_cnn(x)
        x = self.encoder_flatten(x)
        x = self.encoder_lin(x)
        

        return x


class Decoder(nn.Module):
    """"
    This class will contain the Decoder network which decodes the most important features from the
    latent space back into an image.
    TODO: implement the init and the forward method.
    """

    def __init__(self, config: dict):
        """TODO: define your layers here, as described in the assignment."""
        super(Decoder, self).__init__()

        # Linear block
        self.decoder_lin = nn.Sequential(
        nn.Linear(config["latent_dim"], 128),
        nn.ReLU(),
        nn.Linear(128, 288),
        nn.ReLU()
        )

        # Unflatten layer
        self.decoder_unflatten = nn.Unflatten(1, (32, 3, 3))

        # Deconvolutional block
        self.decoder_conv = nn.Sequential(
        nn.ConvTranspose2d(32,16,3,2,0,output_padding=0),
        nn.ReLU(),
        nn.ConvTranspose2d(16,8,3,2,1,output_padding=1),
        nn.ReLU(),
        nn.ConvTranspose2d(8,1,3,2,1,output_padding=1),
        nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """TODO: implement the forward method."""
        x = self.decoder_lin(x)
        x = self.decoder_unflatten(x)
        x = self.decoder_conv(x)
        
        return x


class VariationalEncoder(nn.Module):
    """"
    The VAE uses the same decoder, but changes have to be made to the encoder.
    TODO: implement the init and the forward method.
    """

    def __init__(self, config: dict):
        """TODO: define your layers here, as described in the assignment."""
        super(VariationalEncoder, self).__init__()

        ### Convolutional block
        self.encoder_cnn = nn.Sequential(
        nn.Conv2d(1,8,3,2,1),
        nn.ReLU(),
        nn.Conv2d(8,16,3,2,1),
        nn.ReLU(),
        nn.Conv2d(16,32,3,2,0),
        nn.ReLU()
        )

        ### Flatten layer
        self.encoder_flatten = nn.Flatten(1)
        ### Linear block
        self.encoder_lin = nn.Sequential(
        nn.Linear(3*3*32,128),
        nn.ReLU(),
        
        #nn.Linear(128,config["latent_dim"]) # = mu
        )
        self.fc_mu = nn.Linear(128,config["latent_dim"])
        self.fc_log_var = nn.Linear(128,config["latent_dim"])

    def forward(self, x: torch.Tensor) -> tuple:
        """
        TODO: implement the forward method. Ensure you use the correct return type!
        TODO: Don't forget to use your reparametrization function!
        :rtype: tuple consisting of (latent vector, mu, log_var)
        """
        from utils import reparameterize
        x = self.encoder_cnn(x)
        x = self.encoder_flatten(x)
        x = self.encoder_lin(x)
        
        mu = self.fc_mu(x) # Create a layer that calculates the mean (mu) of the output of the encoder
        log_var = self.fc_log_var(x) # Create a layer that calculates the standard deviation (log_var) of the output of the encoder
        return reparameterize(mu, log_var), mu, log_var
