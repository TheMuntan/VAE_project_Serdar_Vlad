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

        )

        ### Flatten layer

        ### Linear block
        self.encoder_lin = nn.Sequential(

        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """TODO: implement the forward method."""

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

        # Convolutional block
        self.decoder_lin = nn.Sequential(

        )

        # Unflatten layer

        # Deconvolutional block
        self.decoder_conv = nn.Sequential(

        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """TODO: implement the forward method."""

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

        )

        ### Flatten layer

        ### Linear block
        self.encoder_lin = nn.Sequential(

        )
        self.fc_mu = None
        self.fc_log_var = None

    def forward(self, x: torch.Tensor) -> tuple:
        """
        TODO: implement the forward method. Ensure you use the correct return type!
        TODO: Don't forget to use your reparametrization function!
        :rtype: tuple consisting of (latent vector, mu, log_var)
        """
        from utils import reparameterize
        mu = None
        log_var = None
        return reparameterize(mu, log_var), mu, log_var
