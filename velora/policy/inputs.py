from pydantic import BaseModel


class ExtractorInputs(BaseModel):
    """
    A base class for FeatureExtractor input parameters.
    """

    pass


class CNNInputs(ExtractorInputs):
    """
    Input parameters for CNN architectures.

    Args:
        in_channels (int): number of channels in the input image
        n_hidden (list[int]): a list of hidden layer sizes
        img_shape (tuple[int, int]): the shape of a single image `(W, H)`
    """

    in_channels: int
    n_hidden: list[int]
    img_shape: tuple[int, int]


class MLPInputs(ExtractorInputs):
    """
    Input parameters for MLP architectures.

    Args:
        in_features (int): number of input features
        n_hidden (list[int]): a list of hidden layer sizes
    """

    in_features: int
    n_hidden: list[int]
