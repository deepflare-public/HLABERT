from abc import ABC, abstractmethod
from typing import List

import torch
import torch.nn as nn


class Pooling(nn.Module, ABC):
    """ Abstract class for pooling layers.
    """

    def __init__(self, output_no: int) -> None:
        super().__init__()
        self.output_vectors_no = output_no

    @abstractmethod
    def __call__(self, **kwargs) -> List[torch.Tensor]:
        """
        In most cases (B, L, D) -> (B, N, D) where N is the number of pooled vectors but it can be different.
        B - batch size
        D - embedding dimension
        """
        pass


class CLSPooling(Pooling):

    def __init__(self, cls_first: bool = True, **kwargs) -> None:
        super().__init__(1)
        self.cls_idx = 0 if cls_first else -1

    def __call__(self, token_embeddings: torch.Tensor, **kwargs) -> List[torch.Tensor]:
        """ Returns the first token of the sequence.
        token_embeddings: (B, L, D) -> (B, D)
        """
        return [token_embeddings[:, self.cls_idx, :]]


class MeanPooling(Pooling):

    def __init__(self, **kwargs) -> None:
        super().__init__(1)

    def __call__(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor, **kwargs) -> List[torch.Tensor]:
        """ Returns the mean of the sequence.
        token_embeddings: (B, L, D)
        attention_mask: (B, L)

        output: (B, D)
        """
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return [sum_embeddings / sum_mask]


class MeanSqrtPooling(Pooling):

    def __init__(self, **kwargs) -> None:
        super().__init__(1)

    def __call__(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor, **kwargs) -> List[torch.Tensor]:
        """ Returns the mean sqrt of the sequence.
        token_embeddings: (B, L, D)
        attention_mask: (B, L)

        output: (B, D)
        """
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return [sum_embeddings / torch.sqrt(sum_mask)]


class MeanMeanSqrtPooling(Pooling):

    def __init__(self, **kwargs) -> None:
        super().__init__(2)

    def __call__(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor, **kwargs) -> List[torch.Tensor]:
        """ Returns the mean sqrt of the sequence.
        token_embeddings: (B, L, D)
        attention_mask: (B, L)

        output: (B, D)
        """
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return [sum_embeddings / sum_mask, sum_embeddings / torch.sqrt(sum_mask)]


class SelfAttentionPooling(Pooling):
    """https://arxiv.org/pdf/2008.01077v1.pdf"""

    def __init__(self, hidden_dim, **kwargs) -> None:
        super().__init__(1)
        self.W = nn.Linear(hidden_dim, 1)

    def __call__(self, token_embeddings: torch.Tensor, **kwargs) -> List[torch.Tensor]:
        """
        token_embeddings: (B, L, D)

        output: (B, D)
        """
        attention_partial = self.W(token_embeddings)  # (B, L, 1)
        attention_partial = nn.functional.softmax(attention_partial.squeeze(-1)).unsqueeze(-1)  # (B, L, 1)
        # (B, L)
        return [torch.sum(token_embeddings * attention_partial, dim=1)]


class MaxPooling(Pooling):

    def __init__(self, **kwargs) -> None:
        super().__init__(1)

    def __call__(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor, **kwargs) -> List[torch.Tensor]:
        """ Returns the mean sqrt of the sequence.
        token_embeddings: (B, L, D)
        attention_mask: (B, L)

        output: (B, D)
        """
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        # Set padding tokens to large negative value
        token_embeddings[input_mask_expanded == 0] = -1e9
        max_over_time = torch.max(token_embeddings, 1)[0]
        return [max_over_time]


class PoolingFactory:
    """ Factory class for pooling layers. """
    poolings = {
        'max': MaxPooling,
        'cls': CLSPooling,
        'mean': MeanPooling,
        'mean_sqrt': MeanSqrtPooling,
        'mean_mean_sqrt': MeanMeanSqrtPooling,
        'selfattn': SelfAttentionPooling,
    }

    @classmethod
    def create(cls, pooling_type, **kwargs):
        """
        Example usage:
        pooling_args = {
            'hidden_dim': self.encoder_features
        }

        self.poolings = list(map(lambda pooling_type: PoolingFactory.create(pooling_type, **pooling_args)))
        """
        return cls.poolings[pooling_type](**kwargs)

    @classmethod
    def get_pooling_names(cls):
        return list(cls.poolings.keys())

    @classmethod
    def pool(cls, pooling_layers, pooling_args) -> List[torch.Tensor]:
        """
        (B, L, H) --> (B, N_Poolings*H)
        """
        output_vectors = []
        for pooling_layer in pooling_layers:
            results = pooling_layer(**pooling_args)
            for r in results:
                output_vectors.append(r)
        return torch.cat(output_vectors, 1)



class LinearLayer(nn.Module):
    """
    Single linear layer with activation, batch normalization, and dropout.

    Attributes:
        linear_layer (nn.Linear): A PyTorch linear layer.
        activation (nn.Module): The activation function to apply to the linear layer output.
        use_batch_norm (bool): Flag to indicate whether to use batch normalization.
        batch_norm (nn.BatchNorm1d): A PyTorch batch normalization layer.
        use_dropout (bool): Flag to indicate whether to use dropout.
        dropout (nn.Dropout): A PyTorch dropout layer.
    """

    def __init__(
            self,
            input_size: int,
            output_size: int,
            batch_norm: bool = True,
            dropout: float = 0,
            activation=nn.ReLU(),
    ):
        """
        Initializes a new instance of the `LinearLayer` class.

        Args:
            input_size (int): The size of the input data.
            output_size (int): The size of the output data.
            batch_norm (bool, optional): Flag to indicate whether to use batch normalization (default is True).
            dropout (float, optional): The dropout rate between 0 and 1 (default is 0).
            activation (nn.Module, optional): The activation function to apply to the linear layer output (default is ReLU).
        """
        super().__init__()

        self.linear_layer = nn.Linear(input_size, output_size)
        self.activation = activation

        self.use_batch_norm = batch_norm
        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(output_size)

        if 0 < dropout > 1:
            raise ValueError("Dropout has to be between 0 and 1")
        self.use_dropout = dropout > 0
        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the linear layer, activation, batch normalization, and dropout to the input data.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            The processed data.
        """
        x = self.linear_layer(x)
        if self.use_batch_norm:
            x = self.batch_norm(x)
        x = self.activation(x)
        if self.use_dropout:
            x = self.dropout(x)
        return x

from typing import Sequence




class DenseNet(nn.Module):
    """Multiple linear layers with the same activation, batch norm and dropout.

    Attributes:
        layers (nn.ModuleList): A list of `LinearLayer` objects.
    """

    def __init__(self,
                 layer_sizes: Sequence[int],
                 inner_activation=nn.ReLU(),
                 head_activation=nn.Softmax(dim=1),
                 batch_norm_inner: bool = True,
                 batch_norm_head: bool = False,
                 dropout: float = 0):
        """Initialize DenseNet.

        Args:
            layer_sizes (Sequence[int]): List containing widths of linear layers. Minimum 2 values required.
            inner_activation (nn.Module, optional): Activation function for all but the final layer. Default is ReLU.
            head_activation (nn.Module, optional): Activation function for the final layer. Default is Softmax.
            batch_norm_inner (bool, optional): Flag to indicate if inner layers should use batch normalization. Default is True.
            batch_norm_head (bool, optional): Flag to indicate if the final layer should use batch normalization. Default is False.
            dropout (float, optional): Dropout rate, between 0 and 1. Default is 0.

        Raises:
            ValueError: If `layer_sizes` has less than 2 values.
        """
        super().__init__()

        if len(layer_sizes) < 2:
            raise ValueError("Not enough layers to create linear layers module")

        layers = []
        for i in range(len(layer_sizes) - 2):
            layers.append(LinearLayer(layer_sizes[i], layer_sizes[i + 1], batch_norm_inner, dropout, inner_activation))

        layers.append(LinearLayer(layer_sizes[-2], layer_sizes[-1], batch_norm_head, 0, head_activation))

        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through all `LinearLayer` objects in `layers`.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through all `LinearLayer` objects.
        """
        for layer in self.layers:
            x = layer(x)
        return x


def copy_layers(src_layers: nn.ModuleList, dest_layers: nn.ModuleList, layers_to_copy: List[int]) -> None:
    """
    Copies the specified layers from the source layers to the destination layers.

    Args:
        src_layers (nn.ModuleList): The source layers from which to copy.
        dest_layers (nn.ModuleList): The destination layers to which to copy.
        layers_to_copy (List[int]): The indices of the layers to be copied.

    Returns:
        None
    """
    print(f"Copying {layers_to_copy}")
    layers_to_copy = nn.ModuleList([src_layers[i] for i in layers_to_copy])
    assert len(dest_layers) == len(layers_to_copy), f"{len(dest_layers)} != {len(layers_to_copy)}"
    dest_layers.load_state_dict(layers_to_copy.state_dict())


def tokenize(seqs: List[str], tokenizer, device: torch.device, max_len: int, padding_strategy: str = 'longest'):
    """
    This function tokenizes a list of text sequences using a provided tokenizer and converts them to torch tensors.

    Args:
    seqs (List[str]): a list of text sequences to be tokenized
    tokenizer: a tokenizer object to be used for tokenization
    device (torch.device): device on which the tensors should be created
    max_len (int): maximum length of the tokenized sequences

    Returns:
    tuple: a tuple of torch tensors - input_ids, attention_mask
    input_ids (torch.Tensor): tensor of shape (BS, max_len) representing the tokenized input sequences
    attention_mask (torch.Tensor): tensor of shape (BS, max_len) representing the attention mask for input sequences
    """
    inputs = tokenizer.batch_encode_plus(
        seqs, add_special_tokens=True, padding=padding_strategy, truncation=True, max_length=max_len)  # (BS, max_len)

    input_ids = torch.tensor(inputs.input_ids)  # (BS, max_len)
    attention_mask = torch.tensor(inputs.attention_mask)  # (BS, max_len)

    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    return input_ids, attention_mask


class BertForSequenceClassificationBase(nn.Module, ABC):

    def freeze_first_n_blocks(self, layers_with_indexes: nn.Module, n: int) -> None:
        """
        Freeze n first layers in the transformer model.

        Args:
            layers_with_indexes (nn.Module): The transformer model to be modified.
            n (int): The number of first encoder blocks to freeze.

        Returns:
            None
        """
        if n <= 0:
            print("n <= 0, not freezing any blocks")
            return
        else:
            print(f"Freezing first {n} blocks")

        names = list(name for name, _ in layers_with_indexes.named_parameters())
        print(f'names: {names}')
        idxs = sorted(list(set([int(name.split('.')[0]) for name in names])))
        print(f'idxs: {idxs}')

        idxs_to_freeze = list(map(str, idxs[:n]))
        print(f'idxs_to_freeze: {idxs_to_freeze}')

        for name, param in layers_with_indexes.named_parameters():
            if any([name.split('.')[0] == idx for idx in idxs_to_freeze]):
                print(f'Freezing {name}')
                param.requires_grad = False

    @abstractmethod
    def forward(self, sequences) -> torch.Tensor:
        pass

from transformers import (AutoConfig, AutoModelForSequenceClassification, AutoTokenizer)
from transformers.models.bert.modeling_bert import BertEncoder, BertModel


class BertForSequenceClassification(BertForSequenceClassificationBase):

    def __init__(
        self,
        encoder_features: int,
        n_classes: int,
        freeze_blocks_number: int,
        model_name: str,
        max_len: int,
        encoder_block_prefix: str,
        pooling_list: List[str],
        hidden_head_layer_sizes: List[int] = [],
        dropout_head: float = 0.,
    ):
        super().__init__()
        self.model_name = model_name
        self.encoder_features = encoder_features
        self.n_classes = n_classes
        self.max_len = max_len
        self.freeze_blocks_number = freeze_blocks_number
        self.hidden_head_layer_sizes = hidden_head_layer_sizes
        self.dropout_head = dropout_head
        self.pooling_list = pooling_list

        pooling_class_args = {'hidden_dim': self.encoder_features}
        self.pooling_layers = list(
            map(lambda pooling_type: PoolingFactory.create(pooling_type, **pooling_class_args), self.pooling_list))
        self.number_of_poolings_methods = sum(map(lambda layer: layer.output_vectors_no, self.pooling_layers))
        if self.number_of_poolings_methods < 1:
            raise ValueError('At least one pooling method is required')

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, do_lower_case=False)
        config = AutoConfig.from_pretrained(self.model_name)

        self.transformer_body = BertModel(config, add_pooling_layer=False)

        head_architecture = [
            self.encoder_features * self.number_of_poolings_methods, *self.hidden_head_layer_sizes, self.n_classes
        ]

        print("head_architecture", head_architecture)

        self.classification_head = DenseNet(
            head_architecture,
            batch_norm_inner=True,
            batch_norm_head=False,
            dropout=dropout_head,
            inner_activation=nn.ReLU(),
            head_activation=nn.Softmax(dim=1))

        self.encoder_block_prefix = encoder_block_prefix
        self.freeze_first_n_blocks(self.transformer_body.encoder.layer, self.freeze_blocks_number)

    def forward(
            self,
            sequences: List[str]  # len(...) = BS  # ['X Y Z [SEP] A B C']  ,
    ) -> torch.Tensor:
        device = next(self.transformer_body.parameters()).device

        input_ids, attention_mask = tokenize(sequences, self.tokenizer, device, self.max_len)

        word_embeddings = self.transformer_body(input_ids, attention_mask)[0]  # (BS, MaxLen, HiddenDim)

        pooling_args = {
            "token_embeddings": word_embeddings,
            "attention_mask": attention_mask,
        }
        pooling_output = PoolingFactory.pool(self.pooling_layers, pooling_args)  # (BS, N_Poolings*HiddenDim)

        cls_head_output = self.classification_head(pooling_output)  # (bs, n_classes)
        return cls_head_output


class DualBertForSequenceClassification(BertForSequenceClassificationBase):

    def __init__(
        self,
        encoder_features: int,
        n_classes: int,
        freeze_blocks_number: int,
        model_name: str,
        max_len: int,
        n_head_layers: int,
        pooling_list: List[str],
        hidden_head_layer_sizes: List[int] = [],
        dropout_head: float = 0.,
    ):
        super().__init__()
        self.model_name = model_name
        self.encoder_features = encoder_features
        self.n_classes = n_classes
        self.max_len = max_len
        self.freeze_blocks_number = freeze_blocks_number
        self.hidden_head_layer_sizes = hidden_head_layer_sizes
        self.dropout_head = dropout_head
        self.pooling_list = pooling_list

        pooling_class_args = {'hidden_dim': self.encoder_features}
        self.pooling_layers = list(
            map(lambda pooling_type: PoolingFactory.create(pooling_type, **pooling_class_args), self.pooling_list))
        self.number_of_poolings_methods = sum(map(lambda layer: layer.output_vectors_no, self.pooling_layers))
        if self.number_of_poolings_methods < 1:
            raise ValueError('At least one pooling method is required')

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, do_lower_case=False)
        transformer_big = AutoModelForSequenceClassification.from_pretrained(model_name)

        # Dual Bert
        assert n_head_layers <= len(
            transformer_big.bert.encoder.layer), f"{n_head_layers} > {len(transformer_big.bert.encoder.layer)}"

        n_tail_layers = len(transformer_big.bert.encoder.layer) - n_head_layers

        config_tail = AutoConfig.from_pretrained(model_name, num_hidden_layers=n_tail_layers)
        config_head = AutoConfig.from_pretrained(model_name, num_hidden_layers=n_head_layers)

        self.transformer_mhc = BertModel(config_tail, add_pooling_layer=False)
        self.transformer_pep = BertModel(config_tail, add_pooling_layer=False)
        self.transformer_head = BertEncoder(config_head)

        copy_layers(transformer_big.bert.encoder.layer, self.transformer_mhc.encoder.layer, list(range(n_tail_layers)))
        copy_layers(transformer_big.bert.encoder.layer, self.transformer_pep.encoder.layer, list(range(n_tail_layers)))
        copy_layers(transformer_big.bert.encoder.layer, self.transformer_head.layer,
                    list(range(n_tail_layers, len(transformer_big.bert.encoder.layer))))

        head_architecture = [
            self.encoder_features * self.number_of_poolings_methods, *self.hidden_head_layer_sizes, self.n_classes
        ]

        self.classification_head = DenseNet(
            head_architecture,
            batch_norm_inner=True,
            batch_norm_head=False,
            dropout=dropout_head,
            inner_activation=nn.ReLU(),
            head_activation=nn.Softmax(dim=1))

        self.classification_head.train()

        # Freezing first N blocks
        to_freeze_in_tail = min(self.freeze_blocks_number, n_tail_layers)
        to_freeze_in_head = max(0, self.freeze_blocks_number - n_tail_layers)

        self.freeze_first_n_blocks(self.transformer_mhc.encoder.layer, to_freeze_in_tail)
        self.freeze_first_n_blocks(self.transformer_pep.encoder.layer, to_freeze_in_tail)
        self.freeze_first_n_blocks(self.transformer_head.layer, to_freeze_in_head)

    def forward(
            self,
            sequences: List[str]  # len(...) = BS  # ['X Y Z [SEP] A B C']  ,
    ) -> torch.Tensor:
        # Get device
        device = next(self.parameters()).device

        token_ids_mhc, _ = tokenize([seq.split(' [SEP] ')[0] for seq in sequences], self.tokenizer, device,
                                    self.max_len)
        token_ids_pep, _ = tokenize([seq.split(' [SEP] ')[1] for seq in sequences], self.tokenizer, device,
                                    self.max_len)

        embds_mhc = self.transformer_mhc(token_ids_mhc).last_hidden_state
        embds_pep = self.transformer_pep(token_ids_pep).last_hidden_state

        merged_embds = torch.cat([embds_mhc, embds_pep], dim=1)

        embds_head = self.transformer_head(merged_embds).last_hidden_state

        att_m = torch.ones_like(embds_head[:, :, 0])
        pooling_args = {
            "token_embeddings": embds_head,
            "attention_mask": att_m,
        }
        pooling_output = PoolingFactory.pool(self.pooling_layers, pooling_args)  # (BS, N_Poolings*HiddenDim)

        cls_head_output = self.classification_head(pooling_output)  # (bs, n_classes)
        return cls_head_output
