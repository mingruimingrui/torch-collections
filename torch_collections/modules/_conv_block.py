import torch


class ConvBlock2d(torch.nn.Module):
    def __init__(
        self,
        input_feature_size,
        output_feature_size,
        internal_feature_size=None,
        num_layers=4,
        batch_norm=True,
        dropout=None,
        bias=False,
        bias_initializer=None,
        weight_initializer=None
    ):
        """ A trivial implementation of 3x3 conv layers stacked on top of each other
        Implemented only to make code more streamline and easier to read

        Args
            input_feature_size    : Input feature size
            output_feature_size   : Output feature size
            internal_feature_size : Feature size inside the block (if None, uses input_feature_size)
            num_layers : The number of conv layers in this block (excluding final conv layer)
            batch_norm : Flag to raise if batch normalization should be applied at start of conv block
            dropout    : Percent dropout, 0 if None
            bias       : Flag to raise if conv layers should have bias
        """
        super(ConvBlock2d, self).__init__()
        self.num_layers = num_layers
        self.batch_norm = batch_norm
        self.dropout = dropout

        if not internal_feature_size:
            internal_feature_size = input_feature_size

        self.relu = torch.nn.ReLU(inplace=False)

        if self.batch_norm:
            self.batch_norm = torch.nn.BatchNorm2d(input_feature_size)

        for i in range(num_layers):
            current_input_size = input_feature_size if i == 0 else internal_feature_size
            current_output_size = output_feature_size if i == num_layers - 1 else internal_feature_size

            # Create new layer
            conv_layer = torch.nn.Conv2d(
                current_input_size,
                current_output_size,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=bias
            )

            # init bias and weight
            if bias and bias_initializer:
                bias_initializer(conv_layer.bias)
            if weight_initializer:
                weight_initializer(conv_layer.weight)

            # Save layer
            setattr(self, 'conv{}'.format(i), conv_layer)

        if self.dropout:
            self.dropout = torch.nn.Dropout2d(p=self.dropout, inplace=False)

    def forward(self, x):
        if self.batch_norm:
            x = self.batch_norm(x)
            x = self.relu(x)

        for i in range(self.num_layers):
            x = getattr(self, 'conv{}'.format(i))(x)
            x = self.relu(x)

        if self.dropout:
            x = self.dropout(x)

        return x
