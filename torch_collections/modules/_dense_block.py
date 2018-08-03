import torch


class DenseBlock2d(torch.nn.Module):
    def __init__(
        self,
        input_feature_size,
        num_layers=4,
        growth_rate=64,
        batch_norm=True,
        transition=True,
        dropout=None,
        bias=False,
        output_feature_size=256,
        bias_initializer=None,
        weight_initializer=None
    ):
        """ Own implementation of a dense like block
        BN only applied at start and during transition (for efficiency reasons)
        Relu applied after BN and every conv (exception of last conv layer in transition)
        All conv filters have 3x3 kernel sizes, 1 stride and 1 padding
        transition does not apply pooling (go do it yourself)

        Args
            input_feature_size  : Input feature size
            output_feature_size : Output feature size (only applicable if transition is True)
            num_layers  : The number of conv layers in this dense block (not counting transition)
            growth_rate : The feature growth rate
            batch_norm  : Flag to raise if BN layers should be used
            transition  : Flag to raise if transition layer should be applied (it will not apply pooling)
            dropout     : Percent dropout, 0 if None
            bias        : Flag to raise if conv layers should have bias
        """
        super(DenseBlock2d, self).__init__()
        self.num_layers = num_layers
        self.batch_norm = batch_norm
        self.transition = transition
        self.dropout = dropout

        self.relu = torch.nn.ReLU(inplace=False)

        if self.batch_norm:
            self.batch_norm = torch.nn.BatchNorm2d(input_feature_size)

        rolling_feature_size = input_feature_size
        for i in range(self.num_layers):
            # Create new layer
            conv_layer = torch.nn.Conv2d(
                rolling_feature_size,
                growth_rate,
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

            # Save layer and update variable
            setattr(self, 'conv{}'.format(i), conv_layer)
            rolling_feature_size += growth_rate

        if self.transition:
            if self.batch_norm:
                self.batch_norm_final = torch.nn.BatchNorm2d(rolling_feature_size)
            self.conv_transition = torch.nn.Conv2d(
                rolling_feature_size,
                output_feature_size,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=bias
            )

        if self.dropout:
            self.dropout = torch.nn.Dropout2d(p=self.dropout, inplace=False)

    def forward(self, x):
        if self.batch_norm:
            x = self.batch_norm(x)
            x = self.relu(x)

        for i in range(self.num_layers):
            x_ = getattr(self, 'conv{}'.format(i))(x)
            x_ = self.relu(x_)
            x = torch.cat([x, x_], dim=1)

        if self.transition:
            if self.batch_norm:
                x = self.batch_norm_final(x)
                x = self.relu(x)
            x = self.conv_transition(x)

        if self.dropout:
            x = self.dropout(x)

        return x
