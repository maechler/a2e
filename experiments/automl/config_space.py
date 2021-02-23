import ConfigSpace as cs
import ConfigSpace.hyperparameters as csh


def create_config_space(
    hidden_layers: bool = True,
    scaling: bool = True,
    learning: bool = True,
    loss: bool = True,
    easing: bool = True,
    activation_functions: bool = True,
    dropout: bool = True,
    activity_regularizer: bool = True,
    min_dropout_rate_input: float = 0.0,
    min_dropout_rate_hidden_layers: float = 0.0,
    min_dropout_rate_output: float = 0.0,
    max_dropout_rate_input: float = 0.99,
    max_dropout_rate_hidden_layers: float = 0.99,
    max_dropout_rate_output: float = 0.99,
):
    config_space = cs.ConfigurationSpace(seed=1234)

    config_space.add_hyperparameters([
        csh.Constant('input_dimension', value=1025),
        csh.UniformIntegerHyperparameter('latent_dimension', lower=10, upper=1024, log=True),
    ])

    if hidden_layers:
        config_space.add_hyperparameters([
            csh.OrdinalHyperparameter('number_of_hidden_layers', list(range(1, 11, 2))),
        ])

    if scaling:
        config_space.add_hyperparameters([
            csh.CategoricalHyperparameter('_scaler', [
                'none',
                'min_max',
                'std',
            ]),
        ])

    if learning:
        config_space.add_hyperparameters([
            csh.UniformIntegerHyperparameter('_batch_size', lower=16, upper=512),
            csh.UniformFloatHyperparameter('learning_rate', lower=1e-6, upper=1e-1, log=True),
            csh.UniformFloatHyperparameter('learning_rate_decay', lower=1e-8, upper=1e-2, log=True),
        ])

    if loss:
        config_space.add_hyperparameters([
            csh.CategoricalHyperparameter('loss', ['mse', 'mae', 'binary_crossentropy']),
        ])

    if easing:
        config_space.add_hyperparameters([
            csh.CategoricalHyperparameter('easing', ['ease_linear', 'ease_in_quad', 'ease_out_quad']),
        ])

    if activation_functions:
        config_space.add_hyperparameters([
            csh.CategoricalHyperparameter('hidden_layer_activations', ['relu', 'linear', 'sigmoid', 'tanh']),
            csh.CategoricalHyperparameter('output_layer_activation', ['relu', 'linear', 'sigmoid', 'tanh']),
        ])

    if dropout:
        config_space.add_hyperparameters([
            csh.UniformFloatHyperparameter('dropout_rate_input', lower=min_dropout_rate_input, upper=max_dropout_rate_input),
            csh.UniformFloatHyperparameter('dropout_rate_hidden_layers', lower=min_dropout_rate_hidden_layers, upper=max_dropout_rate_hidden_layers),
            csh.UniformFloatHyperparameter('dropout_rate_output', lower=min_dropout_rate_output, upper=max_dropout_rate_output),
        ])

    if activity_regularizer:
        config_space.add_hyperparameters([
            csh.CategoricalHyperparameter('activity_regularizer', ['l1', 'l2']),
            csh.UniformFloatHyperparameter('l1_activity_regularizer_factor', lower=1e-6, upper=1e-1, default_value=1e-2, log=True),
            csh.UniformFloatHyperparameter('l2_activity_regularizer_factor', lower=1e-6, upper=1e-1, default_value=1e-2, log=True),
        ])

        config_space.add_condition(cs.EqualsCondition(config_space.get_hyperparameter('l1_activity_regularizer_factor'), config_space.get_hyperparameter('activity_regularizer'), 'l1'))
        config_space.add_condition(cs.EqualsCondition(config_space.get_hyperparameter('l2_activity_regularizer_factor'), config_space.get_hyperparameter('activity_regularizer'), 'l2'))

    return config_space