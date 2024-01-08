"""Module that stores several useful keys to configure a model.
"""

from . import constants

""" Input pipeline params

batch_size: (int) Size of the mini batches used for training.
shuffle_buffer_size: (int) Size of the buffer to shuffle the data. If 0, no 
    shuffle is applied.
prefetch_buffer_size: (int) Size of the buffer to prefetch the batches. If 0, 
    no prefetch is applied.
page_duration: (int) Size of a EEG page in seconds.
"""
CLIP_VALUE = 'clip_value'
NORM_COMPUTATION_MODE = 'norm_computation_mode'
BATCH_SIZE = 'batch_size'
SHUFFLE_BUFFER_SIZE = 'shuffle_buffer_size'
PREFETCH_BUFFER_SIZE = 'prefetch_buffer_size'
PAGE_DURATION = 'page_duration'
AUG_RESCALE_NORMAL_PROBA = 'aug_rescale_normal_proba'
AUG_RESCALE_NORMAL_STD = 'aug_rescale_normal_std'
AUG_GAUSSIAN_NOISE_PROBA = 'aug_gaussian_noise_proba'
AUG_GAUSSIAN_NOISE_STD = 'aug_gaussian_noise_std'
AUG_RESCALE_UNIFORM_PROBA = 'aug_rescale_uniform_proba'
AUG_RESCALE_UNIFORM_INTENSITY = 'aug_rescale_uniform_intensity'
AUG_ELASTIC_PROBA = 'aug_elastic_proba'
AUG_ELASTIC_ALPHA = 'aug_elastic_alpha'
AUG_ELASTIC_SIGMA = 'aug_elastic_sigma'

AUG_INDEP_GAUSSIAN_NOISE_PROBA = "aug_indep_gaussian_noise_proba"
AUG_INDEP_GAUSSIAN_NOISE_STD = "aug_indep_gaussian_noise_std"

AUG_INDEP_UNIFORM_NOISE_PROBA = "aug_indep_uniform_noise_proba"
AUG_INDEP_UNIFORM_NOISE_INTENSITY = "aug_indep_uniform_noise_intensity"

AUG_RANDOM_WAVES_PROBA = "aug_random_waves_proba"
AUG_RANDOM_WAVES_PARAMS = "aug_random_waves_params"
AUG_RANDOM_ANTI_WAVES_PROBA = "aug_random_anti_waves_proba"
AUG_RANDOM_ANTI_WAVES_PARAMS = "aug_random_anti_waves_params"

AUG_FALSE_SPINDLES_SINGLE_CONT_PROBA = "aug_false_spindles_single_cont_proba"
AUG_FALSE_SPINDLES_SINGLE_CONT_PARAMS = "aug_false_spindles_single_cont_params"

""" Model params
time_resolution_factor: (int) The original sampling frequency for the labels
    is downsampled using this factor.
fs: (float) Sampling frequency of the signals of interest.
border_duration: (int) Non-negative integer that
    specifies the number of seconds to be removed at each border at the
    end. This parameter allows to input a longer signal than the final
    desired size to remove border effects of the CWT.
fb_list: (list of floats) list of values for Fb (one for each scalogram)
n_conv_blocks: ({0, 1, 2, 3}) Indicates the
    number of convolutional blocks to be performed after the CWT and
    before the BLSTM. If 0, no blocks are applied.
n_time_levels: ({1, 2, 3}) Indicates the number
    of stages for the recurrent part, building a ladder-like network.
    If 1, it's a simple 2-layers BLSTM network.
batchnorm_conv: (Optional, {None, BN, BN_RENORM}, defaults to BN_RENORM)
    Type of batchnorm to be used in the convolutional blocks. BN is
    normal batchnorm, and BN_RENORM is a batchnorm with renorm
    activated. If None, batchnorm is not applied.
batchnorm_first_lstm: (Optional, {None, BN, BN_RENORM}, defaults to
    BN_RENORM) Type of batchnorm to be used in the first BLSTM layer.
    BN is normal batchnorm, and BN_RENORM is a batchnorm with renorm
    activated. If None, batchnorm is not applied.
dropout_first_lstm: (Optional, {None REGULAR_DROP, SEQUENCE_DROP},
    defaults to None) Type of dropout to be used in the first BLSTM
    layer. REGULAR_DROP is regular  dropout, and SEQUENCE_DROP is a
    dropout with the same noise shape for each time_step. If None,
    dropout is not applied. The dropout layer is applied after the
    batchnorm.
batchnorm_rest_lstm: (Optional, {None, BN, BN_RENORM}, defaults to
    None) Type of batchnorm to be used in the rest of BLSTM layers.
    BN is normal batchnorm, and BN_RENORM is a batchnorm with renorm
    activated. If None, batchnorm is not applied.
dropout_rest_lstm: (Optional, {None REGULAR_DROP, SEQUENCE_DROP},
    defaults to None) Type of dropout to be used in the rest of BLSTM
    layers. REGULAR_DROP is regular  dropout, and SEQUENCE_DROP is a
    dropout with the same noise shape for each time_step. If None,
    dropout is not applied. The dropout layer is applied after the
    batchnorm.
time_pooling: (Optional, {AVGPOOL, MAXPOOL}, defaults to AVGPOOL)
    Indicates the type of pooling to be performed to downsample
    the time dimension if n_time_levels > 1.
batchnorm_fc: (Optional, {None, BN, BN_RENORM}, defaults to
    None) Type of batchnorm to be used in the output layer.
    BN is normal batchnorm, and BN_RENORM is a batchnorm with renorm
    activated. If None, batchnorm is not applied.
dropout_fc: (Optional, {None REGULAR_DROP, SEQUENCE_DROP},
    defaults to None) Type of dropout to be used in output
    layer. REGULAR_DROP is regular  dropout, and SEQUENCE_DROP is a
    dropout with the same noise shape for each time_step. If None,
    dropout is not applied. The dropout layer is applied after the
    batchnorm.
drop_rate: (Optional, float, defaults to 0.5) Dropout rate. Fraction of
    units to be dropped. If dropout is None, this is ignored.
trainable_wavelet: (Optional, boolean, defaults to False) If True, the
    fb params will be trained with backprop.
type_wavelet: ({CMORLET, SPLINE}) Type of wavelet to be used.
use_log: (boolean) Whether to apply logarithm to the CWT output, after the 
    avg pool.
n_scales: (int) Number of scales to be computed for the scalogram.
lower_freq: (float) Lower frequency to be considered in the scalograms [Hz].
upper_freq: (float) Upper frequency to be considered in the scalograms [Hz].
initial_conv_filters: (int) Number of filters to be used in the first
    convolutional block, if applicable. Subsequent conv blocks will double the
    previous number of filters.
initial_lstm_units: (int) Number of units for lstm layers. If multi stage
    is used (n_time_levels > 1), after every time downsampling operation 
    the number of units is doubled.
"""
# General parameters
FS = 'fs'
MODEL_VERSION = 'model_version'
BORDER_DURATION = 'border_duration'
# Regularization
TYPE_BATCHNORM = 'batchnorm'
TYPE_DROPOUT = 'dropout'
DROP_RATE_BEFORE_LSTM = 'drop_rate_before_lstm'
DROP_RATE_HIDDEN = 'drop_rate_hidden'
DROP_RATE_OUTPUT = 'drop_rate_output'
# CWT stage
FB_LIST = 'fb_list'
TRAINABLE_WAVELET = 'trainable_wavelet'
WAVELET_SIZE_FACTOR = 'wavelet_size_factor'
USE_LOG = 'use_log'
N_SCALES = 'n_scales'
LOWER_FREQ = 'lower_freq'
UPPER_FREQ = 'upper_freq'
USE_RELU = 'use_relu'
CWT_NOISE_INTENSITY = "cwt_noise_intensity"
# Convolutional stage
INITIAL_KERNEL_SIZE = 'initial_kernel_size'
INITIAL_CONV_FILTERS = 'initial_conv_filters'
CONV_DOWNSAMPLING = 'conv_downsampling'
# blstm stage
INITIAL_LSTM_UNITS = 'initial_lstm_units'
# FC units in second to last layer
FC_UNITS = 'fc_units'
OUTPUT_LSTM_UNITS = 'output_lstm_units'
# Time-domain convolutional params
TIME_CONV_FILTERS_1 = 'time_conv_filters_1'
TIME_CONV_FILTERS_2 = 'time_conv_filters_2'
TIME_CONV_FILTERS_3 = 'time_conv_filters_3'
SIGMA_FILTER_NTAPS = 'sigma_filter_ntaps'
# cwt domain convolutional params
CWT_CONV_FILTERS_1 = 'cwt_conv_filters_1'
CWT_CONV_FILTERS_2 = 'cwt_conv_filters_2'
CWT_CONV_FILTERS_3 = 'cwt_conv_filters_3'
# General cwt
CWT_RETURN_REAL_PART = 'cwt_return_real_part'
CWT_RETURN_IMAG_PART = 'cwt_return_imag_part'
CWT_RETURN_MAGNITUDE = 'cwt_return_magnitude'
CWT_RETURN_PHASE = 'cwt_return_phase'
INIT_POSITIVE_PROBA = 'init_positive_proba'
# Upconv output
LAST_OUTPUT_CONV_FILTERS = 'last_output_conv_filters'
# UNET parameters
UNET_TIME_INITIAL_CONV_FILTERS = 'unet_time_initial_conv_filters'
UNET_TIME_LSTM_UNITS = 'unet_time_lstm_units'
UNET_TIME_N_DOWN = 'unet_time_n_down'
UNET_TIME_N_CONV_DOWN = 'unet_time_n_conv_down'
UNET_TIME_N_CONV_UP = 'unet_time_n_conv_up'
# Attention parameters
ATT_DIM = 'att_dim'
ATT_N_HEADS = 'att_n_heads'
ATT_PE_FACTOR = 'att_pe_factor'
ATT_DROP_RATE = 'att_drop_rate'
ATT_LSTM_DIM = 'att_lstm_dim'
ATT_PE_CONCAT_DIM = 'att_pe_concat_dim'
ABLATION_TYPE_BATCHNORM_INPUT = 'ablation_type_batchnorm_input'
ABLATION_TYPE_BATCHNORM_CONV = 'ablation_type_batchnorm_conv'
ABLATION_DROP_RATE = 'ablation_drop_rate'
OUTPUT_RESIDUAL_FC_SIZE = 'output_residual_fc_size'
OUTPUT_USE_BN = 'output_use_bn'
OUTPUT_USE_DROP = 'output_use_drop'
FC_UNITS_1 = 'fc_units_1'
FC_UNITS_2 = 'fc_units_2'
SHIELD_LSTM_DOWN_FACTOR = 'shield_lstm_down_factor'
SHIELD_LSTM_TYPE_POOL = 'shield_lstm_type_pool'
PR_RETURN_RATIOS = 'pr_return_ratios'
PR_RETURN_BANDS = 'pr_return_bands'
LLC_STFT_N_SAMPLES = 'llc_stft_n_samples'
LLC_STFT_FREQ_POOL = 'llc_stft_freq_pool'
LLC_STFT_USE_LOG = 'llc_stft_use_log'
LLC_STFT_N_HIDDEN = 'llc_stft_n_hidden'
LLC_STFT_DROP_RATE = 'llc_stft_drop_rate'
TCN_FILTERS = "tcn_filters"
TCN_KERNEL_SIZE = "tcn_kernel_size"
TCN_DROP_RATE = "tcn_drop_rate"
TCN_N_BLOCKS = "tcn_n_blocks"
TCN_USE_BOTTLENECK = 'tcn_use_bottleneck'
TCN_LAST_CONV_N_LAYERS = "tcn_last_conv_n_layers"
TCN_LAST_CONV_FILTERS = "tcn_last_conv_filters"
TCN_LAST_CONV_KERNEL_SIZE = "tcn_last_conv_kernel_size"
ATT_BANDS_V_INDEP_LINEAR = "att_bands_v_indep_linear"
ATT_BANDS_K_INDEP_LINEAR = "att_bands_k_indep_linear"
ATT_BANDS_V_ADD_BAND_ENC = "att_bands_v_add_band_enc"
ATT_BANDS_K_ADD_BAND_ENC = "att_bands_k_add_band_enc"
A7_WINDOW_DURATION = "a7_window_duration"
A7_WINDOW_DURATION_REL_SIG_POW = "a7_window_duration_rel_sig_pow"
A7_USE_LOG_ABS_SIG_POW = "a7_use_log_abs_sig_pow"
A7_USE_LOG_REL_SIG_POW = "a7_use_log_rel_sig_pow"
A7_USE_LOG_SIG_COV = "a7_use_log_sig_cov"
A7_USE_LOG_SIG_CORR = "a7_use_log_sig_corr"
A7_USE_ZSCORE_REL_SIG_POW = "a7_use_zscore_rel_sig_pow"
A7_USE_ZSCORE_SIG_COV = "a7_use_zscore_rel_sig_cov"
A7_USE_ZSCORE_SIG_CORR = "a7_use_zscore_rel_sig_corr"
A7_REMOVE_DELTA_IN_COV = "a7_remove_delta_in_cov"
A7_DISPERSION_MODE = "a7_dispersion_mode"
A7_CNN_FILTERS = "a7_cnn_filters"
A7_CNN_KERNEL_SIZE = "a7_cnn_kernel_size"
A7_CNN_N_LAYERS = "a7_cnn_n_layers"
A7_CNN_DROP_RATE = "a7_cnn_drop_rate"
A7_RNN_LSTM_UNITS = "a7_rnn_lstm_units"
A7_RNN_DROP_RATE = "a7_rnn_drop_rate"
A7_RNN_FC_UNITS = "a7_rnn_fc_units"
BP_INPUT_LOWCUT = "bp_input_lowcut"
BP_INPUT_HIGHCUT = "bp_input_highcut"
# Multi-kernel 1d conv params
TIME_CONV_MK_PROJECT_FIRST = "time_conv_mk_project_first"
TIME_CONV_MK_FILTERS_1 = 'time_conv_mk_filters_1'  # [(k1, f1), (k2, f2), ...]
TIME_CONV_MK_FILTERS_2 = 'time_conv_mk_filters_2'
TIME_CONV_MK_FILTERS_3 = 'time_conv_mk_filters_3'
TIME_CONV_MK_DROP_RATE = 'time_conv_mk_drop_rate'
TIME_CONV_MK_SKIPS = 'time_conv_mk_skips'
TIME_CONV_MKD_FILTERS_1 = 'time_conv_mkd_filters_1'  # [(k1, f1, d1), (k2, f2, d2), ...]
TIME_CONV_MKD_FILTERS_2 = 'time_conv_mkd_filters_2'
TIME_CONV_MKD_FILTERS_3 = 'time_conv_mkd_filters_3'
# Stat network parameters
# -- Backbone parameters
STAT_NET_CONV_KERNEL_SIZE = 'stat_net_conv_kernel_size'
STAT_NET_CONV_DEPTH = 'stat_net_conv_depth'
STAT_NET_CONV_TYPE_POOL = 'stat_net_conv_type_pool'
STAT_NET_CONV_INITIAL_FILTERS = 'stat_net_conv_initial_filters'
STAT_NET_CONV_MAX_FILTERS = 'stat_net_conv_max_filters'
STAT_NET_LSTM_UNITS = 'stat_net_lstm_units'
# -- General config
STAT_NET_TYPE_BACKBONE = 'stat_net_type_backbone'
STAT_NET_TYPE_COLLAPSE = 'stat_net_type_collapse'
STAT_NET_CONTEXT_DROP_RATE = 'stat_net_context_drop_rate'
STAT_NET_CONTEXT_DIM = 'stat_net_context_dim'
# -- Mod Net specific
STAT_MOD_NET_MODULATE_LOGITS = 'stat_mod_net_modulate_logits'
STAT_MOD_NET_BIASED_SCALE = 'stat_mod_net_biased_scale'
STAT_MOD_NET_BIASED_BIAS = 'stat_mod_net_biased_bias'
STAT_MOD_NET_USE_BIAS = 'stat_mod_net_use_bias'
# -- Dot Net specific
STAT_DOT_NET_PRODUCT_DIM = 'stat_dot_net_product_dim'
STAT_DOT_NET_BIASED_KERNEL = 'stat_dot_net_biased_kernel'
STAT_DOT_NET_BIASED_BIAS = 'stat_dot_net_biased_bias'
STAT_DOT_NET_USE_BIAS = 'stat_dot_net_use_bias'
# Bandpass signal decomposition
DECOMP_BP_USE_DILATION = 'decomp_bp_use_dilation'
DECOMP_BP_INITIAL_FILTERS = 'decomp_bp_initial_filters'
DECOMP_BP_EXTRA_CONV_FILTERS = 'decomp_bp_extra_conv_filters'
# Attention at blstm
ATT_USE_EXTRA_FC = 'att_use_extra_fc'
ATT_USE_ATTENTION_AFTER_BLSTM = 'att_use_attention_after_blstm'
# Expert branch parameters
EXPERT_BRANCH_WINDOW_DURATION = 'expert_branch_window_duration'
EXPERT_BRANCH_REL_POWER_BROAD_LOWCUT = 'expert_branch_rel_power_lowcut'
EXPERT_BRANCH_COVARIANCE_BROAD_LOWCUT = 'expert_branch_covariance_broad_lowcut'
EXPERT_BRANCH_ABS_POWER_TRANSFORMATION = 'expert_branch_abs_power_transformation'
EXPERT_BRANCH_REL_POWER_TRANSFORMATION = 'expert_branch_rel_power_transformation'
EXPERT_BRANCH_COVARIANCE_TRANSFORMATION = 'expert_branch_covariance_transformation'
EXPERT_BRANCH_CORRELATION_TRANSFORMATION = 'expert_branch_correlation_transformation'
EXPERT_BRANCH_REL_POWER_USE_ZSCORE = 'expert_branch_rel_power_use_zscore'
EXPERT_BRANCH_COVARIANCE_USE_ZSCORE = 'expert_branch_covariance_use_zscore'
EXPERT_BRANCH_CORRELATION_USE_ZSCORE = 'expert_branch_correlation_use_zscore'
EXPERT_BRANCH_ZSCORE_DISPERSION_MODE = 'expert_branch_zscore_dispersion_mode'
EXPERT_BRANCH_COLLAPSE_TIME_MODE = 'expert_branch_collapse_time_mode'
EXPERT_BRANCH_USE_ABS_POWER = 'expert_branch_use_abs_power'
EXPERT_BRANCH_USE_REL_POWER = 'expert_branch_use_rel_power'
EXPERT_BRANCH_USE_COVARIANCE = 'expert_branch_use_covariance'
EXPERT_BRANCH_USE_CORRELATION = 'expert_branch_use_correlation'
EXPERT_BRANCH_MODULATION_HIDDEN_FILTERS = 'expert_branch_modulation_hidden_filters'
EXPERT_BRANCH_MODULATION_HIDDEN_KERNEL_SIZE = 'expert_branch_modulation_hidden_kernel_size'
EXPERT_BRANCH_MODULATION_USE_SCALE = 'expert_branch_modulation_use_scale'
EXPERT_BRANCH_MODULATION_USE_BIAS = 'expert_branch_modulation_use_bias'
EXPERT_BRANCH_MODULATION_APPLY_SIGMOID_SCALE = 'expert_branch_modulation_apply_sigmoid_scale'
EXPERT_BRANCH_REGRESSION_HIDDEN_UNITS = 'expert_branch_regression_hidden_units'
EXPERT_BRANCH_REGRESSION_FROM_BLSTM = "expert_branch_regression_from_blstm"
EXPERT_BRANCH_REGRESSION_LOSS_COEFFICIENT = "expert_branch_regression_loss_coefficient"
BIGGER_STEM_KERNEL_SIZE = 'bigger_stem_kernel_size'
BIGGER_STEM_FILTERS = 'bigger_stem_filters'
BIGGER_BLOCKS_KERNEL_SIZE = 'bigger_blocks_kernel_size'
BIGGER_STAGE_1_SIZE = 'bigger_stage_1_size'
BIGGER_STAGE_2_SIZE = 'bigger_stage_2_size'
BIGGER_STAGE_3_SIZE = 'bigger_stage_3_size'
BIGGER_MAX_DILATION = 'bigger_max_dilation'
BIGGER_LSTM_1_SIZE = 'bigger_lstm_1_size'
BIGGER_LSTM_2_SIZE = 'bigger_lstm_2_size'
BIGGER_ATT_N_BLOCKS = 'bigger_att_n_blocks'
BIGGER_ATT_TYPE_NORM = 'bigger_att_type_norm'
BIGGER_CONVOLUTION_PART_OPTION = 'bigger_convolution_part_option'
BIGGER_CONTEXT_PART_OPTION = 'bigger_context_part_option'
BIGGER_STEM_DEPTH = 'bigger_stem_depth'
BIGGER_MULTI_TRANSFORMATION_BEFORE_ADD = 'bigger_multi_transformation_before_add'
# Final models
BORDER_DURATION_CWT = 'border_duration_cwt'
BORDER_DURATION_CONV = 'border_duration_conv'
CWT_EXPANSION_FACTOR = 'cwt_expansion_factor'

""" Loss params

class_weights: ({None, BALANCED, array_like}) Determines the class
    weights to be applied when computing the loss. If None, no weights
    are applied. If BALANCED, the weights balance the class
    frequencies. If is an array of shape [2,], class_weights[i]
    is the weight applied to class i.
type_loss: ({CROSS_ENTROPY_LOSS, DICE_LOSS}) Type of loss to be used 
"""
CLASS_WEIGHTS = 'class_weights'
TYPE_LOSS = 'type_loss'
FOCUSING_PARAMETER = 'focusing_parameter'
WORST_MINING_MIN_NEGATIVE = 'worst_mining_min_negative'
WORST_MINING_FACTOR_NEGATIVE = 'worst_mining_factor_negative'
NEG_ENTROPY_PARAMETER = 'neg_entropy_parameter'
SOFT_LABEL_PARAMETER = 'soft_label_parameter'
MIS_WEIGHT_PARAMETER = 'mis_weight_parameter'
BORDER_WEIGHT_AMPLITUDE = 'border_weight_amplitude'
BORDER_WEIGHT_HALF_WIDTH = 'border_weight_half_width'
MIX_WEIGHTS_STRATEGY = 'mix_weights_strategy'
PREDICTION_VARIABILITY_REGULARIZER = 'prediction_variability_regularizer'
PREDICTION_VARIABILITY_LAG = 'prediction_variability_lag'

SOFT_FOCAL_GAMMA = "soft_focal_gamma"
SOFT_FOCAL_EPSILON = "soft_focal_epsilon"
ANTIBORDER_AMPLITUDE = "antiborder_amplitude"
ANTIBORDER_HALF_WIDTH = "antiborder_half_width"

LOGITS_REG_TYPE = "logits_reg_type"
LOGITS_REG_WEIGHT = "logits_reg_weight"


""" Optimizer params

learning_rate: (float) learning rate for the optimizer
clip_norm: (float) this is the global norm to use to clip gradients. If None,
    no clipping is applied.
momentum: (float) momentum for the SGD optimizer.
use_nesterov: (bool) whether to use nesterov momentum instead of regular
    momentum for SGD optimization.
type_optimizer: ({ADAM_OPTIMIZER, SGD_OPTIMIZER}) Type of optimizer to be used.
"""
LEARNING_RATE = 'learning_rate'
CLIP_NORM = 'clip_norm'
MOMENTUM = 'momentum'
USE_NESTEROV_MOMENTUM = 'use_nesterov'
TYPE_OPTIMIZER = 'type_optimizer'
WEIGHT_DECAY_FACTOR = 'weight_decay_factor'


""" Training params

max_epochs: (int) Maximum numer of epochs to be performed in the training loop.
nstats: (int) Frequency in iterations to display metrics.
"""
MAX_ITERS = 'max_iters'
ITERS_STATS = 'iters_stats'
ITERS_LR_UPDATE = 'iters_lr_update'
REL_TOL_CRITERION = 'rel_tol_criterion'
LR_UPDATE_FACTOR = 'lr_update_factor'
LR_UPDATE_CRITERION = 'lr_update_criterion'
MAX_LR_UPDATES = 'max_lr_updates'
FACTOR_INIT_LR_FINE_TUNE = 'factor_init_lr_fine_tune'
LR_UPDATE_RESET_OPTIMIZER = 'lr_update_reset_optimizer'
KEEP_BEST_VALIDATION = 'keep_best_validation'
FORCED_SEPARATION_DURATION = 'forced_separation_duration'
PRETRAIN_ITERS_INIT = 'pretrain_iters_init'
PRETRAIN_ITERS_ANNEAL = 'pretrain_iters_anneal'
PRETRAIN_MAX_LR_UPDATES = 'pretrain_max_lr_updates'
MAX_EPOCHS = 'max_epochs'
STATS_PER_EPOCH = 'stats_per_epoch'
EPOCHS_LR_UPDATE = 'epochs_lr_update'
PRETRAIN_EPOCHS_INIT = 'pretrain_epochs_init'
PRETRAIN_EPOCHS_ANNEAL = 'pretrain_epochs_anneal'
VALIDATION_AVERAGE_MODE = 'validation_average_mode'

""" Postprocessing params 
"""
TOTAL_DOWNSAMPLING_FACTOR = 'total_downsampling_factor'
ALIGNED_DOWNSAMPLING = 'aligned_downsampling'
SS_MIN_SEPARATION = 'ss_min_separation'
SS_MIN_DURATION = 'ss_min_duration'
SS_MAX_DURATION = 'ss_max_duration'
KC_MIN_SEPARATION = 'kc_min_separation'
KC_MIN_DURATION = 'kc_min_duration'
KC_MAX_DURATION = 'kc_max_duration'
REPAIR_LONG_DETECTIONS = 'repair_long_detections'

"""Inference params"""
PREDICT_WITH_AUGMENTED_PAGE = 'predict_with_augmented_page'


""" Default utility params 
"""
DEFAULT_BORDER_DURATION_V2_TIME = 2.60
DEFAULT_BORDER_DURATION_V2_CWT = 4.91
DEFAULT_AUG_INDEP_UNIFORM_NOISE_INTENSITY_MICROVOLTS = 1
DEFAULT_AUG_RANDOM_WAVES_PARAMS_SPINDLE = [
    dict(
        min_frequency=0.5, max_frequency=2, frequency_bandwidth=1, max_amplitude_microvolts=18,
        min_duration=3, max_duration=5, mask=constants.MASK_NONE),
    # dict(
    #     min_frequency=2, max_frequency=4, frequency_bandwidth=1, max_amplitude_microvolts=13,
    #     min_duration=3, max_duration=5, mask=constants.MASK_NONE),
    dict(
        min_frequency=4, max_frequency=8, frequency_bandwidth=2, max_amplitude_microvolts=20,
        min_duration=1, max_duration=5, mask=constants.MASK_KEEP_BACKGROUND),
    dict(
        min_frequency=7, max_frequency=10, frequency_bandwidth=2, max_amplitude_microvolts=12,
        min_duration=1, max_duration=5, mask=constants.MASK_KEEP_BACKGROUND),
]
DEFAULT_AUG_RANDOM_WAVES_PARAMS_KCOMPLEX = [
    dict(
        min_frequency=11, max_frequency=16, frequency_bandwidth=2, max_amplitude_microvolts=10,
        min_duration=1, max_duration=5, mask=constants.MASK_NONE),
]
DEFAULT_AUG_RANDOM_ANTI_WAVES_PARAMS_SPINDLE = [
    dict(
        lowcut=None, highcut=2, max_attenuation=0.5,
        min_duration=3, max_duration=5, mask=constants.MASK_NONE),
    dict(
        lowcut=4, highcut=8, max_attenuation=1.0,
        min_duration=1, max_duration=5, mask=constants.MASK_KEEP_EVENTS),
    dict(
        lowcut=7, highcut=10, max_attenuation=1.0,
        min_duration=1, max_duration=5, mask=constants.MASK_KEEP_EVENTS),
    # dict(
    #     lowcut=11, highcut=16, max_attenuation=1.0,
    #     min_duration=1, max_duration=5, mask=constants.MASK_KEEP_BACKGROUND),
]
DEFAULT_AUG_RANDOM_ANTI_WAVES_PARAMS_KCOMPLEX = [
    dict(
        lowcut=11, highcut=16, max_attenuation=1.0,
        min_duration=1, max_duration=5, mask=constants.MASK_NONE),
]


""" Default parameter dictionary 
"""
default_params = {
    # Input pipeline
    FS: 200,
    CLIP_VALUE: 10,
    NORM_COMPUTATION_MODE: constants.NORM_GLOBAL,
    SHUFFLE_BUFFER_SIZE: 100000,
    PREFETCH_BUFFER_SIZE: 5,
    PAGE_DURATION: 20,
    BORDER_DURATION_CWT: 2.31,
    BORDER_DURATION_CONV: 0.6,
    BORDER_DURATION: None,
    AUG_INDEP_UNIFORM_NOISE_PROBA: 1.0,
    AUG_INDEP_UNIFORM_NOISE_INTENSITY: None,
    AUG_RANDOM_WAVES_PROBA: 1.0,
    AUG_RANDOM_WAVES_PARAMS: None,
    AUG_RANDOM_ANTI_WAVES_PROBA: 1.0,
    AUG_RANDOM_ANTI_WAVES_PARAMS: None,
    FORCED_SEPARATION_DURATION: 0,
    TOTAL_DOWNSAMPLING_FACTOR: 8,
    ALIGNED_DOWNSAMPLING: True,
    # CWT
    CWT_EXPANSION_FACTOR: 0.9,
    FB_LIST: [0.1323],
    TRAINABLE_WAVELET: True,
    WAVELET_SIZE_FACTOR: 1.5,
    N_SCALES: 32,
    LOWER_FREQ: 0.5,
    UPPER_FREQ: 30,
    CWT_NOISE_INTENSITY: 0.02,
    CWT_RETURN_REAL_PART: True,
    CWT_RETURN_IMAG_PART: True,
    CWT_RETURN_MAGNITUDE: False,
    CWT_RETURN_PHASE: False,
    # Architecture
    MODEL_VERSION: None,  # V2_TIME or V2_CWT1D
    BIGGER_STEM_FILTERS: 64,
    BIGGER_MAX_DILATION: 8,
    BIGGER_LSTM_1_SIZE: 256,
    BIGGER_LSTM_2_SIZE: 256,
    FC_UNITS: 128,
    TYPE_BATCHNORM: constants.BN,
    TYPE_DROPOUT: constants.SEQUENCE_DROP,
    DROP_RATE_BEFORE_LSTM: 0.2,
    DROP_RATE_HIDDEN: 0.5,
    DROP_RATE_OUTPUT: 0.0,
    INIT_POSITIVE_PROBA: 0.1,
    # Loss
    TYPE_LOSS: constants.MASKED_SOFT_FOCAL_LOSS,
    SOFT_FOCAL_GAMMA: 0.0,  # 2.5,
    SOFT_FOCAL_EPSILON: 1.0,  # 0.5,
    CLASS_WEIGHTS: [1.0, 1.0],  # [1.0, 0.5],
    # Optimization
    BATCH_SIZE: 32,
    TYPE_OPTIMIZER: constants.ADAM_OPTIMIZER,
    LEARNING_RATE: 1e-4,
    CLIP_NORM: 1,
    REL_TOL_CRITERION: 0.0,
    LR_UPDATE_FACTOR: 0.5,
    LR_UPDATE_CRITERION: constants.METRIC_CRITERION,
    LR_UPDATE_RESET_OPTIMIZER: False,
    KEEP_BEST_VALIDATION: True,
    MAX_EPOCHS: 200,
    STATS_PER_EPOCH: 5,
    EPOCHS_LR_UPDATE: 5,
    MAX_LR_UPDATES: 4,
    VALIDATION_AVERAGE_MODE: None,
    # Transfer learning
    PRETRAIN_EPOCHS_INIT: None,  # for fit_without_validation
    PRETRAIN_EPOCHS_ANNEAL: None,  # for fit_without_validation
    PRETRAIN_MAX_LR_UPDATES: 3,  # for fit_without_validation
    FACTOR_INIT_LR_FINE_TUNE: 0.5,
    # Inference
    PREDICT_WITH_AUGMENTED_PAGE: True,
    SS_MIN_SEPARATION: 0.3,
    SS_MIN_DURATION: 0.3,
    SS_MAX_DURATION: 3.0,
    KC_MIN_SEPARATION: None,
    KC_MIN_DURATION: 0.3,
    KC_MAX_DURATION: None,
    REPAIR_LONG_DETECTIONS: True,
}
