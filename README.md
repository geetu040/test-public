<img src="assets/gsoc-sktime-banner.png" alt="Typing SVG" width="100%">
<hr>

# About Project

|                    |                                                                                                                                                             |
| ------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Program**        | **Google Summer of Code, 2024**                                                                                                                             |
| **Organisation**   | [Sktime](https://summerofcode.withgoogle.com/programs/2024/organizations/sktime): A unified framework for ML with time series                               |
| **Project**        | Sktime integration with deep learning backends - pytorch and huggingface - [Dashboard](https://summerofcode.withgoogle.com/programs/2024/projects/f5FggiB7) |
| **Mentors**        | [Franz Király](https://github.com/fkiraly) - [Benedikt Heidrich](https://github.com/benHeid) - [Anirban Ray](https://github.com/yarnabrina)                 |
| **Project Length** | 350 hours                                                                                                                                                   |

### Overview

TODO: tell some about opportunity, sktime and gsoc and this submission

### Topics

`Data Science` - `AI` - `Time Series` - `Toolbox Frameworks` - `Machine Learning` - `Deep Learning`

### Technologies

`Python` - `Github` - `Pytorch` - `Huggingface` - `Scikit-Learn`

### Outcomes

# Contributions

## Pull Requests

Major Contributions

|                                                     |            |                                                                                      |                                                       |
| --------------------------------------------------- | ---------- | ------------------------------------------------------------------------------------ | ----------------------------------------------------- |
| **Pull Request**                                    | **Status** | **Title**                                                                            | **Related Issue**                                     |
| [#6928](https://github.com/sktime/sktime/pull/6928) | Draft      | [ENH] Global Forecast API for BaseDeepNetworkPyTorch based interfaces                | [#6836](https://github.com/sktime/sktime/issues/6836) |
| [#6842](https://github.com/sktime/sktime/pull/6842) | Open       | [ENH] Implements Autoregressive Wrapper                                              | [#6802](https://github.com/sktime/sktime/issues/6802) |
| [#6571](https://github.com/sktime/sktime/pull/6571) | Open       | [ENH] interface to TimesFM Forecaster                                                | [#6408](https://github.com/sktime/sktime/issues/6408) |
| [#6791](https://github.com/sktime/sktime/pull/6791) | Merged     | [ENH] Pytorch Classifier & de-novo implementation of Transformer                     | [#6786](https://github.com/sktime/sktime/issues/6786) |
| [#6712](https://github.com/sktime/sktime/pull/6712) | Merged     | [ENH] Interface to TinyTimeMixer foundation model                                    | [#6698](https://github.com/sktime/sktime/issues/6698) |
| [#6202](https://github.com/sktime/sktime/pull/6202) | Merged     | [ENH] de-novo implementation of LTSFTransformer based on cure-lab research code base | [#4939](https://github.com/sktime/sktime/issues/4939) |
| [#6457](https://github.com/sktime/sktime/pull/6457) | Merged     | [ENH] Extend HFTransformersForecaster for PEFT methods                               | [#6435](https://github.com/sktime/sktime/issues/6435) |
| [#6321](https://github.com/sktime/sktime/pull/6321) | Merged     | [BUG] fixes failing test in neuralforecast auto freq, amid pandas freq deprecations  |                                                       |
| [#6237](https://github.com/sktime/sktime/pull/6237) | Merged     | [ENH] Update doc and behavior of freq="auto" in neuralforecast                       |                                                       |

Side Work

|                                                     |            |                                                                                 |                                                       |
| --------------------------------------------------- | ---------- | ------------------------------------------------------------------------------- | ----------------------------------------------------- |
| **Pull Request**                                    | **Status** | **Title**                                                                       | **Related Issue**                                     |
| [#7009](https://github.com/sktime/sktime/pull/7009) | Open       | [ENH] Improve documentation for TinyTimeMixer                                   |                                                       |
| [#6965](https://github.com/sktime/sktime/pull/6965) | Merged     | [BUG] TinyTimeMixerForecaster: fix truncating index and update test_params      |                                                       |
| [#6929](https://github.com/sktime/sktime/pull/6929) | Merged     | [ENH] improve test_global_forecasting_tag                                       |                                                       |
| [#6853](https://github.com/sktime/sktime/pull/6853) | Merged     | [MNT] downgrade pykan version to <0.2.2                                         |                                                       |
| [#6367](https://github.com/sktime/sktime/pull/6367) | Merged     | [MNT] final change cycle (0.30.0) for renaming cINNForecaster to CINNForecaster | [#6120](https://github.com/sktime/sktime/issues/6120) |
| [#6238](https://github.com/sktime/sktime/pull/6238) | Merged     | [MNT] change cycle (0.29.0) for renaming cINNForecaster to CINNForecaster       | [#6120](https://github.com/sktime/sktime/issues/6120) |

Pre-GSoC

|                                                     |            |                                                                                      |                                                       |
| --------------------------------------------------- | ---------- | ------------------------------------------------------------------------------------ | ----------------------------------------------------- |
| **Pull Request**                                    | **Status** | **Title**                                                                            | **Related Issue**                                     |
| [#6121](https://github.com/sktime/sktime/pull/6121) | Merged     | [MNT] initialize change cycle (0.28.0) for renaming cINNForecaster to CINNForecaster | [#6120](https://github.com/sktime/sktime/issues/6120) |
| [#6039](https://github.com/sktime/sktime/pull/6039) | Merged     | [ENH] NeuralForecastRNN should auto-detect freq                                      |                                                       |
| [#6088](https://github.com/sktime/sktime/pull/6088) | Merged     | [MNT] create build tool to check invalid backticks                                   |                                                       |
| [#6023](https://github.com/sktime/sktime/pull/6023) | Merged     | [DOC] Fix invalid use of single-grave in docstrings                                  |                                                       |
| [#6116](https://github.com/sktime/sktime/pull/6116) | Merged     | [ENH] Adds MSTL import statement in detrend                                          | [#6085](https://github.com/sktime/sktime/issues/6085) |
| [#6059](https://github.com/sktime/sktime/pull/6059) | Merged     | [ENH] Examples for YtoX transformer docstring                                        |                                                       |

Issue Opened by Me

|                                                       |            |                                                                                       |
| ----------------------------------------------------- | ---------- | ------------------------------------------------------------------------------------- |
| **Issue**                                             | **Status** | **Title**                                                                             |
| [#6828](https://github.com/sktime/sktime/issues/6828) | Open       | [ENH] time series classifiers: encode or coerce y at base class level                 |
| [#6790](https://github.com/sktime/sktime/issues/6790) | Open       | [ENH] improve sktime/networks folder structure                                        |
| [#6158](https://github.com/sktime/sktime/issues/6158) | Open       | [ENH] cINNForecaster needs some informative error handling and improved documentation |

## Walk Through

### MVTSTransformerClassifier

- **Title:** [ENH] Pytorch Classifier & de-novo implementation of Transformer
- **Status:** Merged
- **Pull Request:** [#6791](https://github.com/sktime/sktime/pull/6791)
- **Related Issue:** [#6786](https://github.com/sktime/sktime/issues/6786)
- **Research Paper**: [A Transformer-based Framework for Multivariate Time Series Representation Learning](https://dl.acm.org/doi/abs/10.1145/3447548.3467401)
- **Official Code**: [gzerveas/mvts_transformer](https://github.com/gzerveas/mvts_transformer)
- **Sktime Source Code**: [sktime/classification/deep_learning/mvts_transformer.py](https://github.com/sktime/sktime/blob/main/sktime/classification/deep_learning/mvts_transformer.py)

```py
from sktime.classification.deep_learning import MVTSTransformerClassifier

model = MVTSTransformerClassifier(
    d_model=256,
    n_heads=4,
    num_layers=4,
    dim_feedforward=128,
    dropout=0.1,
    pos_encoding="fixed",
    activation="relu",
    norm="BatchNorm",
    freeze=False,
)
```

### TinyTimeMixer

- **Title:** [ENH] Interface to TinyTimeMixer foundation model
- **Status:** Merged
- **Pull Request:** [#6712](https://github.com/sktime/sktime/pull/6712)
- **Related Issue:** [#6698](https://github.com/sktime/sktime/issues/6698)
- **Research Paper**: [Tiny Time Mixers (TTMs): Fast Pre-trained Models for Enhanced Zero/Few-Shot Forecasting of Multivariate Time Series](https://www.researchgate.net/profile/Vijay-Ekambaram/publication/381111250_Tiny_Time_Mixers_TTMs_Fast_Pre-trained_Models_for_Enhanced_ZeroFew-Shot_Forecasting_of_Multivariate_Time_Series/links/665d8c5d0b0d2845747de5f5/Tiny-Time-Mixers-TTMs-Fast-Pre-trained-Models-for-Enhanced-Zero-Few-Shot-Forecasting-of-Multivariate-Time-Series.pdf)
- **Official Code**: [ibm-granite/granite-tsfm](https://github.com/ibm-granite/granite-tsfm)
- **Sktime Source Code**: [sktime/forecasting/ttm.py](https://github.com/sktime/sktime/blob/main/sktime/forecasting/ttm.py)

```py
from sktime.forecasting.ttm import TinyTimeMixerForecaster

model = TinyTimeMixerForecaster(
	model_path="ibm/TTM",
	revision="main",
	validation_split=0.2,
	config=None,
	training_args=None,
	compute_metrics=None,
	callbacks=None,
	broadcasting=False,
	use_source_package=False,
)
```

### LTSFTransformer

- **Title:** [ENH] de-novo implementation of LTSFTransformer based on cure-lab research code base
- **Status:** Merged
- **Pull Request:** [#6202](https://github.com/sktime/sktime/pull/6202)
- **Related Issue:** [#4939](https://github.com/sktime/sktime/issues/4939)
- **Research Paper**: [Are Transformers Effective for Time Series Forecasting?](https://ojs.aaai.org/index.php/AAAI/article/view/26317)
- **Official Code**: [cure-lab/LTSF-Linear](https://github.com/cure-lab/LTSF-Linear)
- **Sktime Source Code**: [sktime/forecasting/ltsf.py](https://github.com/sktime/sktime/blob/main/sktime/forecasting/ltsf.py)

```py
from sktime.forecasting.ltsf import LTSFTransformerForecaster

model = LTSFTransformerForecaster(
	seq_len=30,
	context_len=15,
	pred_len=15,
    num_epochs=50,
    batch_size=8,
    in_channels=1,
    individual=False,
    criterion=None,
    criterion_kwargs=None,
    optimizer=None,
    optimizer_kwargs=None,
    lr=0.002,
    position_encoding=True,
    temporal_encoding=True,
    temporal_encoding_type="embed",  # linear, embed, fixed-embed
    d_model=32,
    n_heads=1,
    d_ff=64,
    e_layers=1,
    d_layers=1,
    factor=1,
    dropout=0.1,
    activation="relu",
    freq="M",
)
```

### TimesFM

- **Title:** [ENH] interface to TimesFM Forecaster
- **Status:** Open
- **Pull Request:** [#6571](https://github.com/sktime/sktime/pull/6571)
- **Related Issue:** [#6408](https://github.com/sktime/sktime/issues/6408)
- **Research Paper**: [A decoder-only foundation model for time-series forecasting](https://arxiv.org/abs/2310.10688)
- **Official Code**: [google-research/timesfm](https://github.com/google-research/timesfm)

```py
from sktime.forecasting.timesfm_forecaster import TimesFMForecaster

forecaster = TimesFMForecaster(
    context_len=64,
    horizon_len=32,
)
```

### PEFT for HFTransformersForecaster

- **Title:** [ENH] Extend HFTransformersForecaster for PEFT methods
- **Status:** Merged
- **Pull Request:** [#6457](https://github.com/sktime/sktime/pull/6457)
- **Related Issue:** [#6435](https://github.com/sktime/sktime/issues/6435)

```py
from sktime.forecasting.hf_transformers_forecaster import HFTransformersForecaster
from peft import LoraConfig

forecaster = HFTransformersForecaster(
	model_path="huggingface/autoformer-tourism-monthly",
	fit_strategy="peft",
	training_args={
		"num_train_epochs": 20,
		"output_dir": "test_output",
		"per_device_train_batch_size": 32,
	},
	config={
			"lags_sequence": [1, 2, 3],
			"context_length": 2,
			"prediction_length": 4,
			"use_cpu": True,
			"label_length": 2,
	},
	peft_config=LoraConfig(
		r=8,
		lora_alpha=32,
		target_modules=["q_proj", "v_proj"],
		lora_dropout=0.01,
	)
)
```

### AutoregressiveWrapper

- **Title:** [ENH] Implements Autoregressive Wrapper
- **Status:** Open
- **Pull Request:** [#6842](https://github.com/sktime/sktime/pull/6842)
- **Related Issue:** [#6802](https://github.com/sktime/sktime/issues/6802)

```py
from sktime.forecasting.pytorchforecasting import PytorchForecastingNBeats
from sktime.forecasting.compose import AutoRegressiveWrapper

forecaster = PytorchForecastingNBeats(trainer_params={
    "max_epochs": 20,
})
wrapper = AutoRegressiveWrapper(
    forecaster=forecaster,
    horizon_length=5,
    aggregate_method=np.mean,
)
```

### GlobalForecasting for Pytorch Models

- **Title:** [ENH] Global Forecast API for BaseDeepNetworkPyTorch based interfaces
- **Status:** Draft
- **Pull Request:** [#6928](https://github.com/sktime/sktime/pull/6928)
- **Related Issue:** [#6836](https://github.com/sktime/sktime/issues/6836)

# Achnowledgements