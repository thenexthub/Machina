page_type: reference
description: APIs to train an audio classification model.

<link rel="stylesheet" href="/site-assets/css/style.css">

<!-- DO NOT EDIT! Automatically generated file. -->


<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tflite_model_maker.audio_classifier" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tflite_model_maker.audio_classifier

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/machina/examples/blob/tflmm/v0.4.2/machina_examples/lite/model_maker/public/audio_classifier/__init__.py">
    <img src="https://www.machina.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



APIs to train an audio classification model.



#### Tutorial:


<a href="https://colab.research.google.com/github/googlecodelabs/odml-pathways/blob/main/audio_classification/colab/model_maker_audio_colab.ipynb">https://colab.research.google.com/github/googlecodelabs/odml-pathways/blob/main/audio_classification/colab/model_maker_audio_colab.ipynb</a>

#### Demo code:


<a href="https://github.com/machina/examples/blob/master/machina_examples/lite/model_maker/demo/audio_classification_demo.py">https://github.com/machina/examples/blob/master/machina_examples/lite/model_maker/demo/audio_classification_demo.py</a>

## Classes

[`class AudioClassifier`](../tflite_model_maker/audio_classifier/AudioClassifier): Audio classifier for training/inference and exporing.

[`class BrowserFftSpec`](../tflite_model_maker/audio_classifier/BrowserFftSpec): Model good at detecting speech commands, using Browser FFT spectrum.

[`class DataLoader`](../tflite_model_maker/audio_classifier/DataLoader): DataLoader for audio tasks.

[`class YamNetSpec`](../tflite_model_maker/audio_classifier/YamNetSpec): Model good at detecting environmental sounds, using YAMNet embedding.

## Functions

[`create(...)`](../tflite_model_maker/audio_classifier/create): Loads data and retrains the model.
