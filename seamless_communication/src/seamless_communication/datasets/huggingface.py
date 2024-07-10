# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.

import wave
from scipy.io import wavfile
import logging
import os
from abc import abstractmethod
from typing import Dict, Iterable, Optional

import numpy as np
import torch
from datasets import load_dataset

from .datatypes import LangPairSample, MultimodalSample

logger = logging.getLogger(__name__)


class SpeechTokenizer:
    @abstractmethod
    def encode(self, wav: torch.Tensor, sample_rate: int) -> torch.Tensor:
        ...


class Speech2SpeechFleursDatasetBuilder:
    """Assembles speech2speech dataset from google/fleurs on HuggingFace"""

    HF_FLEURS_DATASET_NAME = "google/fleurs"

    def __init__(
        self,
        source_lang: str,
        target_lang: str,
        split: str = "test",
        skip_source_audio: bool = True,
        skip_target_audio: bool = True,
        audio_dtype: torch.dtype = torch.float32,
        dataset_cache_dir: Optional[str] = None,
        speech_tokenizer: Optional[SpeechTokenizer] = None,
        dataset_dir = None
    ):
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.split = split
        self.dataset_cache_dir = dataset_cache_dir
        self.audio_dtype = audio_dtype
        self.skip_source_audio = skip_source_audio
        self.skip_target_audio = skip_target_audio
        self.speech_tokenizer = speech_tokenizer
        self.dataset_dir = dataset_dir

    def _prepare_sample(
        self,
        sample_id: int,
        lang: str,
        text: str,
        audio_local_path: Optional[str] = None,
        waveform_npy: Optional[np.ndarray] = None,
        sampling_rate: Optional[int] = None,
    ) -> MultimodalSample:
        should_skip_audio = (
            lang == self.target_lang
            and self.skip_target_audio
            or lang == self.source_lang
            and self.skip_source_audio
            or waveform_npy is None
        )
        if not should_skip_audio:
            waveform = torch.from_numpy(waveform_npy).to(self.audio_dtype)
        else:
            waveform = None
        if self.speech_tokenizer is not None and not should_skip_audio:
            assert waveform is not None
            assert sampling_rate is not None
            units_tensor = self.speech_tokenizer.encode(
                waveform, sampling_rate
            ).reshape(-1)
            units = units_tensor.tolist()
        else:
            units = None
        return MultimodalSample(
            id=sample_id,
            lang=lang,
            text=text.strip(),
            audio_local_path=audio_local_path,
            waveform=waveform,
            sampling_rate=sampling_rate,
            units=units,
        )

    def iterate_lang_audio_samples(self, lang: str) -> Iterable[MultimodalSample]:

        ds = load_dataset(
            self.HF_FLEURS_DATASET_NAME,
            lang,
            split=self.split,
            data_dir="/kaggle/working/fleurs",
            streaming=False,
        )
        max_id = max(ds, key=lambda x: int(x["id"]))["id"]
        for idx, item in enumerate(ds):
            audio_path = os.path.join(
                os.path.dirname(item["path"]), item["audio"]["path"]
            )
            (sample_id, audio_local_path, waveform, sampling_rate, text) = (
                item["id"],
                audio_path,
                item["audio"]["array"],
                item["audio"]["sampling_rate"],
                item["transcription"],
            )
            

            yield self._prepare_sample(
                sample_id=sample_id,
                audio_local_path=audio_local_path,
                waveform_npy=waveform,
                sampling_rate=sampling_rate,
                text=text,
                lang=lang,
            )

    def __iter__(self) -> Iterable[LangPairSample]:
        logger.info(f"Loading {self.target_lang} samples")
        target_samples: Dict[int, MultimodalSample] = {}
        for idx, sample in enumerate(
            self.iterate_lang_audio_samples(lang=self.target_lang)
        ):
            if idx and idx % 100 == 0:
                print(f"..loaded {idx} target samples")
            target_samples[sample.id] = sample

        logger.info(f"Loading {self.source_lang} samples")
        for idx, sample in enumerate(
            self.iterate_lang_audio_samples(lang=self.source_lang)
        ):
            if idx and idx % 100 == 0:
                print(f"..loaded {idx} source samples")
            if sample.id in target_samples:
                yield LangPairSample(source=sample, target=target_samples[sample.id])

class Speech2SpeechFleursDatasetBuilderCustom:
    """Assembles speech2speech dataset from google/fleurs on HuggingFace"""

    def __init__(
        self,
        source_lang: str,
        target_lang: str,
        split: str = "test",
        skip_source_audio: bool = True,
        skip_target_audio: bool = True,
        audio_dtype: torch.dtype = torch.float32,
        dataset_cache_dir: Optional[str] = None,
        speech_tokenizer: Optional[SpeechTokenizer] = None,
        dataset_path: Optional[str] = None
    ):
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.split = split
        self.dataset_cache_dir = dataset_cache_dir
        self.audio_dtype = audio_dtype
        self.speech_tokenizer = speech_tokenizer
        self.dataset_path = dataset_path

    def _prepare_sample(
        self,
        sample_id: int,
        lang: str,
        text: str,
        audio_local_path: Optional[str] = None,
        waveform_npy: Optional[np.ndarray] = None,
        sampling_rate: Optional[int] = None,
    ) -> MultimodalSample:
        should_skip_audio = (
            lang == self.target_lang
            or lang == self.source_lang
            or waveform_npy is None
        )
        waveform = torch.from_numpy(waveform_npy).to(self.audio_dtype)

        if self.speech_tokenizer is not None and not should_skip_audio:
            assert waveform is not None
            assert sampling_rate is not None
            units_tensor = self.speech_tokenizer.encode(
                waveform, sampling_rate
            ).reshape(-1)
            units = units_tensor.tolist()
        else:
            units = None
        return MultimodalSample(
            id=sample_id,
            lang=lang,
            text=text.strip(),
            audio_local_path=audio_local_path,
            waveform=waveform,
            sampling_rate=sampling_rate,
            units=units,
        )

    def iterate_lang_audio_samples(self, lang: str, data_dir: str) -> Iterable[MultimodalSample]:
        ds=os.listdir(os.path.join(data_dir, lang, "audio"))
        for item,i in enumerate(ds):
            # Load the file
            sample_rate, data = wavfile.read(item)

            # Check if stereo
            if len(data.shape) > 1:
                # Convert to mono by selecting only the first channel
                data = data[:, 0]
            audio_path = os.path.join(
                item
            )
            (sample_id, audio_local_path, waveform, sampling_rate, text) = (
                i,
                audio_path,
                data,
                sample_rate,
                "",
            )
            yield self._prepare_sample(
                sample_id=sample_id,
                audio_local_path=audio_local_path,
                waveform_npy=waveform,
                sampling_rate=sampling_rate,
                text=text,
                lang=lang,
            )

    def __iter__(self) -> Iterable[LangPairSample]:
        logger.info(f"Loading {self.target_lang} samples")
        target_samples: Dict[int, MultimodalSample] = {}
        for idx, sample in enumerate(
            self.iterate_lang_audio_samples(data_dir=self.dataset_path, lang=self.target_lang)
        ):
            if idx and idx % 100 == 0:
                logger.info(f"..loaded {idx} target samples")
            target_samples[sample.id] = sample

        logger.info(f"Loading {self.source_lang} samples")
        for idx, sample in enumerate(
            self.iterate_lang_audio_samples(data_dir=self.dataset_path, lang=self.source_lang)
        ):
            if idx and idx % 100 == 0:
                logger.info(f"..loaded {idx} source samples")
            if sample.id in target_samples:
                yield LangPairSample(source=sample, target=target_samples[sample.id])
