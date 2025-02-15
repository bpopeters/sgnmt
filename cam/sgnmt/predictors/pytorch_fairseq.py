# -*- coding: utf-8 -*-
# coding=utf-8
# Copyright 2019 The SGNMT Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This is the interface to the fairseq library.

https://github.com/pytorch/fairseq

The fairseq predictor can read any model trained with fairseq.
"""

import logging
import os
from typing import Dict, List, Optional


from cam.sgnmt import utils
from cam.sgnmt.predictors.core import Predictor


try:
    # Requires fairseq
    from fairseq import checkpoint_utils, options, tasks
    from fairseq import utils as fairseq_utils
    from fairseq.sequence_generator import EnsembleModel
    import torch
    from torch import Tensor
    import numpy as np
except ImportError:
    pass  # Deal with it in decode.py


FAIRSEQ_INITIALIZED = False
"""Set to true by _initialize_fairseq() after first constructor call."""


def _initialize_fairseq(user_dir):
    global FAIRSEQ_INITIALIZED
    if not FAIRSEQ_INITIALIZED:
        logging.info("Setting up fairseq library...")
        if user_dir:
            args = type("", (), {"user_dir": user_dir})()
            fairseq_utils.import_user_module(args)
        FAIRSEQ_INITIALIZED = True


class FairseqPredictor(Predictor):
    """Predictor for using fairseq models."""

    def __init__(self, model_path, user_dir, lang_pair, n_cpu_threads=-1, alpha=1.):
        """Initializes a fairseq predictor.

        Args:
            model_path (string): Path to the fairseq model (*.pt). Like
                                 --path in fairseq-interactive.
            lang_pair (string): Language pair string (e.g. 'en-fr').
            user_dir (string): Path to fairseq user directory.
            n_cpu_threads (int): Number of CPU threads. If negative,
                                 use GPU.
            alpha (float): entmax alpha for decoding (defaults to 1, which
                           recovers softmax)
        """
        super(FairseqPredictor, self).__init__()
        _initialize_fairseq(user_dir)
        self.use_cuda = torch.cuda.is_available() and n_cpu_threads < 0

        parser = options.get_generation_parser()
        input_args = ["--path", model_path, os.path.dirname(model_path)]
        if lang_pair:
            src, trg = lang_pair.split("-")
            input_args.extend(["--source-lang", src, "--target-lang", trg])
        args = options.parse_args_and_arch(parser, input_args)

        # Setup task, e.g., translation
        task = tasks.setup_task(args)
        self.src_vocab_size = len(task.source_dictionary)
        self.trg_vocab_size = len(task.target_dictionary)
        self.pad_id = task.source_dictionary.pad()

        # Load ensemble
        logging.info('Loading fairseq model(s) from {}'.format(model_path))
        self.models, _ = checkpoint_utils.load_model_ensemble(
            model_path.split(':'),
            task=task,
        )

        # Optimize ensemble for generation
        for model in self.models:
            model.make_generation_fast_(
                beamable_mm_beam_size=1,
                need_attn=False,
            )
            if self.use_cuda:
                model.cuda()
        self.model = EnsembleModel(self.models)
        self.model.eval()

        self.alpha = alpha

    def get_unk_probability(self, posterior):
        """Fetch posterior[utils.UNK_ID]"""
        return utils.common_get(posterior, utils.UNK_ID, utils.NEG_INF)

    def predict_next(self):
        """Call the fairseq model."""
        with torch.no_grad():
            consumed = torch.tensor([self.consumed], dtype=torch.long, device="cuda" if self.use_cuda else "cpu")
            # note that this does not currently support decoding with temperature
            lprobs, _ = self.model.forward_decoder(
                consumed,
                self.encoder_outs,
                self.incremental_states,
                alpha=self.alpha
            )
            lprobs[0, self.pad_id] = utils.NEG_INF
            return lprobs[0].cpu().numpy()

    def initialize(self, src_sentence):
        """Initialize source tensors, reset consumed."""
        self.consumed = []
        src_tokens = torch.LongTensor([
            utils.oov_to_unk(src_sentence + [utils.EOS_ID],
                             self.src_vocab_size)])
        src_lengths = torch.LongTensor([len(src_sentence) + 1])
        if self.use_cuda:
            src_tokens = src_tokens.cuda()
            src_lengths = src_lengths.cuda()
        with torch.no_grad():
            self.encoder_outs = self.model.forward_encoder({
                'src_tokens': src_tokens,
                'src_lengths': src_lengths})
        self.consumed = [utils.GO_ID or utils.EOS_ID]
        # Reset incremental states
        # do we want this jit?
        self.incremental_states = torch.jit.annotate(
            List[Dict[str, Dict[str, Optional[Tensor]]]],
            [
                torch.jit.annotate(Dict[str, Dict[str, Optional[Tensor]]], {})
                for i in range(self.model.models_size)
            ],
        )

    def consume(self, word):
        """Append ``word`` to the current history."""
        self.consumed.append(word)

    def get_state(self):
        """The predictor state is the complete history."""
        return self.consumed, [state for state in self.incremental_states]

    def set_state(self, state):
        """The predictor state is the complete history."""
        self.consumed, inc_states = state
        self.incremental_states = inc_states

    def is_equal(self, state1, state2):
        """Returns true if the history is the same """
        return state1[0] == state2[0]
