"""Implementation of best-first search"""

import math
import copy
import heapq
import logging

from cam.sgnmt import utils
from cam.sgnmt.decoding.core import Decoder, PartialHypothesis


class PQueue:
    # why? Because I don't like the heapq interface
    def __init__(self, maxheap=True):
        self.maxheap = maxheap
        self.heap = []

    def pop(self):
        score, data = heapq.heappop(self.heap)
        if self.maxheap:
            score = -score
        return score, data

    def append(self, item):
        score, data = item
        if self.maxheap:
            score = -score
        heapq.heappush(self.heap, (score, data))

    def __len__(self):
        return len(self.heap)

    def __bool__(self):
        return bool(self.heap)


class BestFirstDecoder(Decoder):
    """
    Best-first decoding as from single-queue decoding or the
    """

    def __init__(self, decoder_args):
        """Creates a new best-first decoder instance. The following values are
        fetched from `decoder_args`:

            [bpop: need to decide what really goes here]

            beam (int): Maximum number of active hypotheses.
            early_stopping (bool): If this is true, partial hypotheses
                                   with score worse than the current
                                   best complete scores are not
                                   expanded. This applies when nbest is
                                   larger than one and inadmissible
                                   heuristics are used
            nbest (int): If this is set to a positive value, we do not
                         stop decoding at the first complete path, but
                         continue search until we collected this many
                         complete hypothesis. With an admissible
                         heuristic, this will yield an exact n-best
                         list.

        Args:
            decoder_args (object): Decoder configuration passed through
                                   from the configuration API.
        """
        super(BestFirstDecoder, self).__init__(decoder_args)
        self.nbest = max(1, decoder_args.nbest)
        self.capacity = decoder_args.beam

    def decode(self, src_sentence):
        """Decodes a single source sentence using A* search. """
        self.initialize_predictors(src_sentence)
        open_set = PQueue()
        best_score = self.get_lower_score_bound()
        open_set.append((0.0, PartialHypothesis(self.get_predictor_states())))
        # change the while loop to account for maximum
        while open_set:
            # pop the best partial hypothesis
            c, hypo = open_set.pop()

            # there used to be early stopping here

            # log the just-popped (partial) hypothesis
            logging.debug("Expand (score=%f exp=%d best=%f): sentence: %s"
                          % (hypo.score,
                             self.apply_predictors_count,
                             best_score,
                             hypo.trgt_sentence))

            # is the current hypothesis finished?
            if hypo.get_last_word() == utils.EOS_ID:
                if hypo.score > best_score:
                    # update best score
                    logging.debug("New best hypo (score=%f exp=%d): %s" % (
                            hypo.score,
                            self.apply_predictors_count,
                            ' '.join([str(w) for w in hypo.trgt_sentence])))
                    best_score = hypo.score

                # add full hypothesis: I think this is the closed set
                self.add_full_hypo(hypo.generate_full_hypothesis())
                if len(self.full_hypos) >= self.nbest:
                    return self.get_full_hypos_sorted()
                continue

            # if you make it here, the current hypothesis is incomplete
            # I'm not completely sure what set_predictor_states does; it
            # probably advances the decoder by one time step
            self.set_predictor_states(copy.deepcopy(hypo.predictor_states))

            # I do not know what consuming does
            if hypo.word_to_consume is not None:  # Consume if cheap expand
                self.consume(hypo.word_to_consume)
                hypo.word_to_consume = None

            posterior, score_breakdown = self.apply_predictors()
            hypo.predictor_states = self.get_predictor_states()

            # generate filtered list of successors to add to open_set
            children = [i for i in posterior.items() if i[1] > -math.inf]

            # we can use posterior.items() to get children
            # push the successors of the current hypothesis onto the open set.
            for tgt_word, score in children:
                next_hypo = hypo.cheap_expand(
                    tgt_word,
                    score,
                    score_breakdown[tgt_word]
                )
                open_set.append((next_hypo.score, next_hypo))
            # Limit heap capacity
            if self.capacity > 0 and len(open_set) > self.capacity:
                new_open_set = PQueue()
                for _ in range(self.capacity):
                    new_open_set.append(open_set.pop())
                open_set = new_open_set
        return self.get_full_hypos_sorted()
