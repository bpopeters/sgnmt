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
        covered_mass = 0.0
        open_set.append((0.0, PartialHypothesis(self.get_predictor_states())))
        # change the while loop to account for maximum
        while open_set:
            # pop the best partial hypothesis
            curr_priority, hypo = open_set.pop()

            # log the just-popped (partial) hypothesis
            # another thing that should be included: the total mass found
            logging.debug("Expand (score=%f exp=%d best=%f total=%f): sentence: %s"
                          % (hypo.score,
                             self.apply_predictors_count,
                             best_score,
                             covered_mass,
                             hypo.trgt_sentence))

            # is the current hypothesis finished?
            if hypo.get_last_word() == utils.EOS_ID:
                # maybe update best score
                if hypo.score > best_score:
                    best_score = hypo.score

                # add full hypothesis to closed set
                self.add_full_hypo(hypo.generate_full_hypothesis())

                # update total covered mass (this assumes that hypo.score is
                # log probability)
                covered_mass += math.exp(hypo.score)

                # if enough hypotheses have been found already, return them
                if len(self.full_hypos) >= self.nbest:
                    return self.get_full_hypos_sorted()
                continue

            # if you make it here, the current hypothesis is incomplete
            self.set_predictor_states(copy.deepcopy(hypo.predictor_states))

            if hypo.word_to_consume is not None:  # Consume if cheap expand
                self.consume(hypo.word_to_consume)
                hypo.word_to_consume = None

            posterior, score_breakdown = self.apply_predictors()
            hypo.predictor_states = self.get_predictor_states()

            # generate filtered list of successors to add to open_set
            # this technique for generating children might be referred to
            # as "sparse filtering" -> don't consider any hypothesis with
            # probability 0.
            children = sorted(
                [i for i in posterior.items() if i[1] > -math.inf],
                key=lambda child: child[1],
                reverse=True
            )

            # push the successors of the current hypothesis onto the open set.
            for i, (tgt_word, score) in enumerate(children):
                # it's important to distinguish a hypo's score (log prob) from
                # its priority (potentially something else)
                next_hypo = hypo.cheap_expand(
                    tgt_word,
                    score,
                    score_breakdown[tgt_word]
                )
                # the pq score might not be the same as the
                # "true" hypothesis score (the log prob).
                priority = next_hypo.score if i != 0 else math.inf

                open_set.append((priority, next_hypo))
            # Prune hypotheses if too many are stored.
            # heapq.nlargest might be useful here.
            # regardless, is this inefficient? It seems to me that this will
            # be called very frequently unless the capacity is infinite or
            # very large.
            if self.capacity > 0 and len(open_set) > self.capacity:
                new_open_set = PQueue()
                for _ in range(self.capacity):
                    new_open_set.append(open_set.pop())
                open_set = new_open_set
        return self.get_full_hypos_sorted()
