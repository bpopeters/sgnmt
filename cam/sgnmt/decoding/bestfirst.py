"""Implementation of best-first search"""

import math
import copy
import heapq
import logging
from collections import defaultdict

from cam.sgnmt import utils
from cam.sgnmt.decoding.core import Decoder, PartialHypothesis


def ngrams(sequence, n):
    return tuple(zip(*[sequence[i:] for i in range(n)]))


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


class LatticePartialHypothesis(PartialHypothesis):

    def merge(self, other, merge_loc):
        """
        other: another LatticePartialHypothesis
        merge_loc: position in other at which to attach
        """
        merged_hypo = type(self)(None)
        merged_hypo.trgt_sentence = self.trgt_sentence + other.trgt_sentence[merge_loc:]
        merged_hypo.score_breakdown = self.score_breakdown + other.score_breakdown[merge_loc:]
        # hard part: score:
        # score_breakdown is a list.
        # each element in it is a list of tuples
        # each tuple is a (raw_score, weight) tuple
        suffix_breakdown = other.score_breakdown[merge_loc:]
        suffix_score = sum(sum(raw * weight for (raw, weight) in pred_scores) for pred_scores in suffix_breakdown)
        merged_hypo.score = self.score + suffix_score
        return merged_hypo


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
        self.ad_hoc_completion = True
        # sparse is the same as "none" for dense models. Also consider topk,
        # nucleus
        self.child_filtering = "sparse"
        # enabling n and alpha produces RCB. Enabling zip propagates the merge
        # backward.
        self.recomb_n = decoder_args.recomb_n
        self.recomb_alpha = decoder_args.recomb_alpha
        self.recomb_zip = False

        self.ngrams = defaultdict(set)
        # RCB: merge hypotheses if they share their last n-gram and differ
        # from finished hypotheses in length by less than alpha

    def add_full_hypo(self, hypo):
        if self.recomb_n > 0:
            i = len(self.full_hypos)
            hypo_ngrams = ngrams(hypo.trgt_sentence, self.recomb_n)
            for j, ngram in enumerate(hypo_ngrams):
                # j is the start index of the ngram
                self.ngrams[ngram].add((i, j))
        super().add_full_hypo(hypo)

    def recombine(self, hypo):
        """
        If hypo is recombinable, add it (meaning it and its continuation)
        to self.full_list.
        """
        new_hypos = []

        last_ngram = tuple(hypo.trgt_sentence[-self.recomb_n:])
        matches = self.ngrams[last_ngram]
        if matches:
            for finished_index, match_pos in matches:
                finished_hypo = self.full_hypos[finished_index]
                len_diff = abs(len(finished_hypo.trgt_sentence) - len(hypo.trgt_sentence))
                if len_diff < self.recomb_alpha:
                    new_hypo = hypo.merge(finished_hypo, match_pos)
                    new_hypos.append(new_hypo)
        return new_hypos

    def filter_children(self, posterior):
        """
        This also orders children, it doesn't just filter them
        """
        children = sorted(
            [i for i in posterior.items() if i[1] > -math.inf],
            key=lambda child: child[1],
            reverse=True
        )
        return children

    def decode(self, src_sentence):
        """Decodes a single source sentence using A* search. """
        self.initialize_predictors(src_sentence)
        open_set = PQueue()
        best_score = self.get_lower_score_bound()
        covered_mass = 0.0
        nodes_pruned = 0
        open_set.append((0.0, LatticePartialHypothesis(self.get_predictor_states())))
        # change the while loop to account for maximum
        # need to change stopping conditions
        while open_set and len(self.full_hypos) < self.nbest:
            # pop the best partial hypothesis
            curr_priority, hypo = open_set.pop()

            # log the just-popped (partial) hypothesis
            # another thing that should be included: the total mass found
            logging.debug("Expand (score=%f exp=%d pruned=%d best=%f total=%f): sentence: %s"
                          % (hypo.score,
                             self.apply_predictors_count,
                             nodes_pruned,
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

                continue

            recombination_hypos = self.recombine(hypo)
            if recombination_hypos:
                # self.recombine updates self.full_list in place, so it isn't
                # necessary to interact with it here.
                for rec_hypo in recombination_hypos:
                    covered_mass += math.exp(rec_hypo.score)  # might be problematic
                    self.add_full_hypo(rec_hypo.generate_full_hypothesis())
                continue

            # if you make it here, the current hypothesis is incomplete
            self.set_predictor_states(copy.deepcopy(hypo.predictor_states))

            if hypo.word_to_consume is not None:  # Consume if cheap expand
                self.consume(hypo.word_to_consume)
                hypo.word_to_consume = None

            posterior, score_breakdown = self.apply_predictors()
            hypo.predictor_states = self.get_predictor_states()

            # generate filtered list of successors to add to open_set
            children = self.filter_children(posterior)

            # push the successors of the current hypothesis onto the open set.
            for i, (tgt_word, score) in enumerate(children):
                next_hypo = hypo.cheap_expand(
                    tgt_word, score, score_breakdown[tgt_word]
                )

                # a hypo's score (log prob) may be different from
                # its priority (potentially something else)
                if self.ad_hoc_completion:
                    priority = next_hypo.score if i != 0 else math.inf
                else:
                    priority = next_hypo.score

                open_set.append((priority, next_hypo))
            # Prune hypotheses if too many are stored.
            # heapq.nlargest might be useful here.
            # regardless, is this inefficient? It seems to me that this will
            # be called very frequently unless the capacity is infinite or
            # very large.
            if self.capacity > 0 and len(open_set) > self.capacity:
                nodes_pruned += len(open_set) - self.capacity
                new_open_set = PQueue()
                for _ in range(self.capacity):
                    new_open_set.append(open_set.pop())
                open_set = new_open_set
        return self.get_full_hypos_sorted()
