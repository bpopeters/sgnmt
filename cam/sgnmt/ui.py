"""This module handles configuration and user interface when using 
blocks. ``yaml`` and ``ArgumentParser`` are used for parsing config
files and command line arguments.
"""

import argparse
import logging
import os

YAML_AVAILABLE = True
try:
    import yaml
except:
    YAML_AVAILABLE = False

def str2bool(v):
    """For making the ``ArgumentParser`` understand boolean values"""
    return v.lower() in ("yes", "true", "t", "1")


def parse_args(parser):
    """http://codereview.stackexchange.com/questions/79008/parse-a-config-file-
    and-add-to-command-line-arguments-using-argparse-in-python """
    args = parser.parse_args()
    if args.config_file:
        if not YAML_AVAILABLE:
            logging.fatal("Install PyYAML in order to use config files.")
            return args
        paths = args.config_file
        delattr(args, 'config_file')
        arg_dict = args.__dict__
        for path in paths.split(","):
            _load_config_file(arg_dict, path)
    return args


def _load_config_file(arg_dict, path):
    with open(path.strip()) as f:
        data = yaml.load(f)
        for key, value in data.items():
            if key == "config_file":
                for sub_path in value.split(","):
                    _load_config_file(arg_dict, sub_path)
            if isinstance(value, list):
                for v in value:
                    arg_dict[key].append(v)
            else:
                arg_dict[key] = value


def parse_param_string(param):
    """Parses a parameter string such as 'param1=x,param2=y'. Loads 
    config files if specified in the string. If ``param`` points to a
    file, load this file with YAML.
    """
    if not param:
        return {}
    if os.path.isfile(param):
        param = "config_file=%s" % param
    config = {}
    for pair in param.strip().split(","):
        (k,v) = pair.split("=", 1)
        if k == 'config_file':
            if not YAML_AVAILABLE:
                logging.fatal("Install PyYAML in order to use config files.")
            else:
                with open(v) as f:
                    data = yaml.load(f)
                    for config_file_key, config_file_value in data.items():
                        config[config_file_key] = config_file_value
        else:
            config[k] = v
    return config


def get_parser():
    """Get the parser object which is used to build the configuration
    argument ``args``. This is a helper method for ``get_args()``
    TODO: Decentralize configuration
    
    Returns:
        ArgumentParser. The pre-filled parser object
    """
    parser = argparse.ArgumentParser()
    parser.register('type','bool',str2bool)
    
    ## General options
    group = parser.add_argument_group('General options')
    group.add_argument('--config_file', 
                        help="Configuration file in standard .ini format. NOTE:"
                        " Configuration file overrides command line arguments")
    group.add_argument("--verbosity", default="info",
                        choices=['debug', 'info', 'warn', 'error'],
                        help="Log level: debug,info,warn,error")
    group.add_argument("--min_score", default=-1000000.0, type=float,
                        help="Delete all complete hypotheses with total scores"
                        " smaller than this value")
    group.add_argument("--range", default="",
                        help="Defines the range of sentences to be processed. "
                        "Syntax is equal to HiFSTs printstrings and lmerts "
                        "idxrange parameter: <start-idx>:<end-idx> (both "
                        "inclusive, start with 1). E.g. 2:5 means: skip the "
                        "first sentence, process next 4 sentences. If this "
                        "points to a file, we grap sentence IDs to translate "
                        "from that file and delete the decoded IDs. This can "
                        "be used for distributed decoding.")
    group.add_argument("--src_test", default="",
                        help="Path to source test set. This is expected to be "
                        "a plain text file with one source sentence in each "
                        "line. Words need to be indexed, i.e. use word IDs "
                        "instead of their string representations.")
    group.add_argument("--indexing_scheme", default="t2t",
                        choices=['blocks', 'tf', 't2t'],
                        help="This parameter defines the reserved IDs.\n\n"
                        "* 'blocks': eps,unk: 0, <s>: 1, </s>: 2.\n"
                        "* 'tf': unk: 3, <s>: 1, </s>: 2.\n"
                        "* 't2t': unk: 3, <s>: 2, </s>: 1.")
    group.add_argument("--ignore_sanity_checks", default=False, type='bool',
                       help="SGNMT terminates when a sanity check fails by "
                       "default. Set this to true to ignore sanity checks.")
    group.add_argument("--input_method", default="file",
                        choices=['dummy', 'file', 'shell', 'stdin'],
                        help="This parameter controls how the input to SGNMT "
                        "is provided. SGNMT supports three modes:\n\n"
                        "* 'dummy': Use dummy source sentences.\n"
                        "* 'file': Read test sentences from a plain text file"
                            "specified by --src_test.\n"
                        "* 'shell': Start SGNMT in an interactive shell.\n"
                        "* 'stdin': Test sentences are read from stdin\n\n"
                        "In shell and stdin mode you can change SGNMT options "
                        "on the fly: Beginning a line with the string '!sgnmt '"
                        " signals SGNMT directives instead of sentences to "
                        "translate. E.g. '!sgnmt config predictor_weights "
                        "0.2,0.8' changes the current predictor weights. "
                        "'!sgnmt help' lists all available directives. Using "
                        "SGNMT directives is particularly useful in combination"
                        " with MERT to avoid start up times between "
                        "evaluations. Note that input sentences still have to "
                        "be written using word ids in all cases.")
    group.add_argument("--log_sum",  default="log",
                        choices=['tropical', 'log'],
                        help="Controls how to compute the sum in the log "
                        "space, i.e. how to compute log(exp(l1)+exp(l2)) for "
                        "log values l1,l2.\n\n"
                        "* 'tropical': approximate with max(l1,l2)\n"
                        "* 'log': Use logsumexp in scipy")
    group.add_argument("--single_cpu_thread", default=False, type='bool',
                        help="If true, try to prevent libraries like Theano "
                        "or TensorFlow from doing internal multithreading. "
                        "Also, see the OMP_NUM_THREADS environment variable.")
    
    ## Decoding options
    group = parser.add_argument_group('Decoding options')
    group.add_argument("--decoder", default="beam",
                        choices=['greedy',
                                 'beam',
                                 'multisegbeam',
                                 'syncbeam',
                                 'sepbeam',
                                 'mbrbeam',
                                 'syntaxbeam',
                                 'combibeam',
                                 'dfs',
                                 'restarting',
                                 'bow',
                                 'flip',
                                 'bucket',
                                 'bigramgreedy',
                                 'astar',
                                 'vanilla'],
                        help="Strategy for traversing the search space which "
                        "is spanned by the predictors.\n\n"
                        "* 'greedy': Greedy decoding (similar to beam=1)\n"
                        "* 'beam': beam search like in Bahdanau et al, 2015\n"
                        "* 'dfs': Depth-first search. This should be used for "
                        "exact decoding or the complete enumeration of the "
                        "search space, but it cannot be used if the search "
                        "space is too large (like for unrestricted NMT) as "
                        "it performs exhaustive search. If you have not only "
                        "negative predictor scores, set --early_stopping to "
                        "false.\n"
                        "* 'restarting': Like DFS but with better admissible "
                        "pruning behavior.\n"
                        "* 'multisegbeam': Beam search for predictors with "
                        "multiple tokenizations ([sub]word/char-levels).\n"
                        "* 'syncbeam': beam search which compares after "
                        "consuming a special synchronization symbol instead "
                        "of after each iteration.\n"
                        "* 'syntaxbeam': beam search which ensures terminal "
                        "symbol diversity.\n"
                        "* 'mbrbeam': Uses an MBR-based criterion to select "
                        "the next hypotheses at each time step.\n"
                        "* 'sepbeam': Associates predictors with hypos in "
                        "beam search and applies only one predictor instead "
                        "of all for hypo expansion.\n"
                        "* 'combibeam': Applies combination_scheme at each "
                        "time step.\n"
                        "* 'bow': Restarting decoder optimized for bag-of-words "
                        "problems.\n"
                        "* 'flip': This decoder works only for bag problems. "
                        "It traverses the search space by switching two words "
                        "in the hypothesis. Do not use bow predictor.\n"
                        "* 'bucket': Works best for bag problems. Maintains "
                        "buckets for each hypo length and extends a hypo in "
                        "a bucket by one before selecting the next bucket.\n"
                        "* 'bigramgreedy': Works best for bag problems. "
                        "Collects bigram statistics and constructs hypos to "
                        "score by greedily selecting high scoring bigrams. "
                        "Do not use bow predictor with this search strategy.\n"
                        "* 'astar': A* search. The heuristic function is "
                        "configured using the --heuristics options.\n"
                        "* 'vanilla': Original Blocks beam decoder. This "
                        "bypasses the predictor framework and directly "
                        "performs pure NMT beam decoding on the GPU. Use this "
                        "when you do pure NMT decoding as this is usually "
                        "faster then using a single nmt predictor as the "
                        "search can be parallelized on the GPU.")
    group.add_argument("--beam", default=4, type=int,
                        help="Size of beam. Only used if --decoder is set to "
                        "'beam' or 'astar'. For 'astar' it limits the capacity"
                        " of the queue. Use --beam 0 for unlimited capacity.")
    group.add_argument("--sub_beam", default=0, type=int,
                        help="This denotes the maximum number of children of "
                        "a partial hypothesis in beam-like decoders. If zero, "
                        "this is set to --beam to reproduce standard beam "
                        "search.")
    group.add_argument("--hypo_recombination", default=False, type='bool',
                        help="Activates hypothesis recombination. Has to be "
                        "supported by the decoder. Applicable to beam, "
                        "restarting, bow, bucket")
    group.add_argument("--allow_unk_in_output", default=True, type='bool',
                        help="If false, remove all UNKs in the final "
                        "posteriors. Predictor distributions can still "
                        "produce UNKs, but they have to be replaced by "
                        "other words by other predictors")
    group.add_argument("--max_node_expansions", default=0, type=int,
                        help="This parameter allows to limit the total number "
                        "of search space expansions for a single sentence. "
                        "If this is 0 we allow an unlimited number of "
                        "expansions. If it is negative, the maximum number of "
                        "expansions is this times the length of the source "
                        "sentence. Supporting decoders:\n"
                        "bigramgreedy, bow, bucket, dfs, flip, restarting")
    group.add_argument("--max_len_factor", default=2, type=int,
                        help="Limits the length of hypotheses to avoid "
                        "infinity loops in search strategies for unbounded "
                        "search spaces. The length of any translation is "
                        "limited to max_len_factor times the length of the "
                        "source sentence.")
    group.add_argument("--early_stopping", default=True, type='bool',
                        help="Use this parameter if you are only interested in "
                        "the first best decoding result. This option has a "
                        "different effect depending on the used --decoder. For"
                        " the beam decoder, it means stopping decoding when "
                        "the best active hypothesis ends with </s>. If false, "
                        "do not stop until all hypotheses end with EOS. For "
                        "the dfs and restarting decoders, early stopping "
                        "enables admissible pruning of branches when the "
                        "accumulated score already exceeded the currently best "
                        "score. DO NOT USE early stopping in combination with "
                        "the dfs or restarting decoder when your predictors "
                        "can produce positive scores!")
    group.add_argument("--heuristics", default="",
                        help="Comma-separated list of heuristics to use in "
                        "heuristic based search like A*.\n\n"
                        "* 'predictor': Predictor specific heuristics. Some "
                        "predictors come with own heuristics - e.g. the fst "
                        "predictor uses the shortest path to the final state."
                        " Using 'predictor' combines the specific heuristics "
                        "of all selected predictors.\n"
                        "* 'greedy': Do greedy decoding to get the heuristic"
                        " costs. This is expensive but accurate.\n"
                        "* 'lasttoken': Use the single score of the last "
                        "token.\n"
                        "* 'stats': Collect unigram statistics during decoding"
                        "and compare actual hypothesis scores with the product"
                        " of unigram scores of the used words.\n"
                        "* 'scoreperword': Using this heuristic normalizes the"
                        " previously accumulated costs by its length. It can "
                        "be used for beam search with normalized scores, using"
                        " a capacity (--beam), no other heuristic, and setting"
                        "--decoder to astar.\n\n"
                        "Note that all heuristics are inadmissible, i.e. A* "
                        "is not guaranteed to find the globally best path.")
    group.add_argument("--heuristic_predictors", default="all",
                        help="Comma separated list of indices of predictors "
                        "considered by the heuristic. For example, if "
                        "--predictors is set to nmt,length,fst then setting "
                        "--heuristic_predictors to 0,2 results in using nmt "
                        "and fst in the heuristics. Use 'all' to use all "
                        "predictors in the heuristics")
    group.add_argument("--multiseg_tokenizations", default="",
                        help="This argument must be used when the multisegbeam"
                        " decoder is activated. For each predictor, it defines"
                        " the tokenizations used for it (comma separated). If "
                        "a path to a word map file is provided, the "
                        "corresponding predictor is operating on the pure "
                        "word level. The 'mixed:' prefix activates mixed "
                        "word/character models according Wu et al. (2016). "
                        "the 'eow': prefix assumes to find explicit </w>"
                        "specifiers in the word maps which mark end of words. "
                        "This is suitable for subword units, e.g. bpe.")
    group.add_argument("--cache_heuristic_estimates", default=True, type='bool',
                        help="Whether to cache heuristic future cost "
                        "estimates. This is especially useful with the greedy "
                        "heuristic.")
    group.add_argument("--pure_heuristic_scores", default=False, type='bool',
                        help="If this is set to false, heuristic decoders as "
                        "A* score hypotheses with the sum of the partial hypo "
                        "score plus the heuristic estimates (lik in standard "
                        "A*). Set to true to use the heuristic estimates only")
    group.add_argument("--restarting_node_score", default="difference",
                        choices=['difference',
                                 'absolute',
                                 'constant',
                                 'expansions'],
                        help="This parameter defines the strategy how the "
                        "restarting decoder decides from which node to restart"
                        ".\n\n"
                        "* 'difference': Restart where the difference between "
                        "1-best and 2-best is smallest\n"
                        "* 'absolute': Restart from the unexplored node with "
                        "the best absolute score globally.\n"
                        "* 'constant': Constant node score. Simulates FILO or "
                        "uniform distribution with restarting_stochastic.\n"
                        "* 'expansions': Inverse of the number of expansions "
                        "on the node. Discourages expanding arcs on the same "
                        "node repeatedly.\n")
    group.add_argument("--low_decoder_memory", default=True, type='bool',
                        help="Some decoding strategies support modes which do "
                        "not change the decoding logic, but make use of the "
                        "inadmissible pruning parameters like max_expansions "
                        "to reduce memory consumption. This usually requires "
                        "some  computational overhead for cleaning up data "
                        "structures. Applicable to restarting and bucket "
                        "decoders.")
    group.add_argument("--stochastic_decoder", default=False, type='bool',
                        help="Activates stochastic decoders. Applicable to the "
                        "decoders restarting, bow, bucket")
    group.add_argument("--decode_always_single_step", default=False, type='bool',
                        help="If this is set to true, heuristic depth first "
                        "search decoders like restarting or bow always perform "
                        "a single decoding step instead of greedy decoding. "
                        "Handle with care...")
    group.add_argument("--flip_strategy", default="move",
                        choices=['move', 'flip'],
                        help="Defines the hypothesis transition in the flip "
                        "decoder. 'flip' flips two words, 'move' moves a word "
                        "to a different position")
    group.add_argument("--bucket_selector", default="maxscore",
                        help="Defines the bucket selection strategy for the "
                        "bucket decoder.\n\n"
                        "* 'iter': Rotate through all lengths\n"
                        "* 'iter-n': Rotate through all lengths n times\n"
                        "* 'maxscore': Like iter, but filters buckets with "
                            "hypos worse than a threshold. Threshold is "
                            "increased if no bucket found\n"
                        "* 'score': Select bucket with the highest bucket "
                        "score. The bucket score is determined by the "
                        "bucket_score_strategy\n"
                        "* 'score-end': Start with the bucket with highest bucket "
                            "score, and iterate through all subsequent buckets. \n")
    group.add_argument("--bucket_score_strategy", default="difference",
                        choices=['difference', 'heap', 'absolute', 'constant'],
                        help="Defines how buckets are scored for the "
                        "bucket decoder. Usually, the best hypo in the bucket "
                        "is compared to the global best score of that length "
                        "according --collect_statistics.\n\n"
                        "* 'difference': Difference between both hypos\n"
                        "* 'heap': Use best score on bucket heap directly\n"
                        "* 'absolute': Use best hypo score in bucket directly\n"
                        "* 'constant': Uniform bucket scores.")
    group.add_argument("--collect_statistics", default="best",
                       choices=['best', 'full', 'all'],
                        help="Determines over which hypotheses statistics are "
                        "collected.\n\n"
                        "* 'best': Collect statistics from the current best "
                        "full hypothesis\n"
                        "* 'full': Collect statistics from all full hypos\n"
                        "* 'all': Collect statistics also from partial hypos\n"
                        "Applicable to the bucket decoder, the heuristic "
                        "of the bow predictor, and the heuristic 'stats'.")
    group.add_argument("--heuristic_scores_file", default="",
                       help="The bow predictor heuristic and the stats "
                       "heuristic sum up the unigram scores of words as "
                       "heuristic estimate. This option should point to a "
                       "mapping file from word-id to (unigram) score. If this "
                       "is empty, the unigram scores are collected during "
                       "decoding for each sentence separately according "
                       "--collect_statistics.")
    group.add_argument("--score_lower_bounds_file", default="",
                       help="Admissible pruning in some decoding strategies "
                       "can be improved by providing lower bounds on complete "
                       "hypothesis scores. This is useful to improve the "
                       "efficiency of exhaustive search, with lower bounds "
                       "found by e.g. beam search. The expected file format "
                       "is just a text file with line separated scores for "
                       "each sentence. Supported by the following decoders: "
                       "astar, bigramgreedy, bow, bucket, dfs, flip, restarting")
    group.add_argument("--decoder_diversity_factor", default=-1.0, type=float,
                       help="If this is greater than zero, promote diversity "
                       "between active hypotheses during decoding. The exact "
                       "way of doing this depends on --decoder:\n"
                       "* The 'beam' decoder roughly follows the approach in "
                       "Li and Jurafsky, 2016\n"
                       "* The 'bucket' decoder reorders the hypotheses in a "
                       "bucket by penalizing hypotheses with the number of "
                       "expanded hypotheses from the same parent.")
    group.add_argument("--sync_symbol", default=-1, type=int,
                       help="Used for the syncbeam decoder. Synchronization "
                       "symbol for hypothesis comparision. If negative, use "
                       "syntax_[min|max]_terminal_id.")
    group.add_argument("--max_word_len", default=25, type=int,
                       help="Maximum length of a single word. Only applicable "
                       "to the decoders multisegbeam and syncbeam.")
    group.add_argument("--mbrbeam_smooth_factor", default=0.01, type=float,
                       help="If positive, apply mix the evidence space "
                       "distribution with the uniform distribution using "
                       "this factor")
    group.add_argument("--mbrbeam_selection_strategy", default="oracle_bleu",
                        choices=['bleu','oracle_bleu'],
                        help="Defines the hypo selection strategy for mbrbeam."
                        " See the mbrbeam docstring for more information.\n"
                        "'bleu': Select the n best hypotheses with the best "
                        "expected BLEU.\n"
                        "'oracle_bleu': Optimize the expected oracle BLEU "
                        "score of the n-best list.")
    group.add_argument("--mbrbeam_evidence_strategy", default="renorm",
                        choices=['maxent','renorm'],
                        help="Defines the way the evidence space is estimated "
                        "for mbrbeam. See the mbrbeam docstring for more.\n"
                        "'maxent': Maximum entropy criterion on n-gram probs.\n"
                        "'renorm': Only use renormalized scores of the hypos "
                        "which are currently in the beam.")

    ## Output options
    group = parser.add_argument_group('Output options')
    group.add_argument("--nbest", default=0, type=int,
                        help="Maximum number of hypotheses in the output "
                        "files. Set to 0 to output all hypotheses found by "
                        "the decoder. If you use the beam or astar decoder, "
                        "this option is limited by the beam size.")
    group.add_argument("--output_fst_unk_id", default=0, type=int,
                        help="DEPRECATED: Old name for --fst_unk_id")
    group.add_argument("--fst_unk_id", default=999999998, type=int,
                        help="SGNMT uses the ID 0 for UNK. However, this "
                        "clashes with OpenFST when writing FSTs as OpenFST "
                        "reserves 0 for epsilon arcs. Therefore, we use this "
                        "ID for UNK instead. Note that this only applies "
                        "to output FSTs created by the fst or sfst output "
                        "handler, or FSTs used by the fsttok wrapper. Apart "
                        "from that, UNK is still represented by the ID 0.")
    group.add_argument("--output_path", default="sgnmt-out.%s",
                        help="Path to the output files generated by SGNMT. You "
                        "can use the placeholder %%s for the format specifier")
    group.add_argument("--outputs", default="",
                        help="Comma separated list of output formats: \n\n"
                        "* 'text': First best translations in plain text "
                        "format\n"
                        "* 'nbest': Moses' n-best format with separate "
                        "scores for each predictor.\n"
                        "* 'fst': Translation lattices in OpenFST "
                        "format with sparse tuple arcs.\n"
                        "* 'sfst': Translation lattices in OpenFST "
                        "format with standard arcs (i.e. combined scores).\n"
                        "* 'timecsv': Generate CSV files with separate "
                        "predictor scores for each time step.\n"
                        "* 'ngram': MBR-style n-gram posteriors.\n\n"
                        "For extract_scores_along_reference.py, select "
                        "one of the following output formats:\n"
                        "* 'json': Dump data in pretty JSON format.\n"
                        "* 'pickle': Dump data as binary pickle.\n"
                        "The path to the output files can be specified with "
                        "--output_path")
    group.add_argument("--remove_eos", default=True, type='bool',
                        help="Whether to remove </S> symbol on output.")
    group.add_argument("--src_wmap", default="",
                        help="Path to the source side word map (Format: <word>"
                        " <id>). This is used to map the words in --src_test "
                        "to their word IDs. If empty, SGNMT expects the input "
                        "words to be in integer representation.")
    group.add_argument("--trg_wmap", default="",
                        help="Path to the target side word map (Format: <word>"
                        " <id>). This is used to generate log output and the "
                        "output formats text and nbest. If empty, we directly "
                        "write word IDs.")
    group.add_argument("--trg_cmap", default="",
                        help="Path to the target side char map (Format: <char>"
                        " <id>). If this is not empty, all output files are "
                        "converted to character-level. The mapping from word "
                        "to character sequence is read from --trg_wmap. The "
                        "char map must contain an entry for </w> which points "
                        "to the word boundary ID.")
    
    ## Predictor options
    
    # General
    group = parser.add_argument_group('General predictor options')
    group.add_argument("--predictors", default="nmt",
                        help="Comma separated list of predictors. Predictors "
                        "are scoring modules which define a distribution over "
                        "target words given the history and some side "
                        "information like the source sentence. If vocabulary "
                        "sizes differ among predictors, we fill in gaps with "
                        "predictor UNK scores.:\n\n"
                        "* 'nmt': neural machine translation predictor.\n"
                        "         Options: nmt_config, nmt_path, gnmt_beta, "
                        "nmt_model_selector, cache_nmt_posteriors.\n"
                        "* 't2t': Tensor2Tensor predictor.\n"
                        "         Options: t2t_usr_dir, t2t_model, "
                        "t2t_problem, t2t_hparams_set, t2t_checkpoint_dir, "
                        "pred_src_vocab_size, pred_trg_vocab_size\n"
                        "* 'fertt2t': T2T predictor for fertility models.\n"
                        "       Options: syntax_pop_id, t2t_usr_dir, t2t_model,"
                        " t2t_problem, t2t_hparams_set, t2t_checkpoint_dir, "
                        "pred_src_vocab_size, pred_trg_vocab_size\n"
                        "* 'nizza': Nizza alignment models.\n"
                        "           Options: nizza_model, nizza_hparams_set, "
                        "nizza_checkpoint_dir, pred_src_vocab_size, "
                        "pred_trg_vocab_size\n"
                        "* 'lexnizza': Uses Nizza lexical scores for checking "
                        "the source word coverage.\n"
                        "           Options: nizza_model, nizza_hparams_set, "
                        "nizza_checkpoint_dir, pred_src_vocab_size, "
                        "pred_trg_vocab_size, lexnizza_alpha, lexnizza_beta, "
                        "lexnizza_shortlist_strategies, "
                        "lexnizza_max_shortlist_length, lexnizza_trg2src_model, "
                        "lexnizza_trg2src_hparams_set, lexnizza_trg2src_"
                        "checkpoint_dir, lexnizza_min_id\n"
                        "* 'bracket': Well-formed bracketing.\n"
                        "             Options: syntax_max_terminal_id, "
                        "syntax_pop_id, syntax_max_depth, extlength_path\n"
                        "* 'osm': Well-formed operation sequences.\n"
                        "         Options: osm_type\n"
                        "* 'forcedosm': Forced decoding under OSM. Use in "
                        "combination with osm predictor.\n"
                        "         Options: trg_test\n"
                        "* 'kenlm': n-gram language model (KenLM).\n"
                        "          Options: lm_pathr\n"
                        "* 'srilm': n-gram language model (SRILM).\n"
                        "          Options: lm_path, ngramc_order\n"
                        "* 'nplm': neural n-gram language model (NPLM).\n"
                        "          Options: nplm_path, normalize_nplm_probs\n"
                        "* 'rnnlm': RNN language model based on TensorFlow.\n"
                        "          Options: rnnlm_config, rnnlm_path\n"
                        "* 'forced': Forced decoding with one reference\n"
                        "            Options: trg_test\n"
                        "* 'forcedlst': Forced decoding with a Moses n-best "
                        "list (n-best list rescoring)\n"
                        "               Options: trg_test, forcedlst_match_unk"
                        " forcedlst_sparse_feat, use_nbest_weights\n"
                        "* 'bow': Forced decoding with one bag-of-words ref.\n"
                        "         Options: trg_test, heuristic_scores_file, "
                        "bow_heuristic_strategies, bow_accept_subsets, "
                        "bow_accept_duplicates, pred_trg_vocab_size\n"
                        "* 'bowsearch': Forced decoding with one bag-of-words ref.\n"
                        "         Options: hypo_recombination, trg_test, "
                        "heuristic_scores_file, bow_heuristic_strategies, "
                        "bow_accept_subsets, bow_accept_duplicates, "
                        "pred_trg_vocab_size\n"
                        "* 'fst': Deterministic translation lattices\n"
                        "         Options: fst_path, use_fst_weights, "
                        "normalize_fst_weights, fst_to_log, "
                        "fst_skip_bos_weight\n"
                        "* 'nfst': Non-deterministic translation lattices\n"
                        "          Options: fst_path, use_fst_weights, "
                        "normalize_fst_weights, fst_to_log, "
                        "fst_skip_bos_weight\n"
                        "* 'rtn': Recurrent transition networks as created by "
                        "HiFST with late expansion.\n"
                        "         Options: rtn_path, use_rtn_weights, "
                        "minimize_rtns, remove_epsilon_in_rtns, "
                        "normalize_rtn_weights\n"
                        "* 'lrhiero': Direct Hiero (left-to-right Hiero). This "
                        "is an EXPERIMENTAL implementation of LRHiero.\n"
                        "             Options: rules_path, "
                        "grammar_feature_weights, use_grammar_weights\n"
                        "* 'wc': Number of words feature.\n"
                        "        Options: wc_word.\n"
                        "* 'unkc': Poisson model for number of UNKs.\n"
                        "          Options: unk_count_lambdas, "
                        "pred_src_vocab_size.\n"
                        "* 'ngramc': Number of ngram feature.\n"
                        "            Options: ngramc_path, ngramc_order.\n"
                        "* 'length': Target sentence length model\n"
                        "            Options: src_test_raw, "
                        "length_model_weights, use_length_point_probs\n"
                        "* 'extlength': External target sentence lengths\n"
                        "               Options: extlength_path\n"
                        "All predictors can be combined with one or more "
                        "wrapper predictors by adding the wrapper name "
                        "separated by a _ symbol. Following wrappers are "
                        "available:\n"
                        "* 'idxmap': Add this wrapper to predictors which use "
                        "an alternative word map."
                        "            Options: src_idxmap, trg_idxmap\n"
                        "* 'maskvocab': Hides certain words in the SGNMT vocab"
                        "ulary from the masked predictor.\n"
                        "            Options: maskvocab_words\n"
                        "* 'altsrc': This wrapper loads source sentences from "
                        "an alternative source.\n"
                        "            Options: altsrc_test\n"
                        "* 'unkvocab': This wrapper explicitly excludes "
                        "matching word indices higher than pred_trg_vocab_size"
                        " with UNK scores.\n"
                        "             Options: pred_trg_vocab_size\n"
                        "* 'fsttok': Uses an FST to transduce SGNMT tokens to "
                        "predictor tokens.\n"
                        "             Options: fsttok_path, "
                        "fsttok_max_pending_score, fst_unk_id\n"
                        "* 'word2char': Wraps word-level predictors when SGNMT"
                        " is running on character level.\n"
                        "            Options: word2char_map\n"
                        "* 'skipvocab': Skip a subset of the predictor "
                        "vocabulary.\n"
                        "               Options: skipvocab_max_id, "
                        "skipvocab_stop_size\n"
                        "* 'ngramize': Extracts n-gram posterior from "
                        "predictors without token-level history.\n"
                        "               Options: min_ngram_order, "
                        "max_ngram_order, max_len_factor\n"
                        "\n"
                        "Note that you can use multiple instances of the same "
                        "predictor. For example, 'nmt,nmt,nmt' can be used "
                        "for ensembling three NMT systems. You can often "
                        "override parts of the predictor configurations for "
                        "subsequent predictors by adding the predictor "
                        "number (e.g. see --nmt_config2 or --fst_path2)")    
    group.add_argument("--predictor_weights", default="",
                        help="Predictor weights. Have to be specified "
                        "consistently with --predictor, e.g. if --predictor is"
                        " 'bla_fst,nmt' then set their weights with "
                        "--predictor_weights bla-weight_fst-weight,nmt-weight,"
                        " e.g. '--predictor_weights 0.1_0.3,0.6'. Default "
                        "(empty string) means that each predictor gets "
                        "assigned the weight 1. You may specify a single "
                        "weight for wrapped predictors (e.g. 0.3,0.6) if the "
                        "wrapper is unweighted.")
    group.add_argument("--per_sentence_predictor_weights", default=False,
                       type='bool', 
                       help="Assign predictor weights for each sentence. "
                       "Must be set consistent with --predictors as for "
                       "--predictor_weights. Per-sentence weights are set "
                       "by appending a comma-separated string of weights "
                       "to the end of source sentences. e.g. 'X1 X2 X3' "
                       "with two predictors might become "
                       "'X1 X2 X3 pred1_w,pred2_w' "
                       "a sentence with no weight str means each predictor is "
                       "assigned the weights set in --predictor_weights "
                       " or 1 if --predictor_weights is not set ")


    group.add_argument("--interpolation_strategy", default="",
                        help="This parameter specifies how the predictor "
                        "weights are used.\n"
                        "'fixed': Predictor weights do not change.\n"
                        "'entropy': Set predictor weight according the (cross-"
                        ") entropy of its posterior to all other predictors.\n"
                        "'moe': Use a Mixture of Experts gating network "
                        "to decide predictor weights at each time step. See "
                        "the sgnmt_moe project on how to train it.\n"
                        "Interpolation strategies can be specified for each "
                        "predictor separately, eg 'fixed|moe,moe,fixed,moe,moe'"
                        " means that a MoE network with output dimensionality "
                        "4 will decide for the 2nd, 4th, and 5th predictors, "
                        "the 1st predictor mixes the prior weight with the MoE"
                        " prediction, and the rest keep their weight from "
                        "predictor_weights.")
    group.add_argument("--interpolation_weights_mean", default="arith",
                        choices=['arith', 'geo', 'prob'],
                        help="Used when --interpolation_strategy contains |. "
                        "Specifies the way interpolation weights are combined."
                        "'arith'metirc, 'geo'metric, 'prob'abilistic.")
    group.add_argument("--moe_config", default="",
                        help="Only for MoE interpolation strategy: Semicolon-"
                        "separated key=value pairs specifying the MoE network")
    group.add_argument("--moe_checkpoint_dir", default="",
                        help="Only for MoE interpolation strategy: Path to "
                        "the TensorFlow checkpoint directory.")
    group.add_argument("--closed_vocabulary_normalization", default="none",
                        choices=['none', 'exact', 'reduced', 'rescale_unk', 'non_zero'],
                        help="This parameter specifies the way closed "
                        "vocabulary predictors (e.g. NMT) are normalized. "
                        "Closed vocabulary means that they have a predefined "
                        "vocabulary. Open vocabulary predictors (e.g. fst) can"
                        " potentially produce any word, or have a very large "
                        "vocabulary.\n\n"
                        "* 'none': Use unmodified scores for closed "
                        "vocabulary predictors\n"
                        "* 'exact': Renormalize scores depending on the "
                        "probability mass which they distribute to words "
                        "outside the vocabulary via the UNK probability.\n"
                        "* 'rescale_unk': Rescale UNK probabilities and "
                        "leave all other scores unmodified. Results in a "
                        "distribution if predictor scores are stochastic.\n"
                        "* 'reduced': Normalize to vocabulary defined by the "
                        "open vocabulary predictors at each time step.\n"
                       "* 'non_zero': only keep scores which are strictly < 0 "
                       "after combination.")
    group.add_argument("--combination_scheme", default="sum",
                        choices=['sum', 'length_norm', 'bayesian', 
                                 'bayesian_loglin', 'bayesian_state_dependent'],
                        help="This parameter controls how the combined "
                        "hypothesis score is calculated from the predictor "
                        "scores and weights.\n\n"
                        "* 'sum': The combined score is the weighted sum of "
                        "all predictor scores\n"
                        "* 'length_norm': Renormalize scores by the length of "
                        "hypotheses.\n"
                        "* 'bayesian': Apply the Bayesian LM interpolation "
                        "scheme from Allauzen and Riley to interpolate the "
                        "predictor scores\n"
                        "* 'bayesian_state_dependent': Like Bayesian "
                        "but with model-task weights defined by "
                        "'bayesian_domain_task_weights' parameter"
                        "* 'bayesian_loglin': Like bayesian, but retain "
                        "loglinear framework.")
    group.add_argument("--bayesian_domain_task_weights", default=None, 
                       help="comma-separated string of num_predictors^2 "
                       "weights where rows are domains and "
                       "tasks are columns, e.g. w[k, t] gives weight on domain k "
                       "for task t. will be reshaped into "
                       "a num_predictors x num_predictors matrix")
    group.add_argument("--t2t_unk_id", default=-1, type=int,
                        help="unk id for t2t. Used by the t2t predictor")

    group.add_argument("--pred_src_vocab_size", default=30000, type=int,
                        help="Predictor source vocabulary size. Used by the "
                        "bow, bowsearch, t2t, nizza, unkc predictors.")
    group.add_argument("--pred_trg_vocab_size", default=30000, type=int,
                        help="Predictor target vocabulary size. Used by the"
                        "bow, bowsearch, t2t, nizza, unkc predictors.")
    
    # Neural predictors
    group = parser.add_argument_group('Neural predictor options')
    group.add_argument("--length_normalization", default=False, type='bool',
                        help="DEPRECATED. Synonym for --combination_scheme "
                        "length_norm. Normalize n-best hypotheses by sentence "
                        "length. Normally improves pure NMT decoding, but "
                        "degrades performance when combined with predictors "
                        "like fst or multiple NMT systems.")
    group.add_argument("--nmt_config", default="",
                        help="Defines the configuration of the NMT model. This "
                        "can either point to a configuration file, or it can "
                        "directly contain the parameters (e.g. 'src_vocab_size"
                        "=1234,trg_vocab_size=2345'). Use 'config_file=' in "
                        "the parameter string to use configuration files "
                        "with the second method.")
    group.add_argument("--nmt_path", default="",
                        help="Defines the path to the NMT model. If empty, "
                        "the model is loaded from the default location which "
                        "depends on the NMT engine")
    group.add_argument("--nmt_engine", default="blocks",
                        choices=['none', 'blocks', 'tensorflow'],
                        help="NMT implementation which should be used. "
                        "Use 'none' to disable NMT support.")
    group.add_argument("--nmt_model_selector", default="bleu",
                        choices=['params', 'bleu', 'time'],
                        help="NMT training normally creates several files in "
                        "the ./train/ directory from which we can load the NMT"
                        " model. Possible options:\n\n"
                        "* 'params': Load parameters from params.npz. This is "
                        "usually the most recent model.\n"
                        "* 'bleu': Load from the best_bleu_params_* file with "
                        "the best BLEU score.\n"
                        "* 'time': Load from the most recent "
                        "best_bleu_params_* file.")
    group.add_argument("--cache_nmt_posteriors", default=False, type='bool',
                        help="This enables the cache in the [F]NMT predictor. "
                        "Normally, the search procedure is responsible to "
                        "avoid applying predictors to the same history twice. "
                        "However, due to the limited NMT vocabulary, two "
                        "different histories might be the same from the NMT "
                        "perspective, e.g. if they are the same up to words "
                        "which are outside the NMT vocabulary. If this "
                        "parameter is set to true, we cache posteriors with "
                        "histories containing UNK and reload them when needed")
    group.add_argument("--gnmt_beta", default=0.0, type=float,
                       help="If this is greater than zero, add a coverage "
                       "penalization term following Google's NMT (Wu et al., "
                       "2016) to the NMT score. Only works for the Blocks "
                       "NMT predictor.")
    group.add_argument("--gnmt_alpha", default=0.0, type=float,
                       help="If this is greater than zero and the combination "
                       "scheme is set to length_norm, use Google-style length "
                       " normalization (Wu et al., 2016) rather than simply "
                       "dividing by translation length.")
    group.add_argument("--syntax_max_depth", default=30, type=int,
                       help="Maximum depth of generated trees. After this "
                       "depth is reached, only terminals and POP are allowed "
                       "on the next layer.")
    group.add_argument("--syntax_root_id", default=-1, type=int,
                       help="Must be set for the layerbylayer predictor. ID "
                       "of the initial target root label.")
    group.add_argument("--syntax_pop_id", default="-1", type=str,
                       help="ID of the closing bracket in output syntax trees."
                       " layerbylayer and t2t predictors support single "
                       "integer values. The bracket predictor can take a comma"
                       "-separated list of integers.")
    group.add_argument("--syntax_min_terminal_id", default=0,
                       type=int,
                       help="All token IDs smaller than this are considered to "
                       "be non-terminal symbols except the ones specified by "
                       "--syntax_terminal_list")
    group.add_argument("--syntax_max_terminal_id", default=30003,
                       type=int,
                       help="All token IDs larger than this are considered to "
                       "be non-terminal symbols except the ones specified by "
                       "--syntax_terminal_list")
    group.add_argument("--syntax_terminal_list", default="",
                       help="List of IDs which are explicitly treated as "
                       "terminals, in addition to all IDs lower or equal "
                       "--syntax_max_terminal_id. This can be used to "
                       "exclude the POP symbol from the list of non-terminals "
                       "even though it has a ID higher than max_terminal_id.")
    group.add_argument("--syntax_nonterminal_ids", default="",
                       help="Explicitly define non-terminals with a file "
                       "containing their ids. Useful when non-terminals do "
                       "not occur consecutively in data (e.g. internal bpe "
                       "units.)")
    group.add_argument("--osm_type", default="osm", type=str,
                       help="Set of operations used for OSM predictor.\n"
                       "- 'osm': Original OSNMT of Stahlberg et al. (2018)\n"
                       "- 'srcosm': Original OSNMT where IDs>7 are POP\n"
                       "- 'pbosm': Phrase-based OSNMT")
    group.add_argument("--t2t_usr_dir", default="",
                       help="Available for the t2t predictor. See the "
                       "--t2t_usr_dir argument in tensor2tensor.")
    group.add_argument("--t2t_model", default="transformer",
                       help="Available for the t2t predictor. Name of the "
                       "tensor2tensor model.")
    group.add_argument("--t2t_problem", default="translate_ende_wmt32k",
                       help="Available for the t2t predictor. Name of the "
                       "tensor2tensor problem.")
    group.add_argument("--t2t_hparams_set",
                       default="transformer_base_single_gpu",
                       help="Available for the t2t predictor. Name of the "
                       "tensor2tensor hparams set.")
    group.add_argument("--t2t_checkpoint_dir", default="",
                       help="Available for the t2t predictor. Path to the "
                       "tensor2tensor checkpoint directory. Same as "
                       "--output_dir in t2t_trainer.")
    group.add_argument("--t2t_src_vocab_size", default=0, type=int,
                        help="DEPRECATED! Use --pred_src_vocab_size")
    group.add_argument("--t2t_trg_vocab_size", default=0, type=int,
                        help="DEPRECATED! Use --pred_trg_vocab_size")
    group.add_argument("--nizza_model", default="model1",
                       help="Available for the nizza predictor. Name of the "
                       "nizza model.")
    group.add_argument("--nizza_hparams_set",
                       default="model1_default",
                       help="Available for the nizza predictor. Name of the "
                       "nizza hparams set.")
    group.add_argument("--nizza_checkpoint_dir", default="",
                       help="Available for the nizza predictor. Path to the "
                       "nizza checkpoint directory. Same as "
                       "--model_dir in nizza_trainer.")
    group.add_argument("--lexnizza_trg2src_model", default="model1",
                       help="Available for the lexnizza predictor. Name of "
                       "the target-to-source nizza model.")
    group.add_argument("--lexnizza_trg2src_hparams_set",
                       default="model1_default",
                       help="Available for the lexnizza predictor. Name of "
                       "the target-to-source nizza hparams set.")
    group.add_argument("--lexnizza_trg2src_checkpoint_dir", default="",
                       help="Available for the lexnizza predictor. Path to "
                       "the target-to-source nizza checkpoint directory. Same "
                       "as --model_dir in nizza_trainer.")
    group.add_argument("--lexnizza_shortlist_strategies", 
                       default="top10",
                       help="Comma-separated list of strategies to extract "
                       "a short list of likely translations from lexical "
                       "Model1 scores. Strategies are combined using the "
                       "union operation. Available strategies:\n"
                       "* top<N>: Select the top N words.\n"
                       "* prob<p>: Select the top words such that their "
                       " combined probability mass is greater than p.")
    group.add_argument("--lexnizza_alpha", default=0.0, type=float,
                       help="Score of each word which matches a short list.")
    group.add_argument("--lexnizza_beta", default=1.0, type=float,
                       help="Penalty for each uncovered word at the end.")
    group.add_argument("--lexnizza_max_shortlist_length", default=0, type=int,
                        help="If positive and a shortlist is longer than this "
                        "limit, initialize the coverage vector at this "
                        "position with 1")
    group.add_argument("--lexnizza_min_id", default=0, type=int,
                        help="Word IDs lower than this are not considered by "
                        "lexnizza. Can be used to filter out frequent words.")

    # Length predictors
    group = parser.add_argument_group('Length predictor options')
    group.add_argument("--src_test_raw", default="",
                        help="Only required for the 'length' predictor. Path "
                        "to original source test set WITHOUT word indices. "
                        "This is used to extract features for target sentence "
                        "length predictions")
    group.add_argument("--length_model_weights", default="",
                        help="Only required for length predictor. String of "
                        "length model parameters.")
    group.add_argument("--use_length_point_probs", default=False, type='bool',
                        help="If this is true, the length predictor outputs "
                        "probability 1 for all tokens except </S>. For </S> it"
                        " uses the point probability given by the length "
                        "model. If this is set to false, we normalize the "
                        "predictive score by comparing P(l=x) and P(l<x)")
    group.add_argument("--length_model_offset", default=0, type=int,
                        help="The target sentence length model is applied to "
                        "hypothesis length minus length_model_offst")
    group.add_argument("--extlength_path", default="",
                        help="Only required for the 'extlength' predictor. "
                        "This is the path to the file which specifies the "
                        "length distributions for each sentence. Each line "
                        "consists of blank separated '<length>:<logprob>' "
                        "pairs.")
    
    # UNK count predictors
    group = parser.add_argument_group('Count predictor options')
    group.add_argument("--unk_count_lambdas", default="1.0",
                        help="Model parameters for the UNK count model: comma-"
                        "separated list of lambdas for Poisson distributions. "
                        "The first float specifies the Poisson distribution "
                        "over the number of UNKs in the hypotheses given that "
                        "the number of UNKs on the source side is 0. The last "
                        "lambda specifies the distribution given >=n-1 UNKs "
                        "in the source sentence.")
    group.add_argument("--wc_word", default=-1, type=int,
                       help="If negative, the wc predictor counts all "
                       "words. Otherwise, count only the specific word")
    group.add_argument("--wc_nonterminal_penalty", default=False, 
                       action='store_true', help="if true, "
                       "use syntax_[max|min]_terminal_id to apply penalty to "
                       "all non-terminals")

    group.add_argument("--syntax_nonterminal_factor", default=1.0, type=float,
                       help="penalty factor for WeightNonTerminalWrapper to apply")

    group.add_argument("--ngramc_path", default="ngramc/%d.txt",
                        help="Only required for ngramc predictor. The ngramc "
                        "predictor counts the number of ngrams and multiplies "
                        "them with the factors defined in the files. The "
                        "format is one ngram per line '<ngram> : <score>'. "
                        "You can use the placeholder %%d for the sentence "
                        "index.")
    group.add_argument("--ngramc_order", default=0, type=int,
                       help="If positive, count only ngrams of the specified "
                       "Order. Otherwise, count all ngrams")
    group.add_argument("--min_ngram_order", default=1, type=int,
                       help="Minimum ngram order for ngramize wrapper and "
                       "ngram output format")
    group.add_argument("--max_ngram_order", default=4, type=int,
                       help="Maximum ngram order for ngramize wrapper amd "
                       "ngram output format")
    group.add_argument("--ngramc_discount_factor", default=-1.0, type=float,
                       help="If this is non-negative, discount ngram counts "
                       "by this factor each time the ngram is consumed")
    group.add_argument("--skipvocab_max_id", default=30003, type=int,
                        help="All tokens above this threshold are skipped "
                        "by the skipvocab predictor wrapper.")
    group.add_argument("--skipvocab_stop_size", default=1, type=int,
                        help="The internal beam search of the skipvocab "
                        "predictor wrapper stops if the best stop_size "
                         "scores are for in-vocabulary words (ie. with index "
                         "lower or equal skipvocab_max_id")

    # Forced predictors
    group = parser.add_argument_group('Forced decoding predictor options')
    group.add_argument("--trg_test", default="",
                        help="Path to target test set (with integer tokens). "
                        "This is only required for the predictors 'forced' "
                        "and 'forcedlst'. For 'forcedlst' this needs to point "
                        "to an n-best list in Moses format.")
    group.add_argument("--forcedlst_sparse_feat", default="", 
                        help="Per default, the forcedlst predictor uses the "
                        "combined score in the Moses nbest list. Alternatively,"
                        " for nbest lists in sparse feature format, you can "
                        "specify the name of the features which should be "
                        "used instead.")
    group.add_argument("--forcedlst_match_unk", default=False, type='bool',
                        help="Only required for forcedlst predictor. If true, "
                        "allow any word where the n-best list has an UNK.")
    group.add_argument("--use_nbest_weights", default=False, type='bool',
                        help="Only required for forcedlst predictor. Whether "
                        "to use the scores in n-best lists.")
    group.add_argument("--bow_heuristic_strategies", default="remaining",
                       help="Defines the form of heuristic estimates of the "
                       "bow predictor. Comma-separate following values:\n"
                       "* remaining: sum up unigram estimates for all words "
                       "in the bag which haven't been consumed\n"
                       "* consumed: Use the difference between the actual "
                       "hypothesis score and the sum of unigram estimates "
                       "of consumed words as score")
    group.add_argument("--bow_accept_subsets", default=False, type='bool',
                       help="If this is set to false, the bow predictor "
                       "enforces exact correspondence between bag and words "
                       "in complete hypotheses. If false, it ensures that "
                       "hypotheses are consistent with the bag (i.e. do not "
                       "contain words outside the bag) but do not necessarily "
                       "have all words in the bag")
    group.add_argument("--bow_accept_duplicates", default=False, type='bool',
                       help="If this is set to true, the bow predictor "
                       "allows a word in the bag to appear multiple times, "
                       "i.e. the exact count of the word is not enforced. "
                       "Can only be used in conjunction with bow_accept_subsets")
    group.add_argument("--bow_diversity_heuristic_factor", default=-1.0, type=float,
                       help="If this is greater than zero, promote diversity "
                       "between bags via the bow predictor heuristic. Bags "
                       "which correspond to bags of partial bags of full "
                       "hypotheses are penalized by this factor.")
    
    # Wrappers
    group = parser.add_argument_group('Wrapper predictor options')
    group.add_argument("--src_idxmap", default="",
                        help="Only required for idxmap wrapper predictor. Path"
                        " to the source side mapping file. The format is "
                        "'<index> <alternative_index>'. The mapping must be "
                        "complete and should be a bijection.")
    group.add_argument("--trg_idxmap", default="",
                        help="Only required for idxmap wrapper predictor. Path"
                        " to the target side mapping file. The format is "
                        "'<index> <alternative_index>'. The mapping must be "
                        "complete and should be a bijection.")
    group.add_argument("--maskvocab_words", default="",
                        help="Only required for maskvocab wrapper predictor. "
                        "Comma-separated list of token IDs which are masked "
                        "out.")
    group.add_argument("--altsrc_test", default="test_en.alt",
                        help="Only required for altsrc wrapper predictor. Path"
                        " to the alternative source sentences.")
    group.add_argument("--word2char_map", default="word2char.map",
                        help="Only required for word2char wrapper predictor. "
                        "Path to a mapping file from word ID to sequence of "
                        "character IDs (format: <word-id> <char-id1> <char-id2"
                        ">...). All character IDs which do not occur in this "
                        "mapping are treated as word boundary symbols.")
    group.add_argument("--fsttok_path", default="tok.fst",
                        help="For the fsttok wrapper. Defines the path to the "
                        "FSt which transduces sequences of SGNMT tokens (eg. "
                        "characters) to predictor tokens (eg BPEs). FST may "
                        "be non-deterministic and contain epsilons.")
    group.add_argument("--fsttok_max_pending_score", default=5.0, type=float,
                       help="Applicable if an FST used by the fsttok wrapper "
                       "is non-deterministic. In this case, one predictor "
                       "state may correspond to multiple nodes in the FST. "
                       "We prune nodes which are this much worse than the "
                       "best scoring node with the same history.")

    # Hiero predictor
    group = parser.add_argument_group('Hiero predictor options')
    group.add_argument("--rules_path", default="rules/rules",
                        help="Only required for predictor lrhiero. Path to "
                        "the ruleXtract rules file.")
    group.add_argument("--use_grammar_weights", default=False, type='bool',
                        help="Whether to use weights in the synchronous "
                        "grammar for the lrhiero predictor. If set to false, "
                        "use uniform grammar scores.")
    group.add_argument("--grammar_feature_weights", default='',
                        help="If rules_path points to a factorized rules file "
                        "(i.e. containing rules associated with a number of "
                        "features, not only one score) SGNMT uses a weighted "
                        "sum for them. You can specify the weights for this "
                        "summation here (comma-separated) or leave it blank "
                        "to sum them up equally weighted.")
    
    # (NP)LM predictors
    group = parser.add_argument_group('(Neural) LM predictor options')
    group.add_argument("--lm_path", default="lm/ngram.lm.gz",
                        help="Path to the ngram LM file in ARPA format")
    group.add_argument("--srilm_convert_to_ln", default=False,
                        help="Whether to convert srilm scores from log to ln.")
    group.add_argument("--nplm_path", default="nplm/nplm.gz",
                        help="Path to the NPLM language model")
    group.add_argument("--rnnlm_path", default="rnnlm/rnn.ckpt",
                        help="Path to the RNNLM language model")
    group.add_argument("--rnnlm_config", default="rnnlm.ini",
                        help="Defines the configuration of the RNNLM model. This"
                        " can either point to a configuration file, or it can "
                        "directly contain the parameters (e.g. 'src_vocab_size"
                        "=1234,trg_vocab_size=2345'). Use 'config_file=' in "
                        "the parameter string to use configuration files "
                        "with the second method. Use 'model_name=X' in the "
                        "parameter string to use one of the predefined models.")
    group.add_argument("--normalize_nplm_probs", default=False, type='bool',
                        help="Whether to normalize nplm probabilities over "
                        "the current unbounded predictor vocabulary.")
    
    # FSM predictors
    group = parser.add_argument_group('FST and RTN predictor options')
    group.add_argument("--fst_path", default="fst/%d.fst",
                        help="Only required for fst and nfst predictor. Sets "
                        "the path to the OpenFST translation lattices. You "
                        "can use the placeholder %%d for the sentence index.")

    group.add_argument("--syntax_path", default=None,
                        help="Only required for parse predictor. Sets "
                        "the path to the grammar non-terminal map determining"
                        "permitted parses")
    group.add_argument("--syntax_bpe_path", default=None,
                        help="Internal can-follow syntax for subwords")
    group.add_argument("--syntax_word_out", default=True, type='bool',
                        help="Whether to output word tokens only from parse" 
                        "predictor.")
    group.add_argument("--syntax_allow_early_eos", default=False, type='bool',
                        help="Whether to let parse predictor output EOS "
                        "instead of any terminal")
    group.add_argument("--syntax_norm_alpha", default=1.0, type=float,
                        help="Normalizing alpha for internal beam search")
    group.add_argument("--syntax_max_internal_len", default=35, type=int,
                        help="Max length of non-terminal sequences to consider")
    group.add_argument("--syntax_internal_beam", default=1, type=int,
                        help="Beam size when internally searching for words" 
                       "using parse predictor")
    group.add_argument("--syntax_consume_ooc", default=False, type='bool',
                        help="Whether to let parse predictor consume tokens "
                       "which are not permitted by the current LHS")
    group.add_argument("--syntax_tok_grammar", default=False, type='bool',
                        help="Whether to use a token-based grammar."
                        "Default uses no internal grammar")
    group.add_argument("--syntax_terminal_restrict", default=True, type='bool',
                        help="Whether to restrict inside terminals.")
    group.add_argument("--syntax_internal_only", default=False, type='bool',
                        help="Whether to restrict only non-terminals.")
    group.add_argument("--syntax_eow_ids", default=None,
                        help="ids for end-of-word tokens")
    group.add_argument("--syntax_terminal_ids", default=None,
                        help="ids for terminal tokens")
    group.add_argument("--rtn_path", default="rtn/",
                        help="Only required for rtn predictor. Sets "
                        "the path to the RTN directory as created by HiFST")
    group.add_argument("--fst_skip_bos_weight", default=True, type='bool',
                        help="This option applies to fst and nfst "
                        "predictors. Lattices produced by HiFST contain the "
                        "<S> symbol and often have scores on the corresponding"
                        " arc. However, SGNMT skips <S> and this score is not "
                        "regarded anywhere. Set this option to true to add the "
                        "<S> scores. This ensures that the "
                        "complete path scores for the [n]fst and rtn "
                        "predictors match the corresponding path weights in "
                        "the original FST as obtained with fstshortestpath.")
    group.add_argument("--fst_to_log", default=True, type='bool',
                        help="Multiply weights in the FST by -1 to transform "
                        "them from tropical semiring into logprobs.")
    group.add_argument("--use_fst_weights", default=False, type='bool',
                        help="Whether to use weights in FSTs for the"
                        "nfst and fst predictor.")
    group.add_argument("--use_rtn_weights", default=False, type='bool',
                        help="Whether to use weights in RTNs.")
    group.add_argument("--minimize_rtns", default=True, type='bool',
                        help="Whether to do determinization, epsilon removal, "
                        "and minimization after each RTN expansion.")
    group.add_argument("--remove_epsilon_in_rtns", default=True, type='bool',
                        help="Whether to remove epsilons after RTN expansion.")
    group.add_argument("--normalize_fst_weights", default=False, type='bool',
                        help="Whether to normalize weights in FSTs. This "
                        "forces the weights on outgoing edges to sum up to 1. "
                        "Applicable to fst and nfst predictor.")
    group.add_argument("--normalize_rtn_weights", default=False, type='bool',
                        help="Whether to normalize weights in RTNs. This "
                        "forces the weights on outgoing edges to sum up to 1. "
                        "Applicable to rtn predictor.")

    # Adding arguments for overriding when using same predictor multiple times
    group = parser.add_argument_group('Override options')
    for n,w in [('2', 'second'), ('3', 'third'), ('4', '4-th'), ('5', '5-th'), 
                ('6', '6-th'), ('7', '7-th'), ('8', '8-th'), ('9', '9-th'), 
                ('10', '10-th'), ('11', '11-th'), ('12', '12-th')]:
        group.add_argument("--nmt_config%s" % n,  default="",
                        help="If the --predictors string contains more than "
                        "one nmt predictor, you can specify the configuration "
                        "for the %s one with this parameter. The %s nmt "
                        "predictor inherits all previous settings except for "
                        "the ones in this parameter." % (w, w))
        group.add_argument("--nmt_path%s" % n, default="",
                        help="Overrides --nmt_path for the %s nmt" % w)
        group.add_argument("--nmt_engine%s" % n, default="",
                        help="Overrides --nmt_engine for the %s nmt" % w)
        group.add_argument("--t2t_model%s" % n, default="",
                        help="Overrides --t2t_model for the %s t2t predictor"
                        % w)
        group.add_argument("--t2t_problem%s" % n, default="",
                        help="Overrides --t2t_problem for the %s t2t predictor"
                        % w)
        group.add_argument("--t2t_hparams_set%s" % n, default="",
                        help="Overrides --t2t_hparams_set for the %s t2t "
                        "predictor" % w)
        group.add_argument("--t2t_checkpoint_dir%s" % n, default="",
                        help="Overrides --t2t_checkpoint_dir for the %s t2t "
                        "predictor" % w)
        group.add_argument("--pred_src_vocab_size%s" % n, default=0, type=int,
                        help="Overrides --pred_src_vocab_size for the %s t2t "
                        "predictor" % w)
        group.add_argument("--t2t_unk_id%s" % n, default=3, type=int,
                        help="Overrides --t2t_unk_id for the %s t2t "
                        "predictor" % w)

        group.add_argument("--pred_trg_vocab_size%s" % n, default=0, type=int,
                        help="Overrides --pred_trg_vocab_size for the %s t2t "
                        "predictor" % w)
        group.add_argument("--rnnlm_config%s" % n,  default="",
                        help="If the --predictors string contains more than "
                        "one rnnlm predictor, you can specify the configuration "
                        "for the %s one with this parameter. The %s rnnlm "
                        "predictor inherits all previous settings except for "
                        "the ones in this parameter." % (w, w))
        group.add_argument("--rnnlm_path%s" % n, default="",
                        help="Overrides --rnnlm_path for the %s nmt" % w)
        group.add_argument("--altsrc_test%s" % n, default="",
                        help="Overrides --altsrc_test for the %s altsrc" % w)
        group.add_argument("--word2char_map%s" % n, default="",
                        help="Overrides --word2char_map for the %s word2char" % w)
        group.add_argument("--fsttok_path%s" % n, default="",
                        help="Overrides --fsttok_path for the %s fsttok" % w)
        group.add_argument("--src_idxmap%s" % n, default="",
                        help="Overrides --src_idxmap for the %s indexmap" % w)
        group.add_argument("--trg_idxmap%s" % n, default="",
                        help="Overrides --trg_idxmap for the %s indexmap" % w)
        group.add_argument("--fst_path%s" % n, default="",
                        help="Overrides --fst_path for the %s fst "
                        "predictor" % w)
        group.add_argument("--forcedlst_sparse_feat%s" % n, default="",
                        help="Overrides --forcedlst_sparse_feat for the %s "
                        "forcedlst predictor" % w)
        group.add_argument("--ngramc_path%s" % n, default="",
                        help="Overrides --ngramc_path for the %s ngramc" % w)
        group.add_argument("--ngramc_order%s" % n, default=0, type=int,
                        help="Overrides --ngramc_order for the %s ngramc" % w)
    return parser


def get_args():
    """Get the arguments for the current SGNMT run from both command
    line arguments and configuration files. This method contains all
    available SGNMT options, i.e. configuration is not encapsulated e.g.
    by predictors. 
    
    Returns:
        object. Arguments object like for ``ArgumentParser``
    """ 
    parser = get_parser()
    args = parse_args(parser)
    
    # Legacy parameter names
    if args.t2t_src_vocab_size > 0:
        args.pred_src_vocab_size = args.t2t_src_vocab_size
    if args.t2t_trg_vocab_size > 0:
        args.pred_trg_vocab_size = args.t2t_trg_vocab_size
    if args.length_normalization:
        args.combination_scheme = "length_norm"
    if args.output_fst_unk_id:
        args.fst_unk_id = args.output_fst_unk_id 
    return args


def validate_args(args):
    """Some rudimentary sanity checks for configuration options.
    This method directly prints help messages to the user. In case of fatal
    errors, it terminates using ``logging.fatal()``
    
    Args:
        args (object):  Configuration as returned by ``get_args``
    """
    for depr in ['length_normalization', 
                 't2t_src_vocab_size',
                 't2t_trg_vocab_size']:
        if getattr(args, depr):
            logging.warn("Using deprecated argument %s. Please check the "
                         "documentation for the replacement." % depr)
    # Validate --range
    if args.range and args.input_method == 'shell':
        logging.warn("The --range parameter can lead to unintuitive "
                     "behavior in 'shell' mode.")
        
    # Some common pitfalls
    sanity_check_failed = False
    if args.input_method == 'dummy' and args.max_len_factor < 10:
        logging.warn("You are using the dummy input method but a low value "
                     "for max_len_factor (%d). This means that decoding will "
                     "not consider hypotheses longer than %d tokens. Consider "
                     "increasing max_len_factor to the length longest relevant"
                     " hypothesis" % (args.max_len_factor, args.max_len_factor))
        sanity_check_failed = True
    if (args.decoder == "beam" and args.combination_scheme == "length_norm"
                               and args.early_stopping):
        logging.warn("You are using beam search with length normalization but "
                     "with early stopping. All hypotheses found with beam "
                     "search with early stopping have the same length. You "
                     "might want to disable early stopping.")
        sanity_check_failed = True
    if args.combination_scheme != "length_norm" and args.gnmt_alpha != 0.0:
        logging.warn("Setting gnmt_alpha has no effect without using the "
                     "combination scheme length_norm.")
        sanity_check_failed = True
    if "t2t" in args.predictors and args.indexing_scheme != "t2t":
        logging.warn("You are using the t2t predictor, but indexing_scheme "
                     "is not set to t2t.")
        sanity_check_failed = True
    if sanity_check_failed and not args.ignore_sanity_checks:
        raise AttributeError("Sanity check failed (see warnings). If you want "
            "to proceed despite these warnings, use --ignore_sanity_checks.")

