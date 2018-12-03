#!/usr/bin/env python3 -u
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
#
# Modified in 27/11/2018 Marcely Zanon Boito
#

import torch, sys, os
from fairseq import data, options, progress_bar, tasks, tokenizer, utils
from fairseq.meters import StopwatchMeter, TimeMeter
from fairseq.sequence_generator import SequenceGenerator
from fairseq.sequence_scorer import SequenceScorer
import numpy as np
import codecs

def write_attention_textfiles(s_id, src_sentence, trg_sentence, att_dictionary, root_directory):
    #print(s_id)
    #print(src_sentence)
    #print(trg_sentence)
    encoder = att_dictionary[0]
    decoder = att_dictionary[1]
    for coder_dict in [encoder, decoder]:
        name = list(coder_dict)[0] #if it is (en|de)coder
        root_dir = "/".join([root_directory,name])
        check_root(root_dir)
        is_encoder = True if "Encoder" in name else False
        write_coder_attentions(s_id, src_sentence, trg_sentence, coder_dict[name], is_encoder, root_dir)

def write_coder_attentions(s_id, src_sentence, trg_sentence, coder_dict, is_encoder, root_directory):
    for layer in coder_dict:
        root = "/".join([root_directory, str(layer)])
        check_root(root)
        for att_type in list(coder_dict[layer]):
            sub_dir = "/".join([root, att_type]) 
            check_root(sub_dir)
            file_name = "/".join([sub_dir, str(s_id.item())])
            if att_type == "SelfAttention":
                if is_encoder:
                    write_attention_matrices(src_sentence, src_sentence, coder_dict[layer][att_type], file_name)
                else:
                    write_attention_matrices(trg_sentence, trg_sentence, coder_dict[layer][att_type], file_name)
            else:
                write_attention_matrices(src_sentence, trg_sentence, coder_dict[layer][att_type], file_name)


def write_attention_matrices(src_sentence, trg_sentence, matrix, path):
    src = [""] + src_sentence.split(" ") + ["</S>"]
    trg = trg_sentence.split(" ") + ["</S>"]
    #matrix = matrix[0] if len(matrix) == 1 else matrix
    
    avg = matrix[0][0]
    heads = matrix[1][0] #the last [0] is only to remove an empty dimension
    write_matrix(src, trg, avg, path + "_avg")
    for i in range(len(heads)):
        write_matrix(src, trg, heads[i], path + "_head" + str(i))

def write_matrix(src_sentence, trg_sentence, matrix, path):
    with codecs.open(path + ".txt", "w", "utf-8") as output_matrix:
        output_matrix.write("\t".join(src_sentence) + "\n") 
        for i in range(len(matrix)):
            output_matrix.write(trg_sentence[i] + "\t" + line_to_str(matrix[i]) + "\n")

def line_to_str(tensor):
    matrix = torch.Tensor.numpy(tensor)
    values = []
    for i in range(len(matrix)):
        values += [str(matrix[i].item())]
    return "\t".join(values)

def check_root(root_directory):
    try:
        os.stat(root_directory)
    except:
        os.makedirs(root_directory)

def read_gold(f_path):
    return [line.strip("\n") for line in codecs.open(f_path)]

def main(args):
    ####1 ARGS SETTING
    assert args.path is not None, '--path required for generation!'
    assert args.root_directory is not None, '--root directory required for logging!'
    use_cuda = torch.cuda.is_available() and not args.cpu
    args.score_reference = True
    args.print_alignment = True 
    args.max_sentences= 1 #--batch-size 1
    args.beam = 1 #--beam 1
    args.no_progress_bar = True
    args.replace_unk = True
    ####2 LOAD DATASET IN THE RIGHT FORMAT
    task = tasks.setup_task(args)
    task.load_dataset(args.gen_subset)
    print('| {} {} {} examples'.format(args.data, args.gen_subset, len(task.dataset(args.gen_subset))))
    # Set dictionaries
    src_dict = task.source_dictionary
    tgt_dict = task.target_dictionary

    #print("Attention {}".format(args.print_alignment))
    #print("Score reference {}".format(args.score_reference))
    ####3 LOAD MODEL
    print('| loading model(s) from {}'.format(args.path))
    models, _ = utils.load_ensemble_for_inference(args.path.split(':'), task, model_arg_overrides=eval(args.model_overrides))
    '''
    models, args = utils.load_ensemble_for_inference(args.path.split(':'), task, model_arg_overrides=eval(args.model_overrides))
    it returns the list of models, in our case: 
        models: <class 'list'> 
        models[0]: <class 'fairseq.models.transformer.TransformerModel'>
    the second argument is the list of model's parameters (embedding size, number of heads, etc)
    '''
    # Optimize ensemble for generation
    for model in models:
        model.make_generation_fast_(
            beamable_mm_beam_size=None if args.no_beamable_mm else args.beam,
            need_attn=args.print_alignment, #!!
        )
        if args.fp16:
            model.half()
    # Load alignment dictionary for unknown word replacement
    # (None if no unknown word replacement, empty if no path to align dictionary)
    #align_dict = utils.load_align_dict(args.replace_unk)

    ####4 CREATE AN ITERATOR
    # Load dataset (possibly sharded)
    itr = task.get_batch_iterator(
        dataset=task.dataset(args.gen_subset),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=utils.resolve_max_positions(task.max_positions(),*[model.max_positions() for model in models]),
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=8,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
    ).next_epoch_itr(shuffle=False)
    '''
    task.dataset(args.gen_subset): <fairseq.data.language_pair_dataset.LanguagePairDataset object at 0x7fdf4f505a90>
    args.max_sentences: None
    max_positions: (1024, 1024) ??????
    RETURN itr: <class 'fairseq.data.iterators.CountingIterator'>
    '''

    ####5 INIT GENERATOR
    gen_timer = StopwatchMeter()
    translator = SequenceScorer(models, task.target_dictionary)
    if use_cuda:
        translator.cuda()

    #print(models[0])    
    ####6 SCORING
    # Generate and compute BLEU score
    #scorer = bleu.Scorer(tgt_dict.pad(), tgt_dict.eos(), tgt_dict.unk())
    #has_target = True
    s = 0
    check_root(args.root_directory)
    src_gold = read_gold(args.root_directory + "/corpus.gold")

    with progress_bar.build_progress_bar(args, itr) as t: #creates a progress bar, life goes on
        translations = translator.score_batched_itr(t, cuda=use_cuda, timer=gen_timer)
        '''translations: <class 'generator'>
        just defines the generator,does not calculate stuff right here'''
        wps_meter = TimeMeter()
        for sample_id, src_tokens, target_tokens, hypos, acc_attentions in translations:
            s+=1
            src_str = src_dict.string(src_tokens)
            target_str = tgt_dict.string(target_tokens, args.remove_bpe, escape_unk=True)
            if "unk" in src_str:
                s_id = sample_id.item()
                #print(src_str)
                #print(src_dict[s_id])
                assert len(src_str.split(" ")) == len(src_gold[s_id].split(" "))
                src_str = src_gold[s_id]
            
            write_attention_textfiles(sample_id, src_str, target_str, acc_attentions, args.root_directory)
            
            wps_meter.update(src_tokens.size(0))
            t.log({'wps': round(wps_meter.avg)})
            if s > 100:
                break



if __name__ == '__main__':
    parser = options.get_matrices_generator_parser()#get_generation_parser()
    args = options.parse_args_and_arch(parser)
    main(args)
