# This file is the implementation for ensemble evaluation.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *
import misc.utils as utils
from misc.uncertainty_utils import calculate_uncertainties

from .CaptionModel import CaptionModel
from .AttModel import pack_wrapper, AttModel

class AttEnsemble(AttModel):
    def __init__(self, models, weights=None):
        CaptionModel.__init__(self)
        # super(AttEnsemble, self).__init__()

        self.models = nn.ModuleList(models)
        self.vocab_size = models[0].vocab_size
        self.seq_length = models[0].seq_length
        self.bad_endings_ix = models[0].bad_endings_ix
        self.ss_prob = 0
        weights = weights or [1.0] * len(self.models)
        self.register_buffer('weights', torch.tensor(weights))

    def init_hidden(self, batch_size):
        state = [m.init_hidden(batch_size) for m in self.models]
        return self.pack_state(state)

    def pack_state(self, state):
        self.state_lengths = [len(_) for _ in state]
        return sum([list(_) for _ in state], [])

    def unpack_state(self, state):
        out = []
        for l in self.state_lengths:
            out.append(state[:l])
            state = state[l:]
        return out

    def embed(self, it):
        return [m.embed(it) for m in self.models]

    def core(self, *args):
        return zip(*[m.core(*_) for m, _ in zip(self.models, zip(*args))])

    def get_logprobs_state(self, it, tmp_fc_feats, tmp_att_feats, tmp_p_att_feats, tmp_att_masks, state, output_logsoftmax=1):
        # 'it' contains a word index
        xt = self.embed(it)

        state = self.unpack_state(state)
        output, state = self.core(xt, tmp_fc_feats, tmp_att_feats, tmp_p_att_feats, state, tmp_att_masks)
        # (beam_size, vocab_size, num_candidates)
        cand_probs = torch.stack([F.softmax(m.logit(output[i]), dim=1) for i,m in enumerate(self.models)], 2)
        # (beam_size, vocab_size)
        probs = cand_probs.mul(self.weights).div(self.weights.sum()).sum(-1)
        aleatorics, epistemics = calculate_uncertainties(cand_probs, self.weights)
        
        return probs.log(), self.pack_state(state), aleatorics, epistemics

    def _prepare_feature(self, *args):
        return tuple(zip(*[m._prepare_feature(*args) for m in self.models]))

    def _sample_beam(self, fc_feats, att_feats, att_masks=None, opt={}):
        beam_size = opt.get('beam_size', 10)
        batch_size = fc_feats.size(0)

        fc_feats, att_feats, p_att_feats, att_masks = self._prepare_feature(fc_feats, att_feats, att_masks)

        assert beam_size <= self.vocab_size + 1, 'lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed'
        seq = torch.LongTensor(self.seq_length, batch_size).zero_()
        seqLogprobs = torch.FloatTensor(self.seq_length, batch_size, self.vocab_size + 1)
        seqAleatorics = torch.FloatTensor(self.seq_length, batch_size)
        seqEpistemics = torch.FloatTensor(self.seq_length, batch_size)
        
        # lets process every image independently for now, for simplicity
        self.done_beams = [[] for _ in range(batch_size)]
        for k in range(batch_size):
            state = self.init_hidden(beam_size)
            tmp_fc_feats = [fc_feats[i][k:k+1].expand(beam_size, fc_feats[i].size(1)) for i,m in enumerate(self.models)]
            tmp_att_feats = [att_feats[i][k:k+1].expand(*((beam_size,)+att_feats[i].size()[1:])).contiguous() for i,m in enumerate(self.models)]
            tmp_p_att_feats = [p_att_feats[i][k:k+1].expand(*((beam_size,)+p_att_feats[i].size()[1:])).contiguous() for i,m in enumerate(self.models)]
            tmp_att_masks = [att_masks[i][k:k+1].expand(*((beam_size,)+att_masks[i].size()[1:])).contiguous() if att_masks[i] is not None else att_masks[i] for i,m in enumerate(self.models)]

            it = fc_feats[0].data.new(beam_size).long().zero_()
            logprobs, state, aleatorics, epistemics = self.get_logprobs_state(it, tmp_fc_feats, tmp_att_feats, tmp_p_att_feats, tmp_att_masks, state)

            self.done_beams[k] = self.beam_search(state, logprobs, aleatorics, epistemics, tmp_fc_feats, tmp_att_feats, tmp_p_att_feats, tmp_att_masks, opt=opt)
            seq[:, k] = self.done_beams[k][0]['seq'] # the first beam has highest cumulative score
            seqLogprobs[:, k] = self.done_beams[k][0]['logps']
            seqAleatorics[:, k] = self.done_beams[k][0]['aleatorics']
            seqEpistemics[:, k] = self.done_beams[k][0]['epistemics']
            
        # return the samples and their log likelihoods
        return seq.transpose(0, 1), seqLogprobs.transpose(0, 1), seqAleatorics.transpose(0, 1), seqEpistemics.transpose(0, 1)

    
    def beam_search(self, init_state, init_logprobs, init_aleatorics, init_epistemics, *args, **kwargs):

        # function computes the similarity score to be augmented
        def add_diversity(beam_seq_table, logprobsf, t, divm, diversity_lambda, bdash):
            local_time = t - divm
            unaug_logprobsf = logprobsf.clone()
            for prev_choice in range(divm):
                prev_decisions = beam_seq_table[prev_choice][local_time]
                for sub_beam in range(bdash):
                    for prev_labels in range(bdash):
                        logprobsf[sub_beam][prev_decisions[prev_labels]] = logprobsf[sub_beam][prev_decisions[prev_labels]] - diversity_lambda
            return unaug_logprobsf

        # does one step of classical beam search
        def beam_step(logprobsf, unaug_logprobsf, aleatorics, epistemics, beam_size, t, beam_seq, beam_seq_logprobs, beam_logprobs_sum, beam_seq_al, beam_seq_ep, state):
            #INPUTS:
            #logprobsf: probabilities augmented after diversity (beam_size, vocab_size)
            #aleatorics: aleatoric uncertainties evaluated at current step (beam_size,)
            #epistemics: epistemic uncertainties evaluated at current step (beam_size,)            
            #beam_size: obvious
            #t        : time instant
            #beam_seq : tensor contanining the beams
            #beam_seq_logprobs: tensor contanining the beam logprobs
            #beam_logprobs_sum: tensor contanining joint logprobs
            #beam_seq_al: tensor containing the aleatorics uncertainties of the candidates
            #beam_seq_ep: tensor containing the epistemics uncertainties of the candidates
            #OUPUTS:
            #beam_seq : tensor containing the word indices of the decoded captions
            #beam_seq_logprobs : log-probability of each decision made, same size as beam_seq
            #beam_logprobs_sum : joint log-probability of each beam

            ys, ix = torch.sort(logprobsf, 1, True)
            candidates = []
            cols = min(beam_size, ys.size(1))
            rows = beam_size
            if t == 0:
                rows = 1
            for c in range(cols): # for each column (word, essentially)
                for q in range(rows): # for each beam expansion
                    #compute logprob of expanding beam q with word in (sorted) position c
                    local_logprob = ys[q, c].item()
                    candidate_logprob = beam_logprobs_sum[q] + local_logprob
                    # local_unaug_logprob = unaug_logprobsf[q,ix[q,c]]
                    candidates.append({'c': ix[q, c], 'q': q, 'p': candidate_logprob, 'r': unaug_logprobsf[q]})
            candidates = sorted(candidates,  key=lambda x: -x['p'])
            
            new_state = [_.clone() for _ in state]
            #beam_seq_prev, beam_seq_logprobs_prev
            if t >= 1:
            #we''ll need these as reference when we fork beams around
                beam_seq_prev = beam_seq[:t].clone()
                beam_seq_logprobs_prev = beam_seq_logprobs[:t].clone()
                beam_seq_al_prev = beam_seq_al[:t].clone()
                beam_seq_ep_prev = beam_seq_ep[:t].clone()                
            for vix in range(beam_size):
                v = candidates[vix]
                #fork beam index q into index vix
                if t >= 1:
                    beam_seq[:t, vix] = beam_seq_prev[:, v['q']]
                    beam_seq_logprobs[:t, vix] = beam_seq_logprobs_prev[:, v['q']]
                    beam_seq_al[:t, vix] = beam_seq_al_prev[:, v['q']]
                    beam_seq_ep[:t, vix] = beam_seq_ep_prev[:, v['q']]                    
                #rearrange recurrent states
                for state_ix in range(len(new_state)):
                #  copy over state in previous beam q to new beam at vix
                    new_state[state_ix][:, vix] = state[state_ix][:, v['q']] # dimension one is time step
                #append new end terminal at the end of this beam
                beam_seq[t, vix] = v['c'] # c'th word is the continuation
                beam_seq_logprobs[t, vix] = v['r'] # the raw logprob here
                beam_logprobs_sum[vix] = v['p'] # the new (sum) logprob along this beam
                beam_seq_al[t, vix] = aleatorics[v['q']]
                beam_seq_ep[t, vix] = epistemics[v['q']]                
            state = new_state
            return beam_seq, beam_seq_logprobs, beam_logprobs_sum, beam_seq_al, beam_seq_ep, state, candidates

        # Start diverse_beam_search
        opt = kwargs['opt']
        temperature = opt.get('temperature', 1) # This should not affect beam search, but will affect dbs
        beam_size = opt.get('beam_size', 10)
        group_size = opt.get('group_size', 1)
        diversity_lambda = opt.get('diversity_lambda', 0.5)
        uncertainty_lambda = opt.get('uncertainty_lambda', 0)
        decoding_constraint = opt.get('decoding_constraint', 0)
        remove_bad_endings = opt.get('remove_bad_endings', 0)
        suppress_UNK = opt.get('suppress_UNK', 0)
        length_penalty = utils.penalty_builder(opt.get('length_penalty', ''))
        bdash = beam_size // group_size # beam per group

        # INITIALIZATIONS
        beam_seq_table = [torch.LongTensor(self.seq_length, bdash).zero_() for _ in range(group_size)]
        beam_seq_logprobs_table = [torch.FloatTensor(self.seq_length, bdash, self.vocab_size + 1).zero_() for _ in range(group_size)]
        beam_logprobs_sum_table = [torch.zeros(bdash) for _ in range(group_size)]
        beam_seq_aleatorics_table = [torch.FloatTensor(self.seq_length, bdash).zero_() for _ in range(group_size)]
        beam_seq_epistemics_table = [torch.FloatTensor(self.seq_length, bdash).zero_() for _ in range(group_size)]        
        done_beams_table = [[] for _ in range(group_size)]
        # state_table = [list(torch.unbind(_)) for _ in torch.stack(init_state).chunk(group_size, 2)]
        state_table = list(zip(*[_.chunk(group_size, 1) for _ in init_state]))  
        # logprobs # logprobs predicted in last time step, shape (beam_size, vocab_size+1)
        logprobs_table = list(init_logprobs.chunk(group_size, 0))
        # [(beam_size,)]
        aleatorics_table = list(init_aleatorics.chunk(group_size, 0))
        epistemics_table = list(init_epistemics.chunk(group_size, 0))        
        # END INITn

        # Chunk elements in the args
        args = list(args)
        if self.__class__.__name__ == 'AttEnsemble':
            args = [[_.chunk(group_size) if _ is not None else [None]*group_size for _ in args_] for args_ in args] # arg_name, model_name, group_name
            args = [[[args[j][i][k] for i in range(len(self.models))] for j in range(len(args))] for k in range(group_size)] # group_name, arg_name, model_name
        else:
            args = [_.chunk(group_size) if _ is not None else [None]*group_size for _ in args]
            args = [[args[i][j] for i in range(len(args))] for j in range(group_size)]

        for t in range(self.seq_length + group_size - 1):
            for divm in range(group_size): 
                if t >= divm and t <= self.seq_length + divm - 1:
                    # add diversity
                    logprobsf = logprobs_table[divm].float()
                    # suppress previous word
                    if decoding_constraint and t-divm > 0:
                        logprobsf.scatter_(1, beam_seq_table[divm][t-divm-1].unsqueeze(1).cuda(), float('-inf'))
                    if remove_bad_endings and t-divm > 0:
                        logprobsf[torch.from_numpy(np.isin(beam_seq_table[divm][t-divm-1].cpu().numpy(), self.bad_endings_ix)), 0] = float('-inf')
                    # suppress UNK tokens in the decoding
                    if suppress_UNK and hasattr(self, 'vocab') and self.vocab[str(logprobsf.size(1)-1)] == 'UNK':
                        logprobsf[:,logprobsf.size(1)-1] = logprobsf[:, logprobsf.size(1)-1] - 1000  
                    # diversity is added here
                    # the function directly modifies the logprobsf values and hence, we need to return
                    # the unaugmented ones for sorting the candidates in the end. # for historical
                    # reasons :-)
                    unaug_logprobsf = add_diversity(beam_seq_table, logprobsf, t, divm, diversity_lambda, bdash)
                    
                    # get current uncertainties
                    aleatorics = aleatorics_table[divm]
                    epistemics = epistemics_table[divm]

                    # add uncertainty
                    logprobsf = logprobsf - uncertainty_lambda * epistemics.unsqueeze(-1)
                    
                    # infer new beams
                    beam_seq_table[divm],\
                    beam_seq_logprobs_table[divm],\
                    beam_logprobs_sum_table[divm],\
                    beam_seq_aleatorics_table[divm],\
                    beam_seq_epistemics_table[divm],\
                    state_table[divm],\
                    candidates_divm = beam_step(logprobsf,
                                                unaug_logprobsf,
                                                aleatorics,
                                                epistemics,
                                                bdash,
                                                t-divm,
                                                beam_seq_table[divm],
                                                beam_seq_logprobs_table[divm],
                                                beam_logprobs_sum_table[divm],
                                                beam_seq_aleatorics_table[divm],
                                                beam_seq_epistemics_table[divm],
                                                state_table[divm])

                    # if time's up... or if end token is reached then copy beams
                    for vix in range(bdash):
                        if beam_seq_table[divm][t-divm,vix] == 0 or t == self.seq_length + divm - 1:
                            final_beam = {
                                'seq': beam_seq_table[divm][:, vix].clone(),
                                'aleatorics': beam_seq_aleatorics_table[divm][:, vix].clone(),
                                'epistemics': beam_seq_epistemics_table[divm][:, vix].clone(),                                
                                'logps': beam_seq_logprobs_table[divm][:, vix].clone(),
                                'unaug_p': beam_seq_logprobs_table[divm][:, vix].sum().item(),
                                'p': beam_logprobs_sum_table[divm][vix].item()
                            }
                            final_beam['p'] = length_penalty(t-divm+1, final_beam['p'])
                            done_beams_table[divm].append(final_beam)
                            # don't continue beams from finished sequences
                            beam_logprobs_sum_table[divm][vix] = -1000

                    # move the current group one step forward in time
                    it = beam_seq_table[divm][t-divm]
                    logprobs_table[divm], state_table[divm], aleatorics_table[divm], epistemics_table[divm] = self.get_logprobs_state(it.cuda(), *(args[divm] + [state_table[divm]]))
                    logprobs_table[divm] = F.log_softmax(logprobs_table[divm] / temperature, dim=-1)

        # all beams are sorted by their log-probabilities
        done_beams_table = [sorted(done_beams_table[i], key=lambda x: -x['p'])[:bdash] for i in range(group_size)]
        done_beams = sum(done_beams_table, [])
        return done_beams
