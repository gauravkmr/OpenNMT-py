import torch
import torch.nn as nn
import numpy as np
import onmt
from onmt.modules.intra_attention import IntraAttention
from onmt.modules.stacked_LSTM import StackedLSTM
from onmt.models.model import RNNDecoderState

from onmt.utils.statistics import Statistics

class ReinforceDecoder(nn.Module):
    def __init__(self, opt, embeddings, dec_attn=True,
                 exp_bias_reduction=0.25, bidirectional_encoder=False):
        """
        Implementation of a decoder following Paulus et al., (2017)
        By default, we refer to this paper when mentioning a section


        Args:
            opt:
            embeddings: target embeddings
            dec_attn: boolean, use decoder intra attention or not (sect. 2.2)
            exp_bias_reduction: float in [0, 1], exposure bias reduction by
                                feeding predicted token with a given
                                probability as mentionned in sect. 6.1
            bidirectional_encoder
        """
        super(ReinforceDecoder, self).__init__()
        self.opt = opt
        self.embeddings = embeddings

        #w_emb = embeddings.weight
        #self.tgt_vocab_size, self.input_size = w_emb.size()
        self.tgt_vocab_size, self.input_size = embeddings.word_lut.weight.size()

        self.dim = opt.rnn_size

        # TODO use parameter instead of hardcoding nlayer
        self.rnn = StackedLSTM(1, self.input_size, self.dim, opt.dropout)

        self.enc_attn = IntraAttention(self.dim, temporal=True)

        self.dec_attn = None
        if dec_attn:
            self.dec_attn = IntraAttention(self.dim)

        self.pad_id = embeddings.word_padding_idx
        self.exp_bias_reduction = exp_bias_reduction

        # For compatibility reasons, TODO refactor
        self.hidden_size = self.dim
        self.decoder_type = "reinforce"
        self.bidirectional_encoder = bidirectional_encoder

    def mkvar(self, tensor, requires_grad=False):
        v = torch.autograd.Variable(tensor, requires_grad=requires_grad)
        # if use_gpu(self.opt):
        #     v = v.cuda()
        return v

    def _fix_enc_hidden(self, h):
        """
        The encoder hidden is  (layers*directions) x batch x dim.
        We need to convert it to layers x batch x (directions*dim).
        """
        if self.bidirectional_encoder:
            h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        return h

    def init_decoder_state(self, src, context, enc_hidden):
        """
        Args:
            src: For compatibility reasons.......

        """
        if isinstance(enc_hidden, tuple):  # GRU
            return RNNDecoderState(
                self.hidden_size,
                tuple([self._fix_enc_hidden(enc_hidden[i])
                       for i in range(len(enc_hidden))]))
        else:  # LSTM
            return RNNDecoderState(
                self.hidden_size, self._fix_enc_hidden(enc_hidden))

    def forward(self, inputs, src, h_e, state, batch,
                loss_compute=None, tgt=None, generator=None,
                hd_history=None, attn_hist=None, ret_hists=False,
                sampling=False):
        """
        Args:
            inputs (LongTensor): [tgt_len x bs]
            src (LongTensor): [src_len x bs x 1]
            h_e (FloatTensor): [src_len x bs x dim]
            state: onmt.Models.DecoderState
            tgt (LongTensor): [tgt_len x bs]

        Returns:
            stats:
            state:
            scores:
            attns:
            hd_history: memory for decoder intra attention
            attn_hist: memory for temporal attention

        """
        dim = self.dim
        src_len, bs, _ = list(src.size())
        input_size, _bs = list(inputs.size())
        assert bs == _bs, "bs does not match %d, %d" % (bs, _bs)

        if self.training:
            assert tgt is not None
        if tgt is not None:
            assert loss_compute is not None
            if generator is not None:
                print("[WARNING] Parameter 'generator' should not "
                      + "be set at training time")
        else:
            assert generator is not None

        # src as [bs x src_len]
        src = src.transpose(0, 1).squeeze(2).contiguous()

        stats = Statistics()
        hidden = state.hidden
        loss = None
        scores, attns, dec_attns, outputs = [], [], [], []
        preds = []
        inputs_t = inputs[0, :]

        for t in range(input_size):
            # Embedding & intra-temporal attention on source
            emb_t = self.embeddings(inputs_t.view(1, -1, 1)).squeeze(0)

            hd_t, hidden = self.rnn(emb_t, hidden)

            c_e, alpha_e, attn_hist = self.enc_attn(hd_t,
                                                    h_e,
                                                    attn_history=attn_hist)

            # Intra-decoder Attention
            if self.dec_attn is None or hd_history is None:
                # no decoder intra attn at first step
                cd_t = self.mkvar(torch.zeros([bs, dim]))
                alpha_d = cd_t
                hd_history = hd_t.unsqueeze(0)
            else:
                cd_t, alpha_d = self.dec_attn(hd_t, hd_history)
                hd_history = torch.cat([hd_history, hd_t.unsqueeze(0)], dim=0)

            # Prediction - Computing Loss
            sampling = True

            if tgt is not None:
                output = torch.cat([hd_t, c_e, cd_t], dim=1)
                if sampling:
                    prediction_type = "sample"
                    # TODO here 0 and 3 are hardcoded
                    continue_gen = (inputs_t.ne(3) * inputs_t.ne(0))
                    tgt_t = continue_gen.long()
                    align = torch.autograd.Variable(
                        torch.zeros([bs]).long().cuda())
                else:
                    tgt_t = tgt[t, :]
                    prediction_type = "greedy"
                    align = batch.alignment[t, :].contiguous()

                loss_t, pred_t, stats_t = loss_compute.compute_loss(
                    batch,
                    output,
                    tgt_t,
                    copy_attn=alpha_e,
                    align=align,
                    src=src,
                    prediction_type=prediction_type)
                outputs += [output]
                attns += [alpha_e]
                preds += [pred_t]

                stats.update(stats_t)
                loss = loss + loss_t if loss is not None else loss_t

            else:
                # In translation case we just want scores
                # prediction itself will be done with beam search
                output = torch.cat([hd_t, c_e, cd_t], dim=1)
                scores_t = generator(output, alpha_e, batch.src_map)
                scores += [scores_t]
                attns += [alpha_e]
                dec_attns += [alpha_d]

            if sampling:
                # the sampling mode correspond to generating y^s_t as
                # described in sect. 3.2
                inputs_t = preds[-1]

            elif t < input_size - 1:
                if self.training:
                    # Exposure bias reduction by feeding predicted token
                    # with a 0.25 probability as mentionned in sect. 6.1

                    _pred_t = preds[-1].clone()
                    _pred_t = loss_compute.remove_oov(_pred_t)
                    exposure_mask = self.mkvar(
                        torch.rand([bs]).lt(self.exp_bias_reduction).long())
                    inputs_t = exposure_mask * _pred_t.long()
                    inputs_t += (1 - exposure_mask.float()).long() \
                        * inputs[t+1, :]

                else:
                    inputs_t = inputs[t+1, :]

        state.update_state(hidden, None, None)
        if not ret_hists:
            return loss, stats, state, scores, attns, preds
        return stats, state, scores, attns, hd_history, attn_hist