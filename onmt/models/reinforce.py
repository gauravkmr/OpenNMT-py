import torch
import torch.nn as nn
import onmt

class RougeScorer:
    def __init__(self):
        pass
        '''
        import rouge as R
        self.rouge = R.Rouge(stats=["f"], metrics=[
                             "rouge-1", "rouge-2", "rouge-l"])
        '''

    def _score(self, hyps, refs):
        scores = self.rouge.get_scores(hyps, refs)
        # NOTE: here we use score = r1 * r2 * rl
        #       I'm not sure how relevant it is
        metric_weight = {"rouge-1": 0, "rouge-2": 0, "rouge-l": 1}

        scores = [sum([seq[metric]['f'] * metric_weight[metric]
                       for metric in seq.keys()])
                  for seq in scores]
        return scores

    def score(self, sample_pred, greedy_pred, tgt):
        """
            sample_pred: LongTensor [bs x len]
            greedy_pred: LongTensor [bs x len]
            tgt: LongTensor [bs x len]
        """
        def tens2sen(t):
            sentences = []
            for s in t:
                sentence = []
                for wt in s:
                    word = wt.data[0]
                    if word in [0, 3]:
                        break
                    sentence += [str(word)]
                if len(sentence) == 0:
                    # NOTE just a trick not to score empty sentence
                    #      this has not consequence
                    sentence = ["0", "0", "0"]
                sentences += [" ".join(sentence)]
            return sentences

        s_hyps = tens2sen(sample_pred)
        g_hyps = tens2sen(greedy_pred)
        refs = tens2sen(tgt)

        '''
        sample_scores = self._score(s_hyps, refs)
        greedy_scores = self._score(g_hyps, refs)
        ts = torch.Tensor(sample_scores)
        gs = torch.Tensor(greedy_scores)

        return (gs - ts)
        '''

        #Length based score
        sample_lengths = []
        reference_lengths = []
        for sample, ref in zip(s_hyps, refs):
            sample_lengths.append(len(sample))
            reference_lengths.append(len(ref))

        ts = torch.Tensor(sample_lengths)
        gs = torch.Tensor(reference_lengths)

        return 1 - abs(gs-ts)/gs


class ReinforceModel(onmt.models.NMTModel):
    def __init__(self, encoder, decoder, gamma=0.9984):
        """
        Args:
            encoder:
            decoder:
            multigpu: not sure why its here
            gamma: in [0;1] weight between ML and RL
                   loss = gamma * loss_rl + (1 - gamma) * loss_ml
                   (see Paulus et al 2017, sect. 3.3)
        """
        super(ReinforceModel, self).__init__(encoder, decoder)
        self.rouge = RougeScorer()
        self.gamma = gamma
        self.model_type = "Reinforce"

    def forward(self, src, tgt, src_lengths, batch, loss_compute,
                dec_state=None):
        """
        Args:
            src:
            tgt:
            dec_state: A decoder state object
        """
        n_feats = tgt.size(2)
        assert n_feats == 1, "Reinforced model does not handle features"
        tgt = tgt.squeeze(2)
        enc_hidden, enc_out, lengths = self.encoder(src, src_lengths)

        enc_state = self.decoder.init_decoder_state(src=None,
                                                    enc_hidden=enc_hidden,
                                                    context=enc_out)
        state = enc_state if dec_state is None else dec_state

        ml_loss, stats, hidden, _, _, ml_preds = self.decoder(tgt[:-1],
                                                              src,
                                                              enc_out,
                                                              state,
                                                              batch,
                                                              loss_compute,
                                                              tgt=tgt[1:])

        if self.gamma > 0:
            rl_loss, stats2, hidden2, _, _, rl_preds = \
                self.decoder(tgt[:-1],
                             src,
                             enc_out,
                             state,
                             batch,
                             loss_compute,
                             tgt=tgt[1:],
                             sampling=True)

            sample_preds = torch.stack(rl_preds, 1)
            greedy_preds = torch.stack(ml_preds, 1)
            metric = self.rouge.score(sample_preds, greedy_preds, tgt[1:].t())
            metric = torch.autograd.Variable(metric).cuda()

            rl_loss = (rl_loss * metric).sum()
            loss = (self.gamma * rl_loss) - ((1 - self.gamma * ml_loss))
        else:
            loss = ml_loss
        return loss, stats, state