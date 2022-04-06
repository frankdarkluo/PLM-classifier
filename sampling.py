import torch
import numpy as np
import math
import torch.nn as nn
from transformers import pipeline,RobertaTokenizer, RobertaForMaskedLM,GPTNeoForCausalLM,GPT2Tokenizer,\
    GPT2LMHeadModel,GPTJForCausalLM
from utils import predict_next_word,pipe
from utils import pytorch_cos_sim
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipeline_classifier = pipeline("sentiment-analysis", model="siebert/sentiment-roberta-large-english",
                               framework="pt",device=torch.cuda.current_device())
#pipeline_classifier = pipeline("sentiment-analysis")

class SimulatedAnnealing(nn.Module):
    def __init__(self, option,editor,t_init, C, fluency_weight, keyword_weight, sent_weight, style_weight, max_steps):
        super(SimulatedAnnealing,self).__init__()
        self.option=option
        self.editor = editor
        self.t_init = t_init
        self.C = C
        self.fluency_weight = fluency_weight # 3
        self.keyword_weight = keyword_weight # 1
        self.sent_weight = sent_weight
        self.style_weight=style_weight
        self.max_steps = max_steps
        self.stride=1024
        # if 'gpt-neo' in self.option.class_name:
        #     self.plm = GPTNeoForCausalLM.from_pretrained(self.option.class_name)
        # elif 'gpt-j' in self.option.class_name:
        #     self.plm = GPTJForCausalLM.from_pretrained(self.option.class_name)
        # self.plm.eval()
        # self.plm.to(device)
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.option.class_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_len = self.option.max_len
        self.model=GPT2LMHeadModel.from_pretrained('gpt2').to(device)
        self.ppl_max_len=self.model.config.n_positions

    def style_scorer(self,ref_news):

        prefix = 'the sentiment of the text "'
        postfix = '" is'
        prob_new_probs=[]
        for idx, sent in enumerate(ref_news):

            text=ref_news[idx]
            if self.option.style_mode == 'plm':
                # TODO: Define the prompt and the PLM classification score!
                input_candidate_text = prefix + text + postfix
                classifi_tokens, style_prob = predict_next_word(self.plm, self.tokenizer, input_candidate_text,
                                                                k=len(self.tokenizer), direction=self.option.direction)
                if self.option.early_stop==True:
                    res_cand = pipeline_classifier(text)
                    style_label=res_cand[0]['label'].lower()
                else:
                    style_label=None
                prob_new_probs.append(math.pow(style_prob, self.style_weight))

            elif self.option.style_mode == 'pipeline':

                res_cand = pipeline_classifier(text)
                style_prob,style_label=pipe(res_cand,self.option.direction)
                prob_new_probs.append(math.pow(style_prob, self.style_weight))

        prob_new_prob=torch.tensor(prob_new_probs).cuda()

        return prob_new_prob,style_label

    def fluency_scorer(self,ref_news): #Refer to https://huggingface.co/docs/transformers/perplexity
        _, gpt_tokens=self.editor.plm_token(ref_news)
        input_ids = torch.tensor([self.tokenizer.convert_tokens_to_ids(gpt_token) for gpt_token in gpt_tokens]).to(
            device)

        nlls = []
        for i in range(0, input_ids.size(1), self.stride):
            begin_loc = max(i + self.stride - self.ppl_max_len, 0)
            end_loc = min(i + self.stride, input_ids.size(1))
            trg_len = end_loc - i  # may be different from stride on last loop
            input_ids =input_ids[:, begin_loc:end_loc].to(device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = self.model(input_ids, labels=target_ids)
                neg_log_likelihood = outputs[0] * trg_len
            nlls.append(neg_log_likelihood)

        ppl = torch.exp(torch.stack(nlls).sum() / end_loc)

        return 1/ppl

    def keyword_sim(self,ref_new_embeds,ref_old_embeds,state_vec=None):
        e = 1e-5
        emb1 = ref_new_embeds.permute(0, 2, 1)
        emb2 = ref_old_embeds
        emb_mat = torch.bmm(emb2, emb1)
        weight2 = torch.tensor(state_vec[0][:emb2.shape[1]], dtype=torch.bool)
        norm2 = 1 / (torch.norm(emb2, p=2, dim=2) + e)  # K,8,8
        norm1 = 1 / (torch.norm(emb1, p=2, dim=1) + e)  # K,7,7
        diag_norm2 = torch.diag_embed(norm2)  # K,15,15
        diag_norm1 = torch.diag_embed(norm1)
        sim_mat = torch.bmm(torch.bmm(diag_norm2, emb_mat), diag_norm1)  # K,8,7
        sim_vec, _ = torch.max(sim_mat, dim=2)  # K,8
        kw_similarity, _ = torch.min(sim_vec[:, weight2], dim=1)

        return kw_similarity

    def semantic_scorer(self,ref_news, ref_olds,state_vec=None):

        ref_new_embeds, mean_new_embeds = self.editor.get_contextual_word_embeddings(ref_news)
        ref_old_embeds, mean_old_embeds = self.editor.get_contextual_word_embeddings(ref_olds)

        #-----keyword-level sim------
        if self.option.semantic_mode=='kw':
            kw_sim=self.keyword_sim(ref_new_embeds,ref_old_embeds,state_vec)
            similarity = kw_sim.pow(self.keyword_weight)

        #-----sent-level sim------
        elif self.option.semantic_mode=='sent':
            sent_sim=pytorch_cos_sim(mean_new_embeds, mean_old_embeds)
            similarity=sent_sim.pow(self.sent_weight)

        # -----kw-sent level sim------
        elif self.option.semantic_mode=='kw-sent':
            kw_sim=self.keyword_sim(ref_new_embeds,ref_old_embeds,state_vec)
            sent_sim= pytorch_cos_sim(mean_new_embeds, mean_old_embeds)
            similarity = kw_sim.pow(self.keyword_weight)* sent_sim.pow(self.sent_weight)

        return similarity

    def scorer(self, input_news,ref_oris,state_vec=None):
        semantic_scores = self.semantic_scorer(input_news,ref_oris,state_vec)
        fluency_scores = self.fluency_scorer(input_news)
        style_score,style_label=self.style_scorer(input_news)
        total_scores = fluency_scores.pow(self.fluency_weight) * semantic_scores \
                       * style_score

        return total_scores,style_score, style_label

    def choose_action(self,c):
        r = np.random.random()
        c = np.array(c)
        for i in range(1, len(c)):
            c[i] = c[i] + c[i - 1]
        for i in range(len(c)):
            if c[i] >= r:
                return i

    def acceptance_prob(self, input_news, input_olds,ref_oris, T,ops,state_vec=None):
        ref_old_score,old_style_score, _ = self.scorer(input_olds,ref_oris,state_vec)
        ref_old_score=ref_old_score.squeeze()

        ref_new_scores=torch.tensor([self.scorer([ref_hat], ref_oris,state_vec)[0].squeeze() for ref_hat in input_news]).cuda()
        new_style_score=[self.scorer([ref_hat],ref_oris,state_vec)[1].squeeze() for ref_hat in input_news]
        new_style_label=[self.scorer([ref_hat],ref_oris,state_vec)[2] for ref_hat in input_news]
        ref_new_score_index=torch.argmax(ref_new_scores)
        ref_new_score=torch.max(ref_new_scores)

        # seq_len=[]
        # for line in ref_olds:
        #     linewords = [token.replace('Ġ', '') for token in self.tokenizer.tokenize(line.strip())]
        #     seq_len.append(len(linewords))

        # seq_len=[len(ref_old.strip().split()) for ref_old in ref_olds]
        #
        # V_old = np.log(np.maximum(np.power(ref_old_score.cpu().detach().numpy(), 1.0 / seq_len[0]), 1e-200))
        # if ops==0: #replace
        #     V_new = np.log(np.maximum(np.power(ref_new_score.cpu().detach().numpy(), 1.0 / seq_len[0]), 1e-200))
        # elif ops==1: #insert
        #     V_new = np.log(np.maximum(np.power(ref_new_score.cpu().detach().numpy(), 1.0 / (seq_len[0]+1)), 1e-200))
        # else: #delete
        #     V_new = np.log(np.maximum(np.power(ref_new_score.cpu().detach().numpy(), 1.0 / (seq_len[0] - 1)), 1e-200))

        #accept_hat = np.minimum(1, np.exp(np.minimum((V_new - V_old) / T, 100)))

        if ref_new_score-ref_old_score>0:
            accept_hat = [1]
        else:
            accept_hat=[0]

        #accept_hat=min(1,min((ref_new_score-ref_old_score)/T,20))
        #accept_hat = torch.exp((ref_new_score)-ref_new_score/ T)
        return accept_hat,ref_new_score_index,ref_old_score,ref_new_score,old_style_score,new_style_score,new_style_label
