import torch
import train_utils
import os


BERT_HIDDEN_DIM = 768
GPT_HIDDEN_DIM = 768
VOCAB_SIZE = 50257
GEN_CONTEXT_LEN = 127

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BERTLMHead(torch.nn.Module):
    def __init__(self):
        super(BERTLMHead, self).__init__()
        self.fc = torch.nn.Linear(
            BERT_HIDDEN_DIM,
            VOCAB_SIZE, bias=False)

    def forward(self, bert_hidden_pooled, gpt_logits):
        bert_batch_size, bert_hidden_dim = bert_hidden_pooled.shape
        gpt_batch_size, gpt_word_count, gpt_vocab_size = gpt_logits.shape
        assert bert_hidden_dim == BERT_HIDDEN_DIM
        assert gpt_batch_size == bert_batch_size
        assert gpt_vocab_size == VOCAB_SIZE

        bert_hidden_pooled = bert_hidden_pooled.to(DEVICE)
        # (batchsize, bert_hidden_dim)
        bert_lm_logit = self.fc(bert_hidden_pooled)
        # (batchsize, vocabsize)
        bert_lm_logit = bert_lm_logit.unsqueeze(dim=1)
        # (batchsize, 1, vocabsize)

        logit = bert_lm_logit + gpt_logits
        return logit


class BERTGPT2DAUModel(torch.nn.Module):
    def __init__(self, bert_tokenizer, bert_model, gpt_encoder, gpt_model, pool_method):
        super(BERTGPT2DAUModel, self).__init__()
        assert pool_method in ['max', 'avg']
        self.pool_method = pool_method

        self.bert_tokenizer = bert_tokenizer
        self.bert_model = bert_model
        self.gpt_encoder = gpt_encoder
        self.gpt_model = gpt_model

        self.lm_head = BERTLMHead()
        self.lm_head.to(DEVICE)

    def bert_fw_pooled(self, highlight_strs):
        # tokenize, forward, avgpool each highlight
        bert_tokenized = torch.LongTensor([self.bert_tokenizer.encode(
            highlight, add_special_tokens=False, pad_to_max_length=True
        ) for highlight in highlight_strs])
        bert_tokenized = bert_tokenized.to(DEVICE)

        bert_hidden = self.bert_model(bert_tokenized)[0]
        if self.pool_method == 'avg':
            bert_hidden_pooled = bert_hidden.mean(dim=1)
        elif self.pool_method == 'max':
            bert_hidden_pooled = torch.max(bert_hidden, dim=1).values
        else:
            raise AssertionError(f'wrong pool method [{self.pool_method}]')
        return bert_hidden_pooled

    def train_forward(self, highlight_strs, story_gpt_tokens):
        story_gpt_tokens = story_gpt_tokens.to(DEVICE)
        with torch.no_grad():
            bert_hidden_pooled = self.bert_fw_pooled(highlight_strs)
            # (batchsize, bert_hidden_dim)
            # forward each story chunk
            gpt_logit = self.gpt_model(story_gpt_tokens)[0]
            # (batchsize, num_word, vocabsize)

        # the last prediction has no target label, remove it
        gpt_logit = gpt_logit[:, :-1, :]

        bert_adjusted_logits = self.lm_head(bert_hidden_pooled, gpt_logit)
        # (batchsize, GEN_CONTEXT_LEN, vocabsize)
        bert_adjusted_logits = bert_adjusted_logits.reshape(-1, VOCAB_SIZE)
        # (batchsize*GEN_CONTEXT_LEN, vocabsize)
        return bert_adjusted_logits

    def next_word_logit(self, bert_hidden_pooled, context_gpt_tokens):
        '''
        @return a logit vector of (vocabsize,), 
        - this function doesnt support batch
        - since bert forwarding is very expensive, for each passage, bert forward
        - once and use that to call next_word_logit repeatedly with different context
        - returned vector is not softmax'd
        '''
        context = torch.LongTensor([context_gpt_tokens]).to(DEVICE)
        gpt_logit = self.gpt_model(context)[0]
        gpt_logit = gpt_logit[:, -1:, :]  # just keep the last vector
        # (1, 1, vocabsize)
        logits = self.lm_head(bert_hidden_pooled, gpt_logit)
        # (1, 1, vocabsize)
        logits = logits[0, 0, :]
        return logits

    def generate(self, highlight_str, context_gpt_tokens,
                 gen_length=100, use_sample=False, temperature=1):
        bert_hidden_pooled = self.bert_fw_pooled([highlight_str])
        output_tokens = context_gpt_tokens.clone().to(DEVICE)
        prev = output_tokens
        past = None

        with torch.no_grad():
            for _ in range(gen_length):
                latest_token = prev.unsqueeze(0)
                gpt_logits, past = self.gpt_model(
                    latest_token, past=past
                )
                gpt_logits = gpt_logits[:, -1:, :]
                logits = self.lm_head(bert_hidden_pooled, gpt_logits)
                logits = logits / temperature
                probs = torch.torch.nn.functional.softmax(logits, dim=-1)

                if use_sample:
                    generated_token = torch.multinomial(probs, num_samples=1)
                else:
                    generated_token = torch.argmax(probs)
                prev = generated_token.unsqueeze(0)
                output_tokens = torch.cat(
                    [output_tokens, generated_token.unsqueeze(0)]
                )

        return output_tokens.tolist()


def get_bert_gpt_dau_model(pool_method):
    '''
    @param pool_method: "max" or "avg"
    @return (head_model, bert_tokenizer, bert_model, gpt_encoder, gpt_model)
    '''
    bert_tokenizer, bert_model = train_utils.get_bert()
    gpt_encoder, gpt_model = train_utils.get_gpt()

    head_model = BERTGPT2DAUModel(
        bert_tokenizer, bert_model, gpt_encoder, gpt_model, pool_method
    ).to(DEVICE)

    script_dir = os.path.dirname(os.path.realpath(__file__))
    weight_filepath = os.path.join(
        script_dir, '..', 'saved_models', f'bert_gpt_dau_{pool_method}', 'best.pth')
    print(f'[loading bert_gpt_dau_{pool_method}/best.pth]')
    head_model.lm_head.load_state_dict(
        torch.load(
            weight_filepath, map_location=DEVICE
        )
    )
    return (head_model, bert_tokenizer, bert_model, gpt_encoder, gpt_model)


if __name__ == "__main__":
    head_model, bert_tokenizer, bert_model, gpt_encoder, gpt_model = \
        get_bert_gpt_dau_model('avg')

    highlight = 'I am drinking whiskey at 1 am in the morning while writing this code.'
    context = 'I am'
    context_gpt_tokens = torch.LongTensor(gpt_encoder.encode(context))

    generated_gpt_tokens = head_model.generate(
        highlight, context_gpt_tokens,
        gen_length=30, use_sample=True, temperature=1
    )

    print(gpt_encoder.decode(generated_gpt_tokens))
