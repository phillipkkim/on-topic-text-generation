import torch
import train_utils
import os

BERT_HIDDEN_DIM = 768
GPT_HIDDEN_DIM = 768
VOCAB_SIZE = 50257
GEN_CONTEXT_LEN = 127

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BERTGPT2LMHead(torch.nn.Module):
    def __init__(self):
        super(BERTGPT2LMHead, self).__init__()
        self.fc = torch.nn.Linear(
            BERT_HIDDEN_DIM+GPT_HIDDEN_DIM,
            VOCAB_SIZE, bias=False)

    def forward(self, bert_hidden, gpt_hidden):
        bert_batch_size, bert_hidden_dim = bert_hidden.shape
        gpt_batch_size, gpt_word_count, gpt_hidden_dim = gpt_hidden.shape

        assert bert_batch_size == gpt_batch_size
        assert bert_hidden_dim == BERT_HIDDEN_DIM
        assert gpt_hidden_dim == GPT_HIDDEN_DIM

        repeated_bert_hidden = bert_hidden.repeat_interleave(
            gpt_word_count, dim=0)
        reshaped_gpt_hidden = gpt_hidden.reshape(
            gpt_batch_size * gpt_word_count, gpt_hidden_dim)
        if torch.cuda.is_available():
            repeated_bert_hidden = repeated_bert_hidden.cuda()
            reshaped_gpt_hidden = reshaped_gpt_hidden.cuda()
        # (batchsize * gpt_word_count, hidden_dim)

        cat = torch.cat([repeated_bert_hidden, reshaped_gpt_hidden], dim=1)
        # (batchsize * gpt_word_count, bert_hidden_dim + gpt_hidden_dim)

        assert cat.shape[1] == BERT_HIDDEN_DIM + GPT_HIDDEN_DIM
        logit = self.fc(cat)
        return logit


class BERTGPT2LMHeadModel(torch.nn.Module):
    def __init__(self, bert_tokenizer, bert_model, gpt_encoder, gpt_model, pool_method):
        super(BERTGPT2LMHeadModel, self).__init__()
        assert pool_method in ['max', 'avg']
        self.pool_method = pool_method

        self.bert_tokenizer = bert_tokenizer
        self.bert_model = bert_model
        self.gpt_encoder = gpt_encoder
        self.gpt_transformer = gpt_model.transformer

        self.lm_head = BERTGPT2LMHead()
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
            # forward each story chunk
            gpt_hidden = self.gpt_transformer(story_gpt_tokens)[0]

        # the last prediction has no target label, remove it
        gpt_hidden = gpt_hidden[:, :-1, :]
        dau_logits = self.lm_head(bert_hidden_pooled, gpt_hidden)
        return dau_logits

    def next_word_logit(self, bert_hidden_pooled, context_gpt_tokens):
        '''
        @return a logit vector of (vocabsize,), 
        - this function doesnt support batch
        - since bert forwarding is very expensive, for each passage, bert forward
        - once and use that to call next_word_logit repeatedly with different context
        - returned vector is not softmax'd
        '''
        # latest_context = torch.LongTensor(
        #     [[context_gpt_tokens]]
        # )
        latest_context = context_gpt_tokens.unsqueeze(0).unsqueeze(0)
        latest_context = latest_context.to(DEVICE)
        gpt_hidden = self.gpt_transformer(latest_context)[0]
        gpt_hidden = gpt_hidden[:, -1, :]  # keep only the last one
        logits = self.lm_head(bert_hidden_pooled, gpt_hidden)
        return logits

    def generate_slow(self, highlight_str, context_gpt_tokens,
                      gen_length=100, use_sample=False, temperature=1):
        bert_hidden_pooled = self.bert_fw_pooled([highlight_str])

        present_tokens = context_gpt_tokens.clone()
        present_tokens = present_tokens.to(DEVICE)
        with torch.no_grad():
            for _ in range(gen_length):
                logits = self.next_word_logit(
                    bert_hidden_pooled, present_tokens
                )
                logits = logits[-1, :] / temperature
                probs = torch.torch.nn.functional.softmax(logits, dim=-1)

                if use_sample:
                    generated_token = torch.multinomial(probs, num_samples=1)
                else:
                    generated_token = torch.argmax(probs)
                present_tokens = torch.cat(
                    [present_tokens, generated_token.unsqueeze(0)])

        return present_tokens.tolist()

    def generate(self, highlight_str, context_gpt_tokens,
                 gen_length=100, use_sample=False, temperature=1):
        bert_hidden_pooled = self.bert_fw_pooled([highlight_str])
        output_tokens = context_gpt_tokens.clone().to(DEVICE)
        prev = output_tokens

        transformer_past = None
        with torch.no_grad():
            for _ in range(gen_length):
                latest_token = prev.unsqueeze(0).to(DEVICE)
                gpt_hidden, transformer_past = self.gpt_transformer(
                    latest_token, past=transformer_past
                )
                logits = self.lm_head(bert_hidden_pooled, gpt_hidden)
                logits = logits[-1, :] / temperature
                probs = torch.torch.nn.functional.softmax(logits, dim=-1)

                if use_sample:
                    generated_token = torch.multinomial(probs, num_samples=1)
                else:
                    generated_token = torch.argmax(probs)
                output_tokens = torch.cat(
                    [output_tokens, generated_token.unsqueeze(0)])
                prev = generated_token.unsqueeze(0)

        return output_tokens.tolist()


def get_bert_gpt_lm_model(pool_method):
    '''
    @param pool_method: "max" or "avg"
    @return (head_model, bert_tokenizer, bert_model, gpt_encoder, gpt_model)
    '''
    bert_tokenizer, bert_model = train_utils.get_bert()
    gpt_encoder, gpt_model = train_utils.get_gpt()

    head_model = BERTGPT2LMHeadModel(
        bert_tokenizer, bert_model, gpt_encoder, gpt_model, pool_method
    ).to(DEVICE)

    script_dir = os.path.dirname(os.path.realpath(__file__))
    weight_filepath = os.path.join(
        script_dir, '..', 'saved_models', f'bert_gpt_lm_{pool_method}', 'best.pth')
    print(f'[loading bert_gpt_lm_{pool_method}/best.pth]')
    head_model.lm_head.load_state_dict(
        torch.load(
            weight_filepath, map_location=DEVICE
        )
    )
    return (head_model, bert_tokenizer, bert_model, gpt_encoder, gpt_model)


if __name__ == "__main__":
    head_model, bert_tokenizer, bert_model, gpt_encoder, gpt_model = \
        get_bert_gpt_lm_model('avg')

    highlight = 'I am drinking whiskey at 1 am in the morning while writing this code.'
    context = 'I am'
    context_gpt_tokens = torch.LongTensor(gpt_encoder.encode(context))

    generated_gpt_tokens = head_model.generate(
        highlight, context_gpt_tokens,
        gen_length=30, use_sample=True, temperature=1
    )

    print(gpt_encoder.decode(generated_gpt_tokens))
