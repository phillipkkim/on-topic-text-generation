'''
    code by TaeHwan Jung(@graykode)
    Original Paper and repository here : https://github.com/openai/gpt-2
    GPT2 Pytorch Model : https://github.com/huggingface/pytorch-pretrained-BERT
'''
import torch
import torch.nn.functional as F
from tqdm import trange

def top_k_logits(logits, k):
    if k == 0:
        return logits
    values, _ = torch.topk(logits, k)
    min_values = values[:, -1].unsqueeze(1).expand_as(logits)
    return torch.where(logits < min_values, torch.ones_like(logits, dtype=logits.dtype) * -1e10, logits)

def sample_sequence(model, length, start_token=None, context = None, highlights=None, gen_batch_size = 1, temperature=1, top_k=0, device='cuda', sample=True):
    if context == None:
        context = torch.full((highlights.shape[0], 1), start_token, device=device, dtype=torch.long)
        # highlights = highlights.repeat(1,1023,1)
    if start_token == None:
        context = torch.tensor(context, device=device, dtype=torch.long).unsqueeze(0).repeat(gen_batch_size, 1)
    prev = context
    output = context
    past = None

    with torch.no_grad():
        for i in trange(length):
            # highlights = highlights_input.repeat(1, i + 1, 1)
            logits, past = model(prev, highlights, past=past)
            # print(prev.shape, past[-1].shape)
            # print(logits.shape)

            logits = logits[:, -1, :] / temperature
            logits = top_k_logits(logits, k=top_k)
            log_probs = F.softmax(logits, dim=-1)
            if sample:
                prev = torch.multinomial(log_probs, num_samples=1)
            else:
                _, prev = torch.topk(log_probs, k=1, dim=-1)
            output = torch.cat((output, prev), dim=1)
            # print(output.shape)
    return output
