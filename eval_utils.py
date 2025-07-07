import torch
import os
import logging
from tqdm import tqdm


@torch.no_grad()
def evaluator(model, testenc, dev, args):
    model.eval()

    use_cache = model.config.use_cache
    model.config.use_cache = False

    layers = model.model.layers
    model.model.embed_tokens = model.model.embed_tokens.to(dev)

    layers[0] = layers[0].to(dev)
    model.model.rotary_emb = model.model.rotary_emb.to(dev)
    # Convert the whole text of evaluation dataset into batches of sequences.
    input_ids = testenc.input_ids  # (1, text_len)
    nsamples = input_ids.numel() // args.model_max_length  # The tail is truncated.
    input_ids = (
        input_ids[:, : nsamples * args.model_max_length].view(nsamples, args.model_max_length).to(dev)
    )  # (nsamples, seqlen)

    batch_size = args.batch_size
    input_ids = [input_ids[i : i + batch_size] for i in range(0, nsamples, batch_size)]
    nbatches = len(input_ids)

    dtype = next(iter(model.parameters())).dtype
    # The input of the first decoder layer.
    inps = torch.zeros(
        (nbatches, batch_size, args.model_max_length, model.config.hidden_size),
        dtype=dtype,
        device=dev,
    )
    inps = [0] * nbatches
    cache = {"i": 0, "attention_mask": None}

    class Catcher(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            cache["position_ids"] = kwargs["position_ids"]
            cache["cache_position"] = kwargs["cache_position"]
            cache["position_embeddings"] = kwargs["position_embeddings"]

            raise ValueError

    layers[0] = Catcher(layers[0])

    for i in range(nbatches):
        batch = input_ids[i]
        try:
            model(batch)
            print(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()

    model.model.embed_tokens = model.model.embed_tokens.cpu()

    position_ids = cache["position_ids"]
    attention_mask = cache["attention_mask"]
    cache_position= cache["cache_position"]
    position_embeddings= cache["position_embeddings"]
    #print(position_embeddings)
    torch.cuda.empty_cache()
    outs = [0] * nbatches

    for i in tqdm(range(len(layers)), desc="(Eval) Layers"):
        layer = layers[i].to(dev)
        for j in range(nbatches):
            outs[j] = layer(
                inps[j],
                attention_mask=attention_mask,
                #  defined.
                position_ids=position_ids,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(dev)

    model.lm_head.weight.data = model.lm_head.weight.data.to(inps[0].dtype).to(dev)
    nlls = []
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    for i in range(nbatches):
        hidden_states = inps[i]
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :]
        shift_labels = input_ids[i][:, 1:]
        loss = loss_fct(shift_logits.permute(0, 2, 1), shift_labels)
        neg_log_likelihood = loss.float().mean(dim=1)
        nlls.append(neg_log_likelihood)
    nlls_tensor = torch.cat(nlls)

    ppl = torch.exp(nlls_tensor.mean())
    model.config.use_cache = use_cache    return ppl.item()