from typing import Optional
from transformers import CLIPTextModelWithProjection
import torch.nn as nn
import torch
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from dataclasses import dataclass
from typing import Optional, Tuple
from transformers.utils import ModelOutput

@dataclass
class CLIPTextModelOutput(ModelOutput):

    text_embeds: Optional[torch.FloatTensor] = None
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None

def _expand_mask(mask, dtype, tgt_len = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)

def _build_causal_attention_mask(bsz, seq_len, dtype):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(bsz, seq_len, seq_len, dtype=dtype)
        mask.fill_(torch.tensor(torch.finfo(dtype).min))
        mask.triu_(1)  # zero out the lower diagonal
        mask = mask.unsqueeze(1)  # expand mask
        return mask


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError

class FrozenCLIPEmbedderWithProjection(AbstractEncoder):
    """Uses the CLIP transformer encoder for text (from Hugging Face)"""
    def __init__(self, version="", device="cuda", revision=False, 
                 max_length=77, tokenizer=None, instant_prompt=""):
        super().__init__()
        self.text_encoder = CLIPTextModelWithProjection.from_pretrained(
            version,
            subfolder="text_encoder_2",
            revision=revision,
        )
        self.device = device
        self.max_length = max_length
        token = tokenizer(instant_prompt, truncation=True, max_length=self.max_length, return_length=True,
                                return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        self.token = token["input_ids"].to(self.device)
        self.text_encoder.requires_grad_(False)
        self.text_encoder.eval()
        self.text_encoder.train = disabled_train

        def embedding_forward(
                self,
                input_ids = None,
                position_ids = None,
                inputs_embeds = None,
            ) -> torch.Tensor:

                seq_length = input_ids.shape[-1] if input_ids is not None else inputs_embeds.shape[-2]

                if position_ids is None:
                    position_ids = self.position_ids[:, :seq_length]

                if inputs_embeds is None:
                    inputs_embeds = self.token_embedding(input_ids)

                position_embeddings = self.position_embedding(position_ids)
                embeddings = inputs_embeds + position_embeddings
                
                return embeddings

        self.text_encoder.text_model.embeddings.forward = embedding_forward.__get__(self.text_encoder.text_model.embeddings)

        def encoder_forward(
            self,
            inputs_embeds,
            attention_mask = None,
            causal_attention_mask = None,
            output_attentions = None,
            output_hidden_states = None,
            return_dict = None,
        ):
            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            encoder_states = () if output_hidden_states else None
            all_attentions = () if output_attentions else None

            hidden_states = inputs_embeds
            for idx, encoder_layer in enumerate(self.layers):
                if output_hidden_states:
                    encoder_states = encoder_states + (hidden_states,)

                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    causal_attention_mask,
                    output_attentions=output_attentions,
                )

                hidden_states = layer_outputs[0]

                if output_attentions:
                    all_attentions = all_attentions + (layer_outputs[1],)

            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)

            return BaseModelOutput(
                last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
            )
            # return hidden_states

        self.text_encoder.text_model.encoder.forward = encoder_forward.__get__(self.text_encoder.text_model.encoder)

        def text_encoder_forward(
            self,
            input_ids = None,
            attention_mask = None,
            position_ids = None,
            output_attentions = None,
            output_hidden_states = None,
            return_dict = None,
        ):
            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            if input_ids is None:
                raise ValueError("You have to specify either input_ids")

            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])

            hidden_states = self.embeddings(input_ids=input_ids, position_ids=position_ids)

            bsz, seq_len = input_shape
            # CLIP's text model uses causal mask, prepare it here.
            # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
            causal_attention_mask = _build_causal_attention_mask(bsz, seq_len, hidden_states.dtype).to(
                hidden_states.device
            )

            # expand attention_mask
            if attention_mask is not None:
                # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
                attention_mask = _expand_mask(attention_mask, hidden_states.dtype)

            encoder_outputs = self.encoder(
                inputs_embeds=hidden_states,
                attention_mask=attention_mask,
                causal_attention_mask=causal_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            # last_hidden_state = self.final_layer_norm(last_hidden_state)

            # return last_hidden_state
            last_hidden_state = encoder_outputs[0]
            last_hidden_state = self.final_layer_norm(last_hidden_state)
            if self.eos_token_id == 2:
                pooled_output = last_hidden_state[
                    torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
                    input_ids.to(dtype=torch.int, device=last_hidden_state.device).argmax(dim=-1),
                ]
            else:
                pooled_output = last_hidden_state[
                    torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
                    (input_ids.to(dtype=torch.int, device=last_hidden_state.device) == self.eos_token_id)
                    .int()
                    .argmax(dim=-1),
                ]

            if not return_dict:
                return (last_hidden_state, pooled_output) + encoder_outputs[1:]

            return BaseModelOutputWithPooling(
                last_hidden_state=last_hidden_state,
                pooler_output=pooled_output,
                hidden_states=encoder_outputs.hidden_states,
                attentions=encoder_outputs.attentions,
            )

        self.text_encoder.text_model.forward = text_encoder_forward.__get__(self.text_encoder.text_model)

        def transformer_forward(
            self,
            input_ids = None,
            attention_mask = None,
            position_ids = None,
            output_attentions = None,
            output_hidden_states = None,
            return_dict = None,
        ):
        
            text_outputs = self.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            pooled_output = text_outputs[1]

            text_embeds = self.text_projection(pooled_output)

            return CLIPTextModelOutput(
                text_embeds=text_embeds,
                last_hidden_state=text_outputs.last_hidden_state,
                hidden_states=text_outputs.hidden_states,
                attentions=text_outputs.attentions,
            )

        self.text_encoder.forward = transformer_forward.__get__(self.text_encoder)

    def forward(self, 
                input_ids: Optional[torch.Tensor] = None, 
                output_hidden_states: Optional[bool] = True,
                **kwargs):
        input_token = self.token if input_ids is None else input_ids
        prompt_embeds = self.text_encoder(input_ids=input_token, output_hidden_states=output_hidden_states, **kwargs)
        return prompt_embeds
 
    def encode(self, 
               token=None,
               **kwargs):
        if token is None:
            return self(**kwargs)
        else:
            return self(input_ids=token, **kwargs)