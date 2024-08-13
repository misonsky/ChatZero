#coding=utf-8
import warnings
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers import BartForConditionalGeneration,BartConfig

def shift_tokens_right(input_ids, pad_token_id):
    """Shift input ids one token to the right, and wrap the last non pad token (usually <eos>)."""
    prev_output_tokens = input_ids.clone()
    index_of_eos = (input_ids.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
    prev_output_tokens[:, 0] = input_ids.gather(1, index_of_eos).squeeze()
    prev_output_tokens[:, 1:] = input_ids[:, :-1]
    return prev_output_tokens

class BartGenerationModel(BartForConditionalGeneration):
    def __init__(self, config: BartConfig):
        super().__init__(config)
    def forward(
        self,
        input_ids,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **unused,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss.
            Indices should either be in ``[0, ..., config.vocab_size]`` or -100 (see ``input_ids`` docstring).
            Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens
            with labels in ``[0, ..., config.vocab_size]``.

        Returns:

        Conditional generation example::

            >>> # Mask filling only works for bart-large
            >>> from transformers import BartTokenizer, BartForConditionalGeneration
            >>> tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
            >>> TXT = "My friends are <mask> but they eat too many carbs."

            >>> model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
            >>> input_ids = tokenizer([TXT], return_tensors='pt')['input_ids']
            >>> logits = model(input_ids).logits

            >>> masked_index = (input_ids[0] == tokenizer.mask_token_id).nonzero().item()
            >>> probs = logits[0, masked_index].softmax(dim=0)
            >>> values, predictions = probs.topk(5)

            >>> tokenizer.decode(predictions).split()
            >>> # ['good', 'great', 'all', 'really', 'very']
        """
        if "lm_labels" in unused:
            warnings.warn(
                "The `lm_labels` argument is deprecated and will be removed in a future version, use `labels` instead.",
                FutureWarning,
            )
            labels = unused.pop("lm_labels")
        if "decoder_cached_states" in unused:
            warnings.warn(
                "The `decoder_cached_states` argument is deprecated and will be removed in a future version, use `past_key_values` instead.",
                FutureWarning,
            )
            past_key_values = unused.pop("decoder_cached_states")
        if "decoder_past_key_values" in unused:
            warnings.warn(
                "The `decoder_past_key_values` argument is deprecated and will be removed in a future version, use `past_key_values` instead.",
                FutureWarning,
            )
            past_key_values = unused.pop("decoder_past_key_values")
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            use_cache = False
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(labels, self.config.pad_token_id)

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        lm_logits = F.linear(outputs[0], self.model.shared.weight, bias=self.final_logits_bias)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # TODO(SS): do we need to ignore pad tokens in labels?
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )