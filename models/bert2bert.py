#coding=utf-8
from typing import Optional,Union,Tuple
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import Seq2SeqLMOutput,BaseModelOutput
from transformers import PretrainedConfig,PreTrainedModel
from transformers import EncoderDecoderModel

def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    if decoder_start_token_id is None:
        raise ValueError("Make sure to set the decoder_start_token_id attribute of the model's configuration.")
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("Make sure to set the pad_token_id attribute of the model's configuration.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids
class EncoderPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class BERT2BERT(EncoderDecoderModel):
    def __init__(self, config: Optional[PretrainedConfig] = None,
                        encoder: Optional[PreTrainedModel] = None, 
                        decoder: Optional[PreTrainedModel] = None):
        super().__init__(config, encoder, decoder)
        self.loss_fct = CrossEntropyLoss()
    def set_special_tokens(self,start_token,end_token,pad_token,vocab_size):
        self.config.decoder_start_token_id = start_token
        self.config.eos_token_id = end_token
        self.config.pad_token_id = pad_token
        self.config.vocab_size = vocab_size
    def swap_sample(self,inputs,number_examples):
        inputs = inputs.view(-1,number_examples,inputs.size(-1))#batch * number * hidden
        batch_number = inputs.size(0)
        contrast_feature = []
        for i in range(batch_number):
            contrast_feature.append(torch.cat((inputs[0:i],inputs[i+1:]),0).view(-1,inputs.size(-1)))
        contrast_feature = torch.stack(contrast_feature,dim=0)#batch * m * hidden
        scores = torch.einsum("bnd,bmd->bnm",inputs,contrast_feature)
        scores = scores.view(-1,scores.size(-1))/0.07
        labels = torch.cat((torch.ones(1,dtype=scores.dtype),torch.zeros(scores.size(-1)-1,dtype=scores.dtype)),dim=-1).cuda()
        labels = torch.tile(labels,dims=[scores.size(0),1]).view(-1,labels.size(-1))
        instance_loss = self.loss_fct(scores,labels)
        #cluster
        inputs = inputs / inputs.norm(dim=-1, keepdim=True)
        scores = torch.einsum("bsd,btd->bst",inputs,inputs)#batch * max_example * max_example
        scores = torch.div(scores,0.07)#batch * max_example * max_example
        mask = torch.triu(torch.ones(scores.size(-1),scores.size(-1)),diagonal=1).contiguous().cuda()
        scores  = scores * mask
        return instance_loss-torch.sum(scores)


    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        encoder_outputs: Optional[Tuple[torch.FloatTensor]] = None,
        past_key_values: Tuple[Tuple[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, Seq2SeqLMOutput]:
        zero_setting,max_examples=None,None
        if "zero_setting" in kwargs:
            zero_setting = kwargs["zero_setting"]
            kwargs.pop("zero_setting")
        if "max_examples" in kwargs:
            max_examples = kwargs["max_examples"]
            kwargs.pop("max_examples")
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        kwargs_encoder = {argument: value for argument, value in kwargs.items() if not argument.startswith("decoder_")}

        kwargs_decoder = {
            argument[len("decoder_") :]: value for argument, value in kwargs.items() if argument.startswith("decoder_")
        }
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs_encoder,
            )
        elif isinstance(encoder_outputs, tuple):
            encoder_outputs = BaseModelOutput(*encoder_outputs)
        
        encoder_hidden_states = encoder_outputs[0]
        # optionally project encoder_hidden_states
        if (
            self.encoder.config.hidden_size != self.decoder.config.hidden_size
            and self.decoder.config.cross_attention_hidden_size is None
        ):
            encoder_hidden_states = self.enc_to_dec_proj(encoder_hidden_states)

        if (labels is not None) and (decoder_input_ids is None and decoder_inputs_embeds is None):
            decoder_input_ids = shift_tokens_right(
                labels, self.config.pad_token_id, self.config.decoder_start_token_id
            )
        encoder_contrastive_loss = 0
        if zero_setting:
            pool_states = torch.mean(encoder_hidden_states,dim=1)
            encoder_contrastive_loss = self.swap_sample(pool_states,max_examples*2+1)
        
        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            past_key_values=past_key_values,
            return_dict=return_dict,
            **kwargs_decoder,
        )
        #zero_setting
        if zero_setting and labels is not None:
            sample_logits = F.gumbel_softmax(decoder_outputs.logits,hard=False)#batch * s *v
            #label smoothing
            predictions_representation = torch.einsum("bsv,vd->bsd",sample_logits,self.get_output_embeddings())
            gold_representation = self.encoder(
                input_ids=labels,
                attention_mask=None,
                inputs_embeds=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=return_dict,
                **kwargs_encoder,
            )[0].detach()
            predictions_representation = torch.mean(predictions_representation,dim=1)
            gold_representation = torch.mean(gold_representation,dim=1).view(-1,2*max_examples+1,gold_representation.size(-1))[:,-max_examples-1]
            instances = torch.cat((gold_representation,predictions_representation.sunsqueeze(1)),dim=1).view(-1,predictions_representation.size(-1))
            decoder_contrastive_loss = self.swap_sample(instances,max_examples*3+2)
            
            
        # Compute loss independent from decoder (as some shift the logits inside them)
        loss,con_loss = None,None
        if labels is not None:
            logits = decoder_outputs.logits if return_dict else decoder_outputs[0]
            loss = self.loss_fct(logits.reshape(-1, self.decoder.config.vocab_size), labels.view(-1))
        if not return_dict:
            if loss is not None:
                return (loss,encoder_contrastive_loss) + decoder_outputs + encoder_outputs
            else:
                return decoder_outputs + encoder_outputs

        return Seq2SeqLMOutput(
            loss=(loss,encoder_contrastive_loss+decoder_contrastive_loss),
            logits=decoder_outputs.logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )
