#coding=utf-8
import argparse
from distutils.command.config import config
from glob import glob
import logging
import os
import re
import math
import random
import timeit
import json
import pickle as pkl
import numpy as np
from typing import List
from functools import partial
import torch
from torch import distributed
from torch.nn import parallel
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
from nltk import word_tokenize
from datasetBeam import DatasetInstance
from transformers import AdamW,get_linear_schedule_with_warmup,WEIGHTS_NAME
from transformers import BartConfig,BertConfig,GPT2Config
from models.bartmodel import BartGenerationModel
from models.bert2bert import BERT2BERT
from models.dialogGpt import GPT2Generation

from utils.EvaluationUtils import calculationEmbedding,cal_Distinct,compute_bleu_rouge_single_prediction,cal_corpus_bleu

from apex import amp

MODELS={"bert":(BERT2BERT,BertConfig),
        "gpt2":(GPT2Generation,GPT2Config),
        "bart":(BartGenerationModel,BartConfig)}
    
logger = logging.getLogger(__name__)

def validation_parameters(config):
    if not config.do_prepare and not config.do_train and not config.do_eval and not config.do_predict:
        raise ValueError("must specify one of them")
    model_dir = os.path.join(config.model_dir,config.corpus,config.lang,config.model)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    process_dir = os.path.join(config.process_path,config.corpus,config.lang)
    if not os.path.exists(process_dir):
        os.makedirs(process_dir)
    if config.zero_setting:
        process_dir = os.path.join(config.process_path,config.corpus,config.lang,config.zero_path)
        if not os.path.exists(process_dir):
            os.makedirs(process_dir)
        model_dir = os.path.join(config.model_dir,config.corpus,config.lang,config.zero_path,config.model)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
def get_model(config,bos_token_id=0,eos_token_id=0,pad_token_id=0):
    model_cls,config_cls = MODELS[config.model]
    if config.model == "bert":
        model = model_cls.from_encoder_decoder_pretrained(config.model_name_or_path, config.model_name_or_path)
        model.set_special_tokens(start_token = bos_token_id,
                                end_token = eos_token_id,
                                pad_token = pad_token_id,
                                vocab_size = model.config.decoder.vocab_size)
    else:
        ModelConfig = config_cls.from_pretrained(
            config.config_name if config.config_name else config.model_name_or_path,
            cache_dir=config.cache_dir if config.cache_dir else None,
        )
        model = model_cls.from_pretrained(
            config.model_name_or_path,
            from_tf=bool(".ckpt" in config.model_name_or_path),
            config=ModelConfig,
            cache_dir=config.cache_dir if config.cache_dir else None,
        )
    return model


def EvaluateDialogue(config,golds:List[str],predictions:List[str]):
    golds = [gold.split() for gold in golds]
    predictions = [prediction.split() for prediction in predictions]
    metricsResults = calculationEmbedding(config,predictions=predictions,references=golds)
    metricsResults.update(cal_corpus_bleu(golds,predictions))
    metricsResults.update(cal_Distinct(predictions))
    golds_dict = { _index:[" ".join(gold)] for _index, gold in enumerate(golds)}
    con_dict = { _index:[" ".join(pred)] for _index, pred in enumerate(predictions)}
    metricsResults.update(compute_bleu_rouge_single_prediction(con_dict,golds_dict))
    return metricsResults

class Trainer(object):
    def __init__(self,config,model=None,optimizer=None,LRSchedule=None):
        self.lr_scheduler=LRSchedule
        self.config = config
        self.model=model
        self.optimizer=optimizer
        self.global_step = 0
        self.best_metrics = float("inf")
        validation_parameters(self.config)
    def set_seed(self):
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        if self.config.n_gpu > 0:
            torch.cuda.manual_seed_all(self.config.seed)
    def set_fp16(self,model,optimizer):
        try:
            # from apex import amp
            amp.register_half_function(torch, "einsum")
            amp.register_float_function(torch, 'sigmoid')
            amp.register_float_function(torch, 'softmax')
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=self.config.fp16_opt_level)
        return model,optimizer
    def create_optimizer(self,parameters_state,num_training_steps):
        model_dir=os.path.join(self.config.model_dir,self.config.corpus,self.config.model)
        no_decay = ["bias", "LayerNorm.weight"]
        storerName="crossattention"
        parameters_state = [(n, p) for n, p in parameters_state if storerName not in n]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in parameters_state if not any(nd in n for nd in no_decay)],
                "weight_decay": self.config.weight_decay,
                },
            {
                "params": [p for n, p in parameters_state if any(nd in n for nd in no_decay)], 
                "weight_decay": 0.0
                },]
        optimizer = AdamW(optimizer_grouped_parameters, 
                          lr=self.config.learning_rate,
                          betas=(self.config.adam_beta1,self.config.adam_beta2),
                          eps=self.config.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer = optimizer, 
            num_warmup_steps=self.config.warmup_steps, 
            num_training_steps=num_training_steps)
        
        # saved/load optimizer or scheduler states
        if os.path.isfile(os.path.join(model_dir, "optimizer.pt")) and os.path.isfile(os.path.join(model_dir, "scheduler.pt")):
            # Load in optimizer and scheduler states
            optimizer.load_state_dict(torch.load(os.path.join(model_dir, "optimizer.pt")))
            scheduler.load_state_dict(torch.load(os.path.join(model_dir, "scheduler.pt")))
        return optimizer,scheduler
    def prepare(self):
        if self.config.zero_setting:
            datasetPath = os.path.join(self.config.process_path,self.config.corpus,self.config.lang,self.config.zero_path,self.config.tokenizer_path%(self.config.model))
        else:
            datasetPath = os.path.join(self.config.process_path,self.config.corpus,self.config.lang,self.config.tokenizer_path%(self.config.model))
        datasetInstance = None
        if not os.path.exists(datasetPath):
            datasetInstance=DatasetInstance(self.config)
            with open(datasetPath,"wb") as f:
                pkl.dump(datasetInstance,f)
        else:
            with open(datasetPath,"rb") as f:
                datasetInstance = pkl.load(f)
        #construct the evaluation embedding
        if not os.path.exists(os.path.join(self.config.process_path,self.config.corpus,self.config.lang,"corpus.txt")):
            datasetInstance.build_corpus(fileName=self.config.train_files,
                                        outputFile=self.config.train_features%(self.config.model),
                                        file_type="train")

        datasetInstance.convert_examples_features(fileName=self.config.train_files,
                                                                    outputFile=self.config.train_features%(self.config.model),
                                                                    file_type="train",
                                                                    zero_setting=self.config.zero_setting)
        
        datasetInstance.convert_examples_features(fileName=self.config.dev_files,
                                                                    outputFile=self.config.dev_features%(self.config.model),
                                                                    file_type="dev",
                                                                    zero_setting=False)
        

        datasetInstance.convert_examples_features(fileName=self.config.test_files,
                                                                    outputFile=self.config.test_features%(self.config.model),
                                                                    file_type="test",
                                                                    zero_setting=False)
    
    def adapterInputs(self,batch_features,model,**kwargs):
        if self.config.model == "bert":
            outputs = model(input_ids=batch_features["input_ids"],
                            attention_mask=batch_features["attention_mask"],
                            labels=batch_features["target_input_ids"],
                            return_dict=True,
                            zero_setting = self.config.zero_setting,
                            max_examples = self.config.max_examples)
        elif self.config.model== "bart":
            outputs = model(input_ids=batch_features["input_ids"],
                            attention_mask=batch_features["attention_mask"],
                            labels=batch_features["target_input_ids"],
                            return_dict=True)
        elif self.config.model == "gpt2":
            outputs = model(input_ids=batch_features["input_ids"],
                            labels=batch_features["input_ids"],
                            return_dict=True)
        return outputs
    
    def train(self):
        print("local_rank",self.config.local_rank)
        if self.config.local_rank in [-1, 0]:
            tb_writer = SummaryWriter(filename_suffix=self.config.model)

        if self.config.local_rank == -1 or self.config.no_cuda:
            device = torch.device("cuda" if torch.cuda.is_available() and not self.config.no_cuda else "cpu")
            self.config.n_gpu = 0 if self.config.no_cuda else torch.cuda.device_count()
        else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
            torch.cuda.set_device(self.config.local_rank)
            device = torch.device("cuda",self.config.local_rank)
            torch.distributed.init_process_group(backend="nccl")
            self.config.n_gpu = 1
        self.config.device = device
        logger.warning(f"device: {self.config.device}, n_gpu: {self.config.n_gpu}")
        if self.config.local_rank not in [-1, 0]:
            # Make sure only the first process in distributed training will download model & vocab
            distributed.barrier()
        if self.config.zero_setting:
            DatasetPath = os.path.join(self.config.process_path,self.config.corpus,self.config.lang,self.config.zero_path,self.config.tokenizer_path%(self.config.model))
        else:
            DatasetPath = os.path.join(self.config.process_path,self.config.corpus,self.config.lang,self.config.tokenizer_path%(self.config.model))
        with open(DatasetPath,"rb") as f:
            DatasetInstance = pkl.load(f)
        DatasetInstance.update_config(self.config)
        logger.warning(f"load datasets object: {DatasetPath}")
        model = get_model(config=self.config,
                        bos_token_id=DatasetInstance.BOS_ID,
                        eos_token_id=DatasetInstance.EOS_ID,
                        pad_token_id=DatasetInstance.PAD_ID)
        model = self.load_model1(model=model,
                            checkpoint=self.config.model_name_or_path,
                            step="best")
        if self.config.local_rank == 0:
            # Make sure only the first process in distributed training will download model & vocab
            distributed.barrier()
        model.to(self.config.device)
        logger.info("Training/evaluation parameters %s", self.config)
        if self.config.zero_setting:
            train_features = torch.load(os.path.join(self.config.process_path,self.config.corpus,self.config.lang,self.config.zero_path,self.config.train_features%(self.config.model)))
        else:
            train_features = torch.load(os.path.join(self.config.process_path,self.config.corpus,self.config.lang,self.config.train_features%(self.config.model)))
        num_example = len(train_features)
        train_batch_size = self.config.per_gpu_train_batch_size * max(1, self.config.n_gpu)
        t_total = num_example // self.config.gradient_accumulation_steps * self.config.num_train_epochs
        if self.config.model == "gpt2":
            IteratorDataset = DatasetInstance.decoder_batch_features(features = train_features,
                                        batch_size = train_batch_size,
                                        training=True)
        elif self.config.model in ["bert","bart"]:
            if self.config.zero_setting:
                IteratorDataset = DatasetInstance.zero_encoder_decoder_batch_features(features = train_features,
                                                                                        batch_size = train_batch_size,
                                                                                        training=True)
            else:
                IteratorDataset = DatasetInstance.encoder_decoder_batch_features(features = train_features,
                                            batch_size = train_batch_size,
                                            training=True)
        parameters_state = model.named_parameters()
        optimizer,scheduler = self.create_optimizer(parameters_state= parameters_state, num_training_steps = t_total)
        if self.config.fp16:
            self.set_fp16(model, optimizer)
            
        if self.config.n_gpu > 1:
            model = torch.nn.DataParallel(model)
        if self.config.local_rank != -1:
            model = parallel.DistributedDataParallel(model, device_ids=[self.config.local_rank], output_device=self.config.local_rank, find_unused_parameters=True)
        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", num_example)
        logger.info("  Num Epochs = %d", self.config.num_train_epochs)
        logger.info("  Instantaneous batch size per GPU = %d", self.config.per_gpu_train_batch_size)
        logger.info("Total train batch size (w. parallel, distributed & accumulation) = %d", train_batch_size * self.config.gradient_accumulation_steps *(distributed.get_world_size() if self.config.local_rank != -1 else 1))
        logger.info("  Gradient Accumulation steps = %d", self.config.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)
        global_step = 1
        tr_loss, logging_loss, info_loss,logging_info= 0.0, 0.0,0.0,0.0
        model.zero_grad()
        self.set_seed()
        for _ in range(self.config.num_train_epochs):
            for step, batch in enumerate(IteratorDataset):
                model.train()
                batch = {k:v.to(self.config.device) for k, v in batch.items()}
                outputs = self.adapterInputs(batch,model)
                loss1 = outputs.loss[0]
                loss2 = outputs.loss[1]
                if self.config.n_gpu > 1:
                    loss1 = loss1.mean()  # mean() to average on multi-gpu parallel (not distributed) training
                    loss2 = loss2.mean()
                if self.config.gradient_accumulation_steps > 1:
                    loss1 = loss1 / self.config.gradient_accumulation_steps
                    loss2 = loss2 / self.config.gradient_accumulation_steps

                if self.config.fp16:
                    with amp.scale_loss(loss1-loss2, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    (loss1-loss2).backward()
                tr_loss = tr_loss + (loss1-loss2).item()
                info_loss += loss1.item()
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    if self.config.fp16:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), self.config.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.max_grad_norm)
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    global_step += 1
                if self.config.local_rank in [-1, 0] and  global_step % self.config.log_steps == 0 and (step + 1) % self.config.gradient_accumulation_steps == 0:
                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar("loss", (tr_loss - logging_loss) / self.config.log_steps, global_step)
                    logger.info("step {} lr {}  loss {} info_loss {} ".format(global_step,scheduler.get_lr()[0],(tr_loss - logging_loss) / self.config.log_steps,(info_loss - logging_info) /  self.config.log_steps))
                    logging_loss = tr_loss
                    logging_info = info_loss
                if self.config.local_rank in [-1, 0] and global_step % self.config.eval_steps == 0 and (step + 1) % self.config.gradient_accumulation_steps == 0:
                    results = self.evaluate(model, DatasetInstance=DatasetInstance)
                    for key, value in results.items():
                        tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                    self.save_model(model = model, 
                                    optimizer = optimizer,
                                    scheduler = scheduler,
                                    step = global_step)
    def generation_tools(self,generation_fun,max_length,min_length,do_sample,early_stopping,num_beams,repetition_penalty,bos_token_id,pad_token_id,eos_token_id,length_penalty,no_repeat_ngram_size):
        f = partial(generation_fun,
                    max_length=max_length,
                    min_length = min_length,
                    do_sample=do_sample,
                    early_stopping=early_stopping,
                    num_beams=num_beams,
                    repetition_penalty=repetition_penalty,
                    bos_token_id=bos_token_id,
                    pad_token_id=pad_token_id,
                    eos_token_id=eos_token_id,
                    length_penalty=length_penalty,
                    no_repeat_ngram_size=no_repeat_ngram_size)
        return f
    def remove_special_token(self,inputs:List[str]):
        inputs_len = len(inputs)
        for i in range(inputs_len-1,0,-1):
            if inputs[i].startswith("##"):
                inputs[i-1] = inputs[i-1] + inputs[i].replace("##","")
                inputs[i]=[]
        inputs = [item for item in inputs if len(item)>0]
        return " ".join(inputs)
    def evaluate(self,model,DatasetInstance=None):
        if DatasetInstance is None:
            if self.config.zero_setting:
                DatasetPath = os.path.join(self.config.process_path,self.config.corpus,self.config.lang,self.config.zero_path,self.config.tokenizer_path%(self.config.model))
            else:
                DatasetPath = os.path.join(self.config.process_path,self.config.corpus,self.config.lang,self.config.tokenizer_path%(self.config.model))
            with open(DatasetPath,"rb") as f:
                DatasetInstance = pkl.load(f)
        dev_features=torch.load(os.path.join(self.config.process_path,self.config.corpus,self.config.lang,self.config.dev_features%(self.config.model)))
        dev_batch_size = self.config.per_gpu_eval_batch_size * max(1, self.config.n_gpu)
        eval_dataloader = DatasetInstance.encoder_decoder_batch_features(features = dev_features,
                                        batch_size = dev_batch_size,
                                        training=False)
        logger.info("***** Running evaluation  *****")
        logger.info("  Num examples = %d", len(dev_features))
        logger.info("  Batch size = %d", dev_batch_size)
        cands,golds= [],[]
        for step,batch in enumerate(eval_dataloader):
            model.eval()
            batch_features = {k:v.to(self.config.device) for k, v in batch.items()}
            with torch.no_grad():
                max_length = batch_features["input_ids"].size(1) + self.config.max_decode_length if self.config.model == "gpt2" else self.config.max_decode_length
                min_length = batch_features["input_ids"].size(1) + self.config.min_decode_length if self.config.model == "gpt2" else self.config.min_decode_length
                generate_callback = self.generation_tools(generation_fun=model.module.generate if isinstance(model,DistributedDataParallel) else model.generate,
                                                    max_length = max_length,
                                                    min_length = min_length,
                                                    do_sample = False,
                                                    early_stopping = True,
                                                    num_beams = self.config.beam_size,
                                                    repetition_penalty = self.config.repetition_penalty,
                                                    bos_token_id= DatasetInstance.BOS_ID,
                                                    pad_token_id = DatasetInstance.PAD_ID,
                                                    eos_token_id = DatasetInstance.EOS_ID,
                                                    length_penalty = self.config.length_penalty,
                                                    no_repeat_ngram_size = self.config.no_repeat_ngram_size)
                if self.config.model == "gpt2":
                    output_sequences = generate_callback(input_ids=batch_features["input_ids"])
                    output_sequences = output_sequences[:,batch_features["input_ids"].size(1):]
                else:
                    output_sequences = generate_callback(input_ids=batch_features["input_ids"], attention_mask=batch_features["attention_mask"])
                for seq in output_sequences.tolist():
                    decoder_seq = DatasetInstance.tokenizer.decode(seq, skip_special_tokens=True)
                    decoder_seq = self.remove_special_token(DatasetInstance.tokenizer.tokenize(decoder_seq))
                    cands.append(decoder_seq)
        gold_file = os.path.join(self.config.process_path,self.config.corpus,self.config.lang,self.config.dev_files.split(".")[0]+".ref")
        with open(gold_file,"r" ,encoding="utf-8") as f:
            for line in f:
                line = line.rstrip()
                golds.append(line)
        metricsResults = EvaluateDialogue(config=self.config,golds=golds,predictions=cands)
        formatString="the prediction metrics is "
        for _key in metricsResults:
            formatString +=_key
            formatString +=" {} ".format(metricsResults[_key])
        logger.info(formatString)
        predictions = os.path.join(self.config.model_dir,self.config.corpus,self.config.lang,self.config.model,"eval.json")
        with open(predictions,"w",encoding="utf-8") as f:
            json.dump(cands,f,ensure_ascii=False,indent=4)
        return metricsResults
    def save_model(self, model, optimizer,scheduler,step):
        """Save parameters to checkpoint"""
        ckpt_path=os.path.join(self.config.model_dir,self.config.corpus,self.config.lang,self.config.model)
        print(f'Save parameters to {ckpt_path}')
        torch.save(model.state_dict(), os.path.join(ckpt_path,f'{step}.pkl'))
        # torch.save(optimizer.state_dict(), os.path.join(ckpt_path, f'optimizer_{step}.pt'))
        # torch.save(scheduler.state_dict(), os.path.join(ckpt_path, f'scheduler_{step}.pt'))
    def load_model1(self, model,checkpoint,step):
        """Load parameters from checkpoint"""
        logger.info(f'Load parameters from {checkpoint}')
        checkpoints_state_dict = torch.load(os.path.join(checkpoint,f'{step}.pkl'))
        checkpoinss_keys = checkpoints_state_dict.keys()
        for name,p in model.named_parameters():
            if name not in checkpoinss_keys:
                checkpoints_state_dict[name] = p
                print("%s will be  initializes randomly."%(name))
        model.load_state_dict(checkpoints_state_dict)
        return model
    def load_model(self, model, optimizer,scheduler,checkpoint,step):
        """Load parameters from checkpoint"""
        logger.info(f'Load parameters from {checkpoint}')
        model.load_state_dict(torch.load(os.path.join(checkpoint,f'{step}.pkl')))
        optimizer.load_state_dict(torch.load(os.path.join(checkpoint, f'optimizer_{step}.pt')))
        scheduler.load_state_dict(torch.load(os.path.join(checkpoint, f'scheduler_{step}.pt')))
        return model,optimizer,scheduler
    def predictions(self):
        print("local_rank",self.config.local_rank)
        if self.config.local_rank in [-1, 0]:
            tb_writer = SummaryWriter(filename_suffix=self.config.model)

        if self.config.local_rank == -1 or self.config.no_cuda:
            device = torch.device("cuda" if torch.cuda.is_available() and not self.config.no_cuda else "cpu")
            self.config.n_gpu = 0 if self.config.no_cuda else torch.cuda.device_count()
        else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
            torch.cuda.set_device(self.config.local_rank)
            device = torch.device("cuda",self.config.local_rank)
            torch.distributed.init_process_group(backend="nccl")
            self.config.n_gpu = 1
        self.config.device = device
        logger.warning(f"device: {self.config.device}, n_gpu: {self.config.n_gpu}")
        if self.config.local_rank not in [-1, 0]:
            # Make sure only the first process in distributed training will download model & vocab
            distributed.barrier()
        DatasetPath = os.path.join(self.config.process_path,self.config.corpus,self.config.tokenizer_path%(self.config.model))
        with open(DatasetPath,"rb") as f:
            DatasetInstance = pkl.load(f)
        DatasetInstance.update_config(self.config)
        logger.warning(f"load datasets object: {DatasetPath}")
        model = get_model(config=self.config,
                        bos_token_id=DatasetInstance.BOS_ID,
                        eos_token_id=DatasetInstance.EOS_ID,
                        pad_token_id=DatasetInstance.PAD_ID)
        if self.config.local_rank == 0:
            # Make sure only the first process in distributed training will download model & vocab
            distributed.barrier()
        model.to(self.config.device)
        logger.info("Training/evaluation parameters %s", self.config)

        test_features=torch.load(os.path.join(self.config.process_path,self.config.corpus,self.config.lang,self.config.test_features%(self.config.model)))
        num_example = len(test_features)
        dev_batch_size = self.config.per_gpu_eval_batch_size * max(1, self.config.n_gpu)
        t_total = num_example // self.config.gradient_accumulation_steps * self.config.num_train_epochs
        test_dataloader = DatasetInstance.encoder_decoder_batch_features(features = test_features,
                                        batch_size = dev_batch_size,
                                        training=False)
        logger.info("***** Running predictions  *****")
        logger.info("  Num examples = %d", len(test_features))
        logger.info("  Batch size = %d", dev_batch_size)
        parameters_state = model.named_parameters()
        optimizer,scheduler = self.create_optimizer(parameters_state= parameters_state, num_training_steps = t_total)
        checkpoints=os.path.join(self.config.model_dir,self.config.corpus,self.config.lang,self.config.model)
        checkpointFiles = glob(os.path.join(checkpoints,"*.pkl"))
        golds = []
        gold_file = os.path.join(self.config.process_path,self.config.corpus,self.config.test_files.split(".")[0]+".ref")
        with open(gold_file,"r" ,encoding="utf-8") as f:
            for line in f:
                line = line.rstrip()
                golds.append(line)
        for fileName in checkpointFiles:
            step = int(re.match(r"[0-9]*", os.path.basename(fileName)).group(0))
            model,optimizer,scheduler = self.load_model(model=model, 
                            optimizer=optimizer,
                            scheduler=scheduler,
                            checkpoint=checkpoints,
                            step=step)
            model.to(self.config.device) 
            if self.config.fp16:
                self.set_fp16(model, optimizer)
            if self.config.n_gpu > 1:
                model = torch.nn.DataParallel(model)
            if self.config.local_rank != -1:
                model = parallel.DistributedDataParallel(model, device_ids=[self.config.local_rank], output_device=self.config.local_rank, find_unused_parameters=True)
            cands= []
            for _,batch in enumerate(test_dataloader):
                model.eval()
                batch_features = {k:v.to(self.config.device) for k, v in batch.items()}
                with torch.no_grad():
                    max_length = batch_features["input_ids"].size(1) + self.config.max_decode_length if self.config.model == "gpt2" else self.config.max_decode_length
                    min_length = batch_features["input_ids"].size(1) + self.config.min_decode_length if self.config.model == "gpt2" else self.config.min_decode_length
                    generate_callback = self.generation_tools(generation_fun=model.module.generate if isinstance(model,DistributedDataParallel) else model.generate,
                                                        max_length = max_length,
                                                        min_length = min_length,
                                                        do_sample = False,
                                                        early_stopping = True,
                                                        num_beams = self.config.beam_size,
                                                        repetition_penalty = self.config.repetition_penalty,
                                                        bos_token_id= DatasetInstance.BOS_ID,
                                                        pad_token_id = DatasetInstance.PAD_ID,
                                                        eos_token_id = DatasetInstance.EOS_ID,
                                                        length_penalty = self.config.length_penalty,
                                                        no_repeat_ngram_size = self.config.no_repeat_ngram_size)
                    if self.config.model == "gpt2":
                        output_sequences = generate_callback(input_ids=batch_features["input_ids"])
                        output_sequences = output_sequences[:,batch_features["input_ids"].size(1):]
                    else:
                        output_sequences = generate_callback(input_ids=batch_features["input_ids"],
                                                        attention_mask=batch_features["attention_mask"])
                    for seq in output_sequences.tolist():
                        decoder_seq = DatasetInstance.tokenizer.decode(seq, skip_special_tokens=True)
                        decoder_seq = self.remove_special_token(DatasetInstance.tokenizer.tokenize(decoder_seq))
                        cands.append(decoder_seq)
            metricsResults = EvaluateDialogue(config=self.config,golds=golds,predictions=cands)
            formatString="the step {} metrics is  ".format(step)
            for _key in metricsResults:
                formatString +=_key
                formatString +=" {} ".format(metricsResults[_key])
            logger.info(formatString)
            predictions = os.path.join(self.config.model_dir,self.config.corpus,self.config.lang,self.config.model,"predictions_%d.json"%(step))
            with open(predictions,"w",encoding="utf-8") as f:
                json.dump(cands,f,ensure_ascii=False,indent=4)
