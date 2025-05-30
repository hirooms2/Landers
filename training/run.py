import logging
import json
import multiprocessing
import os
from pathlib import Path
import random
import datasets
import torch
import torch.distributed as dist
import numpy as np
from tqdm import tqdm

from pytz import timezone
from datetime import datetime
from dataclasses import dataclass, field

from transformers import AutoConfig, AutoTokenizer, HfArgumentParser, Trainer, set_seed, TrainingArguments, TrainerState, TrainerControl, TrainerCallback

from .arguments import CustomTrainingArguments, DataArguments, ModelArguments
from .data import CustomCollator, CustomDataset, CustomRandomSampler
from .model import GritLMTrainModel



def gritlm_instruction(instruction):
    return "<|user|>\n" + instruction + "\n<|embed|>\n" if instruction else "<|embed|>\n"

BASE_BOS: str = "<s>"
TURN_SEP: str = "\n"

USER_BOS: str = "<|user|>\n"
USER_EOS: str = "" # "</s>" for Zephyr format

EMBED_BOS: str = "\n<|embed|>\n"
# Am embed eos is useless as there is no generative loss on it so it won't be learned
# & it does not add anything new; It only makes sense for lasttoken pooling
EMBED_EOS: str = ""

ASSISTANT_BOS: str = "\n<|assistant|>\n"
ASSISTANT_EOS: str = "</s>"

logger = logging.getLogger(__name__)

from platform import system as sysChecker
if sysChecker() == 'Linux':
    home = os.path.dirname(__file__)
elif sysChecker() == "Windows":
    home = ''

# class QueryEvalCallback(TrainerCallback):
#     def __init__(self, output_dir):
#         self.saved_model_path = output_dir

#     def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
#         model = kwargs['model'].model
#         epoch = state.epoch
#         path = os.path.join(self.saved_model_path, 'E' + str(int(epoch)))
#         logger.info(f'Start saving epoch: {epoch}')   
#         model.save_pretrained(path)
# def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
#     to_return = {k: t for k, t in named_params if "lora_" not in k}
#     if require_grad_only:
#         to_return = {k: t for k, t in to_return.items() if t.requires_grad}
#     to_return = {k: maybe_zero_3(v) for k, v in to_return.items()}
#     return to_return


class QueryEvalCallback(TrainerCallback):
    def __init__(self, output_dir):
        self.saved_model_path = output_dir

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        wrapper_model = kwargs['model']  # 전체 wrapper 모델
        peft_model = wrapper_model.model  # PEFT 모델 내부
        epoch = state.epoch
        path = os.path.join(self.saved_model_path, f'E{round(epoch)}')

        logger.info(f'Start saving epoch: {epoch}')

        # 1. PEFT LoRA 파라미터만 저장
        peft_model.save_pretrained(path)

        # 2. Custom layer 포함한 나머지 파라미터 저장
        # → wrapper_model 전체 기준에서 PEFT가 아닌 것만 추출
        # 전체 모델의 state_dict에서 'lora_'가 포함되지 않은 파라미터만 선택
        non_lora_state_dict = {
            k: v.cpu()
            for k, v in wrapper_model.named_parameters()
            if "lora_" not in k and v.requires_grad
        }
        torch.save(non_lora_state_dict, os.path.join(path, 'non_lora_trainables.bin'))

        # 3. config도 같이 저장
        peft_model.config.save_pretrained(path)


def args_to_dtype(args):
    if args.bf16: return torch.bfloat16
    if args.fp16: return torch.float16
    return torch.float32


def filter_too_long_instructions(tokenizer, dataset, query_max_len, passage_max_len):
    def filter_fn(example):
        # Filter out super long examples to avoid tokenize taking forever
        if (len(example["query"][0]) > query_max_len * 10) or not(example["query"][1]):
            return False
        if len(tokenizer.tokenize(BASE_BOS + USER_BOS + example["query"][0].strip("\t\n :") + USER_EOS + EMBED_BOS)) >= query_max_len:
            return False
        examples = example["pos"] + example["neg"] if "neg" in example else example["pos"]
        for ex in examples:
            if (len(ex[0]) > passage_max_len * 10) or not(ex[1]):
                return False
            if len(tokenizer.tokenize(BASE_BOS + USER_BOS + ex[0].strip("\t\n :") + USER_EOS + EMBED_BOS)) >= passage_max_len:
                return False
        return True
    num_proc = max(multiprocessing.cpu_count()-2, 1) if len(dataset) > 5000 else 1
    return dataset.filter(filter_fn, num_proc=num_proc, load_from_cache_file=True)

@dataclass
class myArgument:
    db_json: str = field(default='')
    target_model_path: str = field(default='')
    home: str = field(default=os.path.dirname(__file__))
    parent_home: str = field(default=os.path.dirname(os.path.dirname(__file__)))
    linear: bool = field(default=False)
    linear_initialize: bool = field(default=False)
    only_linear: bool = field(default=False)
    pooling: str = field(default='')

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, CustomTrainingArguments, myArgument))
    model_args, data_args, training_args, my_args = parser.parse_args_into_dataclasses()
    mdhm = str(datetime.now(timezone('Asia/Seoul')).strftime('%m%d%H%M%S'))
    training_args.output_dir = os.path.join('model_weights', training_args.output_dir, mdhm)
    training_args.save_strategy="no"
    if my_args.target_model_path:
        resume_from_checkpoint = os.path.join(my_args.parent_home, 'model_weights', my_args.target_model_path)
    else:
        resume_from_checkpoint = ''
        
    # db_path = os.path.join(my_args.home, 'crs_data', my_args.db_json)
    # title2feature = json.load(open(db_path, 'r', encoding='utf-8'))
    # # documents = list(title2feature.values())
    # # documents = [doc[:512 * 10] for doc in documents]
    # if data_args.only_title:
    #     feature2idx = {k: idx for idx, (k, v) in enumerate(title2feature.items())}
    #     documents = list(title2feature.keys())
    #     documents = [doc[:512 * 10] for doc in documents]
    # else:
    #     feature2idx = {v: idx for idx, (k, v) in enumerate(title2feature.items())}
    #     documents = list(title2feature.values())
    #     documents = [doc[:512 * 10] for doc in documents]

    # all_items = list(db.keys())
    # num_items = len(all_items)

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to bypass."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )

    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}

    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("Model parameters %s", model_args)
    logger.info("Data parameters %s", data_args)

    # Set seed
    set_seed(training_args.seed)

    # If embedding/unified, handle grad accumulation manually inside forward of GradCacheTrainer.
    gc_chunk_size = None
    if ((training_args.gradient_accumulation_steps > 1) and \
        (training_args.negatives_cross_device) and \
        (training_args.mode in ["embedding", "unified"])) or \
        (training_args.no_gen_gas and training_args.no_emb_gas):
        gc_chunk_size = training_args.per_device_train_batch_size
        training_args.per_device_train_batch_size = \
            training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
        training_args.gradient_accumulation_steps = 4

        logger.info("Using GradCache with chunk size %d", gc_chunk_size)
    elif (training_args.no_gen_gas or training_args.no_emb_gas):
        raise ValueError("Cannot use no_gen_gas or no_emb_gas without GradCache")

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        padding_side="right", # Has to be right so masking of instruction tokens works correctly
    )
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=1,
    )
    logger.info('Config: %s', config)
    
    if not(tokenizer.pad_token) and tokenizer.bos_token:
        tokenizer.pad_token = tokenizer.bos_token
        logger.info('Set pad token to bos token: %s', tokenizer.pad_token)   

    data_files = [os.path.join(data_args.train_data, x) for x in os.listdir(data_args.train_data)] if \
        os.path.isdir(data_args.train_data) else [data_args.train_data]
    train_ds, ds_embedding_lens = [], []
    
    num_samples = None
    if data_args.num_samples:
        with open(data_args.num_samples, "r") as f:
            num_samples = json.load(f)
    
    ds_name_to_samples = {}

    if data_args.generative_max_len is None:
        data_args.generative_max_len = data_args.passage_max_len
    
    for file in data_files:
        logger.info("Loading dataset %s", file)
        tmp_ds = datasets.load_dataset('json', data_files=file, split='train')
        tmp_ds_len = len(tmp_ds)
        # For testing, can add an origin column:
        # origin_col = [file] * len(tmp_ds)
        # tmp_ds = tmp_ds.add_column("origin", origin_col)
        if tmp_ds_len > data_args.max_example_num_per_dataset:
            tmp_ds = tmp_ds.select(
                random.sample(list(range(tmp_ds_len)), data_args.max_example_num_per_dataset)
            )
        # Check if has instructions separated such that they will be masked out later
        # If so filter out samples where the instructions are too long else they will all be 0s
        if training_args.mode in ["embedding", "unified"] and "query" in tmp_ds.features:
            if isinstance(tmp_ds[0]['query'], (tuple, list)):
                logger.info(f"Filtering out embedding samples with too long instructions for {file}")
                tmp_ds = filter_too_long_instructions(
                    tokenizer,
                    tmp_ds,
                    data_args.query_max_len,
                    data_args.passage_max_len,
                )
                if num_samples:
                    assert file.split("/")[-1] in num_samples, f'Missing num_samples for {file.split("/")[-1]}'
                    tmp_ds_len = len(tmp_ds)
                    samples = num_samples[file.split("/")[-1]]
                    if tmp_ds_len > samples:                    
                        tmp_ds = tmp_ds.select(random.sample(list(range(tmp_ds_len)), samples))
            ds_name_to_samples[file.split("/")[-1]] = len(tmp_ds)
            train_ds.append(tmp_ds)
            continue
        if training_args.mode in ["unified", "generative"] and "text" in tmp_ds.features:
            if isinstance(tmp_ds[0]['text'], (tuple, list)):
                logger.info(f"Filtering out generative samples with too long instructions for {file}")
                # Use passage_max_len, as this is the seq len limit for the entire generative snippet
                num_proc = max(multiprocessing.cpu_count()-2, 1) if tmp_ds_len > 5000 else 1
                tmp_ds = tmp_ds.filter(
                    lambda ex: len(tokenizer.tokenize(USER_BOS + ex["text"][0] + USER_EOS + ASSISTANT_BOS)) < data_args.generative_max_len,
                    num_proc=num_proc,
                    load_from_cache_file=True,
                )
            ds_name_to_samples[file.split("/")[-1]] = len(tmp_ds)
            train_ds.append(tmp_ds)
            continue
        logger.info("Skipping dataset %s as its type could not be identified", file)
    if training_args.mode == "embedding":
        ds_embedding_lens = [len(t) for t in train_ds]
        ds = datasets.concatenate_datasets(train_ds)
        logger.info("Embedding mode: %d samples", len(ds))
    elif training_args.mode == "generative":
        ds = datasets.concatenate_datasets(train_ds)
        logger.info("Generative mode: %d samples", len(ds))
    elif training_args.mode == "unified":
        ds_embedding = datasets.concatenate_datasets([
            t for t in train_ds if "query" in t.features
        ])
        ds_generative = datasets.concatenate_datasets([
            t for t in train_ds if "text" in t.features
        ])
        logger.info("Unified mode: %d embedding samples, %d generative samples",
            len(ds_embedding), len(ds_generative)
        )
        for t in train_ds:
            if "query" in t.features:
                num_samples = len(t)
                ds_embedding_lens.append(num_samples)
        ds = [ds_embedding, ds_generative]
    else:
        raise NotImplementedError(training_args.mode)

    os.makedirs(training_args.output_dir, exist_ok=True)
    with open(os.path.join(training_args.output_dir, "dataset_num_samples.json"), "w") as f:
        json.dump(ds_name_to_samples, f)

    if training_args.per_device_generative_bs is not None:
        assert training_args.mode == "unified", "Generative batch size is only supported in unified mode"
        assert training_args.per_device_generative_bs < training_args.per_device_train_batch_size, \
            "Generative batch size must be smaller than regular batch size"
        logger.info("Using generative batch size %d per device", training_args.per_device_generative_bs)


    quantization_config, load_in_4bit = None, False
    if training_args.qlora:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    model = GritLMTrainModel(
        model_name_or_path=model_args.model_name_or_path,
        normalized=model_args.normalized,
        pooling_method=model_args.pooling_method,
        negatives_cross_device=training_args.negatives_cross_device,
        temperature=training_args.temperature,
        mode=training_args.mode,
        projection=model_args.projection,
        attn=model_args.attn,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=args_to_dtype(training_args),
        loss_gen_type=training_args.loss_gen_type,
        loss_gen_factor=training_args.loss_gen_factor,
        use_cache=False,
        # Critical to make Mixtral work
        # low_cpu_mem_usage=True,
        quantization_config=quantization_config,
        load_in_4bit=load_in_4bit,
        num_items = len(data_files) if my_args.linear else 0,
        pooling=my_args.pooling
    )
    # Add special token for embed
    if model_args.pooling_method == "lasttoken":
        embed_eos = "</e>"
        if embed_eos in tokenizer.vocab:
            logger.info("Embed eos token already in vocab: %s", embed_eos)
        else:
            logger.info("Adding embed eos token to vocab: %s", embed_eos)
            tokenizer.add_tokens([embed_eos], special_tokens=True)
            model.model.resize_token_embeddings(len(tokenizer))
        config.num_vocab += 1
    else:
        embed_eos = EMBED_EOS

    if os.getenv("BIDIRECTIONAL_ATTN", False):
        if hasattr(model.model, "model"):
            model.model.model.padding_idx = tokenizer.pad_token_id
        else:
            model.model.padding_idx = tokenizer.pad_token_id

    if (training_args.lora) or (training_args.qlora):
        if training_args.qlora:
            from peft import prepare_model_for_kbit_training
            model.model = prepare_model_for_kbit_training(
                model.model, use_gradient_checkpointing=training_args.gradient_checkpointing
            )

        from peft import get_peft_model, LoraConfig, TaskType, set_peft_model_state_dict
        # https://github.com/texttron/tevatron/blob/2e5d00ee21d5a7db0bd2ea1463c9150a572106d4/examples/repllama/repllama.py#L81
        # https://github.com/allenai/open-instruct/blob/9ebcb582cfc243a6dab75b4302fa432784db26c2/open_instruct/finetune.py#L478
        peft_config = LoraConfig(
            # task_type=TaskType.CAUSAL_LM,
            inference_mode=False, 
            r=16, 
            lora_alpha=64,
            lora_dropout=0.1,
            target_modules=["q_proj", "o_proj", "v_proj", "k_proj", "w1", "w2", "w3"]
        )
        model.model.enable_input_require_grads()
        model.model = get_peft_model(model.model, peft_config)
        model.model.print_trainable_parameters()

        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         print(name, param.shape)

        if resume_from_checkpoint != "":
            # Check the available weights and load them
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "pytorch_model.bin"
            )  # Full checkpoint
            if not os.path.exists(checkpoint_name):
                checkpoint_name = os.path.join(
                    resume_from_checkpoint, "adapter_model.bin"
                )  # only LoRA model - LoRA config above has to fit
                resume_from_checkpoint = (
                    False  # So the trainer won't try loading its state
                )
            # The two files above have a different name depending on how they were saved, but are actually the same.
            if os.path.exists(checkpoint_name):
                print(f"Restarting from {checkpoint_name}")
                # DDP 환경에서 현재 rank에 해당하는 GPU에 로드
                local_rank = int(os.environ.get("LOCAL_RANK", 0))  # torchrun이 자동 설정함
                device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
                adapters_weights = torch.load(checkpoint_name, map_location=device)
                set_peft_model_state_dict(model.model, adapters_weights)
            else:
                print(f"Checkpoint {checkpoint_name} not found")
        else:
            resume_from_checkpoint = None

    if my_args.only_linear:
        for param in model.model.parameters():
            param.requires_grad = False
        model.model.print_trainable_parameters()

    
    db_path = os.path.join(my_args.home, 'crs_data', my_args.db_json)
    print("passage DB loading: ", db_path)
    # print(my_args.db_json, my_args.pooling)
    db_list = None
    if my_args.db_json and my_args.pooling in ['mean', 'attention']:
        db = json.load(open(db_path, 'r', encoding='utf-8'))
        db_list = [data for data in db.values()]
        # print("passage db loading complete", len(db_list), type(db_list))
    
    
    train_dataset = CustomDataset(
        ds,
        args=data_args,
        tokenizer=tokenizer,
        mode=training_args.mode,
        full_bs=training_args.per_device_train_batch_size  ,
        generative_bs=training_args.per_device_generative_bs,
        max_seq_len=max(data_args.query_max_len, data_args.passage_max_len, data_args.generative_max_len),
        pooling = my_args.pooling,
        item_db = db_list if db_list else None
    )

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "data_collator": CustomCollator(
            tokenizer,
            query_max_len=data_args.query_max_len,
            passage_max_len=data_args.passage_max_len,
            generative_max_len=data_args.generative_max_len,
            base_bos=BASE_BOS,
            turn_sep=TURN_SEP,
            user_bos=USER_BOS,
            user_eos=USER_EOS,
            embed_bos=EMBED_BOS,
            embed_eos=embed_eos,
            assistant_bos=ASSISTANT_BOS,
            assistant_eos=ASSISTANT_EOS,
            prefixlm=data_args.prefixlm,
            only_title=data_args.only_title
        ),
        "tokenizer": tokenizer,
        "callbacks": [QueryEvalCallback(training_args.output_dir)]

    }

    if gc_chunk_size is not None:
        from .gradcache_trainer import GradCacheTrainer
        trainer = GradCacheTrainer(**trainer_kwargs)
        trainer.gc_chunk_size = gc_chunk_size
        trainer.emb_loss_fn = model.emb_loss_fn
        trainer.mode = training_args.mode
        trainer.no_gen_gas = training_args.no_gen_gas
        trainer.no_emb_gas = training_args.no_emb_gas
        trainer.split_emb = training_args.split_emb
        trainer.split_emb_full = training_args.split_emb_full
        trainer.emb_p_only = training_args.emb_p_only
        trainer.emb_q_only = training_args.emb_q_only
    else:
        trainer = Trainer(**trainer_kwargs)

    if len(ds_embedding_lens) > 1:
        assert training_args.dataloader_drop_last, "Multiple datasets are only supported with dropping the last incomplete batch, set `--dataloader_drop_last`"
        logger.info("Embedding dataset lengths: %s", ds_embedding_lens)
        # Multiple embedding datasets & we want to make sure each batch mostly comes from one dataset
        # Set custom sampler, see https://github.com/huggingface/transformers/blob/ccb92be23def445f2afdea94c31286f84b89eb5b/src/transformers/trainer.py#L785
        total_bs = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
        total_bs = total_bs * dist.get_world_size() if dist.is_initialized() else total_bs
        trainer._get_train_sampler = lambda: CustomRandomSampler(
            total_batch_size=total_bs, ds_lens=ds_embedding_lens,
            _num_samples=sum(ds_embedding_lens), data_source=train_dataset,
        )

    if training_args.mode == "unified":
        # Track all losses
        from transformers.integrations import WandbCallback
        from transformers.integrations.integration_utils import rewrite_logs
        from transformers.trainer_pt_utils import distributed_concat
        class WandbCustomCallback(WandbCallback):
            def on_log(self, args, state, control, model=None, logs=None, **kwargs):
                if self._wandb is None: return
                if not self._initialized: self.setup(args, state, model)
                if hasattr(state, "loss_emb") and hasattr(state, "loss_gen"):
                    # Gather & avg across gpus like for actual loss
                    # https://github.com/huggingface/transformers/blob/bc72b4e2cdcbc80d5f56731f35dbc9c18b4c8de6/src/transformers/trainer.py#L2257
                    if (args.distributed_state is not None and args.distributed_state.distributed_type != "NO") or (
                        args.distributed_state is None and args.local_rank != -1):
                        state.loss_emb = distributed_concat(state.loss_emb).mean().item()
                        state.loss_gen = distributed_concat(state.loss_gen).mean().item()
                    else:
                        state.loss_emb = state.loss_emb.mean().item()
                        state.loss_gen = state.loss_gen.mean().item()
                    if state.is_world_process_zero:
                        self._wandb.log({
                            **rewrite_logs(logs),
                            "train/global_step": state.global_step,
                            "train/loss_emb": state.loss_emb,
                            "train/loss_gen": state.loss_gen,
                        })
                    del state.loss_emb
                    del state.loss_gen
                else:
                    if state.is_world_process_zero:
                        self._wandb.log({
                            **rewrite_logs(logs),
                            "train/global_step": state.global_step,
                        })

        trainer.add_callback(WandbCustomCallback())

        # Copied from below & added loss_emb/loss_gen
        # https://github.com/huggingface/transformers/blob/cc3e4781854a52cf090ffde28d884a527dab6708/src/transformers/trainer.py#L2699
        def training_step(self, model, inputs):
            model.train()
            inputs = self._prepare_inputs(inputs)

            with self.compute_loss_context_manager():
                out = self.compute_loss(model, inputs, return_outputs=True)
                loss = out[0]
                loss_emb = out[1]["loss_emb"]
                loss_gen = out[1]["loss_gen"]

            if self.args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
                loss_emb = loss_emb.mean()
                loss_gen = loss_gen.mean()

            if self.use_apex:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                self.accelerator.backward(loss) # Includes normalizing by gas

            self.state.loss_emb = getattr(self.state, "loss_emb", torch.tensor(0.0).to(loss.device))
            self.state.loss_gen = getattr(self.state, "loss_gen", torch.tensor(0.0).to(loss.device))
            self.state.loss_emb += loss_emb.detach() / self.args.gradient_accumulation_steps
            self.state.loss_gen += loss_gen.detach() / self.args.gradient_accumulation_steps
            
            return loss.detach() / self.args.gradient_accumulation_steps

        # __get__ is needed to bind the method to the Trainer instance
        trainer.training_step = training_step.__get__(trainer)

    Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)

    if my_args.linear_initialize:
        d_rep= []
        for i in tqdm(range(0, len(documents), 64)):
            batch_documents = documents[i: i + 64]
            batch_documents = [
                f'<s><|user|>\nRepresent the topic for retrieval\n<|embed|>\n{f}' for f in batch_documents
            ]
            passage = tokenizer(
                batch_documents,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
                add_special_tokens=False, # BOS / EOS is already in the prompt
            ).to('cuda')
            with torch.no_grad():
                rep = model.encode(passage).cpu()
                d_rep.append(rep)
        d_rep=np.concatenate(d_rep, axis=0)
        print('document shape:',torch.from_numpy(d_rep).shape)

        # d_rep: numpy array, shape = (num_items, hidden_size)
        d_tensor = torch.from_numpy(d_rep).to(model.item_proj.weight.device)

        # weight 초기화 (학습 가능 상태 유지)
        with torch.no_grad():
            model.item_proj.weight.copy_(d_tensor)

    # Training
    logger.info("Starting training")
    trainer.train()
    
    # The below does not save if state dict type is `SHARDED_STATE_DICT`
    # trainer.save_model()

    # To be safe do another FS save
    if (trainer.is_fsdp_enabled) and (trainer.accelerator.state.fsdp_plugin.state_dict_type != "FULL_STATE_DICT"):
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
        fsd_path = os.path.join(training_args.output_dir, "full_state_dict")
        os.makedirs(fsd_path, exist_ok=True)
        trainer.save_model(fsd_path)

    # Save tokenizer & config for easy usage afterwards
    if trainer.is_world_process_zero(): 
        tokenizer.save_pretrained(training_args.output_dir)
        config.to_json_file(training_args.output_dir + "/config.json")

if __name__ == "__main__":
    main()
