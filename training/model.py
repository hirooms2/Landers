from dataclasses import dataclass
import logging
from typing import Dict, Optional

import torch
import torch.distributed as dist
from torch import Tensor
from transformers import AutoModel
from transformers.file_utils import ModelOutput
from torch import nn

from gritlm import GritLM

logger = logging.getLogger(__name__)


@dataclass
class GritLMTrainOutput(ModelOutput):
    q_reps: Optional[Tensor] = None
    p_reps: Optional[Tensor] = None
    loss: Optional[Tensor] = None
    loss_emb: Optional[Tensor] = None
    loss_gen: Optional[Tensor] = None


class DistributedContrastiveLoss:
    def __init__(self, temperature: float, negatives_cross_device: bool):
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='mean')
        self.temperature = temperature
        self.negatives_cross_device = negatives_cross_device        
        if self.negatives_cross_device:
            if not dist.is_initialized():
                raise ValueError('Cannot do negatives_cross_device without distributed training')
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()

    def __call__(self, q_reps, p_reps):
        if self.negatives_cross_device:
            # This gathers both negatives and positives.
            # It could likely be optimized by only gathering negatives.
            q_reps = self._dist_gather_tensor(q_reps)
            p_reps = self._dist_gather_tensor(p_reps)
        scores = self.compute_similarity(q_reps, p_reps) / self.temperature
        scores = scores.view(q_reps.size(0), -1)
        
        target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
        target *= (p_reps.size(0) // q_reps.size(0))
        # if len(p_reps.size()) == 2:
        #     target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
        #     target *= (p_reps.size(0) // q_reps.size(0))
        # else:
        #     target = torch.zeros(scores.size(0), device=scores.device, dtype=torch.long)
        return self.cross_entropy(scores, target)

    def _dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None: return None
        t = t.contiguous()

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        # All tensors have the same shape, as pooling already applied to them
        dist.all_gather(all_tensors, t)

        all_tensors[self.rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors

    def compute_similarity(self, q_reps, p_reps):
        if len(p_reps.size()) == 2:
            return torch.matmul(q_reps, p_reps.transpose(0, 1))
        return torch.matmul(q_reps.unsqueeze(1), p_reps.transpose(-2, -1))

class NextTokenLoss:
    def __init__(self, vocab_size: int, loss_gen_type: str = "mixed", loss_gen_factor: float = 1.0):
        super().__init__()
        self.vocab_size = vocab_size
        # How to weight loss:
        # a) Each sample gets the same weight (e.g. used in BLOOMZ https://arxiv.org/abs/2211.01786)
        # -> This leads to shorter generations, as short outputs get the same weight as long ones
        # -> The loss curves for this are unstable if there's noisy or very hard short samples
        # b) Each token gets the same weight
        # -> This leads to longer generations, as long outputs get more weight than short ones
        # b.1) Each token gets the same weight globally across batches
        # -> Just sum the loss as is, optionally divide it by a constant. If using Adam, the scale
        # of the loss doesn't matter, so this is only to balance with another loss like in our case.
        # b.2) Each token gets the same weight per batch
        # -> Divide by the number of tokens in the batch
        # Problem: If distributed training, needs all gather of number of tokens on each process        
        # c) Mix of a) and b) which is what you do if you use the loss in transformers as is and 
        # then do grad acc/multi-gpu with bs>1 
        # (https://github.com/huggingface/transformers/issues/24725; https://github.com/pytorch/pytorch/issues/72047)
        self.loss_gen_factor = loss_gen_factor
        self.loss_gen_type = loss_gen_type
        if loss_gen_type == "token": # b.1)
            self.cross_entropy = torch.nn.CrossEntropyLoss(reduction="sum")
        elif loss_gen_type == "mixed": # c)
            self.cross_entropy = torch.nn.CrossEntropyLoss(reduction="mean")
        else:
            raise ValueError(f"Invalid loss_gen_type: {loss_gen_type}")
        
    def __call__(self, labels, logits):
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        shift_logits = shift_logits.view(-1, self.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        # Normalize by number of non-ignored tokens
        if self.loss_gen_type == "token":
            return (self.cross_entropy(shift_logits, shift_labels) / labels.size(0)) * self.loss_gen_factor
        elif self.loss_gen_type == "mixed":
            return self.cross_entropy(shift_logits, shift_labels) * self.loss_gen_factor


class GritLMTrainModel(GritLM):
    TRANSFORMER_CLS = AutoModel
    def __init__(
        self,
        temperature: float = 1.0,
        negatives_cross_device: bool = False,
        loss_gen_type: str = "mixed",
        loss_gen_factor: float = None,
        num_items: int = 0,
        pooling: str = '',
        **kwargs,
    ):
        super().__init__(**kwargs, is_inference=False)
        self.emb_loss_fn = DistributedContrastiveLoss(temperature, negatives_cross_device)
        self.gen_add_kwargs = {"return_dict": True}
        if "mixtral" in kwargs["model_name_or_path"].lower():
            logger.info("Using token loss with routing loss for mixtral")
            self.gen_loss_fn = None
            self.gen_add_kwargs["loss_gen_factor"] = loss_gen_factor
            self.gen_add_kwargs["output_router_logits"] = True
        else:
            self.gen_loss_fn = NextTokenLoss(
                self.model.config.vocab_size, loss_gen_type, loss_gen_factor
            )
        self.config = self.model.config # Required for accelerate DeepSpeed integration

        self.num_items = num_items
        self.item_proj = nn.Linear(self.model.config.hidden_size, self.num_items)
        
        self.pooling_emb = pooling

    def encode(self, features):
        # print('encode 시작!')
        
        if features is None: return None
        # Clone to avoid modifying the original tensor
        attention_mask = features['attention_mask'].clone() if 'attention_mask' in features else None
        instruction_lens = features['instruction_lens'] if 'instruction_lens' in features else None
        kwargs = {'input_ids': features.get('input_ids'), 'attention_mask': attention_mask}

        if self.attn[:2] == 'cb':
            kwargs['instruction_lens'] = instruction_lens
        elif self.attn[:2] == 'bb':
            kwargs['is_causal'] = False
        out = (getattr(self.model, self.embedding_attr) if self.embedding_attr else self.model)(**kwargs)[0]

        if self.projection is not None:
            out = self.projection(out)
        
        # Mask out the instruction tokens for pooling
        if instruction_lens is not None:
            # Make a new copy of attention mask to prevent in-place problems
            attention_mask = features['attention_mask'].clone()
            # Mask out the instruction tokens for pooling
            for i, l in enumerate(instruction_lens):
                attention_mask[i, :l] = 0
                # Make sure not all zeros - If this happens it is a bug
                assert attention_mask[i].sum() > 0, f"All 0: {attention_mask[i]}, l: {l}"

        reps = self.pooling(out, attention_mask)
        # Normalize can change the dtype (https://discuss.pytorch.org/t/tensor-in-float16-is-transformed-into-float32-after-torch-norm/110891)
        if self.normalized: 
            in_dtype = reps.dtype
            return torch.nn.functional.normalize(reps, dim=-1).contiguous().to(in_dtype)
        return reps.contiguous()

    def forward(
        self,
        query: Dict[str, torch.Tensor] = None,
        passage: Dict[str, torch.Tensor] = None,
        generative: Dict[str, torch.Tensor] = None,
        passages_mask: Dict[str, torch.Tensor] = None,
        q_reps: Optional[torch.Tensor] = None,
        p_reps: Optional[torch.Tensor] = None,
        q_grad: bool = True,
        p_grad: bool = True,
        item_labels = None,
    ):
        """
        Args:
            query: [b, n]
            passage: [b*s, m] where s is group size (usually 2)
            generative: [b, m]
        """
        # Do generative first, as emb contains an all-reduce (verified to be faster)
        if generative is not None:
            if self.gen_loss_fn is not None:
                # This pops the labels first, then the rest is passed into model                
                loss_gen = self.gen_loss_fn(
                    generative.pop('labels'), self.model(**generative, **self.gen_add_kwargs).logits
                )
            else:
                loss_gen = self.model(**generative, **self.gen_add_kwargs).loss
        else:
            loss_gen = None

        
        if (q_reps is None) and (query is not None):
            if q_grad:
                q_reps = self.encode(query)
            else:
                with torch.no_grad():
                    q_reps = self.encode(query)

        if self.num_items == 0: # None linear 실행 분기 
            if (p_reps is None) and (passage is not None):
                if p_grad:
                    p_reps = self.encode(passage)
                else:
                    with torch.no_grad():
                        p_reps = self.encode(passage)    
            
            
            # print('##############################################################################')
            # print(q_reps.size())  
            # print(p_reps.size())  
            
            if self.pooling_emb in ['mean', 'attention']:
                batch_size = q_reps.size(0)
                hidden_size = q_reps.size(-1)
                # B = q_reps.size(0)
                # P = p_reps.size(0) // B # 하나의 쿼리에 포함된 passage 개수 (pos+neg)
                # print('original mask: ', passages_mask)
                p_reps = p_reps * passages_mask.view(-1, 1)  # [B * 5, 4096]
                p_reps = p_reps.view(batch_size, -1, hidden_size)  # [B, 5, 4096]
                # num_features = p_reps.size(1) // 2

                pos_reps = p_reps
                pos_mask = passages_mask  # [B, 5]

                # pos_reps = p_reps[:, :num_features, :]  # [B, 5, 4096]
                # neg_reps = p_reps[:, num_features:, :]  # [B, 5, 4096]
                # pos_mask = passages_mask[:, :num_features]  # [B, 5]
                # neg_mask = passages_mask[:, num_features:]  # [B, 5]

                pos_num = torch.sum(pos_mask, dim=-1, keepdim=True)  # [B, 1]
                # neg_num = torch.sum(neg_mask, dim=-1, keepdim=True)  # [B, 1]

                # Mean (opt1)
                if self.pooling_emb == 'mean':
                    pos_reps = torch.sum(pos_reps, dim=1) / pos_num  # [B, 4096]
                    # neg_reps = torch.sum(neg_reps, dim=1) / neg_num  # [B, 4096]

                # # Attention (opt2)
                elif self.pooling_emb == 'attention':
                    pos_key_mask = pos_mask.unsqueeze(1)   # [B, 1, 5] 
                    pos_query_mask = pos_mask.unsqueeze(2) # [B, 5, 1] 
                    # pos_key = self.linear_key(pos_reps)
                    # pos_query = self.linear_query(pos_reps)
                
                    pos_attention = torch.matmul(pos_reps, pos_reps.transpose(-2, -1))  # [B, 5, 5]
                    pos_attention = pos_attention.masked_fill(~pos_key_mask.bool(), float('-inf'))
                    pos_attention = torch.softmax(pos_attention, dim=-1)
                    pos_reps = torch.matmul(pos_attention, pos_reps) # * pos_query_mask  # [B, 5, d]
                    
                    pos_reps = (pos_reps * pos_query_mask).sum(dim=1) / pos_query_mask.sum(dim=1)

                    # neg_key_mask = neg_mask.unsqueeze(1)   # [B, 1, 5]
                    # neg_query_mask = neg_mask.unsqueeze(2) # [B, 5, 1] 
                    
                    # neg_attention = torch.matmul(neg_reps, neg_reps.transpose(-2, -1))  # [B, 5, 5]
                    # neg_attention = neg_attention.masked_fill(~neg_key_mask, float('-inf'))
                    # neg_attention = torch.softmax(neg_attention, dim=-1)

                    # neg_reps = torch.matmul(neg_attention, neg_reps) * neg_query_mask  # [B, 5, d]

                # Concat
                p_reps = pos_reps  # [B, 4096]
                # p_reps = torch.cat([pos_reps.unsqueeze(1), neg_reps.unsqueeze(1)], dim=1)  # [B, 2, 4096]
      
            loss_emb = self.emb_loss_fn(
                q_reps, p_reps
            ) if (q_reps is not None and p_reps is not None) else None
        else:
            logits = self.item_proj(q_reps) # [B, I]
            loss_emb = nn.CrossEntropyLoss()(logits, item_labels)
            
        loss = sum([x for x in [loss_emb, loss_gen] if x is not None])

        # Also return q_reps in case of GradCache
        return GritLMTrainOutput(
            q_reps=q_reps,
            p_reps=p_reps,
            loss=loss,
            loss_emb=loss_emb,
            loss_gen=loss_gen,
        )

    def gradient_checkpointing_enable(self, *args, **kwargs):
        self.model.gradient_checkpointing_enable(*args, **kwargs)
