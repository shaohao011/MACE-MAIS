from torch import nn
import torch
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import AutoModel,AutoTokenizer
from .attention_pooling import GatedAttention

def filter_encoder_inputs(input_dict):
    input_dict = dict(input_dict)
    input_dict.pop("decoder_input_ids", None)
    allowed_keys = {
        "input_ids",
        "attention_mask",
        "inputs_embeds",
        "head_mask",
        "output_attentions",
        "output_hidden_states",
        "return_dict"
    }
    return {k: v for k, v in input_dict.items() if k in allowed_keys}

class SinPositionEncoding(nn.Module):
    def __init__(self, max_sequence_length, d_model, base=10000):
        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.d_model = d_model
        self.base = base

    def forward(self):
        pe = torch.zeros(self.max_sequence_length, self.d_model, dtype=torch.float)  # size(max_sequence_length, d_model)
        exp_1 = torch.arange(self.d_model // 2, dtype=torch.float)  
        exp_value = exp_1 / (self.d_model / 2)

        alpha = 1 / (self.base ** exp_value)  # size(dmodel/2)
        out = torch.arange(self.max_sequence_length, dtype=torch.float)[:, None] @ alpha[None, :]  # size(max_sequence_length, d_model/2)
        embedding_sin = torch.sin(out)
        embedding_cos = torch.cos(out)

        pe[:, 0::2] = embedding_sin  
        pe[:, 1::2] = embedding_cos  
        return pe
    
class BatchFormer(nn.Module):
    def __init__(self, image_dim, text_dim, attention_dim):
        super(BatchFormer, self).__init__()
        self.image_dim = image_dim
        self.text_dim = text_dim
        self.attention_dim = attention_dim
        self.image_projection = nn.Linear(image_dim, attention_dim)
        
        self.text_projection = nn.Linear(text_dim, attention_dim)

        self.q_linear = nn.Linear(attention_dim, attention_dim)
        self.k_linear = nn.Linear(attention_dim, attention_dim)
        self.v_linear = nn.Linear(attention_dim, attention_dim)

        self.output_linear = nn.Linear(attention_dim, text_dim)

    def forward(self, image_features, text_features):
        if image_features.shape[1] == 1:
            image_features = image_features.squeeze(1)
            text_features = text_features.squeeze(1)
        image_proj = self.image_projection(image_features)  
        text_proj = self.text_projection(text_features)    
        
        Q = self.q_linear(image_proj) 
        K = self.k_linear(text_proj)  
        V = self.v_linear(text_proj)  
        
        attention_scores = torch.matmul(Q, K.transpose(1, 0)) / (self.attention_dim ** 0.5)  # [B, B]
        attention_probs = F.softmax(attention_scores, dim=-1)  #
        
        context = torch.matmul(attention_probs, V)  #

        output = self.output_linear(context)  # 

        return output
    
class MacePredictMMF(nn.Module):
    def __init__(self,args,in_channels=5,out_channels=2,model_index=-1) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_index = model_index
        self.llm_name = args.llm_name
        self.attention_dim = 1024
        self.use_text_inputs = args.use_text_inputs
        self.pure_text_inputs = args.pure_text_inputs
        self.device = args.device
        
        self.img_dim = 512
        
        if args.llm_name == "clinicalBERT":
            self.text_model  = AutoModel.from_pretrained("medicalai/ClinicalBERT")
            self.llm_tokenizer = AutoTokenizer.from_pretrained("medicalai/ClinicalBERT")
            self.text_dim=768
        elif args.llm_name == "T5-3B":
            t5_name="t5-base"
            # t5_name="t5-3b"
            args.tokenizer_path = f"llms/google-t5/{t5_name}"
            self.llm_tokenizer = T5Tokenizer.from_pretrained(f"llms/google-t5/{t5_name}")
            self.text_model  = T5ForConditionalGeneration.from_pretrained(f"llms/google-t5/{t5_name}")
            # self.text_dim = 512
            if t5_name=="t5-base":
                self.text_dim = 768
            elif t5_name == "t5-3b":
                self.text_dim = 1024
            else:
                raise ValueError
                
        self.bf = BatchFormer(image_dim=self.text_dim, text_dim=self.text_dim, attention_dim=self.attention_dim)
        self.cls_head = nn.Linear(self.text_dim,self.out_channels)
        self.proj = nn.Linear(self.img_dim,self.text_dim)
        self.gated_attention = GatedAttention(in_dim=self.text_dim,embed_dim=512)
        self.pos_embed = SinPositionEncoding(max_sequence_length=len(args.text_inputs_keys)+1,d_model=self.text_dim, base=10000).forward().to(args.device)
        self.bn = nn.BatchNorm1d(num_features=self.text_dim)
    
    def forward(self,x,text_input=None,keys=[],training=True):
        if text_input!=None:
            text_features_list = []
            for llm_key in keys:
                encoder_outputs = self.text_model.encoder(**text_input[llm_key])
                last_hidden_state = encoder_outputs.last_hidden_state  # [B, seq_len, hidden_dim]
                attention_mask = text_input[llm_key]["attention_mask"]  
                expanded_mask = attention_mask.unsqueeze(-1).float()
                masked_hidden = last_hidden_state * expanded_mask  # [B, seq_len, hidden_dim]
                sum_hidden = masked_hidden.sum(dim=1)              # [B, hidden_dim]
                sum_mask = expanded_mask.sum(dim=1)                # [B, 1]
                mean_hidden = sum_hidden / (sum_mask + 1e-9)       # [B, hidden_dim]
                text_features_list.append(mean_hidden.unsqueeze(1))  # [B, 1, hidden_dim]
            
            if self.pure_text_inputs:
                pass
            else:
                x = x.to(self.device)
                x = self.proj(x) # 8, 1, 768
                # x = self.bn(x.squeeze(1)).unsqueeze(1)
                text_features_list.insert(0,x)
                
            B = text_features_list[0].shape[0]
            x,score = self.gated_attention(torch.cat(text_features_list,dim=1)) # B 1 1024
            if x.shape[1] == 1:
                x = x.squeeze(1)
        
            x = self.bf(x,x)
        else:
            x = x.to(self.device)
            x = self.proj(x)
            x = self.bf(x,x)
            score = None
        x = self.cls_head(x)
        return x,score

if __name__=="__main__":
    data = torch.randn((3,6,336,336))
    B = 32
    image_features = torch.randn(B, 1024)
    text_features = torch.randn(B, 512)
    cross_attention = CrossAttention(image_dim=1024, text_dim=512, attention_dim=256)

    fused_features = cross_attention(image_features, text_features)

    print(fused_features.shape)  
    
        
