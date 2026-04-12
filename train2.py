import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import T5Tokenizer, T5ForConditionalGeneration, get_linear_schedule_with_warmup
import random
import numpy as np
from tqdm import tqdm
import json
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE 
from peft import LoraConfig, get_peft_model, TaskType

import copy
import os
from collections import defaultdict


from matplotlib import font_manager

# 添加你刚上传的字体路径（文件名按你的实际来）
font_path = "times.ttf"

font_manager.fontManager.addfont(font_path)
prop = font_manager.FontProperties(fname=font_path)

# 设置为全局默认字体
plt.rcParams['font.family'] = prop.get_name()



# =========================
# 设置随机种子
# =========================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

# =========================
# 1. 高级渐进式提示模板
# =========================
# class AdvancedPromptManager:
#     def __init__(self):
#         # 多层次渐进式提示设计
#         self.prompts = {
#             # Stage 0: 源域通用提示（任务理解）
#             # 'stage_0': {
#             #     'task_description': "Extract triplets of (aspect_category, opinion, sentiment) from the text.",
#             #     'few_shot_examples': [
#             #         "Input: Build quality seems ok , keyboard is not flimsy or too firm . Output: (LAPTOP#QUALITY, ok, positive) |  (KEYBOARD#GENERAL, not flimsy or too firm, positive)",
#             #         "Input: I like the ease of connecting to the internet wifi . Output: (LAPTOP#CONNECTIVITY, ease, positive)"
#             #     ],
#             #     'instruction': f"Given a laptop review, identify all aspect_category-opinion-sentiment triplets:"
#             # },
#             'stage_0': {
#                 'task_description': "Extract triplets of (aspect_category, opinion, sentiment) from the text.",
#                 'few_shot_examples': [
#                     "Input: The food was delicious and the service was excellent. Output: (FOOD#QUALITY, delicious, positive) | (SERVICE#GENERAL, excellent, positive)",
#                     "Input: The atmosphere is cozy but the prices are too high. Output: (AMBIENCE#GENERAL, cozy, positive) | (RESTAURANT#PRICES, too high, negative)",
#                     "Input: Great pasta, but the wait time was unacceptable. Output: (FOOD#QUALITY, great, positive) | (SERVICE#GENERAL, unacceptable, negative)"
#                 ],
#                 'instruction': f"Given a restaurant review, identify all aspect_category-opinion-sentiment triplets:"
#             },

            
            
#             # Stage 1: 领域适应提示（概念映射）
#             # 'stage_1': {
#             #     'task_description': "Extract triplets of (symptom_category, subjective_description, severity_level) from mental health text.",
#             #     'domain_mapping': "Map laptop aspects to symptom categories: Battery life-> Energy loss, Performance and design->Interest loss, Display/Screen->Concentration",
#             #     'few_shot_examples': [
#             #         "Input: I feel extremely tired and lack motivation. Output: (Energy loss, tired, negative) | (Energy loss, lack motivation, negative)",
#             #         "Input: Sleep is okay but mood is low. Output: (Sleep, sleep is okay, positive) | (Depressed mood, mood is low, negative)"
#             #     ],
#             #     'instruction': "Given a depression-related text, extract symptom triplets with severity (positive=mild or asymptomatic, neutral=moderate, negative=severe):"
#             # },
#             'stage_1': {
#                 'task_description': "Extract triplets of (symptom_category, subjective_description, severity_level) from mental health text.",
#                 'domain_mapping': "Map restaurant aspects to symptom categories: FOOD#QUALITY -> Appetite problem, SERVICE#GENERAL -> Concentration, DRINKS#QUALITY -> Sleep","RESTAURANT#GENERAL -> Interest"
#                 'few_shot_examples': [
#                     "Input: I feel extremely tired and lack motivation. Output: (Energy loss, tired, negative) | (Energy loss, lack motivation, negative)",
#                     "Input: Sleep is okay but mood is low. Output: (Sleep, sleep is okay, positive) | (Depressed mood, mood is low, negative)",
#                     "Input: I have no appetite and feel anxious in crowded places. Output: (Appetite changes, no appetite, negative) | (Depressed mood, anxious in crowded places, negative)"
#                 ],
#                 'instruction': "Given a depression-related text, extract symptom triplets with severity (positive=mild or asymptomatic, neutral=moderate, negative=severe):"
#             },

            
            
            
            
#             # Stage 2: 目标域专家提示（细粒度理解）
#             'stage_2': {
#                 'task_description': "Extract depression symptom triplets with clinical precision.",
#                 'clinical_guidelines': "Severity levels: positive(mild/no symptom), neutral(moderate), negative(severe). Categories: Appetite problem, Depressed mood, Cognitive problem, Interest loss, Congnitive problem, Sleep, Self-evaluation, Retardation, Suicidal ideation, Weight problem, Psychomotor.",
#                 'few_shot_examples': [
#                     "Input: Pandemic burnout and anxiety affect students. Output: (Psychomotor, pandemic anxiety, neutral)",
#                     "Input: Severe insomnia and complete loss of appetite. Output: (Sleep, severe insomnia, negative) | (Appetite problem, loss of appetite, negative)"
#                 ],
#                 'instruction': "Extract all depression symptom triplets from the clinical text:"
#             },
            
#             # Stage 3: 元学习微调提示（上下文学习）
#             'stage_3': {
#                 'task_description': "Few-shot adaptation for depression symptom extraction.",
#                 'meta_instruction': "Learn from the following examples and extract triplets from new text:",
#                 'adaptive_hint': "Pay attention to subtle expressions of depression symptoms and severity indicators."
#             }
#         }
    
#     def get_prompt(self, stage, include_examples=True):
#         """构造多层次提示"""
#         stage_key = f'stage_{stage}'
#         prompt_config = self.prompts[stage_key]
        
#         prompt_parts = []
        
#         # 任务描述
#         if 'task_description' in prompt_config:
#             prompt_parts.append(prompt_config['task_description'])
        
#         # 领域映射（Stage 1）
#         if 'domain_mapping' in prompt_config:
#             prompt_parts.append(prompt_config['domain_mapping'])
        
#         # 临床指南（Stage 2）
#         if 'clinical_guidelines' in prompt_config:
#             prompt_parts.append(prompt_config['clinical_guidelines'])
        
#         # Few-shot示例
#         if include_examples and 'few_shot_examples' in prompt_config:
#             prompt_parts.append("Examples:")
#             prompt_parts.extend(prompt_config['few_shot_examples'])
        
#         # 指令
#         if 'instruction' in prompt_config:
#             prompt_parts.append(prompt_config['instruction'])
        
#         return " ".join(prompt_parts)
    
#     def get_meta_prompt(self, support_examples):
#         """为元学习构造动态提示"""
#         stage_config = self.prompts['stage_3']
        
#         prompt = stage_config['meta_instruction'] + ""
#         for i, (text, triplets) in enumerate(support_examples, 1):
#             triplet_str = " | ".join([f"({t[0]}, {t[1]}, {t[2]})" for t in triplets])
#             prompt += f"Example {i}: {text} → {triplet_str}"
        
#         prompt += stage_config['adaptive_hint'] + "Now extract from: "
#         return prompt



class AdvancedPromptManager:
    """支持多源域的提示管理器"""
    def __init__(self):
        self.prompts = {
            # ==================== Stage 0: 源域预训练 ====================
            # Restaurant 领域,针对（category，Opinion，Sentiment）
            'stage_0_restaurant': {
                'task_description': "Extract triplets of (aspect_category, opinion, sentiment) from restaurant reviews.",
                'few_shot_examples': [
                    "Input: The food was delicious and the service was excellent. Output: (FOOD#QUALITY, delicious, positive) | (SERVICE#GENERAL, excellent, positive)",
                    "Input: The atmosphere is cozy but the prices are too high. Output: (AMBIENCE#GENERAL, cozy, positive) | (RESTAURANT#PRICES, too high, negative)"
                ],
                'instruction': "Given a restaurant review, identify all aspect_category-opinion-sentiment triplets:"
            },
            # Restaurant 领域,针对（aspect term，Opinion term，Sentiment polarity）
            # 'stage_0_restaurant': {
            #     'task_description': "Extract triplets of (aspect term, opinion term, sentiment polarity) from restaurant reviews.",
            #     'few_shot_examples': [
            #         "Input: The food was delicious and the service was excellent. Output: (food, delicious, positive) | (service, excellent, positive)",
            #         "Input: The atmosphere is cozy but the prices are too high. Output: (atmosphere, cozy, positive) | (prices, too high, negative)"
            #     ],
            #     'instruction': "Given a restaurant review, identify all (aspect term, opinion term, sentiment polarity) triplets:"
            # },
            
            # Laptop 领域（新增）
            'stage_0_laptop': {
                'task_description': "Extract triplets of (aspect_category, opinion, sentiment) from laptop reviews.",
                'few_shot_examples': [
                    "Input: Build quality seems ok, keyboard is not flimsy or too firm. Output: (LAPTOP#QUALITY, ok, positive) | (KEYBOARD#GENERAL, not flimsy or too firm, positive)",
                    "Input: I like the ease of connecting to the internet wifi. Output: (LAPTOP#CONNECTIVITY, ease, positive)",
                    "Input: The battery life is terrible but the display is stunning. Output: (BATTERY#OPERATION_PERFORMANCE, terrible, negative) | (DISPLAY#QUALITY, stunning, positive)"
                ],
                'instruction': "Given a laptop review, identify all aspect_category-opinion-sentiment triplets:"
            },
            # 'stage_0_laptop': {
            #     'task_description': "Extract triplets of (aspect term, opinion term, sentiment polarity) from laptop reviews.",
            #     'few_shot_examples': [
            #         "Input: Build quality seems ok, keyboard is not flimsy or too firm. Output: (quality, ok, positive) | (keyboard, not flimsy or too firm, positive)",
            #         "Input: I like the ease of connecting to the internet wifi. Output: (connecting, ease, positive)",
            #         "Input: The battery life is terrible but the display is stunning. Output: (battery life, terrible, negative) | (display, stunning, positive)"
            #     ],
            #     'instruction': "Given a laptop review, identify all (aspect term, opinion term, sentiment polarity) triplets:"
            # },


            # Depression 领域
            'stage_0_depression': {
                'task_description': "Extract triplets of (symptom_category, subjective_description, severity_level) from mental health text.",
                'domain_mapping': "Map aspects to symptoms: FOOD#QUALITY/BATTERY#OPERATION_PERFORMANCE -> Appetite/Energy loss, SERVICE#GENERAL/LAPTOP#CONNECTIVITY -> Concentration, AMBIENCE#GENERAL/DISPLAY#QUALITY -> Mood, RESTAURANT#GENERAL/LAPTOP#GENERAL -> Interest loss",
                'few_shot_examples': [
                    "Input: I feel extremely tired and lack motivation. Output: (Energy loss, tired, negative) | (Energy loss, lack motivation, negative)",
                    "Input: Sleep is okay but mood is low. Output: (Sleep, okay, positive) | (Depressed mood, mood is low, negative)"
                ],
                'instruction': "Given a depression-related text, extract symptom triplets with severity (positive=mild/asymptomatic, neutral=moderate, negative=severe):"
            },

            # ==================== Stage 1: 跨域对齐 ====================
            # Restaurant 领域（保持不变）
            'stage_1_restaurant': {
                'task_description': "Extract triplets of (aspect_category, opinion, sentiment) from restaurant reviews.",
                'instruction': "Given a restaurant review, identify all aspect_category-opinion-sentiment triplets:"
            },
            # 'stage_1_restaurant': {
            #     'task_description': "Extract triplets of (aspect term, opinion term, sentiment polarity) from restaurant reviews.",
            #     'instruction': "Given a restaurant review, identify all (aspect term, opinion term, sentiment polarity) triplets:"
            # },
            
            # Laptop 领域（新增）
            'stage_1_laptop': {
                'task_description': "Extract triplets of (aspect_category, opinion, sentiment) from laptop reviews.",
                'instruction': "Given a laptop review, identify all aspect_category-opinion-sentiment triplets:"
            },
            # 'stage_1_laptop': {
            #     'task_description': "Extract triplets of (aspect term, opinion term, sentiment polarity) from laptop reviews.",
            #     'instruction': "Given a laptop review, identify all (aspect term, opinion term, sentiment polarity) triplets:"
            # },
            
            # Depression 领域
            'stage_1_depression': {
                'task_description': "Extract triplets of (symptom category, symptom description, severity level) from mental health text.",
                'domain_mapping': "Map aspects to symptoms: FOOD#QUALITY/BATTERY#OPERATION_PERFORMANCE -> Appetite/Energy loss, SERVICE#GENERAL/LAPTOP#CONNECTIVITY -> Concentration, AMBIENCE#GENERAL/DISPLAY#QUALITY -> Mood, RESTAURANT#GENERAL/LAPTOP#GENERAL -> Interest loss",
                'few_shot_examples': [
                    "Input: I feel extremely tired and lack motivation. Output: (Energy loss, tired, negative) | (Energy loss, lack motivation, negative)",
                    "Input: Sleep is okay but mood is low. Output: (Sleep, okay, positive) | (Depressed mood, mood is low, negative)"
                ],
                'instruction': "Given a depression-related text, extract symptom triplets with severity (positive=mild/asymptomatic, neutral=moderate, negative=severe):"
            },
            
            
            # ==================== Stage 2: 目标域微调 ====================
            'stage_2_depression': {
                'task_description': "Extract (symptom category, symptom description, severity level) triplets with clinical precision.",
                'clinical_guidelines': "Severity levels: positive(mild/no symptom), neutral(moderate), negative(severe). Categories: Appetite problem, Depressed mood, Cognitive problem, Interest loss, Sleep, Self-evaluation, Retardation, Suicidal ideation, Weight problem, Psychomotor.",
                'few_shot_examples': [
                    "Input: Pandemic burnout and anxiety affect students. Output: (Psychomotor, pandemic anxiety, neutral)",
                    "Input: Severe insomnia and complete loss of appetite. Output: (Sleep, severe insomnia, negative) | (Appetite problem, loss of appetite, negative)"
                ],
                'instruction': "Extract all depression symptom triplets from the clinical text:"
            },
            
            # ## 目标域为餐厅
            # 'stage_2_restaurant': {
            #     'task_description': "Extract triplets of (aspect term, opinion term, sentiment polarity) from restaurant reviews.",
            #     'few_shot_examples': [
            #         "Input: The food was delicious and the service was excellent. Output: (food, delicious, positive) | (service, excellent, positive)",
            #         "Input: The atmosphere is cozy but the prices are too high. Output: (atmosphere, cozy, positive) | (prices, too high, negative)"
            #     ],
            #     'instruction': "Given a restaurant review, identify all (aspect term, opinion term, sentiment polarity) triplets:"
            # },
            
            # ## 目标域为笔记本
            # 'stage_2_laptop': {
            #     'task_description': "Extract triplets of (aspect term, opinion term, sentiment polarity) from laptop reviews.",
            #     'few_shot_examples': [
            #         "Input: Build quality seems ok, keyboard is not flimsy or too firm. Output: (quality, ok, positive) | (keyboard, not flimsy or too firm, positive)",
            #         "Input: The battery life is terrible but the display is stunning. Output: (battery life, terrible, negative) | (display, stunning, positive)"
            #     ],
            #     'instruction': "Given a laptop review, identify all (aspect term, opinion term, sentiment polarity) triplets:"
            # }
        }
    
    def get_prompt(self, stage, domain='restaurant', include_examples=False):
        """
        获取提示词
        
        Args:
            stage: 训练阶段 (0, 1, 2)
            domain: 领域 ('restaurant', 'laptop', 'depression')
            include_examples: 是否包含 few-shot 示例
        
        Returns:
            str: 构造好的提示词
        """
        stage_key = f'stage_{stage}_{domain}'
        
        # 如果找不到对应的提示词，使用默认
        if stage_key not in self.prompts:
            print(f"Warning: Prompt key '{stage_key}' not found, using default")
            stage_key = f'stage_{stage}_restaurant'
        
        prompt_config = self.prompts[stage_key]
        prompt_parts = []
        
        # 1. 任务描述
        if 'task_description' in prompt_config:
            prompt_parts.append(prompt_config['task_description'])
        
        # 2. 领域映射（仅 Stage 1 的 depression）
        if 'domain_mapping' in prompt_config:
            prompt_parts.append(prompt_config['domain_mapping'])
        
        # 3. 临床指南（仅 Stage 2）
        if 'clinical_guidelines' in prompt_config:
            prompt_parts.append(prompt_config['clinical_guidelines'])
        
        # 4. Few-shot 示例（可选）
        if include_examples and 'few_shot_examples' in prompt_config:
            prompt_parts.append("Examples:")
            prompt_parts.extend(prompt_config['few_shot_examples'])
        
        # 5. 指令
        if 'instruction' in prompt_config:
            prompt_parts.append(prompt_config['instruction'])
        
        return " ".join(prompt_parts)
    
    def get_simple_prompt(self, stage, domain='restaurant'):
        """获取极简提示词（推荐用于训练）"""
        simple_prompts = {
            'stage_0_restaurant': "Extract (aspect category, opinion, sentiment) triplets from restaurant review:",
            'stage_0_laptop': "Extract (aspect category, opinion, sentiment) triplets from laptop review:",
            'stage_1_restaurant': "Extract (aspect category, opinion, sentiment) triplets from restaurant review:",
            'stage_1_laptop': "Extract (aspect category, opinion, sentiment) triplets from laptop review:",
            'stage_1_depression': "Extract (symptom category, description, severity) triplets from mental health text:",
            'stage_2_depression': "Extract (symptom category, description, severity) triplets from mental health text:",
            "stage_2_restaurant": "Extract (aspect category, opinion, sentiment) triplets from restaurant review:",
            "stage_2_laptop": "Extract (aspect category, opinion, sentiment) triplets from laptop review:"
        }
        # simple_prompts = {
        #     'stage_0_restaurant': "Extract (aspect term, opinion term, sentiment polarity) triplets from restaurant review:",
        #     'stage_0_laptop': "Extract (aspect term, opinion term, sentiment polarity) triplets from laptop review:",
        #     'stage_1_restaurant': "Extract (aspect term, opinion term, sentiment polarity) triplets from restaurant review:",
        #     'stage_1_laptop': "Extract (aspect term, opinion term, sentiment polarity) triplets from laptop review:",
        #     "stage_2_restaurant": "Extract (aspect term, opinion term, sentiment polarity) triplets from restaurant review:",
        #     "stage_2_laptop": "Extract (aspect term, opinion term, sentiment polarity) triplets from laptop review:"
        # }
        
        stage_key = f'stage_{stage}_{domain}'
        return simple_prompts.get(stage_key, "Extract triplets:")


# =========================
# 2. 增强数据集类（支持验证集）
# =========================
# class EnhancedTripletDataset(Dataset):
#     def __init__(self, file_path, tokenizer, prompt_manager, max_length=256, stage=0):
#         self.tokenizer = tokenizer
#         self.max_length = max_length
#         self.stage = stage
#         self.prompt_manager = prompt_manager
        
#         # 读取数据
#         self.data = []
#         with open(file_path, 'r', encoding='utf-8') as f:
#             for line in f:
#                 if '####' in line:
#                     text, triplets = line.split('####')
#                     self.data.append((text.strip(), eval(triplets.strip())))
            
#     def __len__(self):
#         return len(self.data)
    
#     def __getitem__(self, idx):
#         text, triplets = self.data[idx]
        
#         # 获取阶段性提示
#         prompt = self.prompt_manager.get_prompt(self.stage, include_examples=(idx % 10 == 0))
#         input_text = prompt + " " + text
        
#         # 构造输出
#         target_text = " | ".join([f"({t[0]}, {t[1]}, {t[2]})" for t in triplets])
        
#         # Tokenize
#         input_encoding = self.tokenizer(
#             input_text,
#             max_length=self.max_length,
#             padding='max_length',
#             truncation=True,
#             return_tensors='pt'
#         )
        
#         target_encoding = self.tokenizer(
#             target_text,
#             max_length=self.max_length,
#             padding='max_length',
#             truncation=True,
#             return_tensors='pt'
#         )
        
#         labels = target_encoding['input_ids'].squeeze()
#         labels[labels == self.tokenizer.pad_token_id] = -100
        
#         # 情感标签（用于句级对比）
#         sentiments = [t[2] for t in triplets]
#         sentiment_label = self._get_dominant_sentiment(sentiments)
        
#         return {
#             'input_ids': input_encoding['input_ids'].squeeze(),
#             'attention_mask': input_encoding['attention_mask'].squeeze(),
#             'labels': labels,
#             'sentiment_label': sentiment_label,
#             'raw_text': text,
#             'raw_triplets': triplets
#         }
    
#     def _get_dominant_sentiment(self, sentiments):
#         sentiment_map = {'positive': 0, 'neutral': 1, 'negative': 2}
#         if not sentiments:
#             return 1
#         mapped = [sentiment_map.get(s, 1) for s in sentiments]
#         return max(set(mapped), key=mapped.count)


class EnhancedTripletDataset(Dataset):
    def __init__(self, file_path, tokenizer, prompt_manager, partition, max_length=512, stage=0, domain='restaurant', use_simple_prompt=True):
        """
        Args:
            file_path: 数据文件路径
            tokenizer: T5 tokenizer
            prompt_manager: 提示词管理器
            max_length: 最大输入长度
            stage: 训练阶段 (0, 1, 2)
            domain: 领域 ('restaurant', 'laptop', 'depression')
            use_simple_prompt: 是否使用极简提示词（推荐 True）
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stage = stage
        self.domain = domain
        self.prompt_manager = prompt_manager
        self.partition = partition
        self.use_simple_prompt = use_simple_prompt
        
        # 读取数据
        self.data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if '####' in line:
                    text, triplets = line.split('####')
                    self.data.append((text.strip(), eval(triplets.strip())))
        
        ### 控制抑郁症训练样本的个数（我们按照比例来获取）
        if self.domain == 'depression' and self.partition:
            print(f"Sampling partition is {self.partition}")

            k = int(len(self.data) * self.partition)    # 需要抽取的数量
            print(f"Sampling number is {k}")
            self.data = random.sample(self.data, k)
            
        print(f"Loaded {len(self.data)} samples from {file_path} (domain={domain}, stage={stage})")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text, triplets = self.data[idx]
        
        # 获取提示词
        if self.use_simple_prompt:
            prompt = self.prompt_manager.get_simple_prompt(self.stage, self.domain)
        else:
            # 完整提示词（包含 few-shot 示例）
            include_examples = (idx % 10 == 0)  # 每 10 个样本包含一次示例
            prompt = self.prompt_manager.get_prompt(self.stage, self.domain, include_examples)
        
        input_text = f"{prompt} {text}"
        
        # 构造输出
        # print(text)
        # print(triplets)
        
        target_text = " | ".join([f"({t[0]}, {t[1]}, {t[2]})" for t in triplets])
        # print(target_text)
        # Tokenize 输入
        input_encoding = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize 输出
        target_encoding = self.tokenizer(
            target_text,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        labels = target_encoding['input_ids'].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        # 情感标签（用于对比学习）
        sentiments = [t[2] for t in triplets]
        sentiment_label = self._get_dominant_sentiment(sentiments)
        
        return {
            'input_ids': input_encoding['input_ids'].squeeze(),
            'attention_mask': input_encoding['attention_mask'].squeeze(),
            'labels': labels,
            'sentiment_label': sentiment_label,
            'raw_text': text,
            'raw_triplets': triplets
        }
    
    def _get_dominant_sentiment(self, sentiments):
        """获取主导情感"""
        sentiment_map = {'positive': 0, 'neutral': 1, 'negative': 2}
        if not sentiments:
            return 1
        mapped = [sentiment_map.get(s, 1) for s in sentiments]
        return max(set(mapped), key=mapped.count)




# =========================
# 自定义collate函数
# =========================
def custom_collate_fn(batch):
    """处理变长序列的batch整理"""
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    sentiment_labels = torch.tensor([item['sentiment_label'] for item in batch])
    
    raw_texts = [item['raw_text'] for item in batch]
    raw_triplets = [item['raw_triplets'] for item in batch]
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'sentiment_label': sentiment_labels,
        'raw_text': raw_texts,
        'raw_triplets': raw_triplets
    }


# =========================
# 3. 深度模型
# =========================
class DeepProgressivePromptModel(nn.Module):
    def __init__(self, model_name='t5-base'):
        super().__init__()
        self.t5 = T5ForConditionalGeneration.from_pretrained(model_name)
        
    
    def forward(self, input_ids, attention_mask, labels=None, return_features=False):
        outputs = self.t5(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=return_features
        )
        
        if return_features:
            encoder_hidden = outputs.encoder_last_hidden_state
            pooled_features = encoder_hidden.mean(dim=1)
            return outputs, pooled_features, encoder_hidden
        
        return outputs



# =========================
# 4. 跨域知识蒸馏模块
# =========================
class CrossDomainKnowledgeDistillation:
    def __init__(self, temperature=3.0, alpha=0.5, beta=0.3):
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        self.gamma = 1.0 - alpha - beta
        self.feature_adapter = None
    
    def build_feature_adapter(self, teacher_dim, student_dim):
        if teacher_dim != student_dim:
            self.feature_adapter = nn.Sequential(
                nn.Linear(teacher_dim, student_dim),  # 注意这里，教师维度到学生维度
                nn.ReLU(),
                nn.Dropout(0.1)
            ).cuda()
        else:
            self.feature_adapter = nn.Identity()

    
    def soft_label_distillation_loss(self, student_logits, teacher_logits):
        student_soft = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=-1)
        
        kl_loss = F.kl_div(
            student_soft, 
            teacher_soft, 
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        return kl_loss

    def feature_distillation_loss(self, student_features, teacher_features):
        if self.feature_adapter is not None:
            teacher_features = self.feature_adapter(teacher_features)  # 教师特征映射到学生维度
        mse_loss = F.mse_loss(student_features, teacher_features)
        return mse_loss

    
    def compute_distillation_loss(self, student_outputs, teacher_outputs, hard_labels=None):
        loss_dict = {}
        
        # 软标签蒸馏
        if hasattr(student_outputs, 'logits') and hasattr(teacher_outputs, 'logits'):
            soft_loss = self.soft_label_distillation_loss(
                student_outputs.logits, 
                teacher_outputs.logits
            )
            loss_dict['soft_label'] = soft_loss.item()
        else:
            soft_loss = torch.tensor(0.0).cuda()
        
        # 特征蒸馏
        if (hasattr(student_outputs, 'encoder_last_hidden_state') and 
            hasattr(teacher_outputs, 'encoder_last_hidden_state')):
            student_hidden = student_outputs.encoder_last_hidden_state
            teacher_hidden = teacher_outputs.encoder_last_hidden_state
            
            feature_loss = self.feature_distillation_loss(
                student_hidden, 
                teacher_hidden
            )
            loss_dict['feature'] = feature_loss.item()
        else:
            feature_loss = torch.tensor(0.0).cuda()
        
        # 硬标签损失
        if hard_labels is not None and hasattr(student_outputs, 'loss'):
            hard_loss = student_outputs.loss
            loss_dict['hard_label'] = hard_loss.item()
        else:
            hard_loss = torch.tensor(0.0).cuda()
        
        # 总损失
        total_loss = (
            self.alpha * soft_loss + 
            self.beta * feature_loss + 
            self.gamma * hard_loss
        )
        
        loss_dict['total'] = total_loss.item()
        
        return total_loss, loss_dict



class MultiTeacherEnsembleDistillation:
    """纯集成教师蒸馏（所有教师平等对待）"""
    
    def __init__(self, teachers, teacher_dim, student_dim, 
                 temperature=2.0, alpha=0.5, beta=0.3):
        """
        Args:
            teachers: 教师模型列表 [restaurant_teacher, laptop_teacher]
            teacher_dim: 教师隐藏层维度
            student_dim: 学生隐藏层维度
            temperature: 蒸馏温度（默认2.0）
            alpha: 软标签损失权重（默认0.5）
            beta: 特征损失权重（默认0.3）
        """
        self.teachers = teachers
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        
        # ✅ 教师特征适配器（teacher_dim -> student_dim）
        # 如果教师和学生维度相同，使用Identity（不做任何变换）
        self.teacher_feature_adapters = nn.ModuleList([
            nn.Sequential(
                nn.Linear(teacher_dim, student_dim),
                nn.LayerNorm(student_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ).cuda() if teacher_dim != student_dim else nn.Identity()
            for _ in teachers
        ])
        
        # ✅ 学生特征适配器（student_dim -> teacher_dim）
        # 用于特征蒸馏时的维度对齐
        self.student_feature_adapter = nn.Sequential(
            nn.Linear(student_dim, teacher_dim),
            nn.LayerNorm(teacher_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        ).cuda() if teacher_dim != student_dim else nn.Identity()
        
        # ✅ 冻结所有教师模型
        for teacher in self.teachers:
            teacher.eval()
            for param in teacher.parameters():
                param.requires_grad = False
        
    
    def get_ensemble_outputs(self, input_ids, attention_mask, decoder_input_ids):
        """
        获取所有教师的集成输出
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            decoder_input_ids: [batch_size, decoder_seq_len]
        
        Returns:
            ensemble_logits: [batch_size, decoder_seq_len, vocab_size] - 集成后的logits
            ensemble_features: [batch_size, hidden_dim] - 集成后的特征
        """
        all_logits = []
        all_features = []
        
        # ✅ 遍历所有教师，获取输出
        for i, teacher in enumerate(self.teachers):
            with torch.no_grad():
                # 教师模型前向传播
                outputs = teacher.t5(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    decoder_input_ids=decoder_input_ids,
                    output_hidden_states=True
                )
                
                # 获取logits和编码器特征
                logits = outputs.logits  # [batch_size, decoder_seq_len, vocab_size]
                features = outputs.encoder_last_hidden_state  # [batch_size, seq_len, hidden_dim]
                
                # ✅ 适配教师特征维度（如果需要）
                features = self.teacher_feature_adapters[i](features)
                
                # ✅ 平均池化：将序列特征压缩为单个向量
                # 使用attention_mask加权平均，忽略padding部分
                pooled_features = (features * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1, keepdim=True)
                # pooled_features: [batch_size, hidden_dim]
                
                all_logits.append(logits)
                all_features.append(pooled_features)
        
        # ✅ 集成：简单平均所有教师的输出
        ensemble_logits = torch.stack(all_logits).mean(0)
        # ensemble_logits: [batch_size, decoder_seq_len, vocab_size]
        
        ensemble_features = torch.stack(all_features).mean(0)
        # ensemble_features: [batch_size, hidden_dim]
        
        return ensemble_logits, ensemble_features
    
    def compute_ensemble_distillation_loss(self, student_outputs, input_ids, attention_mask, 
                                          decoder_input_ids, labels=None):
        """
        计算集成蒸馏损失
        
        Args:
            student_outputs: 学生模型的输出（包含logits和hidden_states）
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            decoder_input_ids: [batch_size, decoder_seq_len]
            labels: [batch_size, decoder_seq_len]（可选，未使用）
        
        Returns:
            total_loss: 总蒸馏损失（软标签损失 + 特征损失）
        """
        
        # ============================================
        # 1. 获取教师集成输出
        # ============================================
        teacher_logits, teacher_features = self.get_ensemble_outputs(
            input_ids, attention_mask, decoder_input_ids
        )
        # teacher_logits: [batch_size, decoder_seq_len, vocab_size]
        # teacher_features: [batch_size, hidden_dim]
        
        # ============================================
        # 2. 获取学生输出
        # ============================================
        student_logits = student_outputs.logits
        # student_logits: [batch_size, decoder_seq_len, vocab_size]
        
        # ============================================
        # 3. 计算软标签蒸馏损失（KL散度）
        # ============================================
        # 温度缩放：使概率分布更平滑，包含更多"暗知识"
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
        
        # KL散度：衡量学生和教师概率分布的差异
        soft_loss = F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction='batchmean'
        ) * (self.temperature ** 2)  # 温度平方用于缩放梯度
        
        # ============================================
        # 4. 计算特征蒸馏损失（MSE）
        # ============================================
        # 获取学生的编码器特征
        # if hasattr(student_outputs, 'encoder_last_hidden_state'):
        #     student_features = student_outputs.encoder_last_hidden_state
        # else:
        #     # 兼容性处理：如果没有encoder_last_hidden_state，使用hidden_states
        #     student_features = student_outputs.encoder_hidden_states[-1]
        # # student_features: [batch_size, seq_len, hidden_dim]
        
        # # ✅ 平均池化：与教师特征对齐
        # student_pooled = (student_features * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1, keepdim=True)
        # # student_pooled: [batch_size, hidden_dim]
        
        # # ✅ 适配学生特征维度（如果需要）
        # student_pooled_adapted = self.student_feature_adapter(student_pooled)
        # # student_pooled_adapted: [batch_size, teacher_dim]
        
        # # MSE损失：衡量学生和教师特征的差异
        # feature_loss = F.mse_loss(student_pooled_adapted, teacher_features)
        
        # # ============================================
        # # 5. 加权总损失
        # # ============================================
        # total_loss = self.alpha * soft_loss + self.beta * feature_loss
        
        # return total_loss
        
        
        if hasattr(student_outputs, 'encoder_last_hidden_state'):
            student_features = student_outputs.encoder_last_hidden_state
        else:
            student_features = student_outputs.encoder_hidden_states[-1]
        
        # 平均池化
        student_pooled = (student_features * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1, keepdim=True)
        
        # ✅ 修复：根据维度关系选择适配方向
        if student_pooled.shape[-1] != teacher_features.shape[-1]:
            # 如果维度不同，适配学生特征到教师维度
            if hasattr(self, 'student_feature_adapter'):
                student_pooled_adapted = self.student_feature_adapter(student_pooled)
                feature_loss = F.mse_loss(student_pooled_adapted, teacher_features)
            else:
                # 如果没有适配器，跳过特征损失
                print("⚠️ Warning: Feature dimensions mismatch, skipping feature loss")
                feature_loss = torch.tensor(0.0, device=student_pooled.device)
        else:
            # 维度相同，直接计算 MSE损失：衡量学生和教师特征的差异
            feature_loss = F.mse_loss(student_pooled, teacher_features)
        
        # 4. 总损失
        total_loss = self.alpha * soft_loss + self.beta * feature_loss
        
        return total_loss






# =========================
# 6. Reptile元学习模块
# =========================
class ReptileMetaLearner:
    def __init__(self, model, inner_lr=1e-3, outer_lr=1e-4, num_inner_steps=5):
        self.model = model
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.num_inner_steps = num_inner_steps
        
        self.meta_params = {
            name: param.clone().detach() 
            for name, param in model.named_parameters()
        }
    
    def inner_loop(self, task_data_loader):
        original_params = {
            name: param.clone() 
            for name, param in self.model.named_parameters()
        }
        
        optimizer = optim.SGD(self.model.parameters(), lr=self.inner_lr)
        
        self.model.train()
        for step in range(self.num_inner_steps):
            for batch in task_data_loader:
                input_ids = batch['input_ids'].cuda()
                attention_mask = batch['attention_mask'].cuda()
                labels = batch['labels'].cuda()
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
        
        adapted_params = {
            name: param.clone() 
            for name, param in self.model.named_parameters()
        }
        
        for name, param in self.model.named_parameters():
            param.data = original_params[name]
        
        return adapted_params
    
    def outer_loop(self, adapted_params):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                diff = adapted_params[name] - param.data
                param.data += self.outer_lr * diff
    
    def meta_train_step(self, task_data_loader):
        adapted_params = self.inner_loop(task_data_loader)
        self.outer_loop(adapted_params)



# =========================
# 7. 原有的训练函数
# =========================
def train_stage_enhanced(model, tokenizer, prompt_manager,partition, source_data_path, target_data_path, 
                        stage, epochs=10, patience=3,source_domain='restaurant',target_domain='laptop'):
    """原有的训练函数（保持不变）"""
    print(f"{'='*60}")
    print(f"Enhanced Training Stage {stage}")
    print(f"{'='*60}")
    

    # 构建数据集
    # stage = 0: 只用源域数据进行知识蒸馏 (本函数只执行stage0)
    if stage == 0:
        train_dataset = EnhancedTripletDataset(
            source_data_path + 'train.txt',
            tokenizer, prompt_manager, partition, stage=stage,domain=source_domain,use_simple_prompt=False
        )
        val_dataset = EnhancedTripletDataset(
            source_data_path + 'dev.txt',
            tokenizer, prompt_manager, partition, stage=stage,domain=source_domain,use_simple_prompt=False
        )
    # stage = 1: 混合源域和目标域数据（跨域对齐）
    elif stage == 1:
        source_train = EnhancedTripletDataset(
            source_data_path + 'train.txt',
            tokenizer, prompt_manager, partition,stage=stage,domain=source_domain,use_simple_prompt=False
        )
        target_train = EnhancedTripletDataset(
            target_data_path + 'train.txt',
            tokenizer, prompt_manager, partition,stage=stage,domain=target_domain,use_simple_prompt=False
        )
        train_dataset = torch.utils.data.ConcatDataset([source_train, target_train])
        
        val_dataset = EnhancedTripletDataset(
            target_data_path + 'dev.txt',
            tokenizer, prompt_manager, partition,stage=stage,domain=target_domain,use_simple_prompt=False
        )
     # stage = 2: 只用目标域数据微调
    else:
        train_dataset = EnhancedTripletDataset(
            target_data_path + 'train.txt',
            tokenizer, prompt_manager, partition, stage=stage,domain=target_domain,use_simple_prompt=False
        )
        val_dataset = EnhancedTripletDataset(
            target_data_path + 'dev.txt',
            tokenizer, prompt_manager, partition, stage=stage,domain=target_domain,use_simple_prompt=False
        )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=4, 
        shuffle=True,
        collate_fn=custom_collate_fn
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=4, 
        shuffle=False,
        collate_fn=custom_collate_fn
    )
    
    optimizer = optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps
    )
    
    best_val_f1 = 0.0
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        
        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].cuda()
            attention_mask = batch['attention_mask'].cuda()
            labels = batch['labels'].cuda()
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            total_train_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        # 验证
        model.eval()
        total_val_loss = 0
        all_triplet_f1s = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                input_ids = batch['input_ids'].cuda()
                attention_mask = batch['attention_mask'].cuda()
                labels = batch['labels'].cuda()
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                total_val_loss += outputs.loss.item()
                
                generated = model.t5.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=256,
                    num_beams=3,
                    early_stopping=True
                )
                
                pred_text = tokenizer.batch_decode(generated.tolist(), skip_special_tokens=True)
                pred_triplets = [parse_triplets(text) for text in pred_text]
                ref_triplets = batch['raw_triplets']
                
                metrics = calculate_metrics(pred_triplets, ref_triplets)
                all_triplet_f1s.append(metrics['f1'])
                
        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        avg_val_f1 = sum(all_triplet_f1s) / len(all_triplet_f1s)
        
        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss:   {avg_val_loss:.4f}")
        print(f"  Val F1:     {avg_val_f1:.4f}")
        
        if avg_val_f1 > best_val_f1:
            best_val_f1 = avg_val_f1
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'val_loss': avg_val_loss,
                'val_f1': avg_val_f1
            }, f'best_model_stage_{stage}.pt')
            print(f"  ✓ New best model saved! (F1: {avg_val_f1:.4f})")
        else:
            patience_counter += 1
            print(f"  ✗ No improvement (Best F1: {best_val_f1:.4f}, {patience_counter}/{patience})")
            
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                break
    
    checkpoint = torch.load(f'best_model_stage_{stage}.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best model from stage {stage}")
    print(f"  Best Val F1: {checkpoint['val_f1']:.4f}")



# =========================
# 8. 知识蒸馏+元学习训练函数
# =========================
def train_with_distillation_and_metalearning(
    student_model, 
    teacher_models,
    tokenizer,
    prompt_manager,
    partition,
    source_data_paths,
    target_data_path,
    stage, 
    source_map,
    target_domain,
    epochs=15,
    patience=3,
    use_multi_teacher=True,
    use_meta_learning=True,
    source_domain_index=0,
    target_test = True                ## 是否在目标域上测试
):
    """集成训练：知识蒸馏 + 元学习"""
    print(f"{'='*80}")
    print(f"Training with Knowledge Distillation + Meta-Learning (Stage {stage})")
    print(f"{'='*80}")
    print(f"  Teachers: {len(teacher_models)}")
    print(f"  Multi-Teacher: {use_multi_teacher}")
    print(f"  Meta-Learning: {use_meta_learning}")
    print(f"{'='*80}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 初始化知识蒸馏模块
    kd_module = CrossDomainKnowledgeDistillation(
        temperature=2.0,
        alpha=0.4,
        beta=0.3
    )
    
    teacher_dim = teacher_models[0].t5.config.d_model
    student_dim = student_model.t5.config.d_model
    kd_module.build_feature_adapter(teacher_dim, student_dim)
    
    # 初始化多教师集成
    if use_multi_teacher and len(teacher_models) > 1:
        multi_teacher = MultiTeacherEnsembleDistillation(
            teachers=teacher_models,
            teacher_dim=teacher_dim,  # 传入教师维度
            student_dim=student_dim,  # 传入学生维度
            temperature=2.0,   
            alpha=0.4,
            beta=0.3
        )
        print(f"✓ Multi-teacher ensemble initialized with {len(teacher_models)} teachers")
    else:
        multi_teacher = None
        print(f"✓ Single teacher distillation")
    
    # 初始化元学习模块
    if use_meta_learning:
        meta_learner = ReptileMetaLearner(
            model=student_model,
            inner_lr=1e-3,
            outer_lr=1e-4,
            inner_steps=5   ##5
        )
        print(f"✓ Reptile meta-learner initialized")
    else:
        meta_learner = None
    
    # ✅ 根据stage准备不同的训练数据
    if stage == 0:
        # Stage 0: 只用源域数据进行知识蒸馏  (本函数只执行stage0和stage2)
        print(f"[Stage 0] Using source domain data for distillation")
        
        ## 这里进行单教师的消融实验，需要独自选择对应源域的训练集和验证集

        source_datasets = []
        for source_path in source_data_paths:
            dataset = EnhancedTripletDataset(
                source_path + 'train.txt',
                tokenizer, prompt_manager, partition, stage=stage,domain=source_map[source_path],use_simple_prompt=False
            )
            source_datasets.append(dataset)
            print(f"  ✓ Loaded {len(dataset)} samples from {source_path} + train.txt")
        
        # 验证集也用源域
        val_datasets = []
        for source_path in source_data_paths:
            dataset = EnhancedTripletDataset(
                source_path + 'dev.txt',
                tokenizer, prompt_manager, partition, stage=stage,domain=source_map[source_path],use_simple_prompt=False
            )
            val_datasets.append(dataset)

        ## 定义测试集用于每一个阶段的测试
        test_dataset = EnhancedTripletDataset(
            target_data_path + 'test.txt',
            tokenizer, prompt_manager,stage=stage,domain=target_domain,use_simple_prompt=False,partition=None
        )
        

        ## 如果是使用单教师，则只需要一个源域数据集即可
        if not use_multi_teacher:
            print(f"Using single source train data and dev data: {source_data_paths[source_domain_index]}")
            train_dataset = source_datasets[source_domain_index]
            val_dataset = val_datasets[source_domain_index]
        else:   # 合并所有源域数据
            print("Using Multi-source data")
            train_dataset = torch.utils.data.ConcatDataset(source_datasets)
            val_dataset = torch.utils.data.ConcatDataset(val_datasets)
        

    elif stage == 1:
        # Stage 1: 混合源域和目标域数据（跨域对齐）
        print(f"[Stage 1] Using mixed source and target domain data")
        
        # 加载所有源域数据
        source_datasets = []
        for source_path in source_data_paths:
            dataset = EnhancedTripletDataset(
                source_path + 'train.txt',
                tokenizer, prompt_manager, partition,stage=stage,domain=source_map[source_path],use_simple_prompt=False
            )
            source_datasets.append(dataset)
        source_combined = torch.utils.data.ConcatDataset(source_datasets)
        
        # 加载目标域数据
        target_dataset = EnhancedTripletDataset(
            target_data_path + 'train.txt',
            tokenizer, prompt_manager, partition,stage=stage,domain=target_domain,use_simple_prompt=False
        )
        
        # 混合训练
        train_dataset = torch.utils.data.ConcatDataset([source_combined, target_dataset])
        print(f"  ✓ Source samples: {len(source_combined)}")
        print(f"  ✓ Target samples: {len(target_dataset)}")
        
        # 验证集用目标域
        val_dataset = EnhancedTripletDataset(
            target_data_path + 'dev.txt',
            tokenizer, prompt_manager, partition,stage=stage,domain=target_domain,use_simple_prompt=False
        )
        
    else:  # stage == 2
        # Stage 2: 只用目标域数据（元学习微调）
        print(f"[Stage 2] Using target domain data for meta-learning")
        
        train_dataset = EnhancedTripletDataset(
            target_data_path + 'train.txt',
            tokenizer, prompt_manager, partition,stage=stage,domain=target_domain,use_simple_prompt=False
        )
        val_dataset = EnhancedTripletDataset(
            target_data_path + 'dev.txt',
            tokenizer, prompt_manager,stage=stage,domain=target_domain,use_simple_prompt=False,partition=None
        )
        ## 定义测试集用于每一个阶段的测试
        test_dataset = EnhancedTripletDataset(
            target_data_path + 'test.txt',
            tokenizer, prompt_manager,stage=stage,domain=target_domain,use_simple_prompt=False,partition=None
        )
        print(f"  ✓ Target train samples: {len(train_dataset)}")
    
    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=4 if stage != 2 else 2,  # Stage 2用更小的batch
        shuffle=True,
        collate_fn=custom_collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=custom_collate_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=custom_collate_fn
    )

    # ✅ 元学习任务数据（只在Stage 2使用）
    meta_task_loaders = []
    if use_meta_learning and stage == 2:
        num_tasks = min(10, len(train_dataset) // 20)  # 至少20个样本一个任务
        task_size = len(train_dataset) // num_tasks
        
        for i in range(num_tasks):
            start_idx = i * task_size
            end_idx = min((i + 1) * task_size, len(train_dataset))
            
            task_indices = list(range(start_idx, end_idx))
            task_dataset = torch.utils.data.Subset(train_dataset, task_indices)
            
            task_loader = DataLoader(
                task_dataset,
                batch_size=2,
                shuffle=True,
                collate_fn=custom_collate_fn
            )
            meta_task_loaders.append(task_loader)
        
        print(f"✓ Created {len(meta_task_loaders)} meta-learning tasks")
    
    # 优化器
    optimizer = optim.AdamW(student_model.parameters(), lr=3e-5, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps
    )
    
    # 训练循环
    best_val_f1 = 0.0
    patience_counter = 0

    ## 存放训练过程中的accuracy和F1
    acc_score = []
    pre_score = []
    rec_score = []
    F1_score = []
    
    for epoch in range(epochs):
        print(f"{'='*80}")
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"{'='*80}")
        
        # ✅ 知识蒸馏训练（Stage 0和Stage 1）
        if stage in [0, 1]:
            student_model.train()
            total_train_loss = 0
            total_task_loss = 0
            total_kd_loss = 0
            
            progress_bar = tqdm(train_loader, desc=f"[Distillation Training]")
            
            for batch_idx, batch in enumerate(progress_bar):
                input_ids = batch['input_ids'].cuda()
                attention_mask = batch['attention_mask'].cuda()
                labels = batch['labels'].cuda()
                
                # ✅ 获取领域信息（如果有）
                domain_names = batch.get('domain_name', None)
                
                optimizer.zero_grad()
                
                # ✅ 创建decoder_input_ids（学生和教师都用同一个）
                decoder_input_ids = student_model.t5._shift_right(labels)
                
                # ✅ 学生模型前向传播（使用decoder_input_ids）
                student_outputs = student_model.t5(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    decoder_input_ids=decoder_input_ids,  # ✅ 显式传递
                    labels=labels,
                    output_hidden_states=True
                )
                
                # ✅ 计算蒸馏损失（教师模型在no_grad中，但损失计算在外面）
                if multi_teacher is not None:
                    # print("[Stage 0 Knowledge Distillation] Multiteacher KD")
                    # 多教师集成
                    kd_loss = multi_teacher.compute_ensemble_distillation_loss(
                        student_outputs=student_outputs,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        decoder_input_ids=decoder_input_ids,  # ✅ 传递相同的decoder_input_ids
                        labels=labels
                    )
                else:
                    # print(f"[Stage 0 Knowledge Distillation] single teacher KD with {source_data_paths[source_domain_index]}")
                    # 单教师
                    teacher = teacher_models[source_domain_index]
                    
                    with torch.no_grad():
                        teacher_outputs = teacher.t5(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            decoder_input_ids=decoder_input_ids,  # ✅ 使用相同的decoder_input_ids
                            output_hidden_states=True
                        )
                    
                    # ✅ 蒸馏损失计算在no_grad外面
                    kd_loss, loss_dict = kd_module.compute_distillation_loss(
                        student_outputs, teacher_outputs, labels
                    )
                
                # ✅ 裁剪 KD loss，防止梯度爆炸
                kd_loss = torch.clamp(kd_loss, max=20.0)

                # 总损失 = 任务损失 + 蒸馏损失
                task_loss = student_outputs.loss
                
                # # ✅ 根据stage调整权重
                # if stage == 0:
                #     # Stage 0: 更依赖任务损失
                #     loss = 0.7 * task_loss + 0.3 * kd_loss
                # else:  # stage == 1
                #     # Stage 1: 平衡任务损失和蒸馏损失
                #     loss = 0.5 * task_loss + 0.5 * kd_loss

                # ✅ 渐进式蒸馏：根据 epoch 动态调整权重
                if stage == 0:
                    if epoch < 2:
                        # 前 2 个 epoch：只用 task loss，让模型先学会基本任务
                        loss = task_loss
                        kd_weight = 0.0
                    else:
                        # 后续 epoch：逐步增加 KD loss 权重
                        kd_weight = min(0.3, 0.1 * ((epoch - 1) / 3))
                        loss = (1 - kd_weight) * task_loss + kd_weight * kd_loss
                else:  # stage == 1
                    # Stage 1：平衡任务损失和蒸馏损失
                    loss = 0.5 * task_loss + 0.5 * kd_loss
                    kd_weight = 0.5


                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                total_train_loss += loss.item()
                total_task_loss += task_loss.item()
                total_kd_loss += kd_loss.item()
                
                progress_bar.set_postfix({
                    'loss': loss.item(),
                    'task': task_loss.item(),
                    'kd': kd_loss.item(),
                    'kd_w': f'{kd_weight:.2f}'
                })

                # ✅ 每 100 个 batch 打印详细日志
                if batch_idx % 100 == 0 and batch_idx > 0:
                    print(f"\n[Batch {batch_idx}/{len(train_loader)}]")
                    print(f"  Task Loss: {task_loss.item():.4f}")
                    print(f"  KD Loss: {kd_loss.item():.4f}")
                    print(f"  KD Weight: {kd_weight:.2f}")
                    print(f"  Total Loss: {loss.item():.4f}")
            
            avg_train_loss = total_train_loss / len(train_loader)
            avg_task_loss = total_task_loss / len(train_loader)
            avg_kd_loss = total_kd_loss / len(train_loader)
            
            print(f"[Distillation Training Results]")
            print(f"  Avg Total Loss: {avg_train_loss:.4f}")
            print(f"  Avg Task Loss: {avg_task_loss:.4f}")
            print(f"  Avg KD Loss: {avg_kd_loss:.4f}")


        # ✅ 元学习训练（Stage 2）
        elif stage == 2 and use_meta_learning:
            print(f"[Meta-Learning Training]")
            
            # 每个epoch进行元学习更新
            for task_idx, task_loader in enumerate(meta_task_loaders):
                meta_learner.meta_train_step(task_loader)
                
                if (task_idx + 1) % 3 == 0:
                    print(f"  Completed meta-task {task_idx + 1}/{len(meta_task_loaders)}")
            
            print(f"✓ Meta-learning update completed")
            
            # 普通训练
            student_model.train()
            total_train_loss = 0
            
            for batch in tqdm(train_loader, desc=f"[Fine-tuning]"):
                input_ids = batch['input_ids'].cuda()
                attention_mask = batch['attention_mask'].cuda()
                labels = batch['labels'].cuda()
                
                optimizer.zero_grad()
                
                outputs = student_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                total_train_loss += loss.item()
            
            avg_train_loss = total_train_loss / len(train_loader)
            print(f"\n[Fine-tuning Results]")
            print(f"  Avg Loss: {avg_train_loss:.4f}")
            
        else:
            # Stage 2: 直接微调 Student 模型
            print("[NOT Meta-Learning Training, Fine-Tuning]")
        
            student_model.train()
            total_train_loss = 0
        
            scaler = torch.cuda.amp.GradScaler()  # 可选：混合精度训练，提高显存利用率
        
            for batch in tqdm(train_loader, desc="[Training]"):
                input_ids = batch['input_ids'].cuda()
                attention_mask = batch['attention_mask'].cuda()
                labels = batch['labels'].cuda()
        
                optimizer.zero_grad()
        
                with torch.cuda.amp.autocast():  # 可选
                    outputs = student_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs.loss
        
                # 反向传播
                scaler.scale(loss).backward()
        
                # 梯度裁剪
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
        
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
        
                total_train_loss += loss.item()
        
            avg_train_loss = total_train_loss / len(train_loader)
            print("[Training Results]")
            print(f"  Avg Loss: {avg_train_loss:.4f}")
            


        # ✅ 验证
        print(f"{'='*80}")
        print(f"Validation")
        print(f"{'='*80}")
        
        student_model.eval()
        total_val_loss = 0
        all_triplet_f1s = []
        all_triplet_precisions = []
        all_triplet_recalls = []
        all_triplet_accuracies = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"[Validation]"):
                input_ids = batch['input_ids'].cuda()
                attention_mask = batch['attention_mask'].cuda()
                labels = batch['labels'].cuda()
                
                outputs = student_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                total_val_loss += outputs.loss.item()
                
                # 生成预测
                generated = student_model.t5.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=256,
                    num_beams=3,
                    early_stopping=True
                )
                
                pred_text = tokenizer.batch_decode(generated.tolist(), skip_special_tokens=True)
                pred_triplets = [parse_triplets(text) for text in pred_text]
                ref_triplets = batch['raw_triplets']
                
                # 计算指标
                for pred, ref in zip(pred_triplets, ref_triplets):
                    metrics = calculate_metrics([pred], [ref])
                    all_triplet_accuracies.append(metrics['accuracy'])
                    all_triplet_f1s.append(metrics['f1'])
                    all_triplet_precisions.append(metrics['precision'])
                    all_triplet_recalls.append(metrics['recall'])
        
        # 计算平均指标
        avg_val_loss = total_val_loss / len(val_loader)
        avg_val_accuracy = sum(all_triplet_accuracies) / len(all_triplet_accuracies) if all_triplet_accuracies else 0.0
        avg_val_f1 = sum(all_triplet_f1s) / len(all_triplet_f1s) if all_triplet_f1s else 0.0
        avg_val_precision = sum(all_triplet_precisions) / len(all_triplet_precisions) if all_triplet_precisions else 0.0
        avg_val_recall = sum(all_triplet_recalls) / len(all_triplet_recalls) if all_triplet_recalls else 0.0
        
        print(f"[Validation Results]")
        print(f"  Val Loss:      {avg_val_loss:.4f}")
        print(f" Val Accuracy: {avg_val_accuracy:.4f}")
        print(f"  Val Precision: {avg_val_precision:.4f}")
        print(f"  Val Recall:    {avg_val_recall:.4f}")
        print(f"  Val F1:        {avg_val_f1:.4f}")

       
        
        # ============================================
        # 10. 早停和模型保存
        # ============================================
        if avg_val_f1 > best_val_f1:
            best_val_f1 = avg_val_f1
            patience_counter = 0
            
            save_path = f'best_model_kd_meta_stage_{stage}.pt'
            torch.save({
                'model_state_dict': student_model.state_dict(),
                'epoch': epoch + 1,
                'val_loss': avg_val_loss,
                'val_f1': avg_val_f1,
                'val_precision': avg_val_precision,
                'val_recall': avg_val_recall,
                'stage': stage
            }, save_path)
            
            print(f"  ✓ New best model saved to {save_path}! (F1: {avg_val_f1:.4f})")
        else:
            patience_counter += 1
            print(f"  ✗ No improvement (Best F1: {best_val_f1:.4f}, Patience: {patience_counter}/{patience})")
            
            if patience_counter >= patience:
                print(f"\n{'='*80}")
                print(f"Early stopping triggered at epoch {epoch+1}")
                print(f"{'='*80}")
                break
        
        print(f"{'='*80}\n")


        ## 验证后在目标域测试集上测试

        if target_test:
            # ✅ 验证
            print(f"{'='*80}")
            print(f"Testing")
            print(f"{'='*80}")
            
            student_model.eval()
            total_test_loss = 0

            test_triplet_f1s = []
            test_triplet_precisions = []
            test_triplet_recalls = []
            test_triplet_accuracies = []


            with torch.no_grad():
                for batch in tqdm(test_loader, desc=f"[Testing]"):
                    input_ids = batch['input_ids'].cuda()
                    attention_mask = batch['attention_mask'].cuda()
                    labels = batch['labels'].cuda()
                    
                    outputs = student_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    total_test_loss += outputs.loss.item()
                    
                    # 生成预测
                    generated = student_model.t5.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_length=256,
                        num_beams=3,
                        early_stopping=True
                    )
                    
                    pred_text = tokenizer.batch_decode(generated.tolist(), skip_special_tokens=True)
                    pred_triplets = [parse_triplets(text) for text in pred_text]
                    ref_triplets = batch['raw_triplets']
                    
                    # 计算指标
                    for pred, ref in zip(pred_triplets, ref_triplets):
                        metrics = calculate_metrics([pred], [ref])
                        test_triplet_accuracies.append(metrics['accuracy'])
                        test_triplet_f1s.append(metrics['f1'])
                        test_triplet_precisions.append(metrics['precision'])
                        test_triplet_recalls.append(metrics['recall'])
            
            # 计算平均指标
            avg_test_loss = total_test_loss / len(test_loader)
            avg_test_accuracy = sum(test_triplet_accuracies) / len(test_triplet_accuracies) if test_triplet_accuracies else 0.0
            avg_test_f1 = sum(test_triplet_f1s) / len(test_triplet_f1s) if test_triplet_f1s else 0.0
            avg_test_precision = sum(test_triplet_precisions) / len(test_triplet_precisions) if test_triplet_precisions else 0.0
            avg_test_recall = sum(test_triplet_recalls) / len(test_triplet_recalls) if test_triplet_recalls else 0.0

            ## 存放所有的acc和f1
            acc_score.append(avg_test_accuracy)
            pre_score.append(avg_test_precision)
            rec_score.append(avg_test_recall)
            F1_score.append(avg_test_f1)

            print(f"[Stage {stage}] Average Test metrics:")
            print(f"  Test Accuracy   : {avg_test_accuracy:.4f}")
            print(f"  Test Precision  : {avg_test_precision:.4f}")
            print(f"  Test Recall     : {avg_test_recall:.4f}")
            print(f"  Test F1 Score   : {avg_test_f1:.4f}")

    print(f"[Stage {stage}] metrics list in test dataset:")
    print(f"Test Accuracy list    : {acc_score}")
    print(f"Test Precision list   : {pre_score}")
    print(f"Test Recall list      : {rec_score}")
    print(f"Test F1 Score list    : {F1_score}")

    # 将指标列表保存至txt文件

    metrics_filename = f'test_metrics_stage_{stage}.txt'
    with open(metrics_filename, 'w', encoding='utf-8') as f:
        f.write(f"Test Accuracy list    : {acc_score}\n")
        f.write(f"Test Precision list   : {pre_score}\n")
        f.write(f"Test Recall list      : {rec_score}\n")
        f.write(f"Test F1 Score list    : {F1_score}\n")
    print(f"Test metrics saved to {metrics_filename}")


    # # ============================================
    # # 11. 加载最佳模型
    # # ============================================
    checkpoint_path = f'best_model_kd_meta_stage_{stage}.pt'
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        student_model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"\n{'='*80}")
        print(f"Training Completed!")
        print(f"  Best Val F1:        {checkpoint['val_f1']:.4f}")
        print(f"  Best Val Precision: {checkpoint['val_precision']:.4f}")
        print(f"  Best Val Recall:    {checkpoint['val_recall']:.4f}")
        print(f"  Best Val Loss:      {checkpoint['val_loss']:.4f}")
        print(f"  Best Epoch:         {checkpoint['epoch']}")
        print(f"{'='*80}\n")
    
    return student_model



# =========================
# 9. 评估函数
# =========================
def evaluate_enhanced(model, tokenizer, prompt_manager,partition,target_data_path, stage=2,target_domain='depression'):
    """
    增强版评估：计算三元组级别和单元素级别的Accuracy, Precision, Recall, F1
    """
    print(f"{'='*60}")
    print("Enhanced Evaluation")
    print(f"{'='*60}")
    
    
    ### 单独测试stage2后面的需要
    # checkpoint = torch.load(f'best_model_stage_{stage}.pt')
    # model.load_state_dict(checkpoint['model_state_dict'])
    # print(f"\nLoaded best model from stage {stage}")
    
    test_dataset = EnhancedTripletDataset(
        target_data_path + 'test.txt', tokenizer, prompt_manager, partition,stage=stage,domain=target_domain,use_simple_prompt=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=4, shuffle=False, collate_fn=custom_collate_fn
    )
    
    model.eval()
    
    # 三元组级别指标（每个样本）
    triplet_accuracies = []
    triplet_precisions = []
    triplet_recalls = []
    triplet_f1s = []
    
    # 单元素级别指标（每个样本）
    aspect_accuracies = []
    aspect_precisions = []
    aspect_recalls = []
    aspect_f1s = []
    
    opinion_accuracies = []
    opinion_precisions = []
    opinion_recalls = []
    opinion_f1s = []
    
    sentiment_accuracies = []
    sentiment_precisions = []
    sentiment_recalls = []
    sentiment_f1s = []
    
    predictions_log = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].cuda()
            attention_mask = batch['attention_mask'].cuda()
            
            # 生成预测
            outputs = model.t5.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=256,
                num_beams=5,
                early_stopping=True
            )
            
            # 解码
            pred_text = tokenizer.batch_decode(outputs.tolist(), skip_special_tokens=True)
            # labels = batch['labels'].cuda()
            # labels_for_decode = labels[0].clone()
            # labels_for_decode[labels_for_decode == -100] = tokenizer.pad_token_id
            # ref_text = tokenizer.decode(labels_for_decode, skip_special_tokens=True)
            
            # 解析三元组
            pred_triplets = [parse_triplets(text) for text in pred_text]
            ref_triplets = batch['raw_triplets']
            
            # 1. 计算三元组级别指标
            triplet_metrics = calculate_metrics(pred_triplets, ref_triplets)
            triplet_accuracies.append(triplet_metrics['accuracy'])
            triplet_precisions.append(triplet_metrics['precision'])
            triplet_recalls.append(triplet_metrics['recall'])
            triplet_f1s.append(triplet_metrics['f1'])
            
            # 2. 计算单元素级别指标
            element_metrics = calculate_element_metrics(pred_triplets, ref_triplets)
            
            # Aspect
            aspect_accuracies.append(element_metrics['aspect']['accuracy'])
            aspect_precisions.append(element_metrics['aspect']['precision'])
            aspect_recalls.append(element_metrics['aspect']['recall'])
            aspect_f1s.append(element_metrics['aspect']['f1'])
            
            # Opinion
            opinion_accuracies.append(element_metrics['opinion']['accuracy'])
            opinion_precisions.append(element_metrics['opinion']['precision'])
            opinion_recalls.append(element_metrics['opinion']['recall'])
            opinion_f1s.append(element_metrics['opinion']['f1'])
            
            # Sentiment
            sentiment_accuracies.append(element_metrics['sentiment']['accuracy'])
            sentiment_precisions.append(element_metrics['sentiment']['precision'])
            sentiment_recalls.append(element_metrics['sentiment']['recall'])
            sentiment_f1s.append(element_metrics['sentiment']['f1'])
            
    
    # 计算平均指标
    num_samples = len(test_loader)
    
    # 三元组级别
    avg_triplet_accuracy = sum(triplet_accuracies) / num_samples
    avg_triplet_precision = sum(triplet_precisions) / num_samples
    avg_triplet_recall = sum(triplet_recalls) / num_samples
    avg_triplet_f1 = sum(triplet_f1s) / num_samples
    
    # 单元素级别
    avg_aspect_accuracy = sum(aspect_accuracies) / num_samples
    avg_aspect_precision = sum(aspect_precisions) / num_samples
    avg_aspect_recall = sum(aspect_recalls) / num_samples
    avg_aspect_f1 = sum(aspect_f1s) / num_samples
    
    avg_opinion_accuracy = sum(opinion_accuracies) / num_samples
    avg_opinion_precision = sum(opinion_precisions) / num_samples
    avg_opinion_recall = sum(opinion_recalls) / num_samples
    avg_opinion_f1 = sum(opinion_f1s) / num_samples
    
    avg_sentiment_accuracy = sum(sentiment_accuracies) / num_samples
    avg_sentiment_precision = sum(sentiment_precisions) / num_samples
    avg_sentiment_recall = sum(sentiment_recalls) / num_samples
    avg_sentiment_f1 = sum(sentiment_f1s) / num_samples
    
    # 打印结果
    print(f"{'='*60}")
    print("Evaluation Results:")
    print(f"{'='*60}")
    
    print(f"\n[Triplet-level Metrics]")
    print(f"  Accuracy:  {avg_triplet_accuracy:.4f}")
    print(f"  Precision: {avg_triplet_precision:.4f}")
    print(f"  Recall:    {avg_triplet_recall:.4f}")
    print(f"  F1 Score:  {avg_triplet_f1:.4f}")
    
    print(f"\n[Element-level Metrics]")
    print(f"\n  Aspect:")
    print(f"    Accuracy:  {avg_aspect_accuracy:.4f}")
    print(f"    Precision: {avg_aspect_precision:.4f}")
    print(f"    Recall:    {avg_aspect_recall:.4f}")
    print(f"    F1 Score:  {avg_aspect_f1:.4f}")
    
    print(f"\n  Opinion:")
    print(f"    Accuracy:  {avg_opinion_accuracy:.4f}")
    print(f"    Precision: {avg_opinion_precision:.4f}")
    print(f"    Recall:    {avg_opinion_recall:.4f}")
    print(f"    F1 Score:  {avg_opinion_f1:.4f}")
    
    print(f"\n  Sentiment:")
    print(f"    Accuracy:  {avg_sentiment_accuracy:.4f}")
    print(f"    Precision: {avg_sentiment_precision:.4f}")
    print(f"    Recall:    {avg_sentiment_recall:.4f}")
    print(f"    F1 Score:  {avg_sentiment_f1:.4f}")
    
    # 返回完整指标
    return {
        'triplet': {
            'accuracy': avg_triplet_accuracy,
            'precision': avg_triplet_precision,
            'recall': avg_triplet_recall,
            'f1': avg_triplet_f1
        },
        'element': {
            'aspect': {
                'accuracy': avg_aspect_accuracy,
                'precision': avg_aspect_precision,
                'recall': avg_aspect_recall,
                'f1': avg_aspect_f1
            },
            'opinion': {
                'accuracy': avg_opinion_accuracy,
                'precision': avg_opinion_precision,
                'recall': avg_opinion_recall,
                'f1': avg_opinion_f1
            },
            'sentiment': {
                'accuracy': avg_sentiment_accuracy,
                'precision': avg_sentiment_precision,
                'recall': avg_sentiment_recall,
                'f1': avg_sentiment_f1
            }
        }
    }


# =========================
# 10. 辅助函数
# =========================
def parse_triplets(text):
    """解析生成的三元组文本"""
    triplets = []
    try:
        parts = text.split('|')
        for part in parts:
            part = part.strip()
            if part.startswith('(') and part.endswith(')'):
                content = part[1:-1]
                elements = [e.strip() for e in content.split(',')]
                if len(elements) == 3:
                    triplets.append(tuple(elements))
    except:
        pass
    return triplets




## 三元组指标
# def calculate_metrics(pred_triplets, ref_triplets):
    
#     # 根据输入嵌套列表来计算accuracy和micro三大指标
#     # 支持 pred_triplets 和 ref_triplets 都为嵌套列表（批量）
#     # 单个样本: [(), (), ...]；多样本: [[(), ()], [], ...]
#     # 返回micro的accuracy, precision, recall, f1

#     # 首先判断是否为批量（嵌套列表）:
#     def _is_nested(l):
#         return isinstance(l, list) and len(l) > 0 and isinstance(l[0], (list, tuple))
    
#     # 批量场景
#     if _is_nested(pred_triplets) and _is_nested(ref_triplets):
#         total_correct = 0
#         total_pred = 0
#         total_ref = 0
#         sample_accs = []
#         for preds, refs in zip(pred_triplets, ref_triplets):
#             pred_set = set(preds)
#             ref_set = set(refs)

#             # 完全匹配
#             if len(pred_set) == 0 and len(ref_set) == 0:
#                 acc = 1.0
#             else:
#                 acc = 1.0 if pred_set == ref_set else 0.0
#             sample_accs.append(acc)

#             total_correct += len(pred_set & ref_set)
#             total_pred += len(pred_set)
#             total_ref += len(ref_set)

#         micro_accuracy = sum(sample_accs) / len(sample_accs) if sample_accs else 1.0

#         if total_pred == 0 and total_ref == 0:
#             return {'accuracy': 1.0, 'precision': 1.0, 'recall': 1.0, 'f1': 1.0}
#         if total_pred == 0 or total_ref == 0:
#             return {'accuracy': micro_accuracy, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

#         precision = total_correct / total_pred if total_pred > 0 else 0.0
#         recall = total_correct / total_ref if total_ref > 0 else 0.0
#         f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

#         return {
#             'accuracy': micro_accuracy,
#             'precision': precision,
#             'recall': recall,
#             'f1': f1
#         }


def calculate_metrics(pred_pt, gold_pt):
    """
    Function to compute F1 scores with pred and gold pairs/triplets
    The input needs to be already processed
    """
    # number of true postive, gold standard, predicted aspect terms
    ## 将pred_pt和gold_pt中的元组变成列表
    pred_pt = [list(i) for i in pred_pt]
    gold_pt = [list(i) for i in gold_pt]
    
    n_tp, n_gold, n_pred = 0, 0, 0
    gold_pt = copy.deepcopy(gold_pt)

    for i in range(len(pred_pt)):
        n_gold += len(gold_pt[i])
        n_pred += len(pred_pt[i])

        for t in pred_pt[i]:
            if t in gold_pt[i]:
                # to prevent generate same correct answer and get recall larger than 1
                gold_pt[i].remove(t)
                n_tp += 1

    precision = float(n_tp) / float(n_pred) if n_pred != 0 else 0
    recall = float(n_tp) / float(n_gold) if n_gold != 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision != 0 or recall != 0 else 0
    scores = {"accuracy":0,'precision': precision, 'recall': recall, 'f1': f1}

    return scores 



## 单元素指标
def calculate_element_metrics(pred_triplets, ref_triplets):
    
    # 计算单元素级别的指标，支持单样本和批量
    def _is_nested(l):
        return isinstance(l, list) and len(l) > 0 and isinstance(l[0], (list, tuple))
    
    def calc_single_element(pred_elements, ref_elements):
        """计算单个元素类型的指标"""
        pred_set = set(pred_elements)
        ref_set = set(ref_elements)
        
        # 完全匹配
        if len(pred_elements) == 0 and len(ref_elements) == 0:
            accuracy = 1.0
        else:
            accuracy = 1.0 if pred_set == ref_set else 0.0
        
        # Precision, Recall, F1
        if len(pred_set) == 0 and len(ref_set) == 0:
            return {'accuracy': 1.0, 'precision': 1.0, 'recall': 1.0, 'f1': 1.0}
        if len(pred_set) == 0 or len(ref_set) == 0:
            return {'accuracy': accuracy, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        correct = len(pred_set & ref_set)
        precision = correct / len(pred_set)
        recall = correct / len(ref_set)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    # 支持批量
    if _is_nested(pred_triplets) and _is_nested(ref_triplets):
        # 批量情形：返回每个元素类型的平均值
        aspect_metrics, opinion_metrics, sentiment_metrics = [], [], []
        for preds, refs in zip(pred_triplets, ref_triplets):
            pred_aspects = [t[0] for t in preds]
            ref_aspects = [t[0] for t in refs]
            pred_opinions = [t[1] for t in preds]
            ref_opinions = [t[1] for t in refs]
            pred_sentiments = [t[2] for t in preds]
            ref_sentiments = [t[2] for t in refs]
            
            aspect_metrics.append(calc_single_element(pred_aspects, ref_aspects))
            opinion_metrics.append(calc_single_element(pred_opinions, ref_opinions))
            sentiment_metrics.append(calc_single_element(pred_sentiments, ref_sentiments))
        
        def average_metrics(metrics_list):
            keys = metrics_list[0].keys()
            avg = {k: sum(m[k] for m in metrics_list)/len(metrics_list) if metrics_list else 0.0 for k in keys}
            return avg
        
        return {
            'aspect': average_metrics(aspect_metrics),
            'opinion': average_metrics(opinion_metrics),
            'sentiment': average_metrics(sentiment_metrics)
        }
    else:
        # 单样本情形
        pred_aspects = [t[0] for t in pred_triplets]
        ref_aspects = [t[0] for t in ref_triplets]
        pred_opinions = [t[1] for t in pred_triplets]
        ref_opinions = [t[1] for t in ref_triplets]
        pred_sentiments = [t[2] for t in pred_triplets]
        ref_sentiments = [t[2] for t in ref_triplets]

        return {
            'aspect': calc_single_element(pred_aspects, ref_aspects),
            'opinion': calc_single_element(pred_opinions, ref_opinions),
            'sentiment': calc_single_element(pred_sentiments, ref_sentiments)
        }
    




# =========================
# 对比学习模块（更新版：监督对比学习 + 跨域正样本）
# =========================
class ContrastiveLearningModule(nn.Module):
    """跨域监督对比学习模块"""
    def __init__(self, hidden_dim=768, projection_dim=256, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, projection_dim)
        )

    def forward(self, features):
        """特征投影 + L2归一化"""
        projected = self.projection_head(features)
        return F.normalize(projected, dim=-1)
    
    def compute_supcon_loss(self, source_features, target_features, labels_source, labels_target):
        """
        监督式对比学习损失 (SupCon) + 跨域正样本机制
        """
        device = source_features.device
        
        # 特征投影
        z_s = self.forward(source_features)
        z_t = self.forward(target_features)
        
        # 合并样本
        z_all = torch.cat([z_s, z_t], dim=0)
        labels_all = torch.cat([labels_source, labels_target], dim=0)
        num_samples = z_all.size(0)
        
        # 计算相似度矩阵
        sim_matrix = torch.matmul(z_all, z_all.T) / self.temperature
        
        # 排除自身
        mask_self = torch.eye(num_samples, dtype=torch.bool, device=device)
        
        # 构造正样本集合 P(i)：同标签 + 不同样本
        pos_mask = (labels_all.unsqueeze(0) == labels_all.unsqueeze(1)) & ~mask_self
        
        # 分母集合 A(i)：所有其他样本
        all_mask = ~mask_self
        
        losses = []
        for i in range(num_samples):
            pos_i = pos_mask[i]
            if pos_i.sum() == 0:
                continue
            exp_sim_all = torch.exp(sim_matrix[i][all_mask[i]])
            exp_sim_pos = torch.exp(sim_matrix[i][pos_i])
            
            loss_i = -torch.log(exp_sim_pos.sum() / exp_sim_all.sum())
            losses.append(loss_i)
        
        if len(losses) == 0:
            return torch.tensor(0.0, device=device)
        
        return torch.stack(losses).mean()
    
    def compute_domain_alignment_loss(self, source_features, target_features):
        """域对齐损失（MMD简化版）"""
        z_source = self.forward(source_features)
        z_target = self.forward(target_features)
        mmd_loss = F.mse_loss(z_source.mean(0), z_target.mean(0))
        return mmd_loss

    ## 仅计算源域和目标域内部的监督对比损失（域内正样本），不跨域正样本
    def compute_supcon_loss_intra_domain(self, source_features, target_features, labels_source, labels_target):
        """仅计算源域和目标域内部的监督对比损失（域内正样本），不跨域正样本"""
        device = source_features.device

        def supcon_single(features, labels):
            z = self.forward(features)
            labels = labels
            num_samples = z.size(0)
            sim_matrix = torch.matmul(z, z.T) / self.temperature
            mask_self = torch.eye(num_samples, dtype=torch.bool, device=device)
            pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~mask_self
            all_mask = ~mask_self
            losses = []
            for i in range(num_samples):
                pos_i = pos_mask[i]
                if pos_i.sum() == 0:
                    continue
                exp_sim_all = torch.exp(sim_matrix[i][all_mask[i]])
                exp_sim_pos = torch.exp(sim_matrix[i][pos_i])
                loss_i = -torch.log(exp_sim_pos.sum() / exp_sim_all.sum())
                losses.append(loss_i)
            if len(losses) == 0:
                return torch.tensor(0.0, device=device)
            return torch.stack(losses).mean()

        loss_source = supcon_single(source_features, labels_source)
        loss_target = supcon_single(target_features, labels_target)
        return (loss_source + loss_target) / 2
    



def train_stage1_with_contrastive(model, tokenizer, prompt_manager, partition,source_data_path, target_data_path, 
                    source_domain, target_domain, epochs=20, patience=3,lambda_c=0.1, lambda_m=0.05,use_cross_domain_positive=True,target_test=True,visualize_features=True):
    """
    Stage 1: 跨域对齐 + 监督对比学习（SupCon）
    Loss = L_task + λ_c * L_contrast + λ_m * L_alignment
    """
    print(f"{'='*80}")
    print(f"Stage 1: Cross-Domain Supervised Contrastive Learning (SupCon)")
    print(f"Cross-Domain Positive Samples: {'Enabled' if use_cross_domain_positive else 'Disabled (Intra-domain only)'}")  # ✅ 新增这行
    print(f"{'='*80}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    contrastive_module = ContrastiveLearningModule(
        hidden_dim=model.t5.config.d_model,
        projection_dim=256,
        temperature=0.07
    ).to(device)
    
    # 构建数据集
    source_train = EnhancedTripletDataset(source_data_path + 'train.txt', tokenizer, prompt_manager, partition,stage=1,domain=source_domain,use_simple_prompt=False)
    target_train = EnhancedTripletDataset(target_data_path + 'train.txt', tokenizer, prompt_manager, partition,stage=1,domain=target_domain,use_simple_prompt=False)
    val_dataset = EnhancedTripletDataset(target_data_path + 'dev.txt', tokenizer, prompt_manager, partition,stage=1,domain=target_domain,use_simple_prompt=False)
    
    test_dataset = EnhancedTripletDataset(target_data_path + 'test.txt', tokenizer, prompt_manager, partition,stage=1,domain=target_domain,use_simple_prompt=False)
    

    # DataLoader
    source_loader = DataLoader(source_train, batch_size=3, shuffle=True, collate_fn=custom_collate_fn)
    target_loader = DataLoader(target_train, batch_size=1, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=custom_collate_fn)

    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=custom_collate_fn)
    
    optimizer = optim.AdamW(list(model.parameters()) + list(contrastive_module.parameters()), lr=3e-5)
    
    best_val_f1, patience_counter = 0.0, 0


     ## 存放训练过程中的accuracy和F1
    acc_score = []
    pre_score = []
    rec_score = []
    F1_score = []
    for epoch in range(epochs):
        model.train()
        contrastive_module.train()
        total_task_loss, total_contrastive_loss, total_alignment_loss = 0, 0, 0
        
        progress_bar = tqdm(
            zip(source_loader, target_loader), 
            total=min(len(source_loader), len(target_loader)), 
            desc=f"Epoch {epoch+1}/{epochs}"
        )
            
        for batch_s, batch_t in progress_bar:
            optimizer.zero_grad()
            
            # 源域
            input_s = batch_s['input_ids'].to(device)
            att_s = batch_s['attention_mask'].to(device)
            label_s = batch_s['labels'].to(device)
            sentiment_s = batch_s['sentiment_label'].to(device)
            
            # 目标域
            input_t = batch_t['input_ids'].to(device)
            att_t = batch_t['attention_mask'].to(device)
            label_t = batch_t['labels'].to(device)
            sentiment_t = batch_t['sentiment_label'].to(device)
            
            out_s = model.t5(input_ids=input_s, attention_mask=att_s, labels=label_s, output_hidden_states=True)
            out_t = model.t5(input_ids=input_t, attention_mask=att_t, labels=label_t, output_hidden_states=True)
            
            task_loss = 0.5 * (out_s.loss + out_t.loss)
            
            # 提取编码特征
            feat_s = (out_s.encoder_last_hidden_state * att_s.unsqueeze(-1)).sum(1) / att_s.sum(-1, keepdim=True)
            feat_t = (out_t.encoder_last_hidden_state * att_t.unsqueeze(-1)).sum(1) / att_t.sum(-1, keepdim=True)
            
            # ## 对比损失和对齐损失
            # contrastive_loss = contrastive_module.compute_supcon_loss(feat_s, feat_t, sentiment_s, sentiment_t)
            # alignment_loss = contrastive_module.compute_domain_alignment_loss(feat_s, feat_t)

            # ✅ 根据参数选择对比损失计算方式
            if use_cross_domain_positive:
                # 跨域正样本（原始版本）
                contrastive_loss = contrastive_module.compute_supcon_loss(
                    feat_s, feat_t, sentiment_s, sentiment_t
                )
            else:
                # 仅域内正样本（消融版本）
                contrastive_loss = contrastive_module.compute_supcon_loss_intra_domain(
                    feat_s, feat_t, sentiment_s, sentiment_t
                )
            
            alignment_loss = contrastive_module.compute_domain_alignment_loss(feat_s, feat_t)

            
            ## 总损失 = 任务损失 + 对比损失 + 对齐损失
            total_loss = task_loss + lambda_c * contrastive_loss + lambda_m * alignment_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(contrastive_module.parameters()), 1.0)
            optimizer.step()
            
            total_task_loss += task_loss.item()
            total_contrastive_loss += contrastive_loss.item()
            total_alignment_loss += alignment_loss.item()
            
            progress_bar.set_postfix({
                'task': task_loss.item(),
                'contrast': contrastive_loss.item(),
                'align': alignment_loss.item()
            })
        
        print(f"Epoch {epoch+1} Finished")
        print(f"Task: {total_task_loss/len(source_loader):.4f}")
        print(f"Contrast: {total_contrastive_loss/len(source_loader):.4f}")
        print(f"Align: {total_alignment_loss/len(source_loader):.4f}")
        
        # 验证
        
        print("="*50)
        print("Validation")
        print("="*50)
        model.eval()
        all_triplet_f1s = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating", leave=False):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                # 生成预测
                outputs = model.t5.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=256,
                    num_beams=3,
                    early_stopping=True
                )
                
                # 🔥 修正：批次解码

                pred_text = tokenizer.batch_decode(outputs.tolist(), skip_special_tokens=True)
                pred_triplets = [parse_triplets(text) for text in pred_text]
                ref_triplets = batch['raw_triplets']
                
                metrics = calculate_metrics(pred_triplets, ref_triplets)
                all_triplet_f1s.append(metrics['f1'])
                
        avg_val_f1 = sum(all_triplet_f1s) / len(all_triplet_f1s) if all_triplet_f1s else 0.0
        print(f"Validation F1: {avg_val_f1:.4f}")
        
        if avg_val_f1 > best_val_f1:
            best_val_f1 = avg_val_f1
            patience_counter = 0
            torch.save({
                'model': model.state_dict(),
                'contrastive': contrastive_module.state_dict(),
                'epoch': epoch+1,
                'val_f1': avg_val_f1
            }, 'best_model_stage_1_contrastive.pt')
            
            print(f"✓ New best model saved (F1={avg_val_f1:.4f})")
            
        else:
            patience_counter += 1
            print(f"  ✗ No improvement (Best F1: {best_val_f1:.4f}, {patience_counter}/{patience})")
            
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                break
        

        if target_test:
            # 在目标域测试集上测试
            print("="*50)
            print("Testing")
            print("="*50)
            model.eval()
            test_triplet_accuracy = []
            test_triplet_precision = []
            test_triplet_recall = []
            test_triplet_f1s = []

            with torch.no_grad():
                for batch in tqdm(test_loader, desc="Testing", leave=False):
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    
                    # 生成预测
                    outputs = model.t5.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_length=256,
                        num_beams=3,
                        early_stopping=True
                    )
                    
                    # 🔥 修正：批次解码

                    pred_text = tokenizer.batch_decode(outputs.tolist(), skip_special_tokens=True)
                    pred_triplets = [parse_triplets(text) for text in pred_text]
                    ref_triplets = batch['raw_triplets']
                    
                    metrics = calculate_metrics(pred_triplets, ref_triplets)
                    test_triplet_accuracy.append(metrics['accuracy'])
                    test_triplet_precision.append(metrics['precision'])
                    test_triplet_recall.append(metrics['recall'])
                    test_triplet_f1s.append(metrics['f1'])
                    
            avg_test_f1 = sum(test_triplet_f1s) / len(test_triplet_f1s) if test_triplet_f1s else 0.0
            # 计算 accuracy、Precision 和 Recall
            avg_test_accuracy = sum(test_triplet_accuracy) / len(test_triplet_accuracy) if test_triplet_accuracy else 0.0
            avg_test_precision = sum(test_triplet_precision) / len(test_triplet_precision) if test_triplet_precision else 0.0
            avg_test_recall = sum(test_triplet_recall) / len(test_triplet_recall) if test_triplet_recall else 0.0

            # INSERT_YOUR_CODE
            acc_score.append(avg_test_accuracy)
            pre_score.append(avg_test_precision)
            rec_score.append(avg_test_recall)
            F1_score.append(avg_test_f1)

            print(f"Test Accuracy:  {avg_test_accuracy:.4f}")
            print(f"Test Precision: {avg_test_precision:.4f}")
            print(f"Test Recall:    {avg_test_recall:.4f}")
            print(f"Test F1: {avg_test_f1:.4f}")

  
    print(f"Test Accuracy list    : {acc_score}")
    print(f"Test Precision list   : {pre_score}")
    print(f"Test Recall list      : {rec_score}")
    print(f"Test F1 Score list    : {F1_score}")
    
    # 将指标列表保存至txt文件中
    metrics_filename = 'test_metrics_stage_1.txt'
    with open(metrics_filename, "w", encoding="utf-8") as f:
        f.write(f"Test Accuracy list    : {acc_score}\n")
        f.write(f"Test Precision list   : {pre_score}\n")
        f.write(f"Test Recall list      : {rec_score}\n")
        f.write(f"Test F1 Score list    : {F1_score}\n")


    checkpoint = torch.load('best_model_stage_1_contrastive.pt')
    model.load_state_dict(checkpoint['model'])
    contrastive_module.load_state_dict(checkpoint['contrastive'])
    print(f"✓ Loaded best model (F1={checkpoint['val_f1']:.4f})")


    # ========== 在训练结束后添加 ==========
    if visualize_features:
        print("\n" + "="*80)
        print("Extracting features for t-SNE visualization...")
        print("="*80)
        
        # 加载最佳模型
        checkpoint = torch.load('best_model_stage_1_contrastive.pt')
        model.load_state_dict(checkpoint['model'])
        contrastive_module.load_state_dict(checkpoint['contrastive'])
        
        # 提取特征
        features_dict = extract_features_for_visualization(
            model=model,
            contrastive_module=contrastive_module,
            source_loader=source_loader,
            target_loader=target_loader,
            device=device,
            max_samples=500  # 可调整
        )
        
        # 保存特征（可选）
        import pickle
        with open('features_for_tsne.pkl', 'wb') as f:
            pickle.dump(features_dict, f)
        print("✓ Features saved to features_for_tsne.pkl")
        
        # t-SNE可视化
        visualize_features_tsne(
            features_dict=features_dict,
            save_path=f'tsne_{source_domain}_to_{target_domain}.pdf'
        )


    return model


def extract_features_for_visualization(model, contrastive_module, source_loader, target_loader, device, max_samples=500):
    """
    提取源域和目标域的特征用于可视化
    
    Args:
        model: T5模型
        contrastive_module: 对比学习模块
        source_loader: 源域数据加载器
        target_loader: 目标域数据加载器
        device: 设备
        max_samples: 每个域最多提取的样本数
    
    Returns:
        features_dict: 包含对齐前后特征的字典
    """
    model.eval()
    contrastive_module.eval()
    
    # 存储特征
    source_features_before = []  # 对齐前（编码器输出）
    source_features_after = []   # 对齐后（投影后）
    source_labels = []
    
    target_features_before = []
    target_features_after = []
    target_labels = []
    
    with torch.no_grad():
        # 提取源域特征
        print("Extracting source domain features...")
        for i, batch in enumerate(tqdm(source_loader)):
            if i * source_loader.batch_size >= max_samples:
                break
                
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            sentiment_labels = batch['sentiment_label'].to(device)
            
            # 获取编码器输出
            outputs = model.t5.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            
            # 对齐前特征：编码器最后一层的平均池化
            encoder_hidden = outputs.last_hidden_state
            feat_before = (encoder_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1, keepdim=True)
            
            # 对齐后特征：通过投影头
            feat_after = contrastive_module.forward(feat_before)
            
            source_features_before.append(feat_before.cpu())
            source_features_after.append(feat_after.cpu())
            source_labels.append(sentiment_labels.cpu())
        
        # 提取目标域特征
        print("Extracting target domain features...")
        for i, batch in enumerate(tqdm(target_loader)):
            if i * target_loader.batch_size >= max_samples:
                break
                
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            sentiment_labels = batch['sentiment_label'].to(device)
            
            outputs = model.t5.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            
            encoder_hidden = outputs.last_hidden_state
            feat_before = (encoder_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1, keepdim=True)
            feat_after = contrastive_module.forward(feat_before)
            
            target_features_before.append(feat_before.cpu())
            target_features_after.append(feat_after.cpu())
            target_labels.append(sentiment_labels.cpu())
    
    # 合并所有batch
    features_dict = {
        'source_before': torch.cat(source_features_before, dim=0).numpy(),
        'source_after': torch.cat(source_features_after, dim=0).numpy(),
        'source_labels': torch.cat(source_labels, dim=0).numpy(),
        'target_before': torch.cat(target_features_before, dim=0).numpy(),
        'target_after': torch.cat(target_features_after, dim=0).numpy(),
        'target_labels': torch.cat(target_labels, dim=0).numpy(),
    }
    
    print(f"✓ Feature extraction completed")
    print(f"  Source samples: {features_dict['source_before'].shape[0]}")
    print(f"  Target samples: {features_dict['target_before'].shape[0]}")
    
    return features_dict



def visualize_features_tsne(features_dict, save_path='feature_alignment_tsne.pdf'):
    """
    使用t-SNE可视化对齐前后的特征分布（仅显示正负情感）
    
    Args:
        features_dict: 特征字典
        save_path: 保存路径
    """
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 10
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # ✅ 修改1：情感标签到颜色的映射（只保留正负）
    sentiment_colors = {
        0: '#E74C3C',  # 负面 - 红色
        2: '#27AE60',  # 正面 - 绿色
    }
    sentiment_names = {0: 'Negative', 2: 'Positive'}
    
    # ========== 对齐前 ==========
    print("Computing t-SNE for features before alignment...")
    
    # ✅ 修改2：过滤掉中性样本（label != 1）
    source_mask_before = features_dict['source_labels'] != 1
    target_mask_before = features_dict['target_labels'] != 1
    
    features_before = np.vstack([
        features_dict['source_before'][source_mask_before],
        features_dict['target_before'][target_mask_before]
    ])
    
    labels_before = np.concatenate([
        features_dict['source_labels'][source_mask_before],
        features_dict['target_labels'][target_mask_before]
    ])
    
    domains_before = np.array(
        ['Source'] * source_mask_before.sum() + 
        ['Target'] * target_mask_before.sum()
    )
    
    # t-SNE降维
    tsne_before = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    features_2d_before = tsne_before.fit_transform(features_before)
    
    # 绘制对齐前
    ax = axes[0]
    for domain, marker in [('Source', 'o'), ('Target', '^')]:
        mask = domains_before == domain
        # ✅ 修改3：只遍历0和2（负面和正面）
        for sentiment in [0, 2]:
            sentiment_mask = labels_before == sentiment
            combined_mask = mask & sentiment_mask
            
            if combined_mask.sum() > 0:
                ax.scatter(
                    features_2d_before[combined_mask, 0],
                    features_2d_before[combined_mask, 1],
                    c=sentiment_colors[sentiment],
                    marker=marker,
                    s=30,
                    alpha=0.6,
                    label=f'{domain}-{sentiment_names[sentiment]}',
                    edgecolors='white',
                    linewidths=0.5
                )
    
    ax.set_title('Before Alignment', fontsize=12, fontweight='bold')
    ax.set_xlabel('t-SNE Dimension 1', fontsize=10, fontweight='bold')
    ax.set_ylabel('t-SNE Dimension 2', fontsize=10, fontweight='bold')
    ax.legend(loc='best', fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # ========== 对齐后 ==========
    print("Computing t-SNE for features after alignment...")
    
    # ✅ 修改4：过滤掉中性样本（label != 1）
    source_mask_after = features_dict['source_labels'] != 1
    target_mask_after = features_dict['target_labels'] != 1
    
    features_after = np.vstack([
        features_dict['source_after'][source_mask_after],
        features_dict['target_after'][target_mask_after]
    ])
    
    labels_after = np.concatenate([
        features_dict['source_labels'][source_mask_after],
        features_dict['target_labels'][target_mask_after]
    ])
    
    domains_after = np.array(
        ['Source'] * source_mask_after.sum() + 
        ['Target'] * target_mask_after.sum()
    )
    
    tsne_after = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    features_2d_after = tsne_after.fit_transform(features_after)
    
    # 绘制对齐后
    ax = axes[1]
    for domain, marker in [('Source', 'o'), ('Target', '^')]:
        mask = domains_after == domain
        # ✅ 修改5：只遍历0和2（负面和正面）
        for sentiment in [0, 2]:
            sentiment_mask = labels_after == sentiment
            combined_mask = mask & sentiment_mask
            
            if combined_mask.sum() > 0:
                ax.scatter(
                    features_2d_after[combined_mask, 0],
                    features_2d_after[combined_mask, 1],
                    c=sentiment_colors[sentiment],
                    marker=marker,
                    s=30,
                    alpha=0.6,
                    label=f'{domain}-{sentiment_names[sentiment]}',
                    edgecolors='white',
                    linewidths=0.5
                )
    
    ax.set_title('After Alignment', fontsize=12, fontweight='bold')
    ax.set_xlabel('t-SNE Dimension 1', fontsize=10, fontweight='bold')
    ax.set_ylabel('t-SNE Dimension 2', fontsize=10, fontweight='bold')
    ax.legend(loc='best', fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.pdf', '.svg'), bbox_inches='tight')
    print(f"✓ Visualization saved to {save_path}")
    plt.show()




def main_with_kd_and_meta(
    partition,
    resume_from_phase=None,  # 从哪个阶段开始：1, 2, 3, 4
    teacher_checkpoint_paths=None,  # 教师模型路径列表
    student_checkpoint_path=None,  # 学生模型路径
    source_domain_index=0,    # Stage 1使用哪个源域
    target_domain_index=0       ## stage 1,2阶段使用的目标域
):
    
    """集成知识蒸馏和元学习的完整训练流程"""
    print(f"{'='*80}")
    print("TRAINING WITH KNOWLEDGE DISTILLATION + META-LEARNING")
    if resume_from_phase:
        print(f"RESUMING FROM PHASE {resume_from_phase}")
    print(f"{'='*80}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    teacher_name = 'google/flan-t5-large'
    student_name = 't5-small'
    
    
    teacher_tokenizer = T5Tokenizer.from_pretrained(teacher_name)
    student_tokenizer = T5Tokenizer.from_pretrained(student_name)
    
    teacher = T5ForConditionalGeneration.from_pretrained(teacher_name)
    student = T5ForConditionalGeneration.from_pretrained(student_name)

    def count_params(model):
        return sum(p.numel() for p in model.parameters())

    print(f"Teacher model:{teacher_name},Teacher params:{count_params(teacher)}")
    print(f"Student model:{student_name},Student params:{count_params(student)}" )
    
    ## 提示词
    prompt_manager = AdvancedPromptManager()
    
    source_paths = ['data/laptop/', 'data/rest/','data/rest15/','data/rest16/']
    target_paths = ['data/twitter/','data/reddit/']


    source_map = {'data/laptop/':'laptop','data/rest/':'restaurant','data/rest15/':'restaurant','data/rest16/':'restaurant'}
    target = 'depression'   #目标域
    
    
    ## 针对CD-ASTE任务
    # source_paths = ['data/laptop16_mine/', 'data/rest14_mine/','data/rest15_mine/','data/rest16_mine/']
    # target_paths = ['data/laptop16_mine/', 'data/rest14_mine/','data/rest15_mine/','data/rest16_mine/']
    
    # source_map = {'data/laptop16_mine/':'laptop','data/rest14_mine/':'restaurant','data/rest15_mine/':'restaurant','data/rest16_mine/':'restaurant'}
    # target = 'restaurant'
    # # target = 'laptop'
    

    
    print(f"Cross domain: {source_paths[source_domain_index]}-->{target_paths[target_domain_index]}")
    
    teacher_models = []
    student_model = None
    
    
        
     # ========== 阶段1: 训练教师模型 ==========
    if resume_from_phase is None or resume_from_phase <= 1:
        if resume_from_phase == 1:
            # 加载已训练的教师模型
            print(f"[Phase 1] Loading Pre-trained Teacher Models")
            print(f"{'='*80}\n")
            
            if teacher_checkpoint_paths is None:
                teacher_checkpoint_paths = [f'teacher_model_{i+1}.pt' for i in range(len(source_paths))]
            
            for i, ckpt_path in enumerate(teacher_checkpoint_paths):
                print(f"Loading Teacher {i+1} from {ckpt_path}")
                teacher = DeepProgressivePromptModel(teacher_name).to(device)
                checkpoint = torch.load(ckpt_path, map_location=device)
                teacher.load_state_dict(checkpoint['model_state_dict'])
                teacher_models.append(teacher)
                print(f"✓ Teacher {i+1} loaded\n")
            
    
        else:
            print(f"[Phase 1] Training Teacher Models on Source Domains")
            print(f"{'='*80}\n")
            for i, source_path in enumerate(source_paths):
                print(f"\n[Teacher {i+1}] Training on {source_path}")
                
                teacher = DeepProgressivePromptModel(teacher_name).to(device)
                
                train_stage_enhanced(
                    teacher, teacher_tokenizer, prompt_manager,
                    partition,
                    source_path, source_path,   ##教师训练包括所有源域
                    source_domain = source_map[source_path],
                    target_domain = target,
                    stage=0,
                    epochs=20,
                    patience=3
                )
                
                torch.save({
                    'model_state_dict': teacher.state_dict(),
                    'source_domain': source_path
                }, f'teacher_model_{i+1}.pt')
                
                teacher_models.append(teacher)
                print(f"✓ Teacher {i+1} training completed\n")
        
    
    
    # ========== 阶段2: 知识蒸馏训练学生 ==========
    if resume_from_phase is None or resume_from_phase <= 2:
        if resume_from_phase == 2:
            print(f"[Phase 2] Skipping - Loading Pre-trained Student Model")
            print(f"{'='*80}")
            
            if student_checkpoint_path is None:
                student_checkpoint_path = 'best_model_kd_meta_stage_0.pt'   ## 蒸馏的学生模型
            
            print(f"Loading Student Model from {student_checkpoint_path}")
            student_model = DeepProgressivePromptModel(student_name).to(device)
            checkpoint = torch.load(student_checkpoint_path, map_location=device)
            student_model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✓ Student Model loaded")
        
        else:
            print(f"\n[Phase 2] Training Student Model with Knowledge Distillation")
            print(f"{'='*80}")
            
            student_model = DeepProgressivePromptModel(student_name).to(device)
            
            # 如果蒸馏中断，可以直接加载模型，继续训练
            # checkpoint = torch.load('best_model_kd_meta_stage_0.pt')
            # student_model.load_state_dict(checkpoint['model_state_dict'])
            # print("Loading trained student model: best_model_kd_meta_stage_0.pt")

 
            print(f"[Stage 0] Knowledge Distillation Training")
            student_model = train_with_distillation_and_metalearning(
                student_model=student_model,
                teacher_models=teacher_models,
                tokenizer=student_tokenizer,
                prompt_manager=prompt_manager,
                partition=partition,
                source_data_paths=source_paths,  ##蒸馏使用所有源域
                target_data_path=target_paths[target_domain_index],
                source_map = source_map,
                target_domain = target,
                stage=0,
                epochs=25,
                patience=3,
                use_multi_teacher=True,
                use_meta_learning=False,
                source_domain_index=0,   ##use_multi_teacher为False时，source_domain_index有效
                target_test = True
            )
            
            # 保存Stage 0的学生模型
            torch.save({
                'model_state_dict': student_model.state_dict(),
                'stage': 0
            }, 'best_model_kd_meta_stage_0.pt')
            print(f"✓ Student Stage 0 saved to best_model_kd_meta_stage_0.pt")
            
    
    # ========== 阶段3: 跨域对齐 ==========
    if resume_from_phase is None or resume_from_phase <= 3:
        if resume_from_phase == 3:
            print(f"[Phase 3] Skipping - Loading Stage 1 Model")
            print(f"{'='*80}\n")
            
            if student_checkpoint_path is None:
                student_checkpoint_path = f'student_stage1_source{source_domain_index}.pt'   ##跨域的学生模型
            
            print(f"Loading Student Model from {student_checkpoint_path}")
            student_model = DeepProgressivePromptModel(student_name).to(device)
            checkpoint = torch.load(student_checkpoint_path, map_location=device)
            student_model.load_state_dict(checkpoint['model_state_dict'],strict=False)
            print(f"✓ Student Stage 1 loaded")
        
        else:
            print(f"[Phase 3] Cross-Domain Alignment")
            print(f"Using source domain: {source_paths[source_domain_index]}")
            print(f"{'='*80}")
            
            # if student_model is None:
            #     # 如果没有student_model，加载Stage 0的
            #     print("Loading Student Stage 0 model...")
            #     student_model = DeepProgressivePromptModel(student_name).to(device)
            #     checkpoint = torch.load('student_stage0.pt', map_location=device)
            #     student_model.load_state_dict(checkpoint['model_state_dict'])
            
            # # 领域对齐
            # print(f"[Stage 1] Cross-Domain Alignment")
            # train_stage_enhanced(
            #     student_model, tokenizer, prompt_manager,partition
            #     source_paths[source_domain_index],  # 使用指定的源域
            #     target_path,
            #     stage=1,
            #     epochs=20,
            #     patience=3
            # )
            
            
            ## 领域对齐+对比学习(跨域正样本)
            print(f"[Stage 1] Cross-Domain Alignment + Contracstive Learning")
            train_stage1_with_contrastive(
                student_model,
                student_tokenizer, 
                prompt_manager,
                partition,
                source_paths[source_domain_index],  # 选择一个源域
                target_paths[target_domain_index],  # 选择一个目标域
                source_domain = source_map[source_paths[source_domain_index]],
                target_domain = target,
                epochs=30,
                patience=3,
                lambda_c=0.1,
                lambda_m=0.05,
                use_cross_domain_positive=True,     ##是否使用跨域正样本
                target_test=True,
                visualize_features=False  # ✅ 启用可视化
            )
            
            # ## 领域对齐+对比学习
            # print(f"[Stage 1] Cross-Domain Alignment + Contracstive Learning")
            # train_stage1_with_contrastive(
            #     student_model, 
            #     tokenizer, 
            #     prompt_manager,
            #     source_paths[source_domain_index],  # 选择一个源域
            #     target_path,
            #     epochs=20,
            #     patience=3,
            #     contrastive_weight=0.3
            # )
             
            
            
            # 保存Stage 1的模型（标注使用的源域）
            torch.save({
                'model_state_dict': student_model.state_dict(),
                'stage': 1,
                'source_domain': source_paths[source_domain_index]
            }, f'student_stage1_source{source_domain_index}.pt')
            print(f"✓ Student Stage 1 saved to student_stage1_source{source_domain_index}.pt")
            
            
    
    # ========== 阶段4: 目标域微调 ==========
    if resume_from_phase is None or resume_from_phase <= 4:
        print(f"\n[Phase 4] Target Domain Fine-tuning with Meta-Learning")
        print(f"{'='*80}")
        
        ##加载多教师模型
        teacher_checkpoint_paths = [f'teacher_model_{i+1}.pt' for i in range(len(source_paths))]
            
        for i, ckpt_path in enumerate(teacher_checkpoint_paths):
            print(f"Loading Teacher {i+1} from {ckpt_path}")
            teacher = DeepProgressivePromptModel(teacher_name).to(device)
            checkpoint = torch.load(ckpt_path, map_location=device)
            teacher.load_state_dict(checkpoint['model_state_dict'])
            teacher_models.append(teacher)
            print(f"✓ Teacher {i+1} loaded\n")
        print("All teacher models are load finished!!")
        
        if student_model is None:
            # 加载Stage 1的模型
            print("Loading Student Stage 1 model...")
            student_model = DeepProgressivePromptModel(student_name).to(device)
            checkpoint = torch.load(f'student_stage1_source{source_domain_index}.pt', map_location=device)
            student_model.load_state_dict(checkpoint['model_state_dict'])
            print("Student Stage 1 model is load finished!!")
        
        student_model = train_with_distillation_and_metalearning(
            student_model=student_model,
            teacher_models=teacher_models,
            tokenizer=student_tokenizer,
            prompt_manager=prompt_manager,
            partition=partition,
            source_data_paths=source_paths,  ##实际上不用源域
            target_data_path=target_paths[target_domain_index],  ##目标域
            source_map =  None,     ##stage2不需要源域
            target_domain = target,
            stage=2,
            epochs=30,
            patience=3,
            use_multi_teacher=True,
            use_meta_learning=False,   ##使用微调
            source_domain_index=0,
            target_test = True
        )
        
        # 保存Stage 2的模型
        torch.save({
            'model_state_dict': student_model.state_dict(),
            'stage': 2,
            'source_domain': source_paths[source_domain_index]
        }, f'student_stage2_source{source_domain_index}.pt')
        print(f"✓ Student Stage 2 saved\n")

    
    ## 消融实验 ：w/o stage3(Target-Domain Fine-tuning)
    if resume_from_phase == 5:
        print("==================Ablation study: w/o meta learning====================")
        # 加载Stage 1的模型
        print("Loading Student Stage 1 model for evaluation")
        student_model = DeepProgressivePromptModel(student_name).to(device)
        checkpoint = torch.load(f'student_stage1_source{source_domain_index}.pt', map_location=device)
        student_model.load_state_dict(checkpoint['model_state_dict'])
        print("Student Stage 1 model is load finished!!")
        
    
     ## 消融实验 ：w/o SupCon 移除对比学习
    if resume_from_phase == 6:
        print("==================Ablation study: w/o SupCon====================")
        
        student_checkpoint_path = 'best_model_kd_meta_stage_0.pt'
        
        print(f"Loading Student Model from {student_checkpoint_path} for meta-learning and evaluation")
        
        print("Mete Leaning Training")
        print("Loading multi teacher......")
        ##加载多教师模型
        teacher_checkpoint_paths = [f'teacher_model_{i+1}.pt' for i in range(len(source_paths))]
        for i, ckpt_path in enumerate(teacher_checkpoint_paths):
            print(f"Loading Teacher {i+1} from {ckpt_path}")
            teacher = DeepProgressivePromptModel(teacher_name).to(device)
            checkpoint = torch.load(ckpt_path, map_location=device)
            teacher.load_state_dict(checkpoint['model_state_dict'])
            teacher_models.append(teacher)
            print(f"✓ Teacher {i+1} loaded\n")
        print("All teacher models are load finished!!")
        
        print(f"Loading Student Model from {student_checkpoint_path}")
        student_model = DeepProgressivePromptModel(student_name).to(device)
        checkpoint = torch.load(student_checkpoint_path, map_location=device)
        student_model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Student Model loaded")
        
        
        student_model = train_with_distillation_and_metalearning(
            student_model=student_model,
            teacher_models=teacher_models,
            tokenizer=student_tokenizer,
            prompt_manager=prompt_manager,
            partition=partition,
            source_data_paths=source_paths,  ##实际上不用源域
            target_data_path=target_paths[target_domain_index],  ##目标域
            source_map =  None,     ##stage2不需要源域
            target_domain = target,
            stage=2,
            epochs=15,
            patience=3,
            use_multi_teacher=False,
            use_meta_learning=True
        )
        
        
    # ========== 最终评估 ==========
    print(f"[Final Evaluation] Testing Student Model")
    print(f"{'='*80}")
    
    ### 单独评估
    # print("Loading Student Stage 2 model...")
    # student_model = DeepProgressivePromptModel(student_name).to(device)
    # checkpoint = torch.load(f'best_model_kd_meta_stage_2.pt', map_location=device)
    # student_model.load_state_dict(checkpoint['model_state_dict'])
    
    ### 评估是使用所有样本，partition == None
    final_results = evaluate_enhanced(
        student_model, student_tokenizer, prompt_manager,None,
        target_paths[target_domain_index], stage=2,target_domain=target
    )
    
    # 保存最终结果
    final_checkpoint = {
        'model_state_dict': student_model.state_dict(),
        'teacher_info': {
            f'teacher_{i+1}': path for i, path in enumerate(source_paths)
        },
        'source_domain_used': source_paths[source_domain_index],
        'final_results': final_results,
        'training_method': 'Knowledge Distillation + Meta-Learning'
    }
    
    torch.save(final_checkpoint, f'final_model_source{source_domain_index}.pt')
    
    # 保存JSON结果
    results_json = {
        'model_info': {
            'student_model': student_name,
            'num_teachers': len(teacher_models),
            'teacher_domains': source_paths,
            'source_domain_used_stage1': source_paths[source_domain_index],
            'target_domain': target_paths[target_domain_index],
            'training_method': 'Multi-Teacher KD + Reptile Meta-Learning'
        },
        'final_results': {
            'triplet_level': {
                'accuracy': float(final_results['triplet']['accuracy']),
                'precision': float(final_results['triplet']['precision']),
                'recall': float(final_results['triplet']['recall']),
                'f1': float(final_results['triplet']['f1'])
            },
            'element_level': {
                'aspect': {
                    'accuracy': float(final_results['element']['aspect']['accuracy']),
                    'precision': float(final_results['element']['aspect']['precision']),
                    'recall': float(final_results['element']['aspect']['recall']),
                    'f1': float(final_results['element']['aspect']['f1'])
                },
                'opinion': {
                    'accuracy': float(final_results['element']['opinion']['accuracy']),
                    'precision': float(final_results['element']['opinion']['precision']),
                    'recall': float(final_results['element']['opinion']['recall']),
                    'f1': float(final_results['element']['opinion']['f1'])
                },
                'sentiment': {
                    'accuracy': float(final_results['element']['sentiment']['accuracy']),
                    'precision': float(final_results['element']['sentiment']['precision']),
                    'recall': float(final_results['element']['sentiment']['recall']),
                    'f1': float(final_results['element']['sentiment']['f1'])
                }
            }
        }
    }
    
    with open(f'results_source{source_domain_index}.json', 'w', encoding='utf-8') as f:
        json.dump(results_json, f, indent=2, ensure_ascii=False)
    
    print(f"{'='*80}")
    print("TRAINING COMPLETED!")
    print(f"{'='*80}")
    print(f"  Source Domain Used: {source_paths[source_domain_index]}")
    print(f"  Final Model: final_model_source{source_domain_index}.pt")
    print(f"  Results: results_source{source_domain_index}.json")
    print(f"  Final Triplet F1: {final_results['triplet']['f1']:.4f}")
    print(f"{'='*80}")
    
    return student_model, final_results




# =========================
# 13. 主入口
# =========================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Depression Symptom Triplet Extraction with KD+Meta')
    parser.add_argument('--mode', type=str, default='kd_meta',
                      choices=['train', 'kd_meta'],
                      help='Running mode')
    parser.add_argument('--source', type=str, default='data/laptop/',
                      help='Source domain data path')
    parser.add_argument('--target', type=str, default='data/twitter/',
                      help='Target domain data path')
    
    args = parser.parse_args()
    print(f"========================={args.mode}================================")
    
    
    if args.mode == 'kd_meta':
        # 知识蒸馏+元学习
        # main_with_kd_and_meta()
        
        ## resume_from_phase=0 表示1-4训练
        ## resume_from_phase=1 表示2-4训练
        ## resume_from_phase=2 表示3-4训练
         ## resume_from_phase=3 表示4训练
        # source_paths = ['data/laptop/', 'data/rest/','data/rest15/','data/rest16/']
        # target_paths = ['data/twitter/','data/reddit/']
        
        ## 针对CD-ASTE任务
        # source_paths = ['data/laptop16_mine/', 'data/rest14_mine/','data/rest15_mine/','data/rest16_mine/']
        # target_paths = ['data/laptop16_mine/', 'data/rest14_mine/','data/rest15_mine/','data/rest16_mine/']
        
        # source_map = {'data/laptop16_mine/':'laptop','data/rest14_mine/':'restaurant','data/rest15_mine/':'restaurant','data/rest16_mine/':'restaurant'}
        # target = 'restaurant'
        
        
        main_with_kd_and_meta(
            partition=1,                       ##目标域使用的比例
            resume_from_phase=1,            # 从哪个阶段开始：1, 2, 3, 4
            teacher_checkpoint_paths=None,      # 教师模型路径列表
            student_checkpoint_path=None,       # 学生模型路径
            source_domain_index=2,              # Stage 1使用哪个源域 0：laptop 1：rest
            target_domain_index=1               ## stage 1,2阶段使用的目标域
            )
    
    print("✓ Program finished successfully!")



