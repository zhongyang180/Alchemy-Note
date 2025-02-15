# train_grpo.py
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
import wandb

wandb.init(project="GRPO-visualize")
dataset = load_dataset("trl-lib/tldr", split="train[0:3]")#..后面整理这里的split方法，以及详细观察split之后的数据。https://huggingface.co/docs/datasets/loading#slice-splits
print(dataset[0])
# !!观察一下训练数据长什么样，训练目标是什么。然后再是用各种折线图反应炼丹的内部过程，尝试给出改进启发。
# ！！以及要尝试一下模型部署，观察输出效果
    # 试着用一下trl的chat的方法，看看能不能跑一下！！
# 1. 观察数据
# Define the reward function, which rewards completions that are close to 20 characters 。。后面试试改编奖励函数，看看效果。以及这一点可以拓展到我的课题中。
def reward_len(completions, **kwargs):
    return [abs(20 - len(completion)) for completion in completions]

#?done?我怎么样只训练简单的几轮，只用于观察训练过程
training_args = GRPOConfig(output_dir="Qwen2-0.5B-GRPO-visualize", logging_steps=10)#, log_with="wandb")？？为毛用不了。之后看看代码细节
trainer = GRPOTrainer(#？？！！我能否改造代码，让他支持wandb？可以尝试先研究一下为什么之前的gpt2的代码支持wandb
    model="Qwen/Qwen2-0.5B-Instruct",
    reward_funcs=reward_len,
    args=training_args,
    train_dataset=dataset,
)
trainer.train()
# 。。用wandb来记录训练过程的情况，绘制若干图。否则就是一个纯黑箱的训练过程。

print("训练完成")
# ？？我需要之后去审视训练的这些方面
# 先重复参考wandb的示例输出，至少把一般的训练情况都可视化记录下来。
# 我需要清楚地说明，对于这次炼丹过程，这些参数对应的折线图说明了什么，是否揭示了未来的改进？？

# 需要解释的要点？？！！
# 1. grpo的效果。之前方法的效果。grpo的好处在于
    # ！！这里就是要对比ppo和grpo的效果
# 2. grpo的核心工作原理，说清楚，尽可能可视化出来