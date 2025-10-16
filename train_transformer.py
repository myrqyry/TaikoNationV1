import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import os
import yaml
import argparse
from copy import deepcopy

from transformer_dataset import get_transformer_data_loaders, collate_fn
from transformer_model import TaikoTransformer
from reward_model import RewardModel
from tokenization import TaikoTokenizer
import torch.nn.functional as F

def ppo_finetune(generator_model, tokenizer, config, device, fold_idx, val_loader):
    """
    Fine-tunes the generator model using Proximal Policy Optimization (PPO).
    """
    print(f"\n--- Starting PPO Fine-Tuning for Fold {fold_idx + 1} ---")
    ppo_config = config.get('ppo', {})
    if not ppo_config.get('enabled', False):
        print("PPO fine-tuning is disabled. Skipping.")
        return

    # Clear CUDA cache before starting memory-intensive PPO phase
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    reference_model = deepcopy(generator_model).to(device)
    for param in reference_model.parameters():
        param.requires_grad = False
    reference_model.eval()

    reward_model_path = ppo_config['reward_model_path']
    reward_model = RewardModel(input_size=config['model']['d_model']).to(device)
    if os.path.exists(reward_model_path):
        reward_model.load_state_dict(torch.load(reward_model_path))
        print(f"Loaded reward model from {reward_model_path}.")
    else:
        print(f"WARNING: Reward model not found. Creating a placeholder at {reward_model_path}.")
        os.makedirs(os.path.dirname(reward_model_path), exist_ok=True)
        torch.save(reward_model.state_dict(), reward_model_path)
    reward_model.eval()

    ppo_optimizer = optim.Adam(generator_model.parameters(), lr=ppo_config['ppo_learning_rate'])

    ppo_batch_size = ppo_config.get('num_rollouts', 4)
    ppo_loader = torch.utils.data.DataLoader(
        val_loader.dataset, batch_size=ppo_batch_size, shuffle=True, collate_fn=collate_fn
    )
    ppo_loader_iter = iter(ppo_loader)

    for ppo_epoch in range(ppo_config['ppo_epochs']):
        generator_model.train()
        try:
            batch = next(ppo_loader_iter)
        except StopIteration:
            ppo_loader_iter = iter(ppo_loader)
            batch = next(ppo_loader_iter)

        encoder_input = batch["encoder_input"].to(device)

        with torch.no_grad():
            generated_output = generator_model.generate(
                src=encoder_input, max_len=ppo_config['max_generation_length'], tokenizer=tokenizer
            )

        sequences = generated_output['sequences']
        log_probs = generated_output['log_probs'].sum(dim=-1) # Sum log probs for the sequence
        gen_hidden_states = generated_output['hidden_states']

        with torch.no_grad():
            chart_features = gen_hidden_states.mean(dim=1)
            rewards = reward_model(chart_features).squeeze(-1).detach()

            ref_outputs = reference_model(src=encoder_input, tgt=sequences)
            ref_logits = ref_outputs['logits']
            ref_log_probs_full = F.log_softmax(ref_logits, dim=-1)
            ref_log_probs = torch.gather(ref_log_probs_full, 2, sequences.unsqueeze(-1)).squeeze(-1)
            ref_log_probs = ref_log_probs.sum(dim=-1)

        kl_div = log_probs - ref_log_probs
        advantages = rewards - ppo_config['kl_penalty_weight'] * kl_div
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO Update
        current_outputs = generator_model(src=encoder_input, tgt=sequences)
        current_logits = current_outputs['logits']
        current_log_probs_full = F.log_softmax(current_logits, dim=-1)
        current_log_probs = torch.gather(current_log_probs_full, 2, sequences.unsqueeze(-1)).squeeze(-1)
        current_log_probs = current_log_probs.sum(dim=-1)

        ratio = torch.exp(current_log_probs - log_probs)
        policy_loss_1 = ratio * advantages
        policy_loss_2 = torch.clamp(ratio, 1 - ppo_config['clip_epsilon'], 1 + ppo_config['clip_epsilon']) * advantages
        policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

        entropy = -(torch.exp(current_log_probs_full) * current_log_probs_full).sum(dim=-1).mean()
        entropy_bonus = ppo_config['entropy_bonus_weight'] * entropy
        loss = policy_loss - entropy_bonus

        ppo_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(generator_model.parameters(), 1.0)
        ppo_optimizer.step()

        print(f"[PPO Epoch {ppo_epoch+1}] Loss: {loss.item():.4f}, Reward: {rewards.mean().item():.4f}, KL: {kl_div.mean().item():.4f}")
        wandb.log({"ppo_loss": loss.item(), "ppo_reward": rewards.mean().item(), "ppo_kl_div": kl_div.mean().item()})

    print(f"--- Finished PPO Fine-Tuning ---")
    ppo_save_path = f"{config['training']['save_path']}_fold_{fold_idx + 1}_ppo.pth"
    torch.save(generator_model.state_dict(), ppo_save_path)
    print(f"Saved PPO model to {ppo_save_path}")

def load_config(path="config/default.yaml"):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def train_fold(config, fold_idx):
    run_name = f"fold_{fold_idx + 1}"
    wandb.init(project="TaikoNation-PPO-Fixed", config=config, name=run_name, reinit=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Starting Fold {fold_idx + 1}/{config['training']['k_folds']} on {device} ---")

    train_loader, val_loader, tokenizer = get_transformer_data_loaders(config, fold_idx)
    if train_loader is None:
        wandb.finish()
        return

    model = TaikoTransformer(vocab_size=tokenizer.vocab_size, **config['model']).to(device)
    wandb.watch(model, log="all")

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.vocab["[PAD]"])
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', **config['training']['scheduler'])

    best_val_loss = float('inf')
    epochs_no_improve = 0
    model_save_path = f"{config['training']['save_path']}_fold_{fold_idx + 1}.pth"
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    num_epochs = 1 if config.get('dry_run') else config['training']['num_epochs']

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, batch in enumerate(train_loader):
            if batch is None: continue
            encoder_input, decoder_input, target = batch["encoder_input"].to(device), batch["decoder_input"].to(device), batch["target"].to(device)

            optimizer.zero_grad()
            outputs = model(src=encoder_input, tgt=decoder_input)
            loss = criterion(outputs['logits'].view(-1, tokenizer.vocab_size), target.view(-1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if i % 50 == 49:
                wandb.log({"train_loss": running_loss / 50, "epoch": epoch})
                running_loss = 0.0

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                if batch is None: continue
                encoder_input, decoder_input, target = batch["encoder_input"].to(device), batch["decoder_input"].to(device), batch["target"].to(device)
                outputs = model(src=encoder_input, tgt=decoder_input)
                loss = criterion(outputs['logits'].view(-1, tokenizer.vocab_size), target.view(-1))
                val_loss += loss.item()

        val_loss /= len(val_loader)
        wandb.log({"val_loss": val_loss, "epoch": epoch, "lr": optimizer.param_groups[0]['lr']})
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_save_path)
            print(f"New best model saved with val_loss: {best_val_loss:.4f}")
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= config['training']['early_stopping']['patience']:
            print("Early stopping triggered.")
            break

    print(f"--- Finished Supervised Training ---")

    model.load_state_dict(torch.load(model_save_path))
    ppo_finetune(model, tokenizer, config, device, fold_idx, val_loader)

    wandb.finish()

def main(config):
    os.environ["WANDB_MODE"] = "offline"
    num_folds = 1 if config.get('dry_run') else config['training']['k_folds']
    for i in range(num_folds):
        train_fold(config, i)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Taiko Transformer model.")
    parser.add_argument('--config', type=str, default='config/default.yaml', help='Path to the configuration file.')
    args = parser.parse_args()
    config = load_config(args.config)
    main(config)