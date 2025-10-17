import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import os
import yaml
import argparse
from copy import deepcopy

from transformer_dataset import get_transformer_data_loaders
from transformer_model import TaikoTransformer
from tokenization import TaikoTokenizer

# --- PPO Enhancements ---

class KLController:
    """Adaptive KL controller, as described in Ziegler et al. "Fine-Tuning Language Models from Human Preferences"."""
    def __init__(self, init_kl_coef, target, horizon):
        self.kl_coef = init_kl_coef
        self.target = target
        self.horizon = horizon

    def update(self, current, n_steps):
        """Update KL coefficient."""
        proportional_error = torch.clamp(current / self.target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.kl_coef *= mult

def compute_gae(rewards, values, gamma=0.99, lambda_gae=0.95):
    """Computes Generalized Advantage Estimation (GAE)."""
    advantages = []
    last_advantage = 0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        last_advantage = delta + gamma * lambda_gae * last_advantage
        advantages.insert(0, last_advantage)
    return torch.tensor(advantages, dtype=torch.float32)

def ppo_finetune(generator_model, tokenizer, config, device, fold_idx, val_loader):
    """
    Fine-tunes the generator model using an enhanced PPO implementation
    with GAE and an adaptive KL controller.
    """
    print(f"\n--- Starting PPO Fine-Tuning for Fold {fold_idx + 1} ---")
    rlhf_config = config.get('rlhf', {})
    if not rlhf_config:
        print("RLHF config not found. Skipping PPO.")
        return

    # 1. Models and Controller Setup
    ref_model = deepcopy(generator_model).to(device)
    for param in ref_model.parameters():
        param.requires_grad = False
    ref_model.eval()

    # The reward model is assumed to be part of the generator for simplicity here
    # In a real scenario, it would be a separate, pre-trained model.
    # We will use the value head as a proxy for reward.

    kl_controller = KLController(init_kl_coef=0.2, target=rlhf_config['target_kl'], horizon=10000)
    ppo_optimizer = optim.Adam(generator_model.parameters(), lr=rlhf_config['ppo_learning_rate'])

    # 2. Data Loader for PPO
    ppo_loader = torch.utils.data.DataLoader(
        val_loader.dataset, batch_size=rlhf_config['num_rollouts'], shuffle=True
    )
    ppo_iter = iter(ppo_loader)

    # 3. PPO Training Loop
    for ppo_epoch in range(rlhf_config['ppo_epochs']):
        generator_model.train()
        try:
            batch = next(ppo_iter)
        except StopIteration:
            ppo_iter = iter(ppo_loader)
            batch = next(ppo_iter)

        encoder_input = batch["encoder_input"].to(device)

        # --- Rollout Phase ---
        with torch.no_grad():
            # Generate sequences and get log probs from the current policy
            gen_output = generator_model.generate(
                encoder_input, rlhf_config['max_generation_length'], tokenizer
            )
            sequences = gen_output['sequences']

            # Get log probs and values from the reference model
            ref_output = ref_model(encoder_input, sequences)
            ref_logits, ref_values = ref_output['logits'], ref_output['value']
            ref_log_probs = F.log_softmax(ref_logits, dim=-1)
            ref_log_probs = torch.gather(ref_log_probs, 2, sequences.unsqueeze(-1)).squeeze(-1)

            # Use the value function as a proxy for reward
            rewards = ref_values.squeeze(-1)

        # --- GAE Calculation ---
        advantages = compute_gae(rewards.cpu(), ref_values.squeeze(-1).cpu())
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages = advantages.to(device)
        returns = advantages + ref_values.squeeze(-1)

        # --- PPO Update Phase ---
        # Get log probs and values from the current policy
        output = generator_model(encoder_input, sequences)
        logits, values = output['logits'], output['value']
        log_probs = F.log_softmax(logits, dim=-1)
        log_probs = torch.gather(log_probs, 2, sequences.unsqueeze(-1)).squeeze(-1)

        # Policy Loss (PPO-Clip)
        ratio = torch.exp(log_probs.sum(dim=-1) - ref_log_probs.sum(dim=-1))
        policy_loss_1 = ratio * advantages
        policy_loss_2 = torch.clamp(ratio, 1 - rlhf_config['clip_epsilon'], 1 + rlhf_config['clip_epsilon']) * advantages
        policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

        # Value Loss
        value_loss = F.mse_loss(values.squeeze(-1), returns)

        # Total Loss
        loss = policy_loss + 0.5 * value_loss

        ppo_optimizer.zero_grad()
        loss.backward()
        ppo_optimizer.step()

        # --- KL Controller Update ---
        kl_div = (log_probs - ref_log_probs).sum(dim=-1).mean()
        kl_controller.update(kl_div, rlhf_config['num_rollouts'])

        print(f"[PPO Epoch {ppo_epoch+1}] Loss: {loss.item():.4f}, KL Div: {kl_div.item():.4f}")
        wandb.log({
            "ppo_loss": loss.item(),
            "ppo_kl_div": kl_div.item(),
            "kl_coeff": kl_controller.kl_coef
        })

    print(f"--- Finished PPO Fine-Tuning ---")


def load_config(path="config/default.yaml"):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def train_fold(config, fold_idx):
    run_name = f"fold_{fold_idx + 1}"
    wandb.init(project="TaikoNation-Transformer-GAE", config=config, name=run_name, reinit=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Starting Fold {fold_idx + 1} on {device} ---")

    train_loader, val_loader, tokenizer = get_transformer_data_loaders(config, fold_idx)
    if not train_loader:
        wandb.finish()
        return

    model = TaikoTransformer(vocab_size=tokenizer.vocab_size, **config['model']).to(device)
    wandb.watch(model, log="all")

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.vocab["[PAD]"])
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', **config['training']['scheduler'])

    best_val_loss = float('inf')
    model_save_path = f"{config['training']['save_path']}_fold_{fold_idx + 1}.pth"
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    for epoch in range(config['training']['num_epochs']):
        model.train()
        for batch in train_loader:
            if not batch: continue
            encoder_input, decoder_input, target = batch["encoder_input"].to(device), batch["decoder_input"].to(device), batch["target"].to(device)
            optimizer.zero_grad()
            output = model(encoder_input, decoder_input)
            loss = criterion(output['logits'].view(-1, tokenizer.vocab_size), target.view(-1))
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                if not batch: continue
                encoder_input, decoder_input, target = batch["encoder_input"].to(device), batch["decoder_input"].to(device), batch["target"].to(device)
                output = model(encoder_input, decoder_input)
                loss = criterion(output['logits'].view(-1, tokenizer.vocab_size), target.view(-1))
                val_loss += loss.item()

        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        wandb.log({"val_loss": val_loss, "epoch": epoch})
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"Epoch {epoch+1}: New best model saved with val_loss: {best_val_loss:.4f}")

    print("--- Finished Supervised Training ---")

    model.load_state_dict(torch.load(model_save_path))
    ppo_finetune(model, tokenizer, config, device, fold_idx, val_loader)

    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Taiko Transformer model.")
    parser.add_argument('--config', type=str, default='config/default.yaml', help='Path to config file.')
    args = parser.parse_args()
    config = load_config(args.config)
    os.environ["WANDB_MODE"] = "offline"
    main_config = 1 if config.get('dry_run') else config['training']['k_folds']
    for i in range(main_config):
        train_fold(config, i)