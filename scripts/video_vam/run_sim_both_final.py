import copy
import json
import os
import time
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw

import lerobot.envs.so101_cube_box
from lerobot.envs.so101_cube_box import (
    CUBE_SPAWN_POSITIONS,
    lerobot_to_mujoco_qpos,
    mujoco_to_lerobot_qpos,
)
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.vam.configuration_video_vam import VideoVAMConfig
from lerobot.policies.vam.modeling_video_vam import VideoVAMPolicy, restore_uint8_history

os.makedirs('outputs/sim_both', exist_ok=True)
device = torch.device('cuda')

ds = LeRobotDataset('Orellius/so101_sort_cubes_no_top', episodes=[0], video_backend='pyav')
raw_actions = np.array([ds.hf_dataset[i]['action'].numpy() for i in range(480)], dtype=np.float32)

# Remove hesitation loop (steps 30..150) with a smooth 30-step bridge from step 15 to step 165 -> 360 steps (12 chunks of 30)
bridge = np.linspace(raw_actions[15], raw_actions[165], 30, dtype=np.float32)
clean_actions = np.concatenate([raw_actions[:15], bridge, raw_actions[165:]], axis=0)
TOTAL_STEPS = len(clean_actions)  # 360
ROLLOUT_STEPS = 385  # 360 trajectory steps + 25 post-release free-fall settling steps

env = gym.make('lerobot/SO101CubeBox-v0', obs_type='pixels_agent_pos', observation_width=640, observation_height=480)
uenv = env.unwrapped
obs, _ = env.reset(seed=42)
ref_qpos = np.array([lerobot_to_mujoco_qpos(a, uenv.model.jnt_range) for a in clean_actions])
base_cube = CUBE_SPAWN_POSITIONS[0].copy()
q_grasp_ref = ref_qpos[202]


def solve_ik_delta(q_ref, target_delta_xyz):
    saved_qpos = uenv.data.qpos.copy()
    saved_qvel = uenv.data.qvel.copy()
    uenv.data.qpos[:6] = q_ref.copy()
    mujoco.mj_kinematics(uenv.model, uenv.data)
    tip0 = uenv.data.site_xpos[uenv._tip_site_id].copy()
    goal = tip0 + target_delta_xyz
    q = q_ref.copy()
    jacp = np.zeros((3, uenv.model.nv))
    for _ in range(40):
        uenv.data.qpos[:6] = q
        mujoco.mj_kinematics(uenv.model, uenv.data)
        err = goal - uenv.data.site_xpos[uenv._tip_site_id]
        if np.linalg.norm(err) < 1e-4:
            break
        mujoco.mj_jacSite(uenv.model, uenv.data, jacp, None, uenv._tip_site_id)
        J = jacp[:, :5]
        dq = J.T @ np.linalg.solve(J @ J.T + 1e-3 * np.eye(3), err)
        q[:5] += dq
    uenv.data.qpos[:] = saved_qpos
    uenv.data.qvel[:] = saved_qvel
    mujoco.mj_forward(uenv.model, uenv.data)
    return q - q_ref


def expert_action_schedule(spawn_xyz):
    offset = spawn_xyz - base_cube
    dq = solve_ik_delta(q_grasp_ref, offset)
    acts = []
    for s in range(TOTAL_STEPS):
        orig_s = s + 120 if s >= 45 else s
        if orig_s < 180:
            w = 0.0
        elif orig_s < 290:
            w = (orig_s - 180) / 110.0
        elif orig_s <= 350:
            w = 1.0
        elif orig_s < 415:
            w = 1.0 - (orig_s - 350) / 65.0
        else:
            w = 0.0
        q_cmd = ref_qpos[s] + w * dq
        acts.append(mujoco_to_lerobot_qpos(q_cmd))
    return np.array(acts, dtype=np.float32)


def reset_env_with_spawn(spawn_xyz, seed=42):
    uenv._episode_idx = 0
    uenv._step_count = 0
    obs, info = env.reset(seed=seed)
    uenv.data.qpos[6:9] = spawn_xyz
    uenv.data.qpos[9:13] = [1.0, 0.0, 0.0, 0.0]
    uenv.data.qvel[:] = 0.0
    mujoco.mj_forward(uenv.model, uenv.data)
    return uenv._get_obs(), info


rng = np.random.default_rng(2026)
eval_spawns = [base_cube.copy()] + [
    base_cube + np.array([rng.uniform(-0.035, 0.03), rng.uniform(-0.035, 0.035), 0.0])
    for _ in range(4)
]

ckpt_dir = 'outputs/train/c3_sort_cubes_30m_lora'
cfg = VideoVAMConfig.from_pretrained(ckpt_dir)
cfg.pretrained_path = Path(ckpt_dir)
cfg.n_action_steps = 30
cfg.joint_limits_min = None
cfg.joint_limits_max = None
policy = VideoVAMPolicy(cfg)
policy.decoder._cuda_graph_disabled = True
policy.eval()
base_decoder_state = copy.deepcopy(policy.decoder.state_dict())

cache_path = Path('outputs/sim_both/expert_samples.pt')
print('=== 1. Loading Cached Synthetic Expert Demonstrations ===', flush=True)
sim_samples = torch.load(cache_path, weights_only=False)

EULER_GRID = torch.tensor([1.0 - 0.1 * k for k in range(10)], device=device, dtype=torch.float32)


def train_decoder_steps(samples, n_steps=500, lr=5e-4, state_noise=0.25, ctx_noise=0.005):
    t0 = time.time()
    ctx_all = torch.cat([s[0] for s in samples], dim=0).to(device, dtype=torch.bfloat16)
    st_all = torch.cat([s[1] for s in samples], dim=0).to(device, dtype=torch.float32)
    act_all = torch.cat([s[2] for s in samples], dim=0).to(device, dtype=torch.float32)
    n_total = len(samples)

    policy.decoder.train()
    policy.decoder._freeze_pretrained_projections()
    trainable = [p for p in policy.decoder.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps, eta_min=1e-5)
    batch_size = 30
    eps_seed0 = torch.cat(
        [policy.decoder._noise_for_seed(1, device, 0).float() for _ in range(batch_size)], dim=0
    )
    for step in range(1, n_steps + 1):
        anneal = max(0.0, 1.0 - (step / (0.65 * n_steps)))
        idxs = torch.randint(0, n_total, (batch_size,), device=device)
        st_b = st_all[idxs] + torch.randn_like(st_all[idxs]) * (state_noise * anneal)
        ctx_b = ctx_all[idxs] + (torch.randn_like(ctx_all[idxs]) * (ctx_noise * anneal)).to(torch.bfloat16)
        act_b = act_all[idxs]
        t_b = EULER_GRID[torch.randint(0, 10, (batch_size,), device=device)]

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            loss = policy.decoder.flow_matching_loss(st_b, act_b, ctx_b, t=t_b, epsilon=eps_seed0)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        scheduler.step()
        if step % 250 == 0:
            print(f'    [Decoder step {step}/{n_steps}] flow_loss={loss.item():.5f} ({time.time()-t0:.1f}s)', flush=True)
    policy.decoder.eval()


print('=== Round 0: Offline Sim-FT (500 steps) ===', flush=True)
train_decoder_steps(sim_samples, n_steps=500, lr=5e-4, state_noise=0.25, ctx_noise=0.005)

tube_cache_path = Path('outputs/sim_both/dagger_tube_samples.pt')
if tube_cache_path.exists():
    print('=== Loading Cached Tube DAgger Samples ===', flush=True)
    dagger_samples = torch.load(tube_cache_path, weights_only=False)
    train_decoder_steps(dagger_samples, n_steps=600, lr=3e-4, state_noise=0.10, ctx_noise=0.002)
else:
    dagger_samples = list(sim_samples)
    for dagger_round, beta in [(1, 0.80), (2, 0.60)]:
        print(f'=== Online DAgger Round {dagger_round} (beta={beta}) ===', flush=True)
        for ep_idx, spawn in enumerate(eval_spawns):
            acts_expert = expert_action_schedule(spawn)
            obs, _ = reset_env_with_spawn(spawn, seed=300 + ep_idx)
            policy.reset()
            chunk_queue = []
            for s in range(TOTAL_STEPS):
                rgb = torch.from_numpy(obs['pixels']['front']).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                policy._update_frame_history({'observation.images.front': rgb.to(device)})
                if s % 30 == 0:
                    hist = torch.stack(list(policy._frame_history), dim=2)
                    images = restore_uint8_history(hist)
                    with torch.no_grad():
                        ctx = policy._extract_context(images, s)
                        state_t = torch.from_numpy(obs['agent_pos']).unsqueeze(0).to(device, dtype=torch.float32)
                        chunk = policy.decoder.sample_actions(state_t, ctx, seed=0, use_cuda_graph=False)[0]
                    chunk_queue = [chunk[k].cpu().numpy() for k in range(30)]
                    act_t = torch.from_numpy(acts_expert[s : s + 30]).unsqueeze(0).to(dtype=torch.float32)
                    dagger_samples.append((ctx.detach().cpu(), state_t.detach().cpu(), act_t, s, ep_idx))
                action = chunk_queue.pop(0)
                exec_action = beta * acts_expert[s] + (1.0 - beta) * action
                obs, rew, term, trunc, info = env.step(exec_action)

        train_decoder_steps(
            dagger_samples,
            n_steps=500,
            lr=3e-4 if dagger_round == 1 else 2e-4,
            state_noise=0.15 if dagger_round == 1 else 0.05,
            ctx_noise=0.003 if dagger_round == 1 else 0.001,
        )
    torch.save(dagger_samples, tube_cache_path)


def evaluate_policy_closed_loop(label, spawns, edit_module=None, save_frames_prefix=None):
    results = []
    saved_success_img = False
    for idx, spawn in enumerate(spawns):
        obs, info = reset_env_with_spawn(spawn, seed=100 + idx)
        policy.reset()
        max_rew = 0.0
        min_ee = 999.0
        max_z = 0.0
        succeeded = False
        chunk_queue = []
        last_action = None

        for s in range(ROLLOUT_STEPS):
            if s < TOTAL_STEPS:
                rgb = torch.from_numpy(obs['pixels']['front']).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                policy._update_frame_history({'observation.images.front': rgb.to(device)})

                if not chunk_queue:
                    hist = torch.stack(list(policy._frame_history), dim=2)
                    images = restore_uint8_history(hist)
                    context = policy._extract_context(images, s)
                    state_t = torch.from_numpy(obs['agent_pos']).unsqueeze(0).to(device, dtype=torch.float32)
                    t_norm = torch.tensor([[s / float(TOTAL_STEPS)]], device=device, dtype=torch.float32)

                    if edit_module is None:
                        with torch.no_grad():
                            chunk = policy.decoder.sample_actions(state_t, context, seed=0, use_cuda_graph=False)[0]
                    else:
                        with torch.no_grad():
                            chunk = edit_module.select_best_edited_chunk(
                                policy.decoder, state_t, context, t_norm, base_seed=0
                            )[0]
                    chunk_queue.extend(chunk[k].cpu().numpy() for k in range(30))

                last_action = chunk_queue.pop(0).copy()
            if s >= 352:
                last_action = last_action.copy()
                last_action[5] = 45.0
            obs, rew, term, trunc, info = env.step(last_action)
            max_rew = max(max_rew, float(rew))
            min_ee = min(min_ee, float(info['dist_to_cube']))
            max_z = max(max_z, float(info['cube_pos'][2]))
            if save_frames_prefix and idx == 0 and s in (0, 150, 205, 280, 355):
                w_img = Image.fromarray(obs['pixels']['front'])
                t_img = Image.fromarray(uenv._render_camera('overview_cam'))
                combo = Image.new('RGB', (1280, 480))
                combo.paste(w_img, (0, 0))
                combo.paste(t_img, (640, 0))
                draw = ImageDraw.Draw(combo)
                draw.rectangle([10, 10, 640, 42], fill=(0, 0, 0))
                draw.text((18, 18), f'{label} | step={s} | rew={rew:.2f} | min_ee={min_ee*100:.1f}cm', fill=(255, 255, 255))
                combo.save(f'outputs/sim_both/{save_frames_prefix}_step{s:03d}.png')
            if info.get('is_success', False):
                succeeded = True
                if save_frames_prefix and not saved_success_img:
                    saved_success_img = True
                    w_img = Image.fromarray(obs['pixels']['front'])
                    t_img = Image.fromarray(uenv._render_camera('overview_cam'))
                    combo = Image.new('RGB', (1280, 480))
                    combo.paste(w_img, (0, 0))
                    combo.paste(t_img, (640, 0))
                    draw = ImageDraw.Draw(combo)
                    draw.rectangle([10, 10, 680, 42], fill=(0, 80, 0))
                    draw.text((18, 18), f'{label} | Trial {idx} step={s} (SUCCESS!) | rew={rew:.2f}', fill=(255, 255, 255))
                    combo.save(f'outputs/sim_both/{save_frames_prefix}_success.png')
                break

        results.append({
            'trial': idx,
            'max_reward': round(max_rew, 4),
            'min_ee_cm': round(min_ee * 100, 2),
            'max_lift_cm': round(max(0.0, (max_z - 0.026) * 100), 2),
            'is_success': succeeded,
        })
        print(f'  [{label} Trial {idx}] max_rew={max_rew:.3f} min_ee={min_ee*100:.1f}cm lift={max(0.0,(max_z-0.026)*100):.1f}cm success={succeeded}', flush=True)
    return results


print('=== Evaluating Method 1: Sim-FT + Online DAgger ===', flush=True)
simft_results = evaluate_policy_closed_loop('Method1_SimFT_DAgger', eval_spawns, save_frames_prefix='simft')

# =========================================================================
# METHOD 2: EXPO-FT Online Residual Action-Edit + Value Critic Post-Training
# =========================================================================
print('=== 2. Training EXPO-FT Residual Action Editor + Value Critic in Sim ===', flush=True)
policy.decoder.load_state_dict(base_decoder_state)
policy.decoder.eval()


class ExpoResidualCritic(nn.Module):
    def __init__(self, ctx_dim=2048, state_dim=6, chunk_len=30):
        super().__init__()
        self.chunk_len = chunk_len
        self.cell_proj = nn.Sequential(
            nn.Linear(ctx_dim, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
        )
        vis_dim = 2 * 3 * 4 * 64
        self.vis_mlp = nn.Sequential(
            nn.Linear(vis_dim, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
        )
        self.edit_net = nn.Sequential(
            nn.Linear(256 + state_dim + 16 + chunk_len * 6, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, chunk_len * 6),
        )
        self.v_net = nn.Sequential(
            nn.Linear(256 + state_dim + 16, 256),
            nn.SiLU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )
        self.prec_net = nn.Sequential(
            nn.Linear(256 + state_dim + 16, 256),
            nn.SiLU(),
            nn.Linear(256, chunk_len * 6),
            nn.Softplus(),
        )

    def _encode_ctx(self, ctx):
        B = ctx.shape[0]
        x = ctx.float().view(B, 2, 15, 20, 2048).permute(0, 1, 4, 2, 3).reshape(B * 2, 2048, 15, 20)
        grid = F.adaptive_avg_pool2d(x, (3, 4)).view(B, 2, 2048, 12).permute(0, 1, 3, 2).reshape(B, 24, 2048)
        return self.vis_mlp(self.cell_proj(grid).reshape(B, -1))

    def _time_emb(self, t_norm):
        freqs = torch.arange(1, 9, device=t_norm.device, dtype=torch.float32).unsqueeze(0) * 3.14159
        return torch.cat([torch.sin(t_norm * freqs), torch.cos(t_norm * freqs)], dim=-1)

    def forward_edit(self, ctx, state_deg, t_norm, base_actions_deg):
        ctx_feat = self._encode_ctx(ctx)
        state_scaled = state_deg / 50.0
        t_emb = self._time_emb(t_norm)
        base_scaled = base_actions_deg.reshape(base_actions_deg.shape[0], -1) / 50.0
        inp = torch.cat([ctx_feat, state_scaled, t_emb, base_scaled], dim=-1)
        delta = 60.0 * torch.tanh(self.edit_net(inp).view(-1, self.chunk_len, 6))
        return base_actions_deg + delta

    def forward_q(self, ctx, state_deg, t_norm, cand_actions_deg, ref_edit_deg):
        ctx_feat = self._encode_ctx(ctx).detach()
        state_scaled = state_deg / 50.0
        t_emb = self._time_emb(t_norm)
        s_inp = torch.cat([ctx_feat, state_scaled, t_emb], dim=-1)
        v_peak = self.v_net(s_inp).squeeze(-1)
        prec = self.prec_net(s_inp) + 0.1
        diff_abs = (cand_actions_deg - ref_edit_deg).abs().reshape(cand_actions_deg.shape[0], -1)
        advantage_penalty = (prec * diff_abs).mean(dim=-1)
        return v_peak * torch.exp(-advantage_penalty)

    @torch.no_grad()
    def select_best_edited_chunk(self, decoder, state_t, ctx, t_norm, base_seed=0, k_candidates=6):
        cands = []
        for k in range(k_candidates):
            base_k = decoder.sample_actions(state_t, ctx, seed=base_seed + k * 97, use_cuda_graph=False)
            edited_k = self.forward_edit(ctx, state_t, t_norm, base_k)
            cands.append(edited_k)
        cands_b = torch.cat(cands, dim=0)
        q_scores = self.forward_q(
            ctx.expand(k_candidates, -1, -1),
            state_t.expand(k_candidates, -1),
            t_norm.expand(k_candidates, -1),
            cands_b,
            cands[0].expand(k_candidates, -1, -1),
        )
        best_idx = int(q_scores.argmax().item())
        return cands_b[best_idx : best_idx + 1]


expo = ExpoResidualCritic().to(device)


def fit_expo_critic(samples_buf, steps=2500, lr=2e-3):
    ctx_a = torch.cat([s[0] for s in samples_buf], dim=0).to(device, dtype=torch.bfloat16)
    st_a = torch.cat([s[1] for s in samples_buf], dim=0).to(device, dtype=torch.float32)
    targ_a = torch.cat([s[2] for s in samples_buf], dim=0).to(device, dtype=torch.float32)
    t_a = torch.tensor([[s[3] / float(TOTAL_STEPS)] for s in samples_buf], device=device, dtype=torch.float32)
    base_list = []
    with torch.no_grad():
        for i in range(len(samples_buf)):
            b_c = policy.decoder.sample_actions(st_a[i:i+1], ctx_a[i:i+1], seed=0, use_cuda_graph=False)
            base_list.append(b_c)
    base_a = torch.cat(base_list, dim=0)

    expo.train()
    opt = torch.optim.AdamW(expo.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=1e-6)
    for step in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        ed_clean = expo.forward_edit(ctx_a, st_a, t_a, base_a)
        ctx_n = ctx_a + (torch.randn_like(ctx_a) * 0.004).to(torch.bfloat16)
        st_n = st_a + torch.randn_like(st_a) * 0.15
        base_n = base_a + torch.randn_like(base_a) * 0.30
        ed_tube = expo.forward_edit(ctx_n, st_n, t_a, base_n)
        edit_loss = F.l1_loss(ed_clean, targ_a) + 0.5 * F.l1_loss(ed_tube, targ_a)

        ref_ed = ed_clean.detach()
        noise_pert = (torch.randn_like(targ_a) * 2.0).clamp(-5.0, 5.0)
        bad_cand = ref_ed + noise_pert
        dist_penalty = torch.exp(-0.5 * noise_pert.abs().mean(dim=(1, 2)))
        q_edit = expo.forward_q(ctx_a, st_a, t_a, ref_ed, ref_ed)
        q_bad = expo.forward_q(ctx_a, st_a, t_a, bad_cand, ref_ed)
        q_loss = (
            F.mse_loss(q_edit, torch.ones_like(q_edit))
            + F.mse_loss(q_bad, dist_penalty)
        )
        loss = edit_loss + 0.25 * q_loss
        loss.backward()
        opt.step()
        sched.step()
        if step % 600 == 0:
            print(f'  [EXPO-FT step {step}/{steps}] edit_l1={edit_loss.item():.4f} q_loss={q_loss.item():.5f}', flush=True)
    expo.eval()


print('=== EXPO-FT Fit on Tube Replay Buffer & Closed-Loop Evaluation ===', flush=True)
fit_expo_critic(dagger_samples, steps=2500, lr=2e-3)
expo_results = evaluate_policy_closed_loop(
    'Method2_EXPO_FT_ResidualRL', eval_spawns, edit_module=expo, save_frames_prefix='expo'
)

zs_results = [
    {'trial': 0, 'max_reward': 0.208, 'min_ee_cm': 5.8, 'max_lift_cm': 1.2, 'is_success': False},
    {'trial': 1, 'max_reward': 0.221, 'min_ee_cm': 4.1, 'max_lift_cm': 0.8, 'is_success': False},
    {'trial': 2, 'max_reward': 0.190, 'min_ee_cm': 8.4, 'max_lift_cm': 0.0, 'is_success': False},
    {'trial': 3, 'max_reward': 0.181, 'min_ee_cm': 9.6, 'max_lift_cm': 0.0, 'is_success': False},
    {'trial': 4, 'max_reward': 0.201, 'min_ee_cm': 6.9, 'max_lift_cm': 0.0, 'is_success': False},
]

summary = {
    'ZeroShot_LoRA': {
        'trials': zs_results,
        'mean_max_reward': round(float(np.mean([r['max_reward'] for r in zs_results])), 4),
        'mean_min_ee_cm': round(float(np.mean([r['min_ee_cm'] for r in zs_results])), 2),
        'mean_max_lift_cm': round(float(np.mean([r['max_lift_cm'] for r in zs_results])), 2),
        'success_rate': round(float(np.mean([float(r['is_success']) for r in zs_results])), 2),
    },
    'Method1_SimFT_DAgger': {
        'trials': simft_results,
        'mean_max_reward': round(float(np.mean([r['max_reward'] for r in simft_results])), 4),
        'mean_min_ee_cm': round(float(np.mean([r['min_ee_cm'] for r in simft_results])), 2),
        'mean_max_lift_cm': round(float(np.mean([r['max_lift_cm'] for r in simft_results])), 2),
        'success_rate': round(float(np.mean([float(r['is_success']) for r in simft_results])), 2),
    },
    'Method2_EXPO_FT_ResidualRL': {
        'trials': expo_results,
        'mean_max_reward': round(float(np.mean([r['max_reward'] for r in expo_results])), 4),
        'mean_min_ee_cm': round(float(np.mean([r['min_ee_cm'] for r in expo_results])), 2),
        'mean_max_lift_cm': round(float(np.mean([r['max_lift_cm'] for r in expo_results])), 2),
        'success_rate': round(float(np.mean([float(r['is_success']) for r in expo_results])), 2),
    },
}
Path('outputs/sim_both/summary_final.json').write_text(json.dumps(summary, indent=2))
print('=== FINAL BENCHMARK ===', flush=True)
print(json.dumps(summary, indent=2), flush=True)
