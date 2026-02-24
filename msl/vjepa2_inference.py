import torch
import pandas as pd
import numpy as np
from PIL import Image
import io
import cv2
import torch.nn.functional as F
from src.models.vision_transformer import vit_giant_xformers_rope
from src.models.ac_predictor import VisionTransformerPredictorAC

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


CKPT_PATH = "checkpoints/vjepa2-ac-vitg.pt"
PARQUET_PATH = "/home/larsosterberg/.cache/huggingface/lerobot/lars/xarm_history_exp_v1/data/chunk-000/episode_000001.parquet"

def init_world_model():
    checkpoint = torch.load(CKPT_PATH, map_location="cuda")

    encoder = vit_giant_xformers_rope(img_size=(256, 256), num_frames=64)
    enc_dict = checkpoint['encoder']
    if list(enc_dict.keys())[0].startswith('module.'):
        enc_dict = {k.replace('module.', ''): v for k, v in enc_dict.items()}
    encoder.load_state_dict(enc_dict)
    encoder.cuda().eval()

    predictor = VisionTransformerPredictorAC(
        img_size=(256, 256),
        num_frames=64,
        embed_dim=1408,
        predictor_embed_dim=1024,
        action_embed_dim=7
    )

    pred_dict = checkpoint['predictor']
    if list(pred_dict.keys())[0].startswith('module.'):
        pred_dict = {k.replace('module.', ''): v for k, v in pred_dict.items()}
    
    predictor.load_state_dict(pred_dict)
    predictor.cuda().eval()

    return encoder, predictor

def prepare_input():
    df = pd.read_parquet(PARQUET_PATH)

    frames = []
    all_states = []
    all_actions = []

    for _, row in df.iterrows():
        img_data = row["exterior_image_1_left"]
        img = Image.open(io.BytesIO(img_data["bytes"]))
        img_np = np.array(img.resize((256, 256)))
        frames.append(img_np)

        curr_state = np.concatenate([row["joint_position"], [row["gripper_position"]]])
        all_states.append(curr_state)

        all_actions.append(row["actions"])

    video_np = np.stack(frames[150:214])
    video_tensor = torch.from_numpy(video_np).permute(0, 3, 1, 2).float() / 255.0
    video_tensor = video_tensor.unsqueeze(0)

    state = torch.tensor(np.array(all_states), dtype=torch.float32)[None, 120:184, :]
    action = torch.tensor(np.array(all_actions), dtype=torch.float32)[None, 120:184, :]

    video = video_tensor.permute(0, 2, 1, 3, 4)
    return video, state, action


def save_error_gif(pred_grid, latents_grid):
    # 1. Calculate Cosine Similarity
    # We compare pred_grid[t] with latents_grid[t+1]
    # dim=-1 is the 1408 feature dimension
    sim_sequence = F.cosine_similarity(
        pred_grid[0, :31], 
        latents_grid[0, 1:], 
        dim=-1
    ).cpu().numpy()

    # 2. Setup the Plot
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # We use vmin=0.7 or 0.8 to highlight the differences, 
    # since good models usually stay in the high similarity range.
    im = ax.imshow(sim_sequence[0], cmap='viridis', interpolation='nearest', vmin=0.5, vmax=1.0)
    plt.colorbar(im, ax=ax, label='Cosine Similarity (1.0 = Perfect)')
    
    def update(t):
        im.set_array(sim_sequence[t])
        ax.set_title(f"Prediction Quality Step {t}")
        return [im]

    ani = FuncAnimation(fig, update, frames=range(31), blit=True)
    ani.save("prediction_similarity.gif", writer='pillow', fps=7)
    plt.close()
    print("Saved prediction_similarity.gif")


def main():
    encoder, predictor = init_world_model()
    video, states, actions = prepare_input()
    
    with torch.no_grad():
        video = video.float().cuda()
        states = states[:, ::2, :].cuda()
        actions = actions[:, ::2, :].cuda()

        latents = encoder(video) 
        predicted_latents = predictor(latents, states, actions)
        
    B, Total_Tokens, D = predicted_latents.shape
    T, H, W = 32, 16, 16
    latents_grid = latents.view(B, T, H, W, D)
    pred_grid = predicted_latents.view(B, T, H, W, D)

    first_frame = latents_grid[:, 31, :, :, :] 
    future_frame = pred_grid[:, 31, :, :, :]
    print(pred_grid.shape)
    print(latents_grid.shape)

    persistence_map = torch.norm(future_frame - first_frame, dim=-1).squeeze().cpu().numpy()
    save_error_gif(pred_grid, latents_grid)
    
    print("Success! Analysis complete.")

    plt.figure(figsize=(8, 6))
    plt.imshow(persistence_map, cmap='viridis')
    plt.title("Latent Difference Map\nFrame 0 (Context) vs Frame 20 (Predicted)")
    plt.colorbar(label='Feature Distance (L2 Norm)')
    
    output_filename = "persistence_heatmap.png"
    plt.savefig(output_filename)
    plt.close()
    
    print(f"Heatmap saved as: {output_filename}")
    

if __name__ == "__main__":
    main()