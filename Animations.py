import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.animation


fig = plt.figure(figsize=(40, 30))  # Increased height slightly
wid = 0.26
hei = 0.35  # Reduced height slightly to make room for title and metrics
gap = 0.01  # Gap between plots

# Define the positions for each subplot [left, bottom, width, height]
ax1_pos = [0.02, 0.32, wid, hei]  # Truth (centered vertically)
ax2_pos = [0.35, 0.52, wid, hei]  # Generated with G (top right)
ax3_pos = [0.35, 0.1, wid, hei]  # Generated w/o G (bottom right)
ax4_pos = [0.68, 0.52, wid, hei]  # Error with G (top far right)
ax5_pos = [0.68, 0.1, wid, hei]  # Error w/o G (bottom far right)

# Create the subplots with the defined positions
ax1 = fig.add_axes(ax1_pos)
ax2 = fig.add_axes(ax2_pos)
ax3 = fig.add_axes(ax3_pos)
ax4 = fig.add_axes(ax4_pos)
ax5 = fig.add_axes(ax5_pos)

import seaborn as sns
import numpy as np

# Ticks setting
ticks_64 = np.arange(0, 64, 10 * 64 / 64)
ticks_64_y = np.arange(4, 65, 10 * 64 / 64)[::-1]
tick_labels_64 = [str(int(tick)) for tick in ticks_64]

def animate(t):
    global sol, vorticity_series, vorticity_NoG, sol_t
    fs = 30
    for ax in [ax1, ax2, ax3, ax4, ax5]:
        ax.cla()

    if hasattr(animate, 'colorbar_axes'):
        for cax in animate.colorbar_axes:
            cax.remove()

    if hasattr(animate, 'txt'):
        for txt in animate.txt:
            txt.remove()

    frame_index = min(10 * t, 19999)

    # Define color limits
    vorticity_limits = [-2.0, 2.0]
    error_limits = [0.00, 1.50]

    # Plot for sol tensor (Truth)
    sns.heatmap(sol[k + 7, ..., shifter + 10 * t].cpu().detach(), ax=ax1, cmap='rocket',
                cbar_ax=fig.add_axes([ax1_pos[0] + ax1_pos[2] + gap, ax1_pos[1], 0.01, ax1_pos[3]]),
                vmin=vorticity_limits[0], vmax=vorticity_limits[-1])
    ax1.set_title("Truth", fontsize=fs)
    ax1.collections[0].colorbar.set_ticks(np.linspace(vorticity_limits[0], vorticity_limits[-1], 5))
    ax1.collections[0].colorbar.ax.tick_params(labelsize=fs)

    # Plot for vorticity_series tensor (Generated with G)
    sns.heatmap(vorticity_series[k, ..., frame_index].cpu().detach(), ax=ax2, cmap='rocket',
                cbar_ax=fig.add_axes([ax2_pos[0] + ax2_pos[2] + gap, ax2_pos[1], 0.01, ax2_pos[3]]),
                vmin=vorticity_limits[0], vmax=vorticity_limits[-1])
    ax2.set_title("Generated with G", fontsize=fs)
    ax2.collections[0].colorbar.set_ticks(np.linspace(vorticity_limits[0], vorticity_limits[-1], 5))
    ax2.collections[0].colorbar.ax.tick_params(labelsize=fs)

    # Plot for vorticity_NoG tensor (Generated w/o G)
    sns.heatmap(vorticity_NoG[k, ..., frame_index].cpu().detach(), ax=ax3, cmap='rocket',
                cbar_ax=fig.add_axes([ax3_pos[0] + ax3_pos[2] + gap, ax3_pos[1], 0.01, ax3_pos[3]]),
                vmin=vorticity_limits[0], vmax=vorticity_limits[-1])
    ax3.set_title("Generated w/o G", fontsize=fs)
    ax3.collections[0].colorbar.set_ticks(np.linspace(vorticity_limits[0], vorticity_limits[-1], 5))
    ax3.collections[0].colorbar.ax.tick_params(labelsize=fs)

    # Calculate and plot the absolute error with G
    abs_error_2 = torch.abs(sol[k + 7, ..., shifter + 10 * t].cpu() - vorticity_series[k, ..., frame_index].cpu())
    sns.heatmap(abs_error_2, ax=ax4, cmap='rocket',
                cbar_ax=fig.add_axes([ax4_pos[0] + ax4_pos[2] + gap, ax4_pos[1], 0.01, ax4_pos[3]]),
                vmin=error_limits[0], vmax=error_limits[-1])
    ax4.set_title("Error with G", fontsize=fs)
    ax4.collections[0].colorbar.set_ticks(np.linspace(error_limits[0], error_limits[-1], 5))
    ax4.collections[0].colorbar.ax.tick_params(labelsize=fs)

    # Calculate and plot the absolute error without G
    abs_error = torch.abs(sol[k + 7, ..., shifter + 10 * t].cpu() - vorticity_NoG[k, ..., frame_index].cpu())
    sns.heatmap(abs_error, ax=ax5, cmap='rocket',
                cbar_ax=fig.add_axes([ax5_pos[0] + ax5_pos[2] + gap, ax5_pos[1], 0.01, ax5_pos[3]]),
                vmin=error_limits[0], vmax=error_limits[-1])
    ax5.set_title("Error w/o G", fontsize=fs)
    ax5.collections[0].colorbar.set_ticks(np.linspace(error_limits[0], error_limits[-1], 5))
    ax5.collections[0].colorbar.ax.tick_params(labelsize=fs)

    animate.colorbar_axes = [ax1.collections[0].colorbar.ax, ax2.collections[0].colorbar.ax,
                             ax3.collections[0].colorbar.ax, ax4.collections[0].colorbar.ax,
                             ax5.collections[0].colorbar.ax]

    # Calculate metrics
    re_with_G = relative_mse(sol[k + 7, ..., shifter + 10 * t].cpu(), vorticity_series[k, ..., frame_index].cpu())
    re_without_G = relative_mse(sol[k + 7, ..., shifter + 10 * t].cpu(), vorticity_NoG[k, ..., frame_index].cpu())
    mse_with_G = cal_mse(sol[k + 7, ..., shifter + 10 * t].cpu(), vorticity_series[k, ..., frame_index].cpu())
    mse_without_G = cal_mse(sol[k + 7, ..., shifter + 10 * t].cpu(), vorticity_NoG[k, ..., frame_index].cpu())

    # Update figure title and captions with metrics
    fig.suptitle(r'$\nu = 10^{-3}, \beta = 0.00005, dt = 10^{-3}$' + '\n' + f't = {sol_t[shifter + 10 * t].item():.3f}',
                 fontsize=fs, y=0.95)

    txt1 = fig.text(0.65, 0.48,
                    f'MSE (with G): {mse_with_G.item():.4f}, RE (with G): {re_with_G.mean().item():.4f}',
                    ha='center', fontsize=fs)
    txt2 = fig.text(0.65, 0.06,
                    f'MSE (w/o G): {mse_without_G.item():.4f}, RE (w/o G): {re_without_G.mean().item():.4f}',
                    ha='center', fontsize=fs)

    animate.txt = [txt1, txt2]

    # Adjust x and y axis ticks
    for ax in [ax1, ax2, ax3, ax4, ax5]:
        ax.set_xticks(ticks_64)
        ax.set_yticks(ticks_64_y)
        ax.tick_params(axis='both', which='major', labelsize=fs, rotation=0)
        ax.set_xticklabels(tick_labels_64, rotation=0, ha='center')
        ax.set_yticklabels(tick_labels_64, rotation=0, va='center')

    # Print progress
    progress = (t + 1) / 2000 * 100
    if t % 10 == 0:
        print(f"Progress: {progress:.2f}%")


# Create the animation
Animation1 = matplotlib.animation.FuncAnimation(fig, animate, frames=2000)
plt.close(fig)  # This prevents the static plot from displaying in Jupyter notebooks

# Save the animation
Animation1.save('Animation3050WithAndWithoutG.mp4', writer='ffmpeg', fps=60)