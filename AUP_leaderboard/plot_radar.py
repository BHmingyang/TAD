import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from plot_lines import TECH_STYLE

def plot_radar_chart(data: dict, attributes: list, assigned_colors: dict = None, save_path: str = None, is_dark_mode: bool = True, font_size_axis: int = 12, font_size_legend: int = 12):
    """
    Plot radar chart for AUP results with tech style.
    Normalizes each attribute to [0, 1] based on the maximum value across methods.
    Axes show the real values for each task.
    data: {method_name: [val1, val2, ...]}
    attributes: list of attribute names (e.g., datasets)
    is_dark_mode: whether to use dark mode (True) or light mode (False)
    font_size_axis: font size for axis labels
    font_size_legend: font size for legend
    """
    # Setup angles
    N = len(attributes)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += [angles[0]]  # Close the loop
    
    # Normalize data independently per attribute
    method_names = list(data.keys())
    raw_matrix = np.array([data[m] for m in method_names]) # (num_methods, num_attributes)
    
    # Calculate max for each attribute independently
    attr_maxs = np.max(raw_matrix, axis=0) # (num_attributes,)
    attr_maxs = np.array([np.ceil(m / 5.0) * 5.0 if m > 0 else 5.0 for m in attr_maxs])
    
    # Normalize each attribute independently to [0, 1]
    normalized_matrix = raw_matrix / attr_maxs[np.newaxis, :]
    norm_data = {m: normalized_matrix[i] for i, m in enumerate(method_names)}

    # Determine style settings
    if is_dark_mode:
        style_context = 'dark_background'
        bg_color = 'black'
        fg_color = 'white'
        grid_color = 'white'
        grid_alpha = 0.3
        spine_color = 'white'
    else:
        style_context = 'default'
        bg_color = '#F5F5F7' # Apple light gray
        fg_color = 'black'
        grid_color = '#86868b'
        grid_alpha = 0.3
        spine_color = '#86868b'

    with plt.style.context(style_context):
        fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(polar=True))
        
        if not is_dark_mode:
            fig.patch.set_facecolor(bg_color)
            ax.set_facecolor(bg_color)
        
        for i, (method_name, v) in enumerate(norm_data.items()):
            # Determine color
            if assigned_colors:
                c_name = assigned_colors[i] if isinstance(assigned_colors, list) else assigned_colors.get(method_name, 'grey')
            else:
                c_name = 'grey'
            grad_colors = TECH_STYLE.get(c_name, TECH_STYLE['grey'])
            main_color = grad_colors[1]
            cmap = LinearSegmentedColormap.from_list(f"tech_{c_name}", grad_colors)
            
            # Prepare data for loop
            v_loop = np.concatenate([v, [v[0]]])
            
            # 1. Draw Gradient Line using LineCollection
            points = np.array([angles, v_loop]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            
            norm = plt.Normalize(0, 2 * np.pi)
            lc = LineCollection(segments, cmap=cmap, norm=norm)
            lc.set_array(np.array(angles))
            lc.set_linewidth(4) # Thicker lines
            lc.set_alpha(0.9)
            ax.add_collection(lc)
            
            # 2. Fill area with low alpha
            ax.fill(angles, v_loop, color=main_color, alpha=0.1)
            
            # 3. Add markers
            edge_c = main_color if is_dark_mode else 'white'
            # Actually keep white edge on dark bg, and maybe main_color edge on light?
            # Let's stick to white edge for contrast if filled with white, but here filled with nothing?
            # Previous code: color='white', edgecolors=main_color
            ax.scatter(angles, v_loop, color='white', edgecolors=main_color, s=100, zorder=10, lw=2)
            
            # 4. Add to legend (proxy artist)
            ax.plot([], [], color=main_color, label=method_name, linewidth=4)

        # Styling
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        
        # Grid and Spines
        ax.grid(True, linestyle='--', alpha=grid_alpha, color=grid_color, linewidth=1)
        ax.spines['polar'].set_visible(False)
        
        # X-axis (Attributes) labels
        ax.set_xticks(angles[:-1])
        labels = ax.set_xticklabels(attributes, fontsize=font_size_axis, color=fg_color, fontweight='bold')
        ax.tick_params(axis='x', pad=30)
        
        # Special handling: move 'humaneval+' label right if 4 tasks
        if len(attributes) == 4:
            import matplotlib.transforms as mtransforms
            for label in labels:
                if label.get_text() == 'humaneval+':
                    # Shift the label to the right to prevent occlusion
                    offset = mtransforms.ScaledTranslation(40/72, 0, fig.dpi_scale_trans)
                    label.set_transform(label.get_transform() + offset)
        
        # Y-axis settings
        ax.set_yticklabels([]) # Hide default global ticks
        ax.set_ylim(0, 1.05)
        
        # Add custom ticks for each axis (Real values, independent scales)
        grid_ticks = [0.2, 0.4, 0.6, 0.8, 1.0]
        for i, angle in enumerate(angles[:-1]):
            max_val = attr_maxs[i]
            for t in grid_ticks:
                val = t * max_val
                if val.is_integer():
                    label = f"{int(val)}"
                else:
                    label = f"{val:.1f}"
                
                # Add text with background to improve readability
                box_color = '#202020' if is_dark_mode else '#e0e0e0'
                text_color_tick = 'white' if is_dark_mode else 'black'
                
                ax.text(angle, t, label, 
                        color=text_color_tick, fontsize=font_size_axis, fontweight='bold',
                        ha='center', va='center',
                        bbox=dict(facecolor=box_color, edgecolor='none', alpha=0.8, boxstyle='round,pad=0.2'))
        
        # Legend and Title
        plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1), frameon=False, fontsize=font_size_legend, labelcolor=fg_color)
        # plt.title('Multi-Task AUP Comparison (Normalized)', fontsize=font_size_axis, fontweight='bold', color=fg_color, pad=50)
        
        if save_path:
            fc = bg_color if not is_dark_mode else 'black'
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=fc)
            print(f"Radar chart saved to {save_path}")
            
        # plt.show()
