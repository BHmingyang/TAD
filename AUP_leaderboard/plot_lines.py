import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from aup_utils import get_aup

# Pre-defined Tech Colors (Gradients)
# Format: 'name': [start_color, end_color]
TECH_STYLE = {
    'red':     ['#8B0000', '#FF0011'], # Dark Red -> Neon Red
    'blue':    ['#00008B', '#00CCFF'], # Dark Blue -> Neon Cyan
    'green':   ['#004d00', '#14b881'], # Dark Green -> Neon Green
    'purple':  ['#4B0082', '#D766FF'], # Indigo -> Neon Purple
    'yellow':  ['#8B8000', '#FFD700'], # Dark Yellow -> Gold
    'orange':  ['#8B4500', '#FF5500'], # Saddle Brown -> Neon Orange
    'grey':    ['#404040', '#AAAAAA'], # Dim Grey -> Light Grey
    'cyan':    ['#008B8B', '#00FFFF'], # Dark Cyan -> Neon Cyan
    'magenta': ['#8B008B', '#FF00FF'], # Dark Magenta -> Neon Magenta
    'pink':    ['#C71585', '#FF1493'], # Deep Pink -> Neon Pink
    'lime':    ['#32CD32', '#7FFF00'], # Lime Green -> Chartreuse
    'teal':    ['#008080', '#00CED1'], # Teal -> Dark Turquoise
}

def plot_aup_curve(methods: dict, y_max: float, assigned_colors: dict = None, save_path: str = None, is_dark_mode: bool = True, outlier_threshold: float = None, font_size_axis: int = 12, font_size_legend: int = 12, font_size_tick: int = 12, dataset_name: str = None):
    """
    Plot accuracy-parallelism curves with a high-tech aesthetic.
    
    Args:
        methods: dict of {method_name: [(rho, y), ...]}
        y_max: maximum accuracy across all methods (for AUP calculation)
        assigned_colors: dict of {method_name: color_name} (optional)
        save_path: path to save the figure
        is_dark_mode: whether to use dark mode (True) or light mode (False)
        outlier_threshold: y-value threshold to detect outliers (auto if None)
        font_size_axis: font size for axis labels
        font_size_legend: font size for legend (method names)
        font_size_tick: font size for tick labels (axis numbers)
        dataset_name: name of the dataset to display at the bottom (optional)
    """
    # Default color cycle if not assigned
    default_colors = ['purple', 'blue', 'green', 'orange', 'red', 'yellow', 'grey', 'cyan', 'magenta', 'pink']
    
    # Determine style settings
    if is_dark_mode:
        style_context = 'dark_background'
        bg_color = 'black' # Or default dark
        fg_color = 'white'
        grid_color = 'white'
        grid_alpha = 0.15
        spine_color = 'white'
    else:
        style_context = 'default'
        bg_color = '#F5F5F7' # Apple light gray
        fg_color = 'black'
        grid_color = '#86868b' # Apple gray
        grid_alpha = 0.3
        spine_color = '#86868b'

    with plt.style.context(style_context):
        all_rho = []
        all_y = []
        aup_results = [] # Store (aup, method_name, text_color)
        method_data = [] # Store processed data for plotting
        
        # First Pass: Process Data & AUP
        for i, (method_name, pairs) in enumerate(methods.items()):
            if not pairs:
                continue
                
            # Determine color
            if assigned_colors:
                c_name = assigned_colors[i] if isinstance(assigned_colors, list) else assigned_colors.get(method_name, default_colors[i % len(default_colors)])
            else:
                c_name = default_colors[i % len(default_colors)]
            
            grad_colors = TECH_STYLE.get(c_name, TECH_STYLE['grey'])
            main_color = grad_colors[1] 
            
            if is_dark_mode:
                text_color = grad_colors[1]
            else:
                text_color = grad_colors[0]
            
            cmap = LinearSegmentedColormap.from_list(f"tech_{c_name}", grad_colors)
            
            rho, y = zip(*sorted(pairs, key=lambda x: x[0]))
            rho = np.array(rho)
            y = np.array(y)
            
            if np.max(y) <= 1.0:
                y = y * 100
            
            # Calculate AUP
            aup_val = get_aup(list(rho), list(y), y_max)
            aup_results.append((aup_val, method_name, text_color))
            
            # Store for processing
            method_data.append({
                'name': method_name,
                'rho': rho,
                'y': y,
                'colors': (main_color, text_color),
                'cmap': cmap
            })
            
            # Collect all data for initial stats
            all_rho.extend(rho)
            all_y.extend(y)

        # Detect Outliers
        outlier_names = set()
        if all_y:
            if outlier_threshold is None:
                sorted_y = sorted(all_y)
                q1, q3 = np.percentile(sorted_y, [25, 75])
                iqr = q3 - q1
                outlier_threshold = q3 + 1.5 * iqr
            
            for m in method_data:
                if np.max(m['y']) > outlier_threshold:
                    outlier_names.add(m['name'])

        # Setup Figure (Broken Axis if outliers exist)
        if outlier_names:
            fig, (ax_top, ax_bottom) = plt.subplots(2, 1, sharex=True, figsize=(9, 6), 
                                                  gridspec_kw={'height_ratios': [1, 4], 'hspace': 0.1})
            axes_list = [ax_top, ax_bottom]
            ax = ax_bottom # Main reference
        else:
            fig, ax = plt.subplots(figsize=(9, 6))
            axes_list = [ax]
            ax_bottom = ax

        if not is_dark_mode:
            fig.patch.set_facecolor(bg_color)
            for a in axes_list:
                a.set_facecolor(bg_color)

        # Second Pass: Plotting (draw curves/markers, collect label info)
        non_outlier_rho = []
        non_outlier_y_vals = []
        outlier_y_vals = []
        pending_labels = []  # Collect labels for anti-overlap processing
        
        for m in method_data:
            method_name = m['name']
            is_outlier = method_name in outlier_names
            
            # Select target axis
            if outlier_names:
                target_ax = ax_top if is_outlier else ax_bottom
            else:
                target_ax = ax
            
            rho = m['rho']
            y = m['y']
            main_color, text_color = m['colors']
            cmap = m['cmap']
            
            if is_outlier:
                outlier_y_vals.extend(y)
            else:
                non_outlier_rho.extend(rho)
                non_outlier_y_vals.extend(y)

            # Generate smooth curve points (fitting)
            if len(rho) >= 3:
                z = np.polyfit(rho, y, 2)
                p = np.poly1d(z)
                x_smooth = np.linspace(rho.min(), rho.max(), 300)
                y_smooth = p(x_smooth)
            elif len(rho) == 2:
                # Quadratic with vertex at (rho[0], y[0])
                x_smooth = np.linspace(rho.min(), rho.max(), 300)
                if rho[1] != rho[0]:
                    a = (y[1] - y[0]) / ((rho[1] - rho[0]) ** 2)
                    y_smooth = a * (x_smooth - rho[0]) ** 2 + y[0]
                else:
                    y_smooth = np.linspace(y[0], y[1], 300)
            else:
                x_smooth = rho
                y_smooth = y
            
            # Plotting curves and markers
            x_span = max(all_rho) if all_rho else 1.0
            if len(rho) > 1:
                points = np.array([x_smooth, y_smooth]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                
                norm = plt.Normalize(x_smooth.min(), x_smooth.max())
                lc = LineCollection(segments, cmap=cmap, norm=norm)
                lc.set_array(x_smooth)
                lc.set_linewidth(3)
                lc.set_alpha(0.9)
                target_ax.add_collection(lc)
                
                # Markers
                marker_edge = main_color
                if not is_dark_mode: marker_edge = 'white' 
                target_ax.scatter(rho, y, color='white', edgecolors=main_color, s=60, zorder=10, marker='o', linewidth=1.5)
                
                # Collect label position (defer drawing)
                label_x = rho[-1] + x_span * 0.03
                label_y = y[-1]
                pending_labels.append((label_x, label_y, method_name, text_color, target_ax))
            else:
                target_ax.scatter(rho, y, color=main_color, s=120, marker='o', zorder=10, label=method_name)
                label_x = rho[0] + x_span * 0.03
                label_y = y[0]
                pending_labels.append((label_x, label_y, method_name, text_color, target_ax))

        # Anti-overlap: adjust label y positions
        # Group labels by target axis
        from collections import defaultdict
        labels_by_ax = defaultdict(list)
        for lx, ly, name, color, tax in pending_labels:
            labels_by_ax[id(tax)].append([lx, ly, name, color, tax])
        
        for ax_id, labels in labels_by_ax.items():
            # Sort by y position (ascending)
            labels.sort(key=lambda item: item[1])
            # Estimate minimum spacing in data units
            if all_y:
                y_range = max(all_y) - min(all_y)
            else:
                y_range = 10.0
            min_spacing = y_range * 0.06  # ~6% of y-range as minimum gap
            
            # Push overlapping labels apart (greedy bottom-up)
            for i in range(1, len(labels)):
                prev_y = labels[i-1][1]
                curr_y = labels[i][1]
                if curr_y - prev_y < min_spacing:
                    labels[i][1] = prev_y + min_spacing
        
        # Draw all labels with adjusted positions (from labels_by_ax, not pending_labels)
        for ax_id, labels in labels_by_ax.items():
            for lx, ly, name, color, tax in labels:
                tax.text(lx, ly, name, 
                         color=color, 
                         fontsize=font_size_legend, 
                         fontweight='bold',
                         ha='left', va='center')

        # Styling for all axes
        for a in axes_list:
            a.grid(True, linestyle='--', alpha=grid_alpha, color=grid_color)
            a.spines['top'].set_visible(False)
            a.spines['right'].set_visible(False)
            a.spines['bottom'].set_color(spine_color)
            a.spines['left'].set_color(spine_color)
            a.tick_params(colors=fg_color, labelsize=font_size_tick)
            
        # Labeling
        ax_bottom.set_xlabel(r'Parallelism $\rho$ (TPF, tokens/forward)', fontsize=font_size_axis, color=fg_color)
        # Y label centered? We'll just put it on the bottom ax or create a shared one.
        # Simple approach: label bottom axis
        ax_bottom.set_ylabel(r'Accuracy (%)', fontsize=font_size_axis, color=fg_color)
        
        # Adjust Limits
        if non_outlier_rho:
            max_rho = max(non_outlier_rho)
            for a in axes_list:
                 a.set_xlim(left=0 if max_rho > 5 else 0.8, right=max_rho * 1.25)
        
        # Y-Limits Bottom
        if non_outlier_y_vals:
             min_y = min(non_outlier_y_vals)
             max_y = max(non_outlier_y_vals)
             ax_bottom.set_ylim(bottom=min_y - 1.0, top=max_y + 0.2 if outlier_names else max_y + 1.0)

        # Broken Axis Logic
        if outlier_names:
             # Y-Limits Top
             if outlier_y_vals:
                 tmin, tmax = min(outlier_y_vals), max(outlier_y_vals)
                 # Add some margin
                 margin = 1.0
                 ax_top.set_ylim(tmin - margin, tmax + margin)
            
             # Hide spines
             ax_top.spines['bottom'].set_visible(False)
             ax_bottom.spines['top'].set_visible(False)
             
             # Remove ticks from top ax (prevent white dots)
             ax_top.tick_params(axis='x', which='both', bottom=False, top=False, labeltop=False)
             ax_bottom.xaxis.tick_bottom()
             
             # Diagonal break lines (Left side only)
             d = .015 
             kwargs = dict(transform=ax_top.transAxes, color=fg_color, clip_on=False)
             ax_top.plot((-d, +d), (-d, +d), **kwargs)        

             kwargs.update(transform=ax_bottom.transAxes)  
             ax_bottom.plot((-d, +d), (1 - d, 1 + d), **kwargs) 
            
        # Add AUP Score List in Top Right
        # Sort by AUP descending
        aup_results.sort(key=lambda x: x[0], reverse=True)
        
        text_x = 0.98
        text_y = 0.95
        line_height = 0.06
        
        # for aup_val, m_name, m_color in aup_results:
        #     label_str = f"{m_name}: {aup_val:.2f}"
        #     ax.text(text_x, text_y, label_str, 
        #             transform=ax.transAxes, 
        #             color=m_color, 
        #             fontsize=12, 
        #             fontweight='bold', 
        #             ha='right', 
        #             va='top')
        #     text_y -= line_height

        # Add dataset name at the top
        if dataset_name:
            fig.suptitle(dataset_name, fontsize=font_size_legend+6, color=fg_color, fontweight='bold')

        plt.tight_layout()
        
        if save_path:
            # If light mode, save with correct facecolor
            fc = bg_color if not is_dark_mode else 'black'
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=fc)
            print(f"Figure saved to {save_path}")
