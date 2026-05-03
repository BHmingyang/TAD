import yaml
from aup_utils import get_aup
from plot_lines import plot_aup_curve
from plot_radar import plot_radar_chart
from plot_histogram import plot_histogram

DATA_PATHS = ['data_llada.yaml', 'data_dream.yaml', 'data_dream_coder.yaml']
IS_DARK_MODE = True

# Collect AUP scores across all datasets for leaderboard
leaderboard = {}  # {method_name: [(task_name, aup), ...]}

# Font size settings per dataset: {dataset_key: (axis, legend, tick, radar_axis, radar_legend, hist_axis, hist_legend)}
FONT_SIZES = {
    'data_llada': (22, 20, 18, 18, 21, 19, 17),
    'data_dream': (22, 20, 18, 18, 21, 19, 17),
    'data_dream_coder': (26, 24, 22, 18, 21, 19, 17),
}

# Create output directory for figures
import os
os.makedirs('figs', exist_ok=True)

for DATA_PATH in DATA_PATHS:
    print(f"\n{'='*60}\nProcessing {DATA_PATH}\n{'='*60}")
    
    with open(DATA_PATH, 'r') as f:
        data = yaml.safe_load(f)
    
    TASK_LIST = list(data.keys())
    
    # Get font sizes for current dataset
    dataset_key = DATA_PATH.split('/')[-1].split('.')[0]
    fs_axis, fs_legend, fs_tick, fs_radar_axis, fs_radar_legend, fs_hist_axis, fs_hist_legend = FONT_SIZES.get(dataset_key, (20, 18, 20, 18, 21, 19, 17))
    
    # Auto-detect methods and assign colors (exclude EAGLE-3)
    all_methods = set()
    for task_data in data.values():
        all_methods.update(task_data.keys())
    all_methods.discard('EAGLE-3')  # Remove EAGLE-3 before color assignment
    
    num_methods = len(all_methods)
    if num_methods == 3:
        COLOR_LIST = ['orange', 'grey', 'red']
    elif num_methods == 6:
        COLOR_LIST = ['orange', 'grey', 'purple', 'green', 'blue', 'red']
    else:
        COLOR_LIST = ['orange', 'grey', 'purple', 'blue', 'red', 'green', 'cyan', 'magenta', 'pink', 'lime', 'teal', 'yellow']
    
    radar_data = {}
    
    for TASK in TASK_LIST:
        # Load all methods (including EAGLE-3 for AUP calculation)
        all_methods_data = {}
        methods_for_plot = {}  # Exclude EAGLE-3 for plotting
        
        for method_name, pairs in data[TASK].items():
            all_methods_data[method_name] = [(rho, y) for rho, y in pairs]
            if method_name != 'EAGLE-3':  # Exclude EAGLE-3 from plotting
                methods_for_plot[method_name] = [(rho, y) for rho, y in pairs]
        
        # Calculate y_max from all methods (including EAGLE-3)
        y_max = max(y for pairs in all_methods_data.values() for _, y in pairs)
        
        print(f"\nAUP Results on {TASK.upper()}")
        # Calculate AUP for all methods (including EAGLE-3)
        for method_name, pairs in all_methods_data.items():
            rho, y = zip(*pairs)
            aup = get_aup(list(rho), list(y), y_max, is_print=False)
            print(f"{method_name:25s}: AUP = {aup:.4f}")
            
            # Add to radar data only if not EAGLE-3
            if method_name != 'EAGLE-3':
                if method_name not in radar_data:
                    radar_data[method_name] = []
                radar_data[method_name].append(aup)
            
            # Collect for leaderboard (including EAGLE-3)
            if method_name not in leaderboard:
                leaderboard[method_name] = []
            leaderboard[method_name].append((TASK, aup))
        
        # Plot only methods excluding EAGLE-3
        plot_aup_curve(methods_for_plot, y_max, assigned_colors=COLOR_LIST, save_path=f'figs/{dataset_key}_aup_curve_{TASK}.png', is_dark_mode=IS_DARK_MODE, font_size_axis=fs_axis, font_size_legend=fs_legend, font_size_tick=fs_tick, dataset_name=TASK)
    
    plot_radar_chart(radar_data, TASK_LIST, assigned_colors=COLOR_LIST, save_path=f'figs/{dataset_key}_aup_radar.png', is_dark_mode=IS_DARK_MODE, font_size_axis=fs_radar_axis, font_size_legend=fs_radar_legend)
    plot_histogram(radar_data, TASK_LIST, assigned_colors=COLOR_LIST, save_path=f'figs/{dataset_key}_aup_histogram.png', is_dark_mode=IS_DARK_MODE, font_size_axis=fs_hist_axis, font_size_legend=fs_hist_legend)

# Print leaderboard sorted by average AUP (descending)
print(f"\n{'='*60}\nLEADERBOARD (Average AUP across all datasets)\n{'='*60}")
# Exclude humaneval+ and mbpp+ from average calculation
excluded_tasks = {'humaneval+', 'mbpp+'}
avg_aup = {}
for method, task_scores in leaderboard.items():
    filtered_scores = [aup for task, aup in task_scores if task not in excluded_tasks]
    avg_aup[method] = sum(filtered_scores) / 5.0 if filtered_scores else 0.0

for method, avg in sorted(avg_aup.items(), key=lambda x: x[1], reverse=True):
    print(f"{method:25s}: {avg:.4f}")
print('='*60)
