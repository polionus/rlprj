import glob
import os
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import bootstrap


class custom_plt:
    def __init__(self, input_dir, middle_dir, output_dir, mode="all", n=None):
        self.input_dir = input_dir
        self.middle_dir = middle_dir
        self.output_dir = output_dir
        self.mode = mode
        self.n = n 
        

    
    def _group_files(self):
        self.grouped_files = defaultdict(list)
        file_pattern = os.path.join(self.input_dir, "*RETURNS.npy")
        for file_path in glob.glob(file_pattern):
            filename = os.path.basename(file_path)

            parts = filename.split("_")
            setting_key = "_".join([p for p in parts if not p.startswith("seed")])
            self.grouped_files[setting_key].append(file_path)

    
    def _avg_return(self):
        all_aucs = defaultdict(list)

        for setting_key, file_paths in self.grouped_files.items():
            all_returns = []

            min_length = float('inf')
            for file_path in file_paths:
                returns = np.load(file_path)
                min_length = min(min_length, len(returns))

            for file_path in file_paths:
                returns = np.load(file_path)[:min_length] 
                returns = returns.reshape(returns.shape[0])
                all_returns.append(returns)

            all_returns = np.array(all_returns)
            average_returns = np.mean(all_returns, axis=0)

            # Calculate AUC for each seed
            auc_values = [np.trapz(seed_returns) for seed_returns in all_returns]
            p5 = np.percentile(auc_values, 5)
            p95 = np.percentile(auc_values, 95)

            if self.mode == "all":
                mean_values = np.mean(all_returns, axis=1)
            elif self.mode == "n" and self.n is not None:
                mean_values = np.array([np.mean(seed_returns[-self.n:]) if len(seed_returns) >= self.n else np.mean(seed_returns) for seed_returns in all_returns])

            p5_mean = np.percentile(mean_values, 5)
            p95_mean = np.percentile(mean_values, 95)

            # Calculate bootstrap confidence interval
            confidence_interval = bootstrap((all_returns,), np.mean, axis=0, confidence_level=0.95, n_resamples=1000, method='basic')
            lower_ci = confidence_interval.confidence_interval.low
            upper_ci = confidence_interval.confidence_interval.high

            # Save averaged returns and confidence intervals in columns
            output_data = np.vstack((average_returns, lower_ci, upper_ci, [p5]*len(average_returns), [p95]*len(average_returns), [p5_mean]*len(average_returns), [p95_mean]*len(average_returns))).T

            output_filename = f"{setting_key[:-4]}_AVERAGED_RETURNS.npy"
            output_path = os.path.join(self.middle_dir, output_filename)
            np.save(output_path, output_data)

    def _plot_returns(self):
        file_pattern = os.path.join(self.middle_dir, "*RETURNS.npy")
        files = glob.glob(file_pattern)
    
        print("Available files:")
        file_list = [os.path.basename(file) for file in files]
        for idx, filename in enumerate(file_list):
            params = filename.split("_")
            print(f"{idx}. Alg:{params[0][3:]}, Env: {params[1][3:]}, t: {params[2][11:]}, alpha: {params[3][5:]}")

        choice = input("Do you want to plot all files or a subset? (all/partial): ").strip().lower()

        if choice == "all":
            selected_files = files  # Plot all files
        elif choice == "partial":
            indices = input("Enter the indices of the files to plot, separated by commas: ").strip()
            indices = list(map(int, indices.split(",")))
            selected_files = [files[i] for i in indices]
        else:
            print("Invalid choice. Exiting.")
            return

        # Plot the selected files
        plt.figure(figsize=(10, 6))
        for file_path in selected_files:
            data = np.load(file_path)
            average_returns = data[:, 0]
            lower_ci = data[:, 1]
            upper_ci = data[:, 2]
            
            label = os.path.basename(file_path).replace("_AVERAGED_RETURNS.npy", "")
            plt.plot(average_returns, label=label)
            plt.fill_between(range(len(average_returns)), lower_ci, upper_ci, alpha=0.2)

        plt.title("Averaged Returns")
        plt.xlabel("Episode")
        plt.ylabel("Return")
        plt.legend()
        plt.grid()
        plt.tight_layout()

        # Save the plot to output_dir
        output_path = os.path.join(self.output_dir, "averaged_returns_plot.png")
        plt.savefig(output_path)
        plt.show()
        print(f"Plot saved to: {output_path}")

    def _calculate_auc(self):
        auc_grouped_files = defaultdict(list)
        file_pattern = os.path.join(self.middle_dir, "*RETURNS.npy")
        for file_path in glob.glob(file_pattern):
            filename = os.path.basename(file_path)

            parts = filename.split("_")
            key_without_t = "_".join([p for p in parts if not p.startswith("tmultiplier")])
            auc_grouped_files[key_without_t].append(file_path)

        fig, axs = plt.subplots(1, 2, figsize=(20, 8))

        for setting_key, file_paths in auc_grouped_files.items():
            tmultiplier_files = []
            for file_path in file_paths:
                filename = os.path.basename(file_path)
                for part in filename.split("_"):
                    if part.startswith("tmultiplier"):
                        t_value = float(part[len("tmultiplier"):])
                        tmultiplier_files.append((t_value, file_path))
                        break
            tmultiplier_files.sort(key=lambda x: x[0])  # Sort by tmultiplier

            t_values = []
            auc_values = []
            normalized_auc_values = []

            for t_value, file_path in tmultiplier_files:
                returns = np.load(file_path)
                data = returns[:,0]
                auc = np.trapezoid(data)
                t_values.append(t_value)
                auc_values.append(auc)
                p5 = returns[0,3]
                p95 = returns[0,4]
                if p95 != p5:
                    normalized_auc = (auc - p5) / (p95 - p5)
                else:
                    normalized_auc = 0  # Avoid division by zero
                normalized_auc_values.append(normalized_auc)

            # Plot original AUC values
            axs[0].plot(t_values, auc_values, marker="o", label=setting_key.replace("_", " "))
            axs[0].set_title("AUC vs T Multiplier for All Settings")
            axs[0].set_xlabel("T Multiplier")
            axs[0].set_ylabel("Area Under the Curve (AUC)")
            axs[0].grid()
            axs[0].legend()

            # Plot normalized AUC values
            if normalized_auc_values:
                axs[1].plot(t_values, normalized_auc_values, marker="o", label=setting_key.replace("_", " "))
                axs[1].set_title("Normalized AUC vs T Multiplier for All Settings")
                axs[1].set_xlabel("T Multiplier")
                axs[1].set_ylabel("Normalized AUC")
                axs[1].grid()
                axs[1].legend()

        plt.tight_layout()
        output_filename = "AUC_vs_TMULTIPLIER_COMPARISON.png"
        output_path = os.path.join(self.output_dir, output_filename)
        plt.savefig(output_path)
        plt.show()

    def _calculate_mean_returns(self):
        mean_grouped_files = defaultdict(list)
        file_pattern = os.path.join(self.middle_dir, "*RETURNS.npy")
        for file_path in glob.glob(file_pattern):
            filename = os.path.basename(file_path)

            parts = filename.split("_")
            key_without_t = "_".join([p for p in parts if not p.startswith("tmultiplier")])
            mean_grouped_files[key_without_t].append(file_path)

        fig, axs = plt.subplots(1, 2, figsize=(20, 8))

        for setting_key, file_paths in mean_grouped_files.items():
            tmultiplier_files = []
            for file_path in file_paths:
                filename = os.path.basename(file_path)
                for part in filename.split("_"):
                    if part.startswith("tmultiplier"):
                        t_value = float(part[len("tmultiplier"):])
                        tmultiplier_files.append((t_value, file_path))
                        break
            tmultiplier_files.sort(key=lambda x: x[0])  # Sort by tmultiplier

            t_values = []
            mean_values = []
            normalized_mean_values = []
            for t_value, file_path in tmultiplier_files:
                returns = np.load(file_path)
                data = returns[:,0]

                if self.mode == "all":
                    mean_return = np.mean(returns)
                elif self.mode == "n" and self.n is not None:
                    if self.n > len(returns):
                        mean_return = np.mean(returns)
                    else:
                        mean_return = np.mean(returns[-self.n:])

                t_values.append(t_value)
                mean_values.append(mean_return)

                p5 = returns[0,5]
                p95 = returns[0,6]
                if p95 != p5:
                    normalized_value = (mean_return - p5) / (p95 - p5)
                else:
                    normalized_value = 0  # Avoid division by zero
                normalized_mean_values.append(normalized_value)

            # Plot original AUC values
            axs[0].plot(t_values, mean_values, marker="o", label=setting_key.replace("_", " "))
            axs[0].set_title(f"Mean {self.mode} vs T Multiplier for All Settings")
            axs[0].set_xlabel("T Multiplier")
            axs[0].set_ylabel(f"Mean of {self.mode} steps")
            axs[0].grid()
            axs[0].legend()

            # Plot normalized AUC values
            if normalized_mean_values:
                axs[1].plot(t_values, normalized_mean_values, marker="o", label=setting_key.replace("_", " "))
                axs[1].set_title(f"Normalized Mean {self.mode} vs T Multiplier for All Settings")
                axs[1].set_xlabel("T Multiplier")
                axs[1].set_ylabel(f"Normalized Mean of {self.mode} steps")
                axs[1].grid()
                axs[1].legend()

        plt.tight_layout()
        output_filename = f"Mean_{self.mode}_vs_TMULTIPLIER_COMPARISON.png"
        output_path = os.path.join(self.output_dir, output_filename)
        plt.savefig(output_path)
        plt.show()


if __name__ == "__main__":
    input_dir = "./results"  
    middle_dir = "./plots/avg_npy"
    output_dir = "./plots/png"  
    mode = "all"
    n = 100

    ploter = custom_plt(input_dir,middle_dir, output_dir, mode, n)
    ploter._group_files()
    ploter._avg_return()
    ploter._plot_returns()
    ploter._calculate_auc()
    ploter._calculate_mean_returns()
