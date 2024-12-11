import yaml
import subprocess
import json
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
import logging
from pathlib import Path
import copy
import pandas as pd


class ExperimentRunner:
    def __init__(
        self,
        experiments_dir: str = "experiments_results",
        progress_file: str = "experiment_progress.json",
        results_file: str = "experiment_results.csv",
    ):
        # Initialize paths
        self.base_config_path = Path("cfg/ghostnetv2_clouds.yaml")
        self.experiments_dir = Path(experiments_dir)
        self.progress_file = self.experiments_dir / progress_file
        self.results_file = self.experiments_dir / results_file
        # Create experiments directory if it doesn't exist
        self.experiments_dir.mkdir(exist_ok=True)
        # Set up logging
        self.setup_logging()
        # Load base configuration
        with open(self.base_config_path, "r") as f:
            self.base_config = yaml.safe_load(f)
        # Load or initialize progress and results
        self.progress = self.load_progress()

    def setup_logging(self):
        """Configure logging to both file and console."""
        log_file = self.experiments_dir / "experiment_runner.log"
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(message)s",  # Simplified format
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )

    def load_progress(self) -> Dict:
        """Load progress from file or create new progress tracking."""
        if self.progress_file.exists():
            with open(self.progress_file, "r") as f:
                return json.load(f)
        return {"completed": [], "failed": [], "current_batch": None, "last_run": None}

    def save_progress(self):
        """Save current progress to file."""
        with open(self.progress_file, "w") as f:
            json.dump(self.progress, f, indent=2)

    def generate_experiment_configs(self) -> List[Dict[str, Any]]:
        """Generate different configurations for experiments."""
        configs = []
        # Model variations
        ps = range(-5, 1)
        for p in ps:
            # Create a copy of base config
            config = copy.deepcopy(self.base_config)
            # Update configuration
            config["inject_p"] = 10**p
            # Create identifier
            config_id = f"p{p}"
            configs.append({"id": config_id, "config": config})
        return configs

    def run_experiment(
        self, config: Dict[str, Any], config_id: str
    ) -> Tuple[bool, Optional[float], Optional[float]]:
        """Run a single experiment with the given configuration."""
        try:
            # Copy the config file to the configs/experiment directory
            configs_dir = Path("cfg")
            target_config_path = configs_dir / f"temp_{config_id}.yaml"

            # Ensure the configs/experiment directory exists
            configs_dir.mkdir(exist_ok=True, parents=True)

            # Save configuration
            with open(target_config_path, "w") as f:
                yaml.dump(config, f)

            # Run the experiment using the relative path that Hydra expects
            cmd = f"python -m main -c cfg/temp_{config_id}.yaml"
            logging.info(f"Starting experiment {config_id}")

            # Run with Popen to get real-time output
            process = subprocess.Popen(
                cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Redirect stderr to stdout
                text=True,
                bufsize=1,
                universal_newlines=True,
            )

            # Collect output while printing in real-time
            output_lines = []
            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                if line:
                    print(line.rstrip())  # Print to console in real-time
                    output_lines.append(line.rstrip())

            if process.returncode == 0:
                logging.info(f"Experiment {config_id} completed successfully")
                return True
            else:
                logging.warning(
                    f"Experiment {config_id} failed with return code {process.returncode}"
                )
                return False

        except Exception as e:
            logging.warning(f"Error running experiment {config_id}: {str(e)}")
            return False
        finally:
            # Clean up the temporary config file
            if target_config_path.exists():
                target_config_path.unlink()

    def run_all_experiments(self):
        """Run all experiments with progress tracking and failure handling."""
        configs = self.generate_experiment_configs()
        total_experiments = len(configs)
        completed_count = len(self.progress["completed"])
        failed_count = len(self.progress["failed"])

        logging.info(f"\n{'='*80}")
        logging.info(f"Starting experiments batch")
        logging.info(f"Total experiments to run: {total_experiments}")
        logging.info(f"Already completed: {completed_count}")
        logging.info(f"Previously failed: {failed_count}")
        logging.info(f"Remaining: {total_experiments - completed_count - failed_count}")
        logging.info(f"{'='*80}\n")

        for idx, config_data in enumerate(configs, 1):
            config_id = config_data["id"]

            # Skip if already completed
            if config_id in self.progress["completed"]:
                logging.info(
                    f"[{idx}/{total_experiments}] Skipping completed experiment: {config_id}"
                )
                continue

            # Skip if previously failed
            if config_id in self.progress["failed"]:
                logging.info(
                    f"[{idx}/{total_experiments}] Skipping failed experiment: {config_id}"
                )
                continue

            # Print experiment details
            logging.info(f"\n{'-'*80}")
            logging.info(
                f"[{idx}/{total_experiments}] Starting experiment: {config_id}"
            )
            logging.info(f"P: {config_data['config']['inject_p']}")
            logging.info(f"{'-'*80}\n")

            # Update progress
            self.progress["current_batch"] = config_id
            self.progress["last_run"] = datetime.now().isoformat()
            self.save_progress()

            # Run experiment
            success = self.run_experiment(config_data["config"], config_id)

            if success:
                self.progress["completed"].append(config_id)
            else:
                self.progress["failed"].append(config_id)
                logging.warning(
                    f"\nExperiment {config_id} failed or did not produce valid metrics\n"
                )

            self.save_progress()

        self.progress["current_batch"] = None
        self.save_progress()

        # Print final summary
        logging.info(f"\n{'='*80}")
        logging.info("Experiment Batch Completed!")
        logging.info(f"Total experiments: {total_experiments}")
        logging.info(f"Successfully completed: {len(self.progress['completed'])}")
        logging.info(f"Failed: {len(self.progress['failed'])}")
        logging.info(f"\n{'='*80}")


def main():
    # Initialize and run experiments
    runner = ExperimentRunner(experiments_dir="experiment_runs")

    runner.run_all_experiments()


if __name__ == "__main__":
    main()
