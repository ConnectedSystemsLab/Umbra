import os
import subprocess
import sys

from scripts.util import compute_value_combination


def main():
	job_id = os.environ['SLURM_ARRAY_TASK_ID']
	months = ["Jun", "Jul", "Aug"]
	months_nums = [6, 7, 8]
	bws = ['2G']
	try:
		((month, month_num), bw) = compute_value_combination(int(job_id), [list(zip(months, months_nums)), bws])
		command_line_args = [
			                    'python',
			                    '-m',
			                    'experiments.binary_search',
			                    '--image_mapping_info',
			                    f'data/planet_21{month}_5day_mapping.pkl',
			                    '--cache_file',
			                    f'data/bw_cache_{month}.pkl',
			                    '--gs_config',
			                    f'data/gs_config/gs_config_{bw}.json',
			                    '--start_time',
			                    f'2021-{month_num}-01T00:00:00',
			                    '--time_step',
			                    '60',
			                    '--sat_bw_multiplier',
			                    '0.125',
			                    '--result_dir',
			                    f'result/{month}_{bw}',
			                    '--log_file',
			                    f'log/bin_{month}_{bw}.log',
			                    '--throughput_threshold',
			                    '0.99',
			                    '--max_steps',
			                    '3',
		                    ] + sys.argv[1:]
		with subprocess.Popen(command_line_args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as p:
			for line in p.stdout:
				print(line)
	except:
		exit(0)


if __name__ == '__main__':
	main()
