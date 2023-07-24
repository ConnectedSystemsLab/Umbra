import argparse
import os
import subprocess
import sys

from scripts.util import compute_value_combination


def main():
	job_id = os.environ['SLURM_ARRAY_TASK_ID']
	months = ["Jun", "Jul", "Aug"]
	months_nums = [6, 7, 8]
	parser=argparse.ArgumentParser()
	parser.add_argument('--bws', type=str,nargs='+', default=['hetero1','hetero2','hetero3'])
	args = parser.parse_args()
	bws = args.bws
	try:
		((month, month_num), bw) = compute_value_combination(int(job_id), [list(zip(months, months_nums)), bws])
		command_line_args = [
			                    'python',
			                    '-m',
			                    'experiments.basic_heuristic',
			                    '--image_mapping_info',
			                    f'data/pernament/planet_21{month}_5day_mapping.pkl',
			                    '--cache_file',
			                    f'data/permanent/bw_cache_{month}_dgs.pkl',
			                    '--gs_config',
			                    f'data/dgs_config/gs_config_{bw}.json',
			                    '--start_time',
			                    f'2021-{month_num}-01T00:00:00',
			                    '--time_step',
			                    '60',
			                    '--sat_bw_multiplier',
			                    '0.125',
			                    '--result_dir',
			                    f'result/{month}_{bw}_basic_heuristics',
			                    '--log_file',
			                    f'log/baseline_{month}_{bw}.log',
		                    ] + sys.argv[1:]
		print(subprocess.call(command_line_args))
	except:
		exit(0)


if __name__ == '__main__':
	main()
