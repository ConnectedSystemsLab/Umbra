import argparse
import os
import subprocess
import sys

from scripts.util import compute_value_combination


def main():
	job_id = os.environ['SLURM_ARRAY_TASK_ID']
	parser=argparse.ArgumentParser()
	parser.add_argument('--bws', type=str,nargs='+', default=['hetero1','hetero2','hetero3'])
	parser.add_argument('--experiment', type=str, default='basic_heuristic')
	parser.add_argument('--experiment_suffix', type=str, default='')
	parser.add_argument('--months', type=str, nargs='+', default=['Jun', 'Jul', 'Aug'])
	parser.add_argument('--month_nums', type=int, nargs='+', default=[6, 7, 8])
	args,unknown_args = parser.parse_known_args()
	bws = args.bws
	try:
		((month, month_num), bw) = compute_value_combination(int(job_id), [list(zip(args.months, args.month_nums)), bws])
		command_line_args = [
			                    'python',
			                    '-m',
			                    f'experiments.{args.experiment}',
			                    '--image_mapping_info',
			                    f'data/permanent/planet_21{month}_5day_mapping.pkl',
			                    '--cache_file',
			                    f'data/permanent/bw_cache_{month}{args.experiment_suffix}.pkl',
			                    '--gs_config',
			                    f'data/permanent/gs_config/{bw}.json',
			                    '--start_time',
			                    f'2021-{month_num}-01T00:00:00',
			                    '--time_step',
			                    '60',
			                    '--sat_bw_multiplier',
			                    '0.125',
			                    '--result_dir',
			                    f'result/{args.experiment}{args.experiment_suffix}_{month}_{bw}',
			                    '--log_file',
			                    f'log/{args.experiment}{args.experiment_suffix}_{month}_{bw}.log',
		                    ] + unknown_args
		with subprocess.Popen(command_line_args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as p:
			for line in p.stdout:
				print(line.decode('utf-8'), end='')
	except Exception as e:
		print(e)
		exit(0)


if __name__ == '__main__':
	main()
