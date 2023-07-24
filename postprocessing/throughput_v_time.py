import pickle

import matplotlib.pyplot as plt
import numpy as np


def get_throughput(cloud):
	time_list = [x[1].timestamp() / 60 for x in cloud.image_list]
	x, y = np.unique(time_list, return_counts=True)
	y = y * 300e6 / 60 / 1e9 * 8
	x -= np.min(x)
	return x, y


def main():
	plt.rcParams['font.size'] = 14
	plt.rcParams['legend.title_fontsize'] = 14
	plt.rcParams["figure.figsize"] = (5, 4)
	# plt.rcParams["font.family"] = "sans-serif"
	months = ['Jun', 'Jul', 'Aug']
	month_nums = [6, 7, 8]
	bws = ['1.2G', '1.5G', '1.8G', '2.0G']
	for (month, month_num) in list(zip(months, month_nums)):
		for bw in bws:
			try:
				plt.figure()
				cloud = pickle.load(open(f'result/{month}_{bw}/best_delay_cloud.pkl', 'rb'))
				baseline_cloud = pickle.load(open(f'result/{month}_{bw}/baseline_cloud.pkl', 'rb'))
				heuristics_cloud = pickle.load(open(f'result/{month}_{bw}_baseline_dgs/cloud.pkl', 'rb'))
				base_x, base_y = get_throughput(baseline_cloud)
				heuristic_x, heuristic_y = get_throughput(heuristics_cloud)
				best_x, best_y = get_throughput(cloud)
				plt.plot(base_x, base_y, 'b-', label='Baseline', linewidth=2)
				plt.plot(heuristic_x, 'g--', heuristic_y, label='Heuristics', linewidth=2)
				plt.plot(best_x, best_y, 'k', label='Umbra', linewidth=2)
				l = plt.legend(title='Method')
				l.get_title().set_fontsize('14')
				plt.grid()
				plt.xlabel('Time/h', fontsize=14)
				plt.ylabel('Throughput/Gbps', fontsize=14)
				plt.xlim(left=0)
				plt.savefig(f'{month}_{bw}_throughput_v_time.pdf', bbox_inches='tight')
				plt.show()
			except:
				continue
		print('1')


if __name__ == '__main__':
	main()
