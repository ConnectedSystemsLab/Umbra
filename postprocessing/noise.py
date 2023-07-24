import pickle
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def get_cdf(data, bins=100):
	"""
	Get the CDF of the data.

	@param data: the data
	@type data: list
	@return: the CDF
	@rtype: list
	:param bins:
	"""
	count, bins_count = np.histogram(data, bins=bins)
	pdf = count / sum(count)
	cdf = np.cumsum(pdf)
	return bins_count[1:], cdf


def process(paths, levels, output_name):
	plt.figure()
	baseline_median, baseline_90, baseline_25, baseline_75 = [], [], [], []
	best_median, best_90, best_25, best_75 = [], [], [], []
	for (path, name) in zip(paths, levels):
		dir = Path(path)
		baseline_cloud = pickle.load((dir / 'baseline_cloud.pkl').open('rb'))
		best_average_delay_cloud = pickle.load((dir / 'best_delay_cloud.pkl').open('rb'))
		_baseline_delays = [(x[1] - x[0].time).total_seconds() / 3600 for x in baseline_cloud.image_list]
		_best_delays = [(x[1] - x[0].time).total_seconds() / 3600 for x in best_average_delay_cloud.image_list]
		baseline_median.append(np.median(_baseline_delays))
		baseline_90.append(np.percentile(_baseline_delays, 90))
		best_median.append(np.median(_best_delays))
		best_90.append(np.percentile(_best_delays, 90))
		baseline_25.append(np.percentile(_baseline_delays, 25))
		baseline_75.append(np.percentile(_baseline_delays, 75))
		best_25.append(np.percentile(_best_delays, 25))
		best_75.append(np.percentile(_best_delays, 75))
		print(f'{name} median baseline: {np.median(_baseline_delays)}, median best: {np.median(_best_delays)}')
		print(
			f'{name} baseline 90 percentile: {np.percentile(_baseline_delays, 90)}, Umbras 90 percentile: {np.percentile(_best_delays, 90)},')
	pickle.dump([baseline_25, baseline_median, baseline_75, baseline_90, best_25, best_median, best_75, best_90],
	            open(output_name + '.pkl', 'wb+'))


if __name__ == '__main__':
	process(
		['data/1G_Aug_0.05', 'data/1G_Aug_0.1', 'data/1G_Aug_0.2', 'data/1G_Aug_0.4', 'data/1G_Aug_0.8'],
		['0.05', '0.1', '0.2', '0.4', '0.8'],
		'Noise'
	)
