import pickle
from collections import defaultdict
from datetime import timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
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


def filter_images(img_list, start_time: timedelta = None, end_time: timedelta = None):
    if start_time is None:
        return img_list
    first_time = img_list[0][0].time
    return [x for x in img_list if start_time <= x[0].time - first_time <= end_time]


def process(path_config, heuristic_paths, name, output_name, start_time=None, end_time=None):
    plt.figure()
    plot_record = defaultdict(dict)
    baseline_delays, best_delays, basic_heuristic_delays, smart_heuristic_delays = [], [], [], []
    for (path, (basic_heuristic_path, smart_heuristic_path)) in zip(path_config, heuristic_paths):
        dir = Path(path)
        baseline_cloud = pickle.load((dir / 'baseline_cloud.pkl').open('rb'))
        best_average_delay_cloud = pickle.load(
            (dir / 'best_delay_cloud.pkl').open('rb'))
        basic_heuristic_cloud = pickle.load(
            open(basic_heuristic_path + 'cloud.pkl', 'rb')) if basic_heuristic_path else None
        smart_heuristic_cloud = pickle.load(
            open(smart_heuristic_path + 'cloud.pkl', 'rb')) if smart_heuristic_path else None
        _baseline_delays = [(x[1] - x[0].time).total_seconds() / 3600 for x in
                            filter_images(baseline_cloud.image_list, start_time, end_time)]
        _best_delays = [(x[1] - x[0].time).total_seconds() / 3600 for x in
                        filter_images(best_average_delay_cloud.image_list, start_time, end_time)]
        _basic_heuristic_delays = [(x[1] - x[0].time).total_seconds() / 3600 for x in
                                   filter_images(basic_heuristic_cloud.image_list, start_time,
                                                 end_time)] if basic_heuristic_cloud else None
        _smart_heuristic_delays = [(x[1] - x[0].time).total_seconds() / 3600 for x in
                                   filter_images(smart_heuristic_cloud.image_list, start_time,
                                                 end_time)] if smart_heuristic_cloud else None

        baseline_delays.extend(_baseline_delays)
        best_delays.extend(_best_delays)
        if _basic_heuristic_delays:
            basic_heuristic_delays.extend(_basic_heuristic_delays)
            smart_heuristic_delays.extend(_smart_heuristic_delays)
    baseline_x, baseline_y = get_cdf(baseline_delays)
    best_x, best_y = get_cdf(best_delays)
    plt.plot(baseline_x, baseline_y, 'b-.', label='Greedy', linewidth=2)
    plt.plot(best_x, best_y, 'k', label='Umbra', linewidth=2)
    if basic_heuristic_delays:
        basic_heuristic_x, basic_heuristic_y = get_cdf(basic_heuristic_delays)
        smart_heuristic_x, smart_heuristic_y = get_cdf(smart_heuristic_delays)
        plt.plot(basic_heuristic_x, basic_heuristic_y, 'g--',
                 label='Withhold - Naive', linewidth=2)
        plt.plot(smart_heuristic_x, smart_heuristic_y, 'r:',
                 label='Withhold - Smart', linewidth=2)
    plot_record[name]['baseline'] = [baseline_x, baseline_y]
    plot_record[name]['umbra'] = [best_x, best_y]
    print(
        f'{name} median baseline: {np.median(baseline_delays)}, '
        f'median best: {np.median(best_delays)}, '
        f'median heuristic: {np.median(basic_heuristic_delays) if basic_heuristic_delays else None}')
    print(
        f'{name} baseline 90 percentile: {np.percentile(baseline_delays, 90)}, '
        f'Umbras 90 percentile: {np.percentile(best_delays, 90)}, '
        f'basic heuristic 90 percentile: {np.percentile(basic_heuristic_delays, 90) if basic_heuristic_delays else None}'
        f'smart heuristic 90 percentile: {np.percentile(smart_heuristic_delays, 90) if smart_heuristic_delays else None}')
    l = plt.legend(title='Method')
    l.get_title().set_fontsize('14')
    plt.grid()
    plt.xlabel('Latency in Hours', fontsize=14)
    plt.ylabel('CDF', fontsize=14)
    plt.xlim(left=0)
    plt.ylim([0, 1])
    plt.savefig(output_name + '.pdf', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    plt.rcParams['font.size'] = 14
    plt.rcParams['legend.title_fontsize'] = 14
    plt.rcParams["figure.figsize"] = (5, 4)
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    # plt.rcParams["font.family"] = "sans-serif"
    months = ['Jun', 'Jul', 'Aug']
    month_nums = [6, 7, 8]
    bws = ['1.2G', '1.5G', '1.8G']
    for bw in bws:
        umbra_dir, heuristic_dir = [], []
        for (month, month_num) in (zip(months, month_nums)):
            umbra_dir.append(f'result/binary_search_{month}_{bw}')
            heuristic_dir.append(
                [f'result/basic_heuristic_{month}_{bw}/', f'result/smart_heuristic_{month}_{bw}/'])
        try:
            process(umbra_dir, heuristic_dir, bw, f'{bw}_latency')
        except Exception as e:
            print(e)
            continue
