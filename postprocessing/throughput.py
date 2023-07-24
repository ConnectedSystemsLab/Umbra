import datetime
import pickle


def get_throughput(cloud, start_time, end_time):
    return sum([x[0].size for x in cloud.image_list if start_time < x[1] < end_time])


def main():
    months = ['Jun', 'Jul', 'Aug']
    month_num = [6, 7, 8]
    bws = ['1.2G', '1.5G', '1.8G']
    for (month, month_num) in list(zip(months, month_num)):
        for bw in bws:
            try:
                cloud = pickle.load(
                    open(f'result/binary_search_{month}_{bw}/best_delay_cloud.pkl', 'rb'))
                baseline_cloud = pickle.load(
                    open(f'result/binary_search_{month}_{bw}/baseline_cloud.pkl', 'rb'))
                basic_heuristic_cloud = pickle.load(
                    open(f'result/basic_heuristic_{month}_{bw}/cloud.pkl', 'rb'))
                smart_heuristic_cloud = pickle.load(
                    open(f'result/smart_heuristic_{month}_{bw}/cloud.pkl', 'rb'))
                baseline_throughput = get_throughput(baseline_cloud, datetime.datetime(2021, month_num, 2, 0, 0, 0),
                                                     datetime.datetime(2021, month_num, 5, 0, 0, 0))
                basic_heuristics_throughput = get_throughput(basic_heuristic_cloud, datetime.datetime(2021, month_num, 2, 0, 0, 0),
                                                             datetime.datetime(2021, month_num, 5, 0, 0, 0))
                smart_heuristics_throughput = get_throughput(smart_heuristic_cloud, datetime.datetime(2021, month_num, 2, 0, 0, 0),
                                                             datetime.datetime(2021, month_num, 5, 0, 0, 0))
                best_throughput = get_throughput(cloud, datetime.datetime(2021, month_num, 2, 0, 0, 0),
                                                 datetime.datetime(2021, month_num, 5, 0, 0, 0))
                print(
                    f'{bw} {month} {baseline_throughput / 1e12:.1f} & {basic_heuristics_throughput / 1e12:.1f} & {best_throughput / 1e12:.1f}& {smart_heuristics_throughput / 1e12:.1f}\\\\')
            except:
                raise
        print('\n')


if __name__ == '__main__':
    main()
