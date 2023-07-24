import json
from pathlib import Path


def main():
	output_dir = Path('../data/gs_config')
	sample_file = Path('../data/gs_config_1G.json')
	output_dir.mkdir(exist_ok=True)
	for bw in [1.2, 1.5, 1.8, 2.0]:
		sample_data = json.load(sample_file.open('r'))
		out_file = output_dir / f'gs_config_{bw:.1f}G.json'
		for gs_id in sample_data:
			sample_data[gs_id]['bandwidth'] = bw * 1e9 / 8
		json.dump(sample_data, out_file.open('w'), indent=4)


if __name__ == '__main__':
	main()
