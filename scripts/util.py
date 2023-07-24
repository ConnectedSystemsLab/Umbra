def compute_value_combination(id, combination_list):
	dimensions = [len(x) for x in combination_list]
	total_num = 1
	for dim in dimensions:
		total_num *= dim
	if id >= total_num:
		return None
	results = []
	for i in range(len(dimensions)):
		results.append(combination_list[i][id % dimensions[i]])
		id //= dimensions[i]
	return results
