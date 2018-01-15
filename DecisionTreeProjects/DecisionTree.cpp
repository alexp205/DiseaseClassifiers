#include "DecisionTree.h"

// Constructor
/*

*/
decisionTree::decisionTree(vvd& train_dataset, node* root, bool discrete, bool classification, bool forest)
{
	is_discrete = discrete;
	is_classification = classification;
	is_in_forest = forest;

	root_node = buildTree(train_dataset, root);
	if (!is_in_forest) {
		root_node = pruneTree();
	}
}

// Destructor
decisionTree::~decisionTree()
{

}

node* decisionTree::buildTree(vvd& input_data, node* node_ref)
{
	// if there is no input data, something went wrong
	if (input_data.empty()) {
		wcout << L"ERROR: empty data detected, please check for errors" << endl;
		return NULL;
	}
	// check if the tree is at a leaf
	if (checkLeaf(input_data)) {
		node_ref->is_leaf = true;
		node_ref->label = input_data[0][input_data[0].size() - 1];
		return node_ref;
	}
	data_info = getDatasetInfo(input_data);
	labels = getLabelInfo(input_data);
	// choose how to split based on if the data is discrete or continuous
	// for reference:
	//   - discrete - split on all possible values for that variable
	//   - continuous - find the best threshold for that variable and make a 
	//                  binary split
	vvd split_data;
	if (is_in_forest) {
		int num_data = round(sqrt(input_data.size()));
		split_data = getForestNodeData(input_data, num_data);
	} else {
		split_data = input_data;
	}
	if (is_discrete) {
		auto split_info = bestSplitVar(split_data);
		int split_var = get<0>(split_info);
		if (split_var == -1) {
			wcout << L"ERROR: no split variable detected, please check for errors" << endl;
			return NULL;
		}
		node_ref->split_label = split_var;
		for (int x = 0; x < data_info[split_var].size(); x++) {
			node* child = new node;
			child->split_label = split_var;
			child->is_leaf = false;
			vvd data_subset = subsetDiscreteData(input_data, split_var, data_info[split_var][x]);
			node_ref->children.push_back(buildTree(data_subset, child));
		}
		return node_ref;
	}
	else {
		auto split_info = bestSplitVar(split_data);
		int split_var = get<0>(split_info);
		double threshold = get<1>(split_info);
		if (split_var == -1) {
			wcout << L"ERROR: no split variable detected, please check for errors" << endl;
			return NULL;
		}
		node_ref->split_label = split_var;
		vector<vvd> data_subsets = subsetContinuousData(input_data, split_var, threshold);
		node* left_child = new node;
		left_child->split_label = split_var;
		left_child->is_leaf = false;
		node* right_child = new node;
		right_child->split_label = split_var;
		right_child->is_leaf = false;
		node_ref->children.push_back(buildTree(data_subsets[0], left_child));
		node_ref->children.push_back(buildTree(data_subsets[1], right_child));
		return node_ref;
	}
}

vvd decisionTree::getDatasetInfo(vvd& input_data)
{
	vvd data_info;

	for (int y = 0; y < input_data[0].size() - 1; y++) {
		vd var_info;
		for (int x = 0; x < input_data.size(); x++) {
			double val = input_data[x][y];
			if (find(var_info.begin(), var_info.end(), val) == var_info.end()) {
				var_info.push_back(val);
			}
		}
		data_info.push_back(var_info);
	}

	return data_info;
}

map<double,double> decisionTree::getLabelInfo(vvd& input_data)
{
	map<double, double> label_info;

	for (int x = 0; x < input_data.size(); x++) {
		double label = input_data[x][input_data[x].size() - 1];
		if (label_info.count(label) == 0) {
			label_info[label] = 1;
		} else {
			label_info[label]++;
		}
	}

	return label_info;
}

/*
This function checks two things in order to determine if the node is a leaf:
  1. if all of the remaining data has the same label
  2. if all of the remaining data has the same values for every variable
*/
bool decisionTree::checkLeaf(vvd& input_data)
{
	if (input_data.size() == 1) return true;

	double sample_label = input_data[0][input_data[0].size() - 1];
	for (int x = 1; x < input_data.size(); x++) {
		double test_label = input_data[x][input_data[x].size() - 1];
		if (test_label != sample_label) return false;
	}

	for (int y = 0; y < input_data[0].size() - 1; y++) {
		double sample_var_val = input_data[0][y];
		for (int x = 1; x < input_data.size(); x++) {
			double test_var_val = input_data[x][y];
			if (test_var_val != sample_var_val) return false;
		}
	}

	return true;
}

tuple<int,double> decisionTree::bestSplitVar(vvd& input_data)
{
	int best_split_var = -1;
	double best_threshold = -1;

	double label_entropy = calculateEntropy(input_data, input_data[0].size() - 1, -1); // H(Y)

	double max_info_gain = -numeric_limits<double>::infinity();
	if (is_discrete) {
		for (int y = 0; y < input_data[0].size() - 1; y++) {
			double var_info_gain = calculateInfoGain(input_data, y, -1, label_entropy);
			if (var_info_gain > max_info_gain) best_split_var = y;
		}
	} else {
		for (int y = 0; y < input_data[0].size() - 1; y++) {
			vd thresholds = getThresholds(input_data, y);
			for (int x = 0; x < thresholds.size(); x++) {
				double var_info_gain = calculateInfoGain(input_data, y, thresholds[x], label_entropy);
				if (var_info_gain > max_info_gain) {
					best_split_var = y;
					best_threshold = thresholds[x];
				}
			}
		}
	}

	return make_tuple(best_split_var, best_threshold);
}

double decisionTree::calculateEntropy(vvd& input_data, int idx, double threshold)
{
	double entropy = 0;

	if (idx = input_data[0].size() - 1) {
		for (map<double, double>::iterator itr = labels.begin(); itr != labels.end(); ++itr) {
			double prob = itr->second / input_data.size();
			double label_entropy = -prob * log2(prob);
			entropy += label_entropy;
		}
	} else {
		if (is_discrete) {
			vector<int> var_val_counts(data_info[idx].size());
			vector<vector<int>> var_label_counts(data_info[idx].size(), vector<int>(labels.size()));
			for (int x = 0; x < input_data.size(); x++) {
				double val = input_data[x][idx];
				ptrdiff_t val_pos = distance(data_info[idx].begin(), find(data_info[idx].begin(), data_info[idx].end(), val));
				var_val_counts[val_pos]++;
				double label = input_data[x][input_data[x].size() - 1];
				ptrdiff_t label_pos = distance(labels.begin(), labels.find(label));
				var_label_counts[val_pos][label_pos]++;
			}
			// H(Y|X)
			for (int y = 0; y < data_info[idx].size(); y++) {
				// P(X = x_j)
				double var_prob = var_val_counts[y] / data_info.size();
				// calculating conditional entropy
				double cond_entropy;
				for (int lbl = 0; lbl < var_label_counts[y].size(); lbl++) {
					// P(Y = y_i | X = x_j)
					double lable_prob = var_label_counts[y][lbl] / var_val_counts[y];
					double lable_entropy = -lable_prob * log2(lable_prob);
					cond_entropy += lable_entropy;
				}
				double var_entropy = -var_prob * cond_entropy;
				entropy += var_entropy;
			}
		} else {
			vector<int> var_val_counts(2);
			vector<vector<int>> var_label_counts(2, vector<int>(labels.size()));
			for (int x = 0; x < input_data.size(); x++) {
				double val = input_data[x][idx];
				int val_pos;
				if (val < threshold) {
					val_pos = 0;
				} else {
					val_pos = 1;
				}
				var_val_counts[val_pos]++;
				double label = input_data[x][input_data[x].size() - 1];
				ptrdiff_t label_pos = distance(labels.begin(), labels.find(label));
				var_label_counts[val_pos][label_pos]++;
			}
			// H(Y|X)
			for (int y = 0; y < 2; y++) {
				// P(X = x_j)
				double var_prob = var_val_counts[y] / data_info.size();
				// calculating conditional entropy
				double cond_entropy;
				for (int lbl = 0; lbl < var_label_counts[y].size(); lbl++) {
					// P(Y = y_i | X = x_j)
					double lable_prob = var_label_counts[y][lbl] / var_val_counts[y];
					double lable_entropy = -lable_prob * log2(lable_prob);
					cond_entropy += lable_entropy;
				}
				double var_entropy = -var_prob * cond_entropy;
				entropy += var_entropy;
			}
		}
	}

	return entropy;
}

double decisionTree::calculateInfoGain(vvd& input_data, int idx, double threshold, double base_entropy)
{
	double info_gain = 0;

	double entropy = calculateEntropy(input_data, idx, threshold);
	info_gain = base_entropy - entropy;

	return info_gain;
}

vd decisionTree::getThresholds(vvd& input_data, int idx)
{
	vd thresholds;

	double split = input_data[0][input_data.size() - 1];
	for (int x = 1; x < input_data.size(); x++) {
		double next_candidate = input_data[x][input_data.size() - 1];
		if (next_candidate != split) {
			double threshold = (input_data[x][idx] + input_data[x - 1][idx]) / 2;
			thresholds.push_back(threshold);
			split = next_candidate;
		}
	}

	return thresholds;
}

vvd decisionTree::subsetDiscreteData(vvd& input_data, int var, double var_value)
{
	vvd subset_data;

	for (int x = 0; x < input_data.size(); x++) {
		if (input_data[x][var] == var_value) {
			vd data_copy = input_data[x];
			data_copy.erase(data_copy.begin() + var);
			subset_data.push_back(data_copy);
		}
	}

	return subset_data;
}

vector<vvd> decisionTree::subsetContinuousData(vvd& input_data, int var, double threshold)
{
	vector<vvd> subset_data;

	vvd left_subtree_data;
	vvd right_subtree_data;

	for (int x = 0; x < input_data.size(); x++) {
		double val = input_data[x][var];
		if (val < threshold) {
			left_subtree_data.push_back(input_data[x]);
		} else {
			right_subtree_data.push_back(input_data[x]);
		}
	}

	subset_data.push_back(left_subtree_data);
	subset_data.push_back(right_subtree_data);

	return subset_data;
}

node* decisionTree::pruneTree()
{

}

vvd decisionTree::getForestNodeData(vvd& input_data, int size)
{
	vvd random_data;
	vector<int> used_nums;

	while (random_data.size() < size) {
		int rand_idx = rand() % input_data.size();
		if (find(used_nums.begin(), used_nums.end(), rand_idx) == used_nums.end()) {
			random_data.push_back(input_data[rand_idx]);
			used_nums.push_back(rand_idx);
		}
	}

	return random_data;
}