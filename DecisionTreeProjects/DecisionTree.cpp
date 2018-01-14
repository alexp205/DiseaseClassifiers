#include "DecisionTree.h"

// Constructor
decisionTree::decisionTree(vvd& train_dataset, node* root, bool is_discrete, bool is_classification)
{
	data_info = getDatasetInfo(train_dataset);
	labels = getLabelInfo(train_dataset);

	root_node = buildTree(train_dataset, root, is_discrete, is_classification);
	root_node = pruneTree();
}

// Destructor
decisionTree::~decisionTree()
{

}

node* decisionTree::buildTree(vvd& input_data, node* node_ref, bool discrete, bool classification)
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
	// choose how to split based on if the data is discrete or continuous
	// for reference:
	//   - discrete - split on all possible values for that variable
	//   - continuous - find the best threshold for that variable and make a 
	//                  binary split
	if (discrete) {
		auto split_info = bestSplitVar(input_data, discrete);
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
			vvd data_subset = discreteSubsetData(input_data, split_var, data_info[split_var][x]);
			node_ref->children.push_back(buildTree(data_subset, child, discrete));
		}
		return node_ref;
	}
	else {
		auto split_info = bestSplitVar(input_data, discrete);
		int split_var = get<0>(split_info);
		double threshold = get<1>(split_info);
		if (split_var == -1) {
			wcout << L"ERROR: no split variable detected, please check for errors" << endl;
			return NULL;
		}
		node_ref->split_label = split_var;
		vector<vvd> data_subsets = continuousSubsetData(input_data, split_var, threshold);
		node* left_child = new node;
		left_child->split_label = split_var;
		left_child->is_leaf = false;
		node* right_child = new node;
		right_child->split_label = split_var;
		right_child->is_leaf = false;
		node_ref->children.push_back(buildTree(data_subsets[0], left_child, discrete));
		node_ref->children.push_back(buildTree(data_subsets[1], right_child, discrete));
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

vector<int> decisionTree::getLabelInfo(vvd& input_data)
{
	vector<int> label_info;

	for (int x = 0; x < input_data.size(); x++) {
		int label = input_data[x][input_data[x].size() - 1];
		if (find(label_info.begin(), label_info.end(), label) == label_info.end()) {
			label_info.push_back(label);
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

	int sample_label = input_data[0][input_data[0].size() - 1];
	for (int x = 1; x < input_data.size(); x++) {
		int test_label = input_data[x][input_data[x].size() - 1];
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

tuple<int,double> decisionTree::bestSplitVar(vvd& input_data, bool discrete)
{
	int best_split_var = -1;
	double best_threshold = -1;

	double label_entropy = calculateEntropy(input_data, input_data.size() - 1); //H(Y)

	double max_info_gain = -numeric_limits<double>::infinity();
	if (discrete) {
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

double decisionTree::calculateEntropy(vvd& input_data, int idx)
{

}

double decisionTree::calculateInfoGain(vvd& input_data, int idx, double threshold, double base_entropy)
{

}

vd decisionTree::getThresholds(vvd& input_data, int idx)
{

}

vvd decisionTree::discreteSubsetData(vvd& input_data, int var, double var_value)
{

}

vector<vvd> decisionTree::continuousSubsetData(vvd& input_data, int var, double threshold)
{

}