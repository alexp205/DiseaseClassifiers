#include "DecisionTree.h"

// Constructor
/*
*  Constructor for a decision tree object.
*
*  There are a few key components/member "structures" to note:
*    - data_info:
*      Scans input data and notes the unique values in each column of input data.
*      so, ---           ---
*          | [ a b ... c ] | <--- column 1 unique values
*          | [ d e ... f ] | <--- column 2 unique values
*          | etc.          |
*          .               .
*          .               .
*          .               .
*          |               |
*          ---           ---
*
*    - label_info:
*      Dictionary structure that maps each unique label the occurs w/ the number of occurrences.
*      so, ( [ x --- y ] ) <--- label 1
*          ( [ w --- z ] ) <--- label 2
*          ( etc.        )
*          .             .
*          .             .
*          .             .
*          (             )
*
*/
decisionTree::decisionTree(vvd& train_dataset, int data_cutoff, bool discrete, bool classification, bool forest)
{
	is_discrete = discrete;
	is_classification = classification;
	is_in_forest = forest;
	min_data_size = data_cutoff;
    if (is_discrete) data_info = getDatasetInfo(train_dataset);
	labels = getLabelInfo(train_dataset);

	vector<int> original_indices;
	for (size_t y = 0; y < train_dataset[0].size() - 1; y++) {
		original_indices.push_back(y);
	}
	node root;
	root.frequency = train_dataset.size();
	root_node = buildTree(train_dataset, original_indices, root);
	/*
	if (!is_in_forest) {
		root_node = pruneTree();
	}
	*/
}

// Private (Internal) Functions
node decisionTree::buildTree(vvd& input_data, vector<int> indices, node node_ref)
{
	// if there is no input data, something went wrong
	if (input_data.empty()) {
		wcout << L"ERROR: empty data detected, please check for errors" << endl;
		exit(-1);
	}
	
    vvd bkp_data_info;
    if (is_discrete) {
        bkp_data_info = data_info;
        data_info = getDatasetInfo(input_data);
    }
	map<double,int> bkp_labels = labels;
	labels = getLabelInfo(input_data);
	// check if the tree is at a leaf
	auto is_leaf = checkLeaf(input_data);
	if (get<0>(is_leaf)) {
		node_ref.is_leaf = true;
		node_ref.label = get<1>(is_leaf);
		if (is_discrete) data_info = bkp_data_info;
		labels = bkp_labels;
		return node_ref;
	}
    if (input_data.size() < (size_t)min_data_size) {
        node_ref.is_leaf = true;
        node_ref.label = getCutoffLeafLabel();
        if (is_discrete) data_info = bkp_data_info;
        labels = bkp_labels;
        return node_ref;
    }

	// choose how to split based on if the tree is part of a forest 
    // OR if the data is discrete or continuous
	// for reference:
	//   - discrete - split on all possible values for that variable
	//   - continuous - find the best threshold for that variable and make a 
	//                  binary split
	// also, if the tree is part of a forest, split on a random subset of 
	// the data
	vvd split_data;
    vvd ref_data_info;
    map<double, int> ref_labels;
	if (is_in_forest) {
		int num_data = (int) ceil(sqrt(input_data.size()));
		split_data = getForestNodeData(input_data, num_data);
		// the following is necessary because of the references to data_info and 
		// labels in bestSplitVar (and all reliant functions), this system can 
		// almost certainly be improved and eventually maybe possibly will
        if (is_discrete) {
            ref_data_info = data_info;
            data_info = getDatasetInfo(split_data);
        }
        ref_labels = labels;
		labels = getLabelInfo(split_data);
	} else {
		split_data = input_data;
	}
    
    auto split_info = bestSplitVar(split_data);
    int split_var = get<0>(split_info);
    if (is_in_forest) {
        if (is_discrete) data_info = ref_data_info;
        labels = ref_labels;
    }
	if (is_discrete) {
		if (split_var == -1) {
			wcout << L"ERROR: no split variable detected, please check for errors" << endl;
			exit(-1);
		}
		node_ref.split_var = indices[split_var];
		for (size_t x = 0; x < data_info[split_var].size(); x++) {
			node child;
			child.split_var = indices[split_var];
			child.split_val = data_info[split_var][x];
			child.is_leaf = false;
			vvd data_subset = subsetDiscreteData(input_data, split_var, data_info[split_var][x]);
			child.frequency = data_subset.size();
			vector<int> subset_indices = indices;
			subset_indices.erase(subset_indices.begin() + split_var);
			node_ref.children.push_back(buildTree(data_subset, subset_indices, child));
		}
		data_info = bkp_data_info;
		labels = bkp_labels;
		return node_ref;
	} else {
		double split_threshold = get<1>(split_info);
        // check for and handle rare exact-same data case
        if ((-1 == split_var) && (-1 == split_threshold)) {
            node_ref.is_leaf = true;
            node_ref.label = input_data[0][input_data[0].size() - 1];
            labels = bkp_labels;
            return node_ref;
        } else {
            if (split_var == -1) {
                wcout << L"ERROR: no split variable detected, please check for errors" << endl;
                exit(-1);
            }
            if (split_threshold == -1) {
                wcout << L"ERROR: no split threshold detected in continuous mode, please check for errors" << endl;
                exit(-1);
            }
        }
		node_ref.split_var = indices[split_var];
		node_ref.threshold = split_threshold;
		node left_child;
		left_child.split_var = indices[split_var];
		left_child.is_leaf = false;
		node right_child;
		right_child.split_var = indices[split_var];
		right_child.is_leaf = false;
		vector<vvd> data_subsets = subsetContinuousData(input_data, split_var, split_threshold);
		left_child.frequency = data_subsets[0].size();
		right_child.frequency = data_subsets[1].size();
		node_ref.children.push_back(buildTree(data_subsets[0], indices, left_child));
		node_ref.children.push_back(buildTree(data_subsets[1], indices, right_child));
		labels = bkp_labels;
		return node_ref;
	}
}

vvd decisionTree::getDatasetInfo(vvd& input_data)
{
	vvd data_info;

	for (size_t y = 0; y < input_data[0].size() - 1; y++) {
		vd var_info;
		for (size_t x = 0; x < input_data.size(); x++) {
			double val = input_data[x][y];
			if (find(var_info.begin(), var_info.end(), val) == var_info.end()) {
				var_info.push_back(val);
			}
		}
		data_info.push_back(var_info);
	}

	return data_info;
}

map<double,int> decisionTree::getLabelInfo(vvd& input_data)
{
	map<double,int> label_info;

	for (size_t x = 0; x < input_data.size(); x++) {
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
* This function checks two things in order to determine if the node is a leaf:
*  1. if all of the remaining data has the same label
*  2. if all of the remaining data has the same values for every feature
*/
tuple<bool,double> decisionTree::checkLeaf(vvd& input_data)
{
	if (input_data.size() == 1) return make_tuple(true, input_data[0][input_data[0].size() - 1]);

	if (labels.size() == 1) {
		return make_tuple(true, input_data[0][input_data[0].size() - 1]);
	} else {
		for (size_t y = 0; y < input_data[0].size() - 1; y++) {
			double sample_var_val = input_data[0][y];
			for (size_t x = 1; x < input_data.size(); x++) {
				double test_var_val = input_data[x][y];
				if (test_var_val != sample_var_val) return make_tuple(false, -1);
			}
		}

		return make_tuple(true, getCutoffLeafLabel());
	}
}

tuple<int,double> decisionTree::bestSplitVar(vvd& input_data)
{
	int best_split_var = -1;
	double best_threshold = -1;

	double label_entropy = calculateEntropy(input_data, input_data[0].size() - 1, -1); // H(Y)

	double max_info_gain = -numeric_limits<double>::infinity();
	if (is_discrete) {
		for (size_t y = 0; y < input_data[0].size() - 1; y++) {
			double var_info_gain = calculateInfoGain(input_data, y, -1, label_entropy);
			if (var_info_gain > max_info_gain) {
				best_split_var = y;
				max_info_gain = var_info_gain;
			}
		}
	} else {
		for (size_t y = 0; y < input_data[0].size() - 1; y++) {
			vd thresholds = getThresholds(input_data, y);
			for (size_t x = 0; x < thresholds.size(); x++) {
				double var_info_gain = calculateInfoGain(input_data, y, thresholds[x], label_entropy);
				if (var_info_gain > max_info_gain) {
					best_split_var = y;
					best_threshold = thresholds[x];
					max_info_gain = var_info_gain;
				}
			}
		}
	}

	return make_tuple(best_split_var, best_threshold);
}

/*
* NOTE: future improvements could include restructuring the procedure to use a general (i.e. base) 
*       case entropy calculation function then have other calculations (e.g. conditional entropy) 
*       call that base case function
*/
double decisionTree::calculateEntropy(vvd& input_data, int idx, double threshold)
{
	double entropy = 0;

	if (idx == input_data[0].size() - 1) {
		for (map<double,int>::iterator itr = labels.begin(); itr != labels.end(); ++itr) {
			double prob = (double) itr->second / input_data.size();
			double label_entropy = 0;
			if (prob != 0) {
				label_entropy = -prob * log2(prob);
			}
			entropy += label_entropy;
		}
	} else {
		if (is_discrete) {
            // some helpful information regarding the variables var_val_counts and var_label_counts:
            //   - var_val_counts:
            //     Tracks the number of occurrences of every possible discrete value corresponding to 
            //     the feature at idx. References data_info which, in turn, stores the relevant 
            //     possible discrete values.
            //
            //   - var_label_counts:
            //     Tracks the number of occurrences of each label per every possible discrete value 
            //     corresponding to the feature at idx. So, for example, the feature at idx could 
            //     have the following possible values with the following frequencies of labels:
            //       - 0 -> label 1: 3
            //           -> label 2: 0
            //           -> label 3: 1
            //       - 3 -> label 1: 1
            //           -> label 2: 0
            //           -> label 3: 0
            //       - 4 -> label 1: 0
            //           -> label 2: 0
            //           -> label 3: 4
			vector<int> var_val_counts(data_info[idx].size());
			vector<vector<int>> var_label_counts(data_info[idx].size(), vector<int>(labels.size()));
			for (size_t x = 0; x < input_data.size(); x++) {
				double val = input_data[x][idx];
				ptrdiff_t val_pos = distance(data_info[idx].begin(), find(data_info[idx].begin(), data_info[idx].end(), val));
				var_val_counts[val_pos]++;
				double label = input_data[x][input_data[x].size() - 1];
				ptrdiff_t label_pos = distance(labels.begin(), labels.find(label));
				var_label_counts[val_pos][label_pos]++;
			}
			// H(Y|X)
			for (size_t y = 0; y < data_info[idx].size(); y++) {
				// P(X = x_j)
				double var_prob = (double) var_val_counts[y] / input_data.size();
				// calculating conditional entropy
				double cond_entropy = 0;
				for (size_t lbl = 0; lbl < var_label_counts[y].size(); lbl++) {
					// P(Y = y_i | X = x_j)
                    double label_prob = 0;
                    if (var_val_counts[y] != 0) {
                        label_prob = (double) var_label_counts[y][lbl] / var_val_counts[y];
                    }
					double label_entropy = 0;
					if (label_prob != 0) {
						label_entropy = label_prob * log2(label_prob);
					}
					cond_entropy += label_entropy;
				}
				double var_entropy = -var_prob * cond_entropy;
				entropy += var_entropy;
			}
		} else {
            // the variables var_val_counts and var_label counts work similarly to the discrete 
            // case except they only consider the variables greater than or equal to and less 
            // than the given threshold
			vector<int> var_val_counts(2);
			vector<vector<int>> var_label_counts(2, vector<int>(labels.size()));
			for (size_t x = 0; x < input_data.size(); x++) {
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
				double var_prob = (double) var_val_counts[y] / input_data.size();
				// calculating conditional entropy
				double cond_entropy = 0;
				for (size_t lbl = 0; lbl < var_label_counts[y].size(); lbl++) {
					// P(Y = y_i | X = x_j)
					double label_prob = 0;
					if (var_val_counts[y] != 0) {
						label_prob = (double) var_label_counts[y][lbl] / var_val_counts[y];
					}
					double label_entropy = 0;
					if (label_prob != 0) {
						label_entropy = label_prob * log2(label_prob);
					}
					cond_entropy += label_entropy;
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

    vd candidates;
    for (size_t i = 0; i < input_data.size(); i++) {
        candidates.push_back(input_data[i][idx]);
    }
    sort(candidates.begin(), candidates.end());

    double split = candidates[0];
    for (size_t x = 1; x < candidates.size(); x++) {
		double next_candidate = candidates[x];
		if (next_candidate != split) {
			double threshold = (next_candidate + split) / (double) 2;
			thresholds.push_back(threshold);
			split = next_candidate;
		}
	}

	return thresholds;
}

vvd decisionTree::subsetDiscreteData(vvd& input_data, int var, double var_value)
{
	vvd subset_data;

	for (size_t x = 0; x < input_data.size(); x++) {
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

	for (size_t x = 0; x < input_data.size(); x++) {
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

/*
* Currently not implemented. Prospective improvement involves using misclassification 
* for the error while iterating over subtrees to detect the best accuracy improvements/ 
* error reductions.
*
* In place, if the tree is not part of a random forest, minimum dataset sizes are used 
* to limit the size of the tree.
*/
/*
node* decisionTree::pruneTree()
{

}
*/

double decisionTree::getCutoffLeafLabel()
{
	double best_label;

	int best_count = -1;
	for (map<double,int>::iterator itr = labels.begin(); itr != labels.end(); ++itr) {
		int count = itr->second;
		if (count > best_count) {
			best_label = itr->first;
			best_count = count;
		}
	}

	return best_label;
}

vvd decisionTree::getForestNodeData(vvd& input_data, int size)
{
	vvd random_data;
	vector<int> used_nums;

	while (random_data.size() < (size_t) size) {
		int rand_idx = rand() % input_data.size();
		if (find(used_nums.begin(), used_nums.end(), rand_idx) == used_nums.end()) {
			random_data.push_back(input_data[rand_idx]);
			used_nums.push_back(rand_idx);
		}
	}

	return random_data;
}

double decisionTree::predict(vd& data, node current_node)
{
	if (current_node.is_leaf) return current_node.label;
	if (is_discrete) {
		double data_val = data[current_node.split_var];
		int child_idx;
		int max_freq = -1;
		for (size_t x = 0; x < current_node.children.size(); x++) {
			double val = current_node.children[x].split_val;
			if (val == data_val) return predict(data, current_node.children[x]);
			int freq = current_node.children[x].frequency;
			if (freq > max_freq) {
				max_freq = freq;
				child_idx = x;
			}
		}
		// if the data contains a value never seen before, throw it into the node 
		// with the highest frequency
		return predict(data, current_node.children[child_idx]);
	}
	else {
		if (data[current_node.split_var] < current_node.threshold) return predict(data, current_node.children[0]);
		return predict(data, current_node.children[1]);
	}
}

void decisionTree::printTree(node node_ref, int depth)
{
	if (node_ref.is_leaf) {
		printSpacing(depth, true);
		wcout << L"Split Variable: " << node_ref.split_var << "\n";
        if (is_discrete) {
            printSpacing(depth, false);
            wcout << L"Split Value: " << node_ref.split_val << "\n";
        }
        else {
            printSpacing(depth, false);
            wcout << L"Threshold: " << node_ref.threshold << "\n";
        }
		printSpacing(depth, false);
		wcout << L"Predicted Label: " << node_ref.label << endl;
		return;
	} else {
		printSpacing(depth, true);
		wcout << L"Split Variable: " << node_ref.split_var << "\n";
		if (is_discrete) {
			printSpacing(depth, false);
			wcout << L"Split Value: " << node_ref.split_val << "\n";
		}
		else {
			printSpacing(depth, false);
			wcout << L"Threshold: " << node_ref.threshold << "\n";
		}
		printSpacing(depth, false);
		wcout << L"Subtree Size: " << node_ref.frequency << "\n";
		depth++;
		for (size_t x = 0; x < node_ref.children.size(); x++) {
			printTree(node_ref.children[x], depth);
		}
		return;
	}
}

void decisionTree::printSpacing(int depth, bool is_top)
{
	if (depth > 0) {
		for (int layer = 1; layer < depth; layer++) {
			wcout << L"     ";
		}
		if (is_top) {
			wcout << L"---> ";
		}
		else {
			wcout << L"     ";
		}
	}
}

double decisionTree::processStats(vd& test_labels, vd& test_predictions, wstring filename)
{
	ofstream output_file;
	output_file.open(filename);
	int correct = 0;

	output_file << setw(2) << "#" << setw(10) << "True Label" << setw(30) << right << "Predicted Label\n";
	output_file << "----------------------------------------------------------------" << endl;
	for (size_t x = 0; x < test_labels.size(); x++) {
		output_file << setw(2) << x + 1 << setw(10) << test_labels[x];
		if (test_labels[x] == test_predictions[x]) {
			correct++;
			output_file << "            ";
		}
		else {
			output_file << "  ********  ";
		}
		output_file << test_predictions[x] << endl;
	}

	output_file << "----------------------------------------------------------------" << endl;
	output_file << "Size of the test dataset: " << test_labels.size() << "\n";
	output_file << "Number of correctly predicted labels: " << correct << endl;
	output_file.close();

	return (double) correct / test_labels.size();
}

// Public Functions
/*
* Returns: - [double] the predicted label for the input data
*/
double decisionTree::predict(vd& data)
{
	return predict(data, root_node);
}

/*
* Overloaded version of predict that can handle sets of data.
*
* Returns: - [vd] the list of predicted labels for each data point in the dataset
*/
vd decisionTree::predict(vvd& dataset)
{
	vd predicted_labels;

	for (size_t x = 0; x < dataset.size(); x++) {
		predicted_labels.push_back(predict(dataset[x]));
	}

	return predicted_labels;
}

void decisionTree::print()
{
	wcout << L"Tree Structure:\n";
	wcout << L"----------------------------------------------------------------\n";
	printTree(root_node, 0);
}

double decisionTree::getStatsInfo(vd& test_labels, vd& test_predictions, wstring filename)
{
	wcout << L"Statistics:\n";
	wcout << L"----------------------------------------------------------------\n";
	double accuracy = processStats(test_labels, test_predictions, filename);
	wcout << L"NOTE: testing results recorded at " << filename << "\n";
	wcout << L"\nModel Accuracy on Test Data: " << accuracy << endl;

	return accuracy;
}