#include "RandomForest.h"

// Constructor
/*

*/
randomForest::randomForest(vvd& dataset, int forest_size, int bag_size, bool discrete, bool classification)
{
	for (int x = 0; x < forest_size; x++) {
		vvd bootstrap_data = getBootstrapSample(dataset, bag_size);
		decisionTree forest_tree(bootstrap_data, (int)sqrt(dataset.size()), discrete, classification, true);
		forest.push_back(forest_tree);
	}
}

// Private (Internal) Functions
vvd randomForest::getBootstrapSample(vvd& input_data, int size)
{
	vvd bootstrap_data;

	if (input_data.size() < (size_t) size) {
		bootstrap_data = input_data;
	} else {
		while (bootstrap_data.size() < (size_t) size) {
			int rand_idx = rand() % input_data.size();
			bootstrap_data.push_back(input_data[rand_idx]);
		}
	}

	return bootstrap_data;
}

void randomForest::printForestSample(int tree_idx)
{
	forest[tree_idx].print();
	return;
}

double randomForest::processStats(vd& test_labels, vd& test_predictions, wstring filename)
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
double randomForest::predict(vd& data)
{
	double label;
	map<double,int> predictions;
	
	for (size_t x = 0; x < forest.size(); x++) {
		double prediction = forest[x].predict(data);
		if (predictions.count(prediction) == 0) {
			predictions[prediction] = 1;
		} else {
			predictions[prediction]++;
		}
	}

	int max_count = -1;
	// NOTE: ties are broken "randomly" (i.e. first visited is chosen)
	for (map<double,int>::iterator itr = predictions.begin(); itr != predictions.end(); ++itr) {
		int label_count = itr->second;
		if (label_count > max_count) label = itr->first;
	}

	return label;
}

vd randomForest::predict(vvd& dataset)
{
	vd predicted_labels;

	for (size_t x = 0; x < dataset.size(); x++) {
		predicted_labels.push_back(predict(dataset[x]));
	}

	return predicted_labels;
}

void randomForest::print(int sample_size)
{
	wcout << L"Taking Sample of Size " << sample_size << " from the Forest:\n";
	wcout << L"----------------------------------------------------------------" << endl;
	int step_size = (int) floor(forest.size() / (double) sample_size);
	for (int x = 0; x < sample_size; x += step_size) {
		wcout << L"Random Forest: Tree " << x << "\n";
		printForestSample(x);
		wcout << endl;
	}
}

double randomForest::getStatsInfo(vd& test_labels, vd& test_predictions, wstring filename)
{
	wcout << L"Statistics:\n";
	double accuracy = processStats(test_labels, test_predictions, filename);
	wcout << L"NOTE: testing results recorded at " << filename << "\n";
	wcout << L"Model Accuracy on Test Data: " << accuracy << endl;

	return accuracy;
}