#include "DecisionForest.h"

// Constructor
/*

*/
randomForest::randomForest(vvd& dataset, int forest_size, int bag_size, bool discrete, bool classification)
{
	for (int x = 0; x < forest_size; x++) {
		vvd bootstrap_data = getBootstrapSample(dataset, bag_size);
		decisionTree forest_tree(bootstrap_data, -1, discrete, classification, true);
		forest.push_back(forest_tree);
	}
}

// Destructor
randomForest::~randomForest()
{

}

// Private (Internal) Functions
vvd randomForest::getBootstrapSample(vvd& input_data, int size)
{
	vvd bootstrap_data;

	while (bootstrap_data.size() < size) {
		int rand_idx = rand() % input_data.size();
		bootstrap_data.push_back(input_data[rand_idx]);
	}

	return bootstrap_data;
}


// Public Functions
double randomForest::predict(vd data)
{
	double label;
	map<double,int> predictions;
	
	for (int x = 0; x < forest.size(); x++) {
		double prediction = forest[x].predict(data);
		if (predictions.count(prediction) == 0) {
			predictions[prediction] = 1;
		} else {
			predictions[prediction]++;
		}
	}

	int max_count = -1;
	// NOTE: ties are broken "randomly"
	for (map<double,int>::iterator itr = predictions.begin(); itr != predictions.end(); ++itr) {
		int label_count = itr->second;
		if (label_count > max_count) label = itr->first;
	}

	return label;
}