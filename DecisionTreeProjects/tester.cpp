#include "tester.h"

vvd train_data;
vvd test_data;
bool use_forest;
bool is_discrete;
bool is_classification;
int forest_size_default = 1000;
int bag_size_default;

/*
Args: 1. [string] the path to the training data csv file
	  2. [string] the path to the testing data csv file
	  3. [bool] determines whether or not the data is discrete or continuous
	  4. [bool] determines whether or not the task is classification or regression
				(i.e. discrete, finite or continous labels)
	  5. [bool] determines whether or not to use a random forest
	  6. [int] number of trees in random forest
	  7. [int] bagging size of tree data in random forest
*/
int main(int argc, char* argv[])
{
	wcout << L"Extracting training and testing data from files\n";
	use_forest = getBoolArg(argv[5]);
	is_discrete = getBoolArg(argv[3]);
	is_classification = getBoolArg(argv[4]);
	auto datasets = parseData(string(argv[1]), string(argv[2]));
	train_data = get<0>(datasets);
	test_data = get<1>(datasets);

	wcout << L"Training Data Sample:\n[";
	for (size_t x = 0; x < train_data[0].size() - 1; x++) {
		wcout << train_data[0][x] << ", ";
	}
	wcout << train_data[0][train_data[0].size() - 1] << "]\n";

	wcout << L"Testing Data Sample:\n[";
	for (size_t x = 0; x < test_data[0].size() - 1; x++) {
		wcout << test_data[0][x] << ", ";
	}
	wcout << test_data[0][test_data[0].size() - 1] << "]\n";

	if (use_forest) {
		int forest_size;
		int bag_size;
		if (argc < 7) {
			wcout << L"Random forest size and tree bag size not specified, using default values\n";
			forest_size = forest_size_default;
			int subset_size = 3;
			if (train_data.size() > 15) {
				subset_size = train_data.size() / 5;
			}
			bag_size = subset_size;
		} else if (argc == 7) {
			wcout << L"Random forest tree bag size not specified, using default value\n";
			forest_size = strtol(argv[6], NULL, 10);
			int subset_size = 3;
			if (train_data.size() > 15) {
				subset_size = train_data.size() / 5;
			}
			bag_size = subset_size;
		} else {
			forest_size = strtol(argv[6], NULL, 10);
			bag_size = strtol(argv[7], NULL, 10);
		}

		wcout << L"Building random forest...\n";
		randomForest forest(train_data, forest_size, bag_size, is_discrete, is_classification);
		forest.print(10);

		vd predictions = forest.predict(test_data);
		vd test_labels;
		for (size_t x = 0; x < test_data.size(); x++) {
			test_labels.push_back(test_data[x][test_data[x].size() - 1]);
		}
		wstring filename = L"random_forest_output.txt";
		double accuracy = forest.getStatsInfo(test_labels, predictions, filename);
	} else {
		wcout << L"Building decision tree...\n";
		decisionTree tree(train_data, (int) sqrt(train_data.size()), is_discrete, is_classification, use_forest);
		tree.print();

		vd predictions = tree.predict(test_data);
		vd test_labels;
		for (size_t x = 0; x < test_data.size(); x++) {
			test_labels.push_back(test_data[x][test_data[x].size() - 1]);
		}
		wstring filename = L"decision_tree_output.txt";
		double accuracy = tree.getStatsInfo(test_labels, predictions, filename);
	}

	return 0;
}

bool getBoolArg(char* arg)
{
	if (string(arg) == "true" || string(arg) == "True") {
		return true;
	} else if (string(arg) == "false" || string(arg) == "False") {
		return false;
	} else {
		wcout << L"Error processing boolean arguments, please check the program call" << endl;
		exit(-1);
	}
}

tuple<vvd, vvd> parseData(string train_data_csv, string test_data_csv)
{
	vvd extracted_train_data;
	vvd extracted_test_data;
	ifstream input_file;
	string data_string;

	// parse training data
	input_file.open(train_data_csv);
	input_file.ignore(3, EOF);
	if (!input_file) {
		wcerr << L"Error: Invalid path to training data file" << endl;
		exit(-1);
	}
	while (getline(input_file, data_string)) {
		extracted_train_data.push_back(parseDataLine(data_string));
	}
	input_file.close();
	input_file.clear();

	// parse testing data
	input_file.open(test_data_csv);
	input_file.ignore(3, EOF);
	if (!input_file) {
		wcerr << L"Error: Invalid path to testing data file" << endl;
		exit(-1);
	}
	while (getline(input_file, data_string)) {
		extracted_test_data.push_back(parseDataLine(data_string));
	}

	return make_tuple(extracted_train_data, extracted_test_data);
}

vd parseDataLine(string data)
{
	vd parsed_data;

	while ((data.length() != 0) && (data.find(",") != string::npos)) {
		size_t pos = data.find_first_of(",");
		string var_val_str = data.substr(0, pos);
		double var_val = atof(var_val_str.c_str());
		parsed_data.push_back(var_val);
		data.erase(0, pos + 1);
	}
	double var_val = atof(data.c_str());
	parsed_data.push_back(var_val);

	return parsed_data;
}