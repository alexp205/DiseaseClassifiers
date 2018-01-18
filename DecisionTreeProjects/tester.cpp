#include "tester.h"

vvd train_data;
vvd test_data;
bool use_forest;
bool is_discrete;
bool is_classification;

int main(int argc, char* argv[])
{
	wcout << L"Extracting training and testing data from files\n";
	use_forest = argv[3];
	is_discrete = argv[4];
	is_classification = argv[5];
	auto datasets = parseData(argv[1], argv[2]);
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
		wcout << L"Building random forest...\n";
		randomForest forest(train_data, 1000, 100, is_discrete, is_classification);
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
		decisionTree tree(train_data, 5, is_discrete, is_classification, use_forest);
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

tuple<vvd, vvd> parseData(string train_data_csv, string test_data_csv)
{
	vvd extracted_train_data;
	vvd extracted_test_data;
	ifstream input_file;
	string data_string;
	string skip;

	// parse training data
	input_file.open(train_data_csv);
	if (!input_file) {
		wcerr << L"Error: Invalid path to training data file" << endl;
		exit(-1);
	}
	getline(input_file, skip);
	while (getline(input_file, data_string)) {
		extracted_train_data.push_back(parseDataLine(data_string));
	}
	input_file.close();
	input_file.clear();

	// parse testing data
	input_file.open(test_data_csv);
	if (!input_file) {
		wcerr << L"Error: Invalid path to testing data file" << endl;
		exit(-1);
	}
	getline(input_file, skip);
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