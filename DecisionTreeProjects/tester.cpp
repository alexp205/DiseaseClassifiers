#include "tester.h"

vvd train_data;
vvd test_data;
vd test_labels;
bool use_forest;
bool is_discrete;
bool is_classification;
int forest_size_default = 1000;
int bag_size_default;

/*
* Args: 1. [string] the path to the training data csv file
*       2. [string] the path to the testing data csv file
*       3. [string] the path to the testing data labels csv file
*       4. [bool] determines whether or not the data is discrete [T] or continuous [F]
*       5. [bool] determines whether or not the task is classification [T] or regression [F]
*                 (i.e. discrete, finite or continous labels)
*	    6. [bool] determines whether or not to use a random forest
*       7. [int] number of trees in random forest
*       8. [int] <optional> amount of bootstrap data per forest tree
*
* Sample Args:
*  - Discrete
*   "C:\Users\ap\Documents\Visual Studio 2017\Projects\DecisionTreeProjects\DecisionTreeProjects\data\TEST-discrete_train_data.csv" "C:\Users\ap\Documents\Visual Studio 2017\Projects\DecisionTreeProjects\DecisionTreeProjects\data\TEST-discrete_test_data.csv" "C:\Users\ap\Documents\Visual Studio 2017\Projects\DecisionTreeProjects\DecisionTreeProjects\data\TEST-discrete_test_labels.csv" true true false
*   "C:\Users\ap\Documents\Visual Studio 2017\Projects\DecisionTreeProjects\DecisionTreeProjects\data\TEST-discrete_train_data.csv" "C:\Users\ap\Documents\Visual Studio 2017\Projects\DecisionTreeProjects\DecisionTreeProjects\data\TEST-discrete_test_data.csv" "C:\Users\ap\Documents\Visual Studio 2017\Projects\DecisionTreeProjects\DecisionTreeProjects\data\TEST-discrete_test_labels.csv" true true true
*   "C:\Users\ap\Documents\Visual Studio 2017\Projects\DecisionTreeProjects\DecisionTreeProjects\data\TEST-discrete_train_data.csv" "C:\Users\ap\Documents\Visual Studio 2017\Projects\DecisionTreeProjects\DecisionTreeProjects\data\TEST-discrete_test_data.csv" "C:\Users\ap\Documents\Visual Studio 2017\Projects\DecisionTreeProjects\DecisionTreeProjects\data\TEST-discrete_test_labels.csv" true true true 200
*
*  - Continuous
*   "C:\Users\ap\Documents\Visual Studio 2017\Projects\DecisionTreeProjects\DecisionTreeProjects\data\TEST-continuous_train_data.csv" "C:\Users\ap\Documents\Visual Studio 2017\Projects\DecisionTreeProjects\DecisionTreeProjects\data\TEST-continuous_test_data.csv" "C:\Users\ap\Documents\Visual Studio 2017\Projects\DecisionTreeProjects\DecisionTreeProjects\data\TEST-continuous_test_labels.csv" false true false
*   "C:\Users\ap\Documents\Visual Studio 2017\Projects\DecisionTreeProjects\DecisionTreeProjects\data\TEST-continuous_train_data.csv" "C:\Users\ap\Documents\Visual Studio 2017\Projects\DecisionTreeProjects\DecisionTreeProjects\data\TEST-continuous_test_data.csv" "C:\Users\ap\Documents\Visual Studio 2017\Projects\DecisionTreeProjects\DecisionTreeProjects\data\TEST-continuous_test_labels.csv" false true true 200
*/
int main(int argc, char* argv[])
{
    wcout << L"Extracting training and testing data from files\n";
    use_forest = getBoolArg(argv[6]);
    is_discrete = getBoolArg(argv[4]);
    is_classification = getBoolArg(argv[5]);
    auto datasets = parseData(string(argv[1]), string(argv[2]));
    train_data = get<0>(datasets);
    test_data = get<1>(datasets);
    test_labels = parseData(string(argv[3]));

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
        if (train_data.size() < 500) {
            wcout << L"WARNING: amount of input data is smaller than the minimum recommended (500), please consider providing at least that amount of data before attempting to run random forests\n";
        }

	    int forest_size;
	    int bag_size = train_data.size();
	    if (argc < 8) {
		    wcout << L"Random forest size and bootstrap bag size not specified, using default values\n";
		    forest_size = forest_size_default;
	    }
	    else if (argc == 8) {
		    wcout << L"Random forest bootstrap bag size not specified, using default value\n";
		    forest_size = strtol(argv[7], NULL, 10);
            if (forest_size < 500) {
                wcout << L"WARNING: small forest size detected, consider using more than 500 trees\n";
            }
	    }
	    else {
		    forest_size = strtol(argv[7], NULL, 10);
            if (forest_size < 500) {
                wcout << L"WARNING: small forest size detected, consider using more than 500 trees\n";
            }
		    bag_size = strtol(argv[8], NULL, 10);
            if (bag_size > train_data.size()) {
                wcout << L"Specified bootstrap bag size is larger than size of input data\n";
                exit(-1);
            }
            wcout << L"WARNING: small bootstrap bag sizes increase performance BUT can drastically worsen prediction accuracy\n";
	    }

	    wcout << L"Building random forest...\n";
	    randomForest forest(train_data, forest_size, bag_size, is_discrete, is_classification);
	    forest.print(3);

	    vd predictions = forest.predict(test_data);
	    wstring filename = L"random_forest_output.txt";
	    double accuracy = forest.getStatsInfo(test_labels, predictions, filename);
    }
    else {
	    wcout << L"Building decision tree...\n";
	    decisionTree tree(train_data, (int)sqrt(train_data.size()), is_discrete, is_classification, use_forest);
	    tree.print();

	    vd predictions = tree.predict(test_data);
	    wstring filename = L"decision_tree_output.txt";
	    double accuracy = tree.getStatsInfo(test_labels, predictions, filename);
    }

    cin.get();
    return 0;
}

bool getBoolArg(char* arg)
{
	if (string(arg) == "true" || string(arg) == "True") {
		return true;
	}
	else if (string(arg) == "false" || string(arg) == "False") {
		return false;
	}
	else {
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
	if (!input_file) {
		wcerr << L"Error: Invalid path to testing data file" << endl;
		exit(-1);
	}
	while (getline(input_file, data_string)) {
		extracted_test_data.push_back(parseDataLine(data_string));
	}

	return make_tuple(extracted_train_data, extracted_test_data);
}

vd parseData(string test_labels_csv)
{
    vd extracted_test_labels;
    ifstream input_file;
    string data_string;

    input_file.open(test_labels_csv);
    if (!input_file) {
        wcerr << L"Error: Invalid path to testing data file" << endl;
        exit(-1);
    }
    while (getline(input_file, data_string)) {
        extracted_test_labels.push_back(atof(data_string.c_str()));
    }

    return extracted_test_labels;
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