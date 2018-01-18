#include "tester.h"

vvd train_data;
vvd test_data;

int main(int argc, char* argv[])
{
	wcout << L"Extracting training and testing data from files\n";
	auto datasets = parseData(argv[1], argv[2]);
	train_data = get<0>(datasets);
	test_data = get<1>(datasets);

	wcout << L"Training Data:\n";

	wcout << L"Testing Data:\n";



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