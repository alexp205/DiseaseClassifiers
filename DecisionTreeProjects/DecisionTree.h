#pragma once

#ifndef DECISION_TREE_H_
#define DECISION_TREE_H_

#include <string>
#include <vector>
#include <algorithm>
#include <tuple>
#include <iostream>

using namespace std;

typedef vector<double> vd;
typedef vector<vd> vvd;

struct node
{
	bool is_leaf;
	int label; // if leaf node, contains label, else...
	int split_label; // contains attribute split label
	vector<node*> children;
};

class decisionTree
{
	vvd data_info; // list (in order of input format) of all variables and possible values
	vector<int> labels;
	node* root_node;

	vvd getDatasetInfo(vvd&);
	vector<int> getLabelInfo(vvd&);
	node* buildTree(vvd&, node*, bool, bool);
	bool checkLeaf(vvd&);
	tuple<int, double> bestSplitVar(vvd&, bool);
	double calculateEntropy(vvd&, int);
	double calculateInfoGain(vvd&, int, double, double);
	vd getThresholds(vvd&, int);
	vvd discreteSubsetData(vvd&, int, double);
	vector<vvd> continuousSubsetData(vvd&, int, double);

public:
	decisionTree(vvd&, node*, bool, bool);
	~decisionTree();
};

#endif