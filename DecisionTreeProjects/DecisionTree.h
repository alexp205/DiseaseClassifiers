#pragma once

#ifndef DECISION_TREE_H_
#define DECISION_TREE_H_

#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <tuple>
#include <iostream>

using namespace std;

typedef vector<double> vd;
typedef vector<vd> vvd;

struct node
{
	bool is_leaf;
	double label; // if leaf node, contains label, else...
	double split_label; // contains attribute split label
	vector<node*> children;
};

class decisionTree
{
	vvd data_info; // list (in order of input format) of all variables and possible values
	map<double, double> labels;
	node* root_node;
	bool is_discrete;
	bool is_classification;
	bool is_in_forest;
	vvd random_data; // this holds the random data collected for each node in trees in a random forest

	vvd getDatasetInfo(vvd&);
	map<double, double> getLabelInfo(vvd&);
	node* buildTree(vvd&, node*);
	bool checkLeaf(vvd&);
	tuple<int, double> bestSplitVar(vvd&);
	double calculateEntropy(vvd&, int, double);
	double calculateInfoGain(vvd&, int, double, double);
	vd getThresholds(vvd&, int);
	vvd subsetDiscreteData(vvd&, int, double);
	vector<vvd> subsetContinuousData(vvd&, int, double);
	node* pruneTree();
	vvd getForestNodeData(vvd&, int);

public:
	decisionTree(vvd&, node*, bool, bool, bool);
	~decisionTree();
};

#endif