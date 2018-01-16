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
	int split_label; // contains attribute split label
	double split_val; // NOTE: only used in discrete data trees
	double threshold; // NOTE: only used in continous data trees
	int frequency = 1;
	vector<node*> children;
};

class decisionTree
{
	vvd parent_data_info; // list (in order of input format) of all variables and possible values
	vvd data_info; // same as above but mutable
	map<double,int> labels;
	node* root_node;
	int min_data_size;
	bool is_discrete;
	bool is_classification;
	bool is_in_forest;
	vvd random_data; // this holds the random data collected for each node in trees in a random forest

	vvd getDatasetInfo(vvd&);
	map<double,int> getLabelInfo(vvd&);
	node* buildTree(vvd&, node*);
	bool checkLeaf(vvd&);
	tuple<int, double> bestSplitVar(vvd&);
	double calculateEntropy(vvd&, int, double);
	double calculateInfoGain(vvd&, int, double, double);
	vd getThresholds(vvd&, int);
	vvd subsetDiscreteData(vvd&, int, double);
	vector<vvd> subsetContinuousData(vvd&, int, double);
	//node* pruneTree();
	double getCutoffLeafLabel(vvd&);
	vvd getForestNodeData(vvd&, int);
	double predict(vd, node*);

public:
	decisionTree(vvd&, int, bool, bool, bool);
	~decisionTree();
	double predict(vd);
};

#endif