#pragma once

#ifndef DECISION_TREE_H_
#define DECISION_TREE_H_

#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <tuple>
#include <iostream>
#include <iomanip>
#include <fstream>

using namespace std;

typedef vector<double> vd;
typedef vector<vd> vvd;

struct node
{
	bool is_leaf = false;
	double label; // if leaf node, contains label, else...
	int split_var; // contains attribute split label
	double split_val = -1; // NOTE: only used in discrete data trees
	double threshold = -1; // NOTE: only used in continous data trees
	int frequency = 1;
	vector<node> children;
};

class decisionTree
{
	vvd data_info; // list (in order of input format) of all variables and possible values
	map<double,int> labels;
	node root_node;
	int min_data_size;
	bool is_discrete;
	bool is_classification;
	bool is_in_forest;

	vvd getDatasetInfo(vvd&);
	map<double,int> getLabelInfo(vvd&);
	node buildTree(vvd&, vector<int>, node);
	tuple<bool,double> checkLeaf(vvd&);
	tuple<int,double> bestSplitVar(vvd&);
	double calculateEntropy(vvd&, int, double);
	double calculateInfoGain(vvd&, int, double, double);
	vd getThresholds(vvd&, int);
	vvd subsetDiscreteData(vvd&, int, double);
	vector<vvd> subsetContinuousData(vvd&, int, double);
	//node* pruneTree();
	double getCutoffLeafLabel();
	vvd getForestNodeData(vvd&, int);
	double predict(vd&, node);
	void printTree(node, int);
	void printSpacing(int, bool);
	double processStats(vd&, vd&, wstring);

public:
	decisionTree(vvd&, int, bool, bool, bool);
	//decisionTree(const decisionTree&);
	//decisionTree& operator=(const decisionTree&);
	//~decisionTree();
	double predict(vd&);
	vd predict(vvd&);
	void print();
	double getStatsInfo(vd&, vd&, wstring);
};

#endif