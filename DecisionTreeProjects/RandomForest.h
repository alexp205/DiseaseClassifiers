#pragma once

#ifndef DECISION_FOREST_H_
#define DECISION_FOREST_H_

#include "DecisionTree.h"

class randomForest
{
	vector<decisionTree> forest;

	vvd getBootstrapSample(vvd&, int);
	void printForestSample(int);
	double processStats(vd&, vd&, wstring);

public:
	randomForest(vvd&, int, int, bool, bool);
	double predict(vd&);
	vd predict(vvd&);
	void print(int);
	double getStatsInfo(vd&, vd&, wstring);
};

#endif