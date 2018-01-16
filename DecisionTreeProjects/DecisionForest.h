#pragma once

#ifndef DECISION_FOREST_H_
#define DECISION_FOREST_H_

#include "DecisionTree.h"

class randomForest
{
	vector<decisionTree> forest;

	vvd getBootstrapSample(vvd&, int);

public:
	randomForest(vvd&, int, int, bool, bool);
	~randomForest();
	double predict(vd);
};

#endif