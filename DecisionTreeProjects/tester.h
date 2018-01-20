#pragma once

#ifndef TREES_AND_FORESTS_TESTER_H_
#define TREES_AND_FORESTS_TESTER_H_

#include "DecisionTree.h"
#include "RandomForest.h"

bool getBoolArg(char*);
tuple<vvd, vvd> parseData(string, string);
vd parseDataLine(string);

#endif