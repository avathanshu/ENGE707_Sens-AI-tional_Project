"""
elcs_custom.py -- trimmed, self-contained eLCS + Decision Tree factories for the ENGE707 Phase II notebook.

Built from scikit-eLCS (skeLCS) v1.2.4. The learning algorithm (matching, covering, GA, subsumption, deletion,
prediction) is the ORIGINAL code, unchanged. Removed as irrelevant to this project: timers and iteration
tracking, population reboot/pickling, parameter-validation boilerplate, debug printing, roulette selection,
CSV iteration export, attribute-specificity/accuracy reports and the data-cleanup helpers.
Everything is in this one file, so no skeLCS package or GitHub copy is needed.

The improved system in this project is FEATURE SELECTION (see SELECTED_FEATURE_COLUMNS below), not a change to the
algorithm: eLCS itself is identical for every run and only the columns it is given differ.
"""
import csv
import copy
import math
import random

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import balanced_accuracy_score
from sklearn.tree import DecisionTreeClassifier

# =============================================================================
# PROJECT SETTINGS (the "top parameters" -- edit here, the notebook imports them)
# =============================================================================
N_SEEDS = 10                                      # one full experiment per seed 0..9
TRAIN_SUBSAMPLE_SIZE = 1000                       # eLCS training rows per seed (eLCS is slow in pure Python)
TEST_EVAL_SIZE = 5000                             # held-out rows scored per seed (same slice for every model)
POPULATION_SIZE_N = round(1.5 * TRAIN_SUBSAMPLE_SIZE)   # N ~ 1.5x training instances (lecturer guideline)
LEARNING_ITERATIONS = 5 * TRAIN_SUBSAMPLE_SIZE    # same iteration budget for original and improved eLCS
MAX_DEPTH = 8                                     # Decision Tree depth cap

# -----------------------------------------------------------------------------
# Feature preprocessing / feature selection (Task 3 and the Task 4 improved system)
# -----------------------------------------------------------------------------
# Identifiers, plus the column the class label was cut from -- never usable as predictors.
ID_AND_TARGET_COLUMNS = ["response_id", "company_id", "productivity_change_percent", "productivity_class"]
# "AI paid off" outcome measures: the same kind of quantity as the target, so keeping them would leak the answer.
OUTCOME_FAMILY_COLUMNS = [
    "revenue_growth_percent", "cost_reduction_percent", "customer_satisfaction", "time_saved_per_week",
    "employee_satisfaction_score", "jobs_created", "jobs_displaced", "reskilled_employees",
    "remote_work_percentage", "innovation_score",
]
# Core numeric predictors that are winsorised (capped at their IQR fences) during preprocessing.
NUMERIC_COLUMNS_TO_CAP = [
    "ai_adoption_rate", "ai_maturity_score", "ai_failure_rate", "ai_training_hours", "task_automation_rate",
]
# The curated feature subset kept by the improved eLCS (5 numeric + 1 ordinal + 1 nominal = 7 columns).
SELECTED_FEATURE_COLUMNS = NUMERIC_COLUMNS_TO_CAP + ["ai_adoption_stage", "industry"]
STAGE_ORDER = {"none": 0, "pilot": 1, "partial": 2, "full": 3}   # ai_adoption_stage is ordinal


def make_elcs(seed):
    """Project eLCS: unmodified algorithm and original settings. The original and the improved system share
    this factory; they differ only in the feature matrix the notebook passes to fit()."""
    return eLCS(learning_iterations=LEARNING_ITERATIONS, N=POPULATION_SIZE_N, random_state=seed)


def make_decision_tree(seed):
    """Project Decision Tree (trained on the full training partition by the notebook)."""
    return DecisionTreeClassifier(max_depth=MAX_DEPTH, random_state=seed)



# =============================================================================
# DATA MANAGEMENT / ENVIRONMENT
# =============================================================================
class DataManagement:
    def __init__(self, dataFeatures, dataPhenotypes, elcs):
        # About Attributes
        self.savedRawTrainingData = [dataFeatures,dataPhenotypes]
        self.numAttributes = dataFeatures.shape[1]  # The number of attributes in the input file.
        self.attributeInfoType = [0]*self.numAttributes #stores false (d) or true (c) depending on its type, which points to parallel reference in one of the below 2 arrays
        self.attributeInfoContinuous = [[np.inf,-np.inf] for _ in range(self.numAttributes)] #stores continuous ranges and NaN otherwise
        self.attributeInfoDiscrete = [0]*self.numAttributes #stores arrays of discrete values or NaN otherwise.
        for i in range(0,self.numAttributes):
            self.attributeInfoDiscrete[i] = AttributeInfoDiscreteElement()

        # About Phenotypes
        self.discretePhenotype = True  # Is the Class/Phenotype Discrete? (False = Continuous)
        self.phenotypeList = []  # Stores all possible discrete phenotype states/classes or maximum and minimum values for a continuous phenotype
        self.classCount = {}
        self.majorityClass = None
        self.phenotypeRange = None  # Stores the difference between the maximum and minimum values for a continuous phenotype
        self.isDefault = True #Is discrete attribute limit an int or string
        try:
            int(elcs.discrete_attribute_limit)
        except:
            self.isDefault = False

        #About Dataset
        self.numTrainInstances = dataFeatures.shape[0]  # The number of instances in the training data
        self.discriminateClasses(dataPhenotypes)

        self.discriminateAttributes(dataFeatures, elcs)
        self.characterizeAttributes(dataFeatures, elcs)
        self.trainFormatted = self.formatData(dataFeatures,dataPhenotypes,elcs) #The only np array

    def discriminateClasses(self,phenotypes):
        currentPhenotypeIndex = 0

        while (currentPhenotypeIndex < self.numTrainInstances):
            target = phenotypes[currentPhenotypeIndex]
            if target in self.phenotypeList:
                self.classCount[target]+=1
            else:
                self.phenotypeList.append(target)
                self.classCount[target] = 1
            currentPhenotypeIndex+=1
        self.majorityClass = max(self.classCount)

    def characterizePhenotype(self,phenotypes,elcs):
        for target in phenotypes:
            if np.isnan(target):
                pass
            elif float(target) > self.phenotypeList[1]:
                self.phenotypeList[1] = float(target)
            elif float(target) < self.phenotypeList[0]:
                self.phenotypeList[0] = float(target)
            else:
                pass
        self.phenotypeRange = self.phenotypeList[1] - self.phenotypeList[0]

    def discriminateAttributes(self,features,elcs):
        for att in range(self.numAttributes):
            attIsDiscrete = True
            if self.isDefault:
                currentInstanceIndex = 0
                stateDict = {}
                while attIsDiscrete and len(list(stateDict.keys())) <= elcs.discrete_attribute_limit and currentInstanceIndex < self.numTrainInstances:
                    target = features[currentInstanceIndex,att]
                    if target in list(stateDict.keys()):
                        stateDict[target] += 1
                    elif np.isnan(target):
                        pass
                    else:
                        stateDict[target] = 1
                    currentInstanceIndex+=1

                if len(list(stateDict.keys())) > elcs.discrete_attribute_limit:
                    attIsDiscrete = False
            elif elcs.discrete_attribute_limit == "c":
                if att in elcs.specified_attributes:
                    attIsDiscrete = False
                else:
                    attIsDiscrete = True
            elif elcs.discrete_attribute_limit == "d":
                if att in elcs.specified_attributes:
                    attIsDiscrete = True
                else:
                    attIsDiscrete = False

            if attIsDiscrete:
                self.attributeInfoType[att] = False
            else:
                self.attributeInfoType[att] = True

    def characterizeAttributes(self,features,elcs):
        for currentFeatureIndexInAttributeInfo in range(self.numAttributes):
            for currentInstanceIndex in range(self.numTrainInstances):
                target = features[currentInstanceIndex,currentFeatureIndexInAttributeInfo]
                if not self.attributeInfoType[currentFeatureIndexInAttributeInfo]:#if attribute is discrete
                    if target in self.attributeInfoDiscrete[currentFeatureIndexInAttributeInfo].distinctValues or np.isnan(target):
                        pass
                    else:
                        self.attributeInfoDiscrete[currentFeatureIndexInAttributeInfo].distinctValues.append(target)
                else: #if attribute is continuous
                    if np.isnan(target):
                        pass
                    elif float(target) > self.attributeInfoContinuous[currentFeatureIndexInAttributeInfo][1]:
                        self.attributeInfoContinuous[currentFeatureIndexInAttributeInfo][1] = float(target)
                    elif float(target) < self.attributeInfoContinuous[currentFeatureIndexInAttributeInfo][0]:
                        self.attributeInfoContinuous[currentFeatureIndexInAttributeInfo][0] = float(target)
                    else:
                        pass

    def formatData(self,features,phenotypes,elcs):
        formatted = np.insert(features,self.numAttributes,phenotypes,1) #Combines features and phenotypes into one array
        np.random.shuffle(formatted)
        shuffledFeatures = formatted[:,:-1].tolist()
        shuffledLabels = formatted[:,self.numAttributes].tolist()
        for i in range(len(shuffledFeatures)):
            for j in range(len(shuffledFeatures[i])):
                if np.isnan(shuffledFeatures[i][j]):
                    shuffledFeatures[i][j] = None
            if np.isnan(shuffledLabels[i]):
                shuffledLabels[i] = None
        return [shuffledFeatures,shuffledLabels]

class AttributeInfoDiscreteElement():
    def __init__(self):
        self.distinctValues = []


class OfflineEnvironment:
    def __init__(self,features,phenotypes,eLCS):
        """Initialize Offline Environment"""
        self.dataRef = 0
        self.formatData = DataManagement(features,phenotypes,eLCS)

        self.currentTrainState = self.formatData.trainFormatted[0][self.dataRef]
        self.currentTrainPhenotype = self.formatData.trainFormatted[1][self.dataRef]

    def getTrainInstance(self):
        return (self.currentTrainState,self.currentTrainPhenotype)

    def newInstance(self):
        if self.dataRef < self.formatData.numTrainInstances-1:
            self.dataRef+=1
            self.currentTrainState = self.formatData.trainFormatted[0][self.dataRef]
            self.currentTrainPhenotype = self.formatData.trainFormatted[1][self.dataRef]
        else:
            self.resetDataRef()

    def resetDataRef(self):
        self.dataRef = 0
        self.currentTrainState = self.formatData.trainFormatted[0][self.dataRef]
        self.currentTrainPhenotype = self.formatData.trainFormatted[1][self.dataRef]


# =============================================================================
# CLASSIFIER (RULE)
# =============================================================================
class Classifier:
    def __init__(self,elcs,a=None,b=None,c=None,d=None):
        #Major Parameters
        self.specifiedAttList = []
        self.condition = []
        self.phenotype = None #arbitrary

        self.fitness = elcs.init_fit
        self.accuracy = 0.0
        self.numerosity = 1
        self.aveMatchSetSize = None
        self.deletionProb = None

        # Experience Management
        self.timeStampGA = None
        self.initTimeStamp = None

        # Classifier Accuracy Tracking --------------------------------------
        self.matchCount = 0  # Known in many LCS implementations as experience i.e. the total number of times this classifier was in a match set
        self.correctCount = 0  # The total number of times this classifier was in a correct set

        if isinstance(c, list):
            self.classifierCovering(elcs, a, b, c, d)
        elif isinstance(a, Classifier):
            self.classifierCopy(a, b)

    # Classifier Construction Methods
    def classifierCovering(self, elcs, setSize, exploreIter, state, phenotype):
        # Initialize new classifier parameters----------
        self.timeStampGA = exploreIter
        self.initTimeStamp = exploreIter
        self.aveMatchSetSize = setSize
        dataInfo = elcs.env.formatData

        # -------------------------------------------------------
        # DISCRETE PHENOTYPE
        # -------------------------------------------------------
        if dataInfo.discretePhenotype:
            self.phenotype = phenotype
        # -------------------------------------------------------
        # CONTINUOUS PHENOTYPE
        # -------------------------------------------------------
        else:
            phenotypeRange = dataInfo.phenotypeList[1] - dataInfo.phenotypeList[0]
            rangeRadius = random.randint(25,75) * 0.01 * phenotypeRange / 2.0  # Continuous initialization domain radius.
            Low = float(phenotype) - rangeRadius
            High = float(phenotype) + rangeRadius
            self.phenotype = [Low, High]

        while len(self.specifiedAttList) < 1:
            for attRef in range(len(state)):
                if random.random() < elcs.p_spec and not(state[attRef] == None):
                    self.specifiedAttList.append(attRef)
                    self.buildMatch(elcs, attRef, state)  # Add classifierConditionElement

    def classifierCopy(self, toCopy, exploreIter):
        self.specifiedAttList = copy.deepcopy(toCopy.specifiedAttList)
        self.condition = copy.deepcopy(toCopy.condition)

        self.phenotype = copy.deepcopy(toCopy.phenotype)
        self.timeStampGA = exploreIter
        self.initTimeStamp = exploreIter
        self.aveMatchSetSize = copy.deepcopy(toCopy.aveMatchSetSize)
        self.fitness = toCopy.fitness
        self.accuracy = toCopy.accuracy

    def buildMatch(self, elcs, attRef, state):
        attributeInfoType = elcs.env.formatData.attributeInfoType[attRef]
        if not(attributeInfoType): #Discrete
            attributeInfoValue = elcs.env.formatData.attributeInfoDiscrete[attRef]
        else:
            attributeInfoValue = elcs.env.formatData.attributeInfoContinuous[attRef]

        # Continuous attribute
        if attributeInfoType:
            attRange = attributeInfoValue[1] - attributeInfoValue[0]
            rangeRadius = random.randint(25, 75) * 0.01 * attRange / 2.0  # Continuous initialization domain radius.
            ar = state[attRef]
            Low = ar - rangeRadius
            High = ar + rangeRadius
            condList = [Low, High]
            self.condition.append(condList)

        # Discrete attribute
        else:
            condList = state[attRef]
            self.condition.append(condList)

    # Matching
    def match(self, state, elcs):
        for i in range(len(self.condition)):
            specifiedIndex = self.specifiedAttList[i]
            attributeInfoType = elcs.env.formatData.attributeInfoType[specifiedIndex]

            # Continuous
            if attributeInfoType:
                instanceValue = state[specifiedIndex]
                if elcs.match_for_missingness:
                    if instanceValue == None:
                        pass
                    elif self.condition[i][0] < instanceValue < self.condition[i][1]:
                        pass
                    else:
                        return False
                else:
                    if instanceValue == None:
                        return False
                    elif self.condition[i][0] < instanceValue < self.condition[i][1]:
                        pass
                    else:
                        return False

            # Discrete
            else:
                stateRep = state[specifiedIndex]
                if elcs.match_for_missingness:
                    if stateRep == self.condition[i] or stateRep == None:
                        pass
                    else:
                        return False
                else:
                    if stateRep == self.condition[i]:
                        pass
                    elif stateRep == None:
                        return False
                    else:
                        return False
        return True

    def equals(self, elcs, cl):
        if cl.phenotype == self.phenotype and len(cl.specifiedAttList) == len(self.specifiedAttList):
            clRefs = sorted(cl.specifiedAttList)
            selfRefs = sorted(self.specifiedAttList)
            if clRefs == selfRefs:
                for i in range(len(cl.specifiedAttList)):
                    tempIndex = self.specifiedAttList.index(cl.specifiedAttList[i])
                    if not (cl.condition[i] == self.condition[tempIndex]):
                        return False
                return True
        return False

    def updateNumerosity(self, num):
        """ Updates the numberosity of the classifier.  Notice that 'num' can be negative! """
        self.numerosity += num

    def updateExperience(self):
        """ Increases the experience of the classifier by one. Once an epoch has completed, rule accuracy can't change."""
        self.matchCount += 1

    def updateCorrect(self):
        """ Increases the correct phenotype tracking by one. Once an epoch has completed, rule accuracy can't change."""
        self.correctCount += 1

    def updateMatchSetSize(self, elcs, matchSetSize):
        """  Updates the average match set size. """
        if self.matchCount < 1.0 / elcs.beta:
            self.aveMatchSetSize = (self.aveMatchSetSize * (self.matchCount - 1) + matchSetSize) / float(
                self.matchCount)
        else:
            self.aveMatchSetSize = self.aveMatchSetSize + elcs.beta * (matchSetSize - self.aveMatchSetSize)

    def updateAccuracy(self):
        """ Update the accuracy tracker """
        self.accuracy = self.correctCount / float(self.matchCount)

    def updateFitness(self, elcs):
        """ Update the fitness parameter. """
        if elcs.env.formatData.discretePhenotype or (
                self.phenotype[1] - self.phenotype[0]) / elcs.env.formatData.phenotypeRange < 0.5:
            self.fitness = pow(self.accuracy, elcs.nu)
        else:
            if (self.phenotype[1] - self.phenotype[0]) >= elcs.env.formatData.phenotypeRange:
                self.fitness = 0.0
            else:
                self.fitness = math.fabs(pow(self.accuracy, elcs.nu) - (
                            self.phenotype[1] - self.phenotype[0]) / elcs.env.formatData.phenotypeRange)

    def isSubsumer(self, elcs):
        if self.matchCount > elcs.theta_sub and self.accuracy > elcs.acc_sub:
            return True
        return False

    def isMoreGeneral(self, cl, elcs):
        if len(self.specifiedAttList) >= len(cl.specifiedAttList):
            return False
        for i in range(len(self.specifiedAttList)):
            attributeInfoType = elcs.env.formatData.attributeInfoType[self.specifiedAttList[i]]
            if self.specifiedAttList[i] not in cl.specifiedAttList:
                return False

            # Continuous
            if attributeInfoType:
                otherRef = cl.specifiedAttList.index(self.specifiedAttList[i])
                if self.condition[i][0] < cl.condition[otherRef][0]:
                    return False
                if self.condition[i][1] > cl.condition[otherRef][1]:
                    return False
        return True

    def uniformCrossover(self, elcs, cl):
        if elcs.env.formatData.discretePhenotype or random.random() < 0.5:
            p_self_specifiedAttList = copy.deepcopy(self.specifiedAttList)
            p_cl_specifiedAttList = copy.deepcopy(cl.specifiedAttList)

            # Make list of attribute references appearing in at least one of the parents.-----------------------------
            comboAttList = []
            for i in p_self_specifiedAttList:
                comboAttList.append(i)
            for i in p_cl_specifiedAttList:
                if i not in comboAttList:
                    comboAttList.append(i)
                elif not elcs.env.formatData.attributeInfoType[i]:
                    comboAttList.remove(i)
            comboAttList.sort()

            changed = False
            for attRef in comboAttList:
                attributeInfoType = elcs.env.formatData.attributeInfoType[attRef]
                probability = 0.5
                ref = 0
                if attRef in p_self_specifiedAttList:
                    ref += 1
                if attRef in p_cl_specifiedAttList:
                    ref += 1

                if ref == 0:
                    pass
                elif ref == 1:
                    if attRef in p_self_specifiedAttList and random.random() > probability:
                        i = self.specifiedAttList.index(attRef)
                        cl.condition.append(self.condition.pop(i))

                        cl.specifiedAttList.append(attRef)
                        self.specifiedAttList.remove(attRef)
                        changed = True

                    if attRef in p_cl_specifiedAttList and random.random() < probability:
                        i = cl.specifiedAttList.index(attRef)
                        self.condition.append(cl.condition.pop(i))

                        self.specifiedAttList.append(attRef)
                        cl.specifiedAttList.remove(attRef)
                        changed = True
                else:
                    # Continuous Attribute
                    if attributeInfoType:
                        i_cl1 = self.specifiedAttList.index(attRef)
                        i_cl2 = cl.specifiedAttList.index(attRef)
                        tempKey = random.randint(0, 3)
                        if tempKey == 0:
                            temp = self.condition[i_cl1][0]
                            self.condition[i_cl1][0] = cl.condition[i_cl2][0]
                            cl.condition[i_cl2][0] = temp
                        elif tempKey == 1:
                            temp = self.condition[i_cl1][1]
                            self.condition[i_cl1][1] = cl.condition[i_cl2][1]
                            cl.condition[i_cl2][1] = temp
                        else:
                            allList = self.condition[i_cl1] + cl.condition[i_cl2]
                            newMin = min(allList)
                            newMax = max(allList)
                            if tempKey == 2:
                                self.condition[i_cl1] = [newMin, newMax]
                                cl.condition.pop(i_cl2)

                                cl.specifiedAttList.remove(attRef)
                            else:
                                cl.condition[i_cl2] = [newMin, newMax]
                                self.condition.pop(i_cl1)

                                self.specifiedAttList.remove(attRef)

                    # Discrete Attribute
                    else:
                        pass

            tempList1 = copy.deepcopy(p_self_specifiedAttList)
            tempList2 = copy.deepcopy(cl.specifiedAttList)
            tempList1.sort()
            tempList2.sort()

            if changed and len(set(tempList1) & set(tempList2)) == len(tempList2):
                changed = False

            return changed
        else:
            return self.phenotypeCrossover(cl)

    def phenotypeCrossover(self, cl):
        changed = False
        if self.phenotype == cl.phenotype:
            return changed
        else:
            tempKey = random.random() < 0.5  # Make random choice between 4 scenarios, Swap minimums, Swap maximums, Children preserve parent phenotypes.
            if tempKey:  # Swap minimum
                temp = self.phenotype[0]
                self.phenotype[0] = cl.phenotype[0]
                cl.phenotype[0] = temp
                changed = True
            elif tempKey:  # Swap maximum
                temp = self.phenotype[1]
                self.phenotype[1] = cl.phenotype[1]
                cl.phenotype[1] = temp
                changed = True

        return changed

    def Mutation(self, elcs, state, phenotype):
        changed = False
        # Mutate Condition
        for attRef in range(elcs.env.formatData.numAttributes):
            attributeInfoType = elcs.env.formatData.attributeInfoType[attRef]
            if not (attributeInfoType):  # Discrete
                attributeInfoValue = elcs.env.formatData.attributeInfoDiscrete[attRef]
            else:
                attributeInfoValue = elcs.env.formatData.attributeInfoContinuous[attRef]

            if random.random() < elcs.mu and not(state[attRef] == None):
                # Mutation
                if attRef not in self.specifiedAttList:
                    self.specifiedAttList.append(attRef)
                    self.buildMatch(elcs, attRef, state)
                    changed = True
                elif attRef in self.specifiedAttList:
                    i = self.specifiedAttList.index(attRef)

                    if not attributeInfoType or random.random() > 0.5:
                        del self.specifiedAttList[i]
                        del self.condition[i]
                        changed = True
                    else:
                        attRange = float(attributeInfoValue[1]) - float(attributeInfoValue[0])
                        mutateRange = random.random() * 0.5 * attRange
                        if random.random() > 0.5:
                            if random.random() > 0.5:
                                self.condition[i][0] += mutateRange
                            else:
                                self.condition[i][0] -= mutateRange
                        else:
                            if random.random() > 0.5:
                                self.condition[i][1] += mutateRange
                            else:
                                self.condition[i][1] -= mutateRange
                        self.condition[i] = sorted(self.condition[i])
                        changed = True

                else:
                    pass

        # Mutate Phenotype
        if elcs.env.formatData.discretePhenotype:
            nowChanged = self.discretePhenotypeMutation(elcs)
        else:
            nowChanged = self.continuousPhenotypeMutation(elcs, phenotype)

        if changed or nowChanged:
            return True

    def discretePhenotypeMutation(self, elcs):
        changed = False
        if random.random() < elcs.mu:
            phenotypeList = copy.deepcopy(elcs.env.formatData.phenotypeList)
            phenotypeList.remove(self.phenotype)
            newPhenotype = random.choice(phenotypeList)
            self.phenotype = newPhenotype
            changed = True
        return changed

    def continuousPhenotypeMutation(self, elcs, phenotype):
        changed = False
        if random.random() < elcs.mu:
            phenRange = self.phenotype[1] - self.phenotype[0]
            mutateRange = random.random() * 0.5 * phenRange
            tempKey = random.randint(0,2)  # Make random choice between 3 scenarios, mutate minimums, mutate maximums, mutate both
            if tempKey == 0:  # Mutate minimum
                if random.random() > 0.5 or self.phenotype[0] + mutateRange <= phenotype:  # Checks that mutated range still contains current phenotype
                    self.phenotype[0] += mutateRange
                else:  # Subtract
                    self.phenotype[0] -= mutateRange
                changed = True
            elif tempKey == 1:  # Mutate maximum
                if random.random() > 0.5 or self.phenotype[1] - mutateRange >= phenotype:  # Checks that mutated range still contains current phenotype
                    self.phenotype[1] -= mutateRange
                else:  # Subtract
                    self.phenotype[1] += mutateRange
                changed = True
            else:  # mutate both
                if random.random() > 0.5 or self.phenotype[0] + mutateRange <= phenotype:  # Checks that mutated range still contains current phenotype
                    self.phenotype[0] += mutateRange
                else:  # Subtract
                    self.phenotype[0] -= mutateRange
                if random.random() > 0.5 or self.phenotype[1] - mutateRange >= phenotype:  # Checks that mutated range still contains current phenotype
                    self.phenotype[1] -= mutateRange
                else:  # Subtract
                    self.phenotype[1] += mutateRange
                changed = True
            self.phenotype.sort()
        return changed

    def updateTimeStamp(self, ts):
        """ Sets the time stamp of the classifier. """
        self.timeStampGA = ts

    def setAccuracy(self, acc):
        """ Sets the accuracy of the classifier """
        self.accuracy = acc

    def setFitness(self, fit):
        """  Sets the fitness of the classifier. """
        self.fitness = fit

    def subsumes(self, elcs, cl):
        # Discrete Phenotype
        if elcs.env.formatData.discretePhenotype:
            if cl.phenotype == self.phenotype:
                if self.isSubsumer(elcs) and self.isMoreGeneral(cl, elcs):
                    return True
            return False

        # Continuous Phenotype
        else:
            if self.phenotype[0] >= cl.phenotype[0] and self.phenotype[1] <= cl.phenotype[1]:
                if self.isSubsumer(elcs) and self.isMoreGeneral(cl, elcs):
                    return True
                return False

    def getDelProp(self, elcs, meanFitness):
        """  Returns the vote for deletion of the classifier. """
        if self.fitness / self.numerosity >= elcs.delta * meanFitness or self.matchCount < elcs.theta_del:
            deletionVote = self.aveMatchSetSize * self.numerosity
        elif self.fitness == 0.0:
            deletionVote = self.aveMatchSetSize * self.numerosity * meanFitness / (elcs.init_fit / self.numerosity)
        else:
            deletionVote = self.aveMatchSetSize * self.numerosity * meanFitness / (self.fitness / self.numerosity)
        return deletionVote


# =============================================================================
# PREDICTION
# =============================================================================
class Prediction():
    def __init__(self,elcs,population):
        self.decision = None
        self.probabilities = {}
        self.hasMatch = len(population.matchSet) != 0

        #Discrete Phenotypes
        if elcs.env.formatData.discretePhenotype:
            self.vote = {}
            self.tieBreak_Numerosity = {}
            self.tieBreak_TimeStamp = {}

            for eachClass in elcs.env.formatData.phenotypeList:
                self.vote[eachClass] = 0.0
                self.tieBreak_Numerosity[eachClass] = 0.0
                self.tieBreak_TimeStamp[eachClass] = 0.0

            for ref in population.matchSet:
                cl = population.popSet[ref]
                self.vote[cl.phenotype] += cl.fitness * cl.numerosity
                self.tieBreak_Numerosity[cl.phenotype] += cl.numerosity
                self.tieBreak_TimeStamp[cl.phenotype] += cl.initTimeStamp

            #Populate Probabilities
            sProb = 0
            for k,v in sorted(self.vote.items()):
                self.probabilities[k] = v
                sProb += v
            if sProb == 0: #In the case the match set doesn't exist
                for k, v in sorted(self.probabilities.items()):
                    self.probabilities[k] = 0
            else:
                for k,v in sorted(self.probabilities.items()):
                    self.probabilities[k] = v/sProb

            highVal = 0.0
            bestClass = []
            for thisClass in elcs.env.formatData.phenotypeList:
                if self.vote[thisClass] >= highVal:
                    highVal = self.vote[thisClass]

            for thisClass in elcs.env.formatData.phenotypeList:
                if self.vote[thisClass] == highVal:  # Tie for best class
                    bestClass.append(thisClass)

            if highVal == 0.0:
                self.decision = None

            elif len(bestClass) > 1:
                bestNum = 0
                newBestClass = []
                for thisClass in bestClass:
                    if self.tieBreak_Numerosity[thisClass] >= bestNum:
                        bestNum = self.tieBreak_Numerosity[thisClass]

                for thisClass in bestClass:
                    if self.tieBreak_Numerosity[thisClass] == bestNum:
                        newBestClass.append(thisClass)

                if len(newBestClass) > 1:
                    bestStamp = 0
                    newestBestClass = []
                    for thisClass in newBestClass:
                        if self.tieBreak_TimeStamp[thisClass] >= bestStamp:
                            bestStamp = self.tieBreak_TimeStamp[thisClass]

                    for thisClass in newBestClass:
                        if self.tieBreak_TimeStamp[thisClass] == bestStamp:
                            newestBestClass.append(thisClass)
                    # -----------------------------------------------------------------------
                    if len(newestBestClass) > 1:  # Prediction is completely tied - eLCS has no useful information for making a prediction
                        self.decision = 'Tie'
                else:
                    self.decision = newBestClass[0]
            else:
                self.decision = bestClass[0]

        #Continuous Phenotypes
        else:
            if len(population.matchSet) < 1:
                self.decision = None
            else:
                phenotypeRange = elcs.env.formatData.phenotypeList[1] - elcs.env.formatData.phenotypeList[0]
                predictionValue = 0
                valueWeightSum = 0
                for ref in population.matchSet:
                    cl = population.popSet[ref]
                    localRange = cl.phenotype[1] - cl.phenotype[0]
                    valueWeight = (phenotypeRange / float(localRange))
                    localAverage = cl.phenotype[1] + cl.phenotype[0] / 2.0

                    valueWeightSum += valueWeight
                    predictionValue += valueWeight * localAverage
                if valueWeightSum == 0.0:
                    self.decision = None
                else:
                    self.decision = predictionValue / float(valueWeightSum)

        if self.decision == None or self.decision == 'Tie':
            if elcs.env.formatData.discretePhenotype:
                self.decision = elcs.env.formatData.majorityClass
                #self.decision = random.choice(elcs.env.formatData.phenotypeList)
            else:
                self.decision = random.randrange(elcs.env.formatData.phenotypeList[0],elcs.env.formatData.phenotypeList[1],(elcs.env.formatData.phenotypeList[1]-elcs.env.formatData.phenotypeList[0])/float(1000))

    def getFitnessSum(self, population, low, high):
        """ Get the fitness sum of rules in the rule-set. For continuous phenotype prediction. """
        fitSum = 0
        for ref in population.matchSet:
            cl = population.popSet[ref][0]
            if cl.phenotype[0] <= low and cl.phenotype[1] >= high:  # if classifier range subsumes segment range.
                fitSum += cl.fitness
        return fitSum

    def getDecision(self):
        """ Returns prediction decision. """
        return self.decision

    def getProbabilities(self):
        ''' Returns probabilities of each phenotype from the decision'''
        a = np.empty(len(sorted(self.probabilities.items())))
        counter = 0
        for k,v in sorted(self.probabilities.items()):
            a[counter] = v
            counter += 1
        return a


# =============================================================================
# CLASSIFIER SET (POPULATION)
# =============================================================================
class ClassifierSet:
    def __init__(self):
        #Major Parameters
        self.popSet = []
        self.matchSet = []
        self.correctSet = []
        self.microPopSize = 0

    def makeMatchSet(self,state_phenotype,exploreIter,elcs):
        state = state_phenotype[0]
        phenotype = state_phenotype[1]
        doCovering = True
        setNumerositySum = 0

        #Matching
        for i in range(len(self.popSet)):
            cl = self.popSet[i]
            if cl.match(state,elcs):
                self.matchSet.append(i)
                setNumerositySum += cl.numerosity

                #Covering Check
                if elcs.env.formatData.discretePhenotype:
                    if cl.phenotype == phenotype:
                        doCovering = False
                else:
                    if float(cl.phenotype[0]) <= float(phenotype) <= float(cl.phenotype[1]):
                        doCovering = False
        #Covering
        while doCovering:
            newCl = Classifier(elcs,setNumerositySum+1,exploreIter,state,phenotype)
            self.addClassifierToPopulation(elcs,newCl,True)
            self.matchSet.append(len(self.popSet) - 1)
            doCovering = False

    def getIdenticalClassifier(self,elcs,newCl):
        for cl in self.popSet:
            if newCl.equals(elcs,cl):
                return cl
        return None

    def addClassifierToPopulation(self,elcs,cl,covering):
        oldCl = None
        if not covering:
            oldCl = self.getIdenticalClassifier(elcs,cl)
        if oldCl != None:
            oldCl.updateNumerosity(1)
            self.microPopSize += 1
        else:
            self.popSet.append(cl)
            self.microPopSize += 1

    def makeCorrectSet(self,elcs,phenotype):
        for i in range(len(self.matchSet)):
            ref = self.matchSet[i]
            #Discrete Phenotype
            if elcs.env.formatData.discretePhenotype:
                if self.popSet[ref].phenotype == phenotype:
                    self.correctSet.append(ref)

            #Continuous Phenotype
            else:
                if float(phenotype) <= float(self.popSet[ref].phenotype[1]) and float(phenotype) >= float(self.popSet[ref].phenotype[0]):
                    self.correctSet.append(ref)

    def updateSets(self,elcs,exploreIter):
        matchSetNumerosity = 0
        for ref in self.matchSet:
            matchSetNumerosity += self.popSet[ref].numerosity

        for ref in self.matchSet:
            self.popSet[ref].updateExperience()
            self.popSet[ref].updateMatchSetSize(elcs,matchSetNumerosity)
            if ref in self.correctSet:
                self.popSet[ref].updateCorrect()

            self.popSet[ref].updateAccuracy()
            self.popSet[ref].updateFitness(elcs)

    def do_correct_set_subsumption(self,elcs):
        subsumer = None
        for ref in self.correctSet:
            cl = self.popSet[ref]
            if cl.isSubsumer(elcs):
                if subsumer == None or cl.isMoreGeneral(subsumer,elcs):
                    subsumer = cl

        if subsumer != None:
            i = 0
            while i < len(self.correctSet):
                ref = self.correctSet[i]
                if subsumer.isMoreGeneral(self.popSet[ref],elcs):
                    subsumer.updateNumerosity(self.popSet[ref].numerosity)
                    self.removeMacroClassifier(ref)
                    self.deleteFromMatchSet(ref)
                    self.deleteFromCorrectSet(ref)
                    i -= 1
                i+=1

    def removeMacroClassifier(self,ref):
        del self.popSet[ref]

    def deleteFromMatchSet(self,deleteRef):
        if deleteRef in self.matchSet:
            self.matchSet.remove(deleteRef)

        for j in range(len(self.matchSet)):
            ref = self.matchSet[j]
            if ref > deleteRef:
                self.matchSet[j] -=1

    def deleteFromCorrectSet(self,deleteRef):
        if deleteRef in self.correctSet:
            self.correctSet.remove(deleteRef)

        for j in range(len(self.correctSet)):
            ref = self.correctSet[j]
            if ref > deleteRef:
                self.correctSet[j] -= 1

    def runGA(self,elcs,exploreIter,state,phenotype):
        #GA Run Requirement
        if (exploreIter - self.getIterStampAverage()) < elcs.theta_GA:
            return
        self.setIterStamps(exploreIter)
        changed = False

        #Select Parents
        selectList = self.selectClassifierT(elcs)   # tournament selection only
        clP1 = selectList[0]
        clP2 = selectList[1]
        #Initialize Offspring
        cl1 = Classifier(elcs,clP1,exploreIter)
        if clP2 == None:
            cl2 = Classifier(elcs,clP1, exploreIter)
        else:
            cl2 = Classifier(elcs,clP2, exploreIter)

        #Crossover Operator (uniform crossover)
        if not cl1.equals(elcs,cl2) and random.random() < elcs.chi:
            changed = cl1.uniformCrossover(elcs,cl2)

        #Initialize Key Offspring Parameters
        if changed:
            cl1.setAccuracy((cl1.accuracy + cl2.accuracy) / 2.0)
            cl1.setFitness(elcs.fitness_reduction * (cl1.fitness + cl2.fitness) / 2.0)
            cl2.setAccuracy(cl1.accuracy)
            cl2.setFitness(cl1.fitness)
        else:
            cl1.setFitness(elcs.fitness_reduction * cl1.fitness)
            cl2.setFitness(elcs.fitness_reduction * cl2.fitness)

        #Mutation Operator
        nowchanged = cl1.Mutation(elcs,state,phenotype)
        howaboutnow = cl2.Mutation(elcs,state,phenotype)

        #Add offspring to population
        if changed or nowchanged or howaboutnow:

            self.insertDiscoveredClassifiers(elcs,cl1, cl2, clP1, clP2, exploreIter)  # Subsumption

    def getIterStampAverage(self):
        sumCl = 0.0
        numSum = 0.0
        for i in range(len(self.correctSet)):
            ref = self.correctSet[i]
            sumCl += self.popSet[ref].timeStampGA * self.popSet[ref].numerosity
            numSum += self.popSet[ref].numerosity
        if numSum != 0:
            return sumCl/float(numSum)
        else:
            return 0


    def setIterStamps(self,exploreIter):
        for i in range(len(self.correctSet)):
            ref = self.correctSet[i]
            self.popSet[ref].updateTimeStamp(exploreIter)


    def getFitnessSum(self, setList):
        """ Returns the sum of the fitnesses of all classifiers in the set. """
        sumCl = 0.0
        for i in range(len(setList)):
            ref = setList[i]
            sumCl += self.popSet[ref].fitness
        return sumCl

    def selectClassifierT(self,elcs):
        selectList = [None, None]
        currentCount = 0
        setList = self.correctSet

        while currentCount < 2:
            tSize = int(len(setList) * elcs.theta_sel)

            #Select tSize elements from correctSet
            posList = random.sample(setList,tSize)

            bestF = 0
            bestC = self.correctSet[0]
            for j in posList:
                if self.popSet[j].fitness > bestF:
                    bestF = self.popSet[j].fitness
                    bestC = j

            selectList[currentCount] = self.popSet[bestC]
            currentCount += 1

        return selectList

    def insertDiscoveredClassifiers(self,elcs,cl1,cl2,clP1,clP2,exploreIter):
        if elcs.do_GA_subsumption:
            if len(cl1.specifiedAttList) > 0:
                self.subsumeClassifier(elcs,cl1,clP1,clP2)
            if len(cl2.specifiedAttList) > 0:
                self.subsumeClassifier(elcs,cl2, clP1, clP2)
        else:
            if len(cl1.specifiedAttList) > 0:
                self.addClassifierToPopulation(elcs,cl1,False)
            if len(cl2.specifiedAttList) > 0:
                self.addClassifierToPopulation(elcs,cl2, False)

    def subsumeClassifier(self,elcs,cl=None,cl1P=None,cl2P=None):
        if cl1P != None and cl1P.subsumes(elcs,cl):
            self.microPopSize += 1
            cl1P.updateNumerosity(1)
        elif cl2P != None and cl2P.subsumes(elcs,cl):
            self.microPopSize += 1
            cl2P.updateNumerosity(1)
        else:
            if len(cl.specifiedAttList) > 0:
                self.addClassifierToPopulation(elcs, cl, False)

    def deletion(self,elcs,exploreIter):
        while (self.microPopSize > elcs.N):
            self.deleteFromPopulation(elcs)

    def deleteFromPopulation(self,elcs):
        meanFitness = self.getPopFitnessSum() / float(self.microPopSize)

        sumCl = 0.0
        voteList = []
        for cl in self.popSet:
            vote = cl.getDelProp(elcs,meanFitness)
            sumCl += vote
            voteList.append(vote)
        i = 0
        for cl in self.popSet:
            cl.deletionProb = voteList[i]/sumCl
            i+=1
        choicePoint = sumCl * random.random()  # Determine the choice point

        newSum = 0.0
        for i in range(len(voteList)):
            cl = self.popSet[i]
            newSum = newSum + voteList[i]
            if newSum > choicePoint:  # Select classifier for deletion
                # Delete classifier----------------------------------
                cl.updateNumerosity(-1)
                self.microPopSize -= 1
                if cl.numerosity < 1:  # When all micro-classifiers for a given classifier have been depleted.
                    self.removeMacroClassifier(i)
                    self.deleteFromMatchSet(i)
                    self.deleteFromCorrectSet(i)
                return
        return

    def getPopFitnessSum(self):
        """ Returns the sum of the fitnesses of all classifiers in the set. """
        sumCl = 0.0
        for cl in self.popSet:
            sumCl += cl.fitness * cl.numerosity
        return sumCl

    def clearSets(self):
        """ Clears out references in the match and correct sets for the next learning iteration. """
        self.matchSet = []
        self.correctSet = []




    def makeEvalMatchSet(self,state,elcs):
        for i in range(len(self.popSet)):
            cl = self.popSet[i]
            if cl.match(state,elcs):
                self.matchSet.append(i)


# =============================================================================
# eLCS (trimmed)
# =============================================================================
class eLCS(BaseEstimator, ClassifierMixin):
    def __init__(self, learning_iterations=10000, N=1000, p_spec=0.5, discrete_attribute_limit=10,
                 specified_attributes=np.array([]), nu=5, chi=0.8, mu=0.04, theta_GA=25, theta_del=20,
                 theta_sub=20, acc_sub=0.99, beta=0.2, delta=0.1, init_fit=0.01, fitness_reduction=0.1,
                 do_correct_set_subsumption=False, do_GA_subsumption=True, theta_sel=0.5,
                 random_state=None, match_for_missingness=False):
        """Defaults are the library defaults. Parameters as in scikit-eLCS:
        learning_iterations = training cycles; N = max micro-classifier population; p_spec = covering specificity;
        nu = fitness pressure; chi = crossover prob.; mu = mutation prob.; theta_GA = GA threshold;
        theta_del/theta_sub = deletion/subsumption experience thresholds; acc_sub = subsumption accuracy;
        beta = correct-set-size learning rate; delta = deletion param.; init_fit / fitness_reduction = new-rule fitness;
        do_correct_set_subsumption / do_GA_subsumption = subsumption switches; theta_sel = tournament fraction;
        match_for_missingness = allow NaN to match specified values."""
        self.learning_iterations = learning_iterations
        self.N = N
        self.p_spec = p_spec
        self.discrete_attribute_limit = discrete_attribute_limit
        self.specified_attributes = specified_attributes
        self.nu = nu
        self.chi = chi
        self.mu = mu
        self.theta_GA = theta_GA
        self.theta_del = theta_del
        self.theta_sub = theta_sub
        self.acc_sub = acc_sub
        self.beta = beta
        self.delta = delta
        self.init_fit = init_fit
        self.fitness_reduction = fitness_reduction
        self.do_correct_set_subsumption = do_correct_set_subsumption
        self.do_GA_subsumption = do_GA_subsumption
        self.theta_sel = theta_sel
        self.random_state = random_state
        self.match_for_missingness = match_for_missingness
        self.hasTrained = False

    def fit(self, X, y):
        """Supervised training. X: numeric array (NaN allowed); y: numeric class labels."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if self.random_state is not None:
            random.seed(int(self.random_state))
            np.random.seed(int(self.random_state))
        self.population = ClassifierSet()
        self.explorIter = 0
        self.env = OfflineEnvironment(X, y, self)
        while self.explorIter < self.learning_iterations:
            self.runIteration(self.env.getTrainInstance(), self.explorIter)
            self.explorIter += 1
            self.env.newInstance()
        self.hasTrained = True
        return self

    def runIteration(self, state_phenotype, exploreIter):
        self.population.makeMatchSet(state_phenotype, exploreIter, self)      # form [M] (+ covering)
        self.population.makeCorrectSet(self, state_phenotype[1])              # form [C]
        self.population.updateSets(self, exploreIter)                         # update rule parameters
        if self.do_correct_set_subsumption:
            self.population.do_correct_set_subsumption(self)
        self.population.runGA(self, exploreIter, state_phenotype[0], state_phenotype[1])
        self.population.deletion(self, exploreIter)
        self.population.clearSets()

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        preds = []
        for state in X:
            self.population.makeEvalMatchSet(state, self)
            preds.append(Prediction(self, self.population).getDecision())
            self.population.clearSets()
        return np.array(preds)

    def score(self, X, y):
        return balanced_accuracy_score(y, self.predict(X))

    def get_final_instance_coverage(self):
        """Fraction of training instances matched by at least one rule."""
        data = self.env.formatData.savedRawTrainingData[0]
        covered = 0
        for state in data:
            self.population.makeEvalMatchSet(state, self)
            if Prediction(self, self.population).hasMatch:
                covered += 1
            self.population.clearSets()
        return covered / len(data)

    def export_final_rule_population(self, headerNames=np.array([]), className="phenotype",
                                     filename="populationData.csv", DCAL=True):
        """Write the rule population to CSV in the readable 'Detailed Compact Attribute List' format."""
        if not self.hasTrained:
            raise Exception("No rule population to export: the eLCS model has not been trained")
        numAttributes = self.env.formatData.numAttributes
        headerNames = list(headerNames) or ["N" + str(i) for i in range(numAttributes)]
        if len(headerNames) != numAttributes:
            raise Exception("# of header names does not match the number of attributes")
        with open(filename, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Specified Values", "Specified Attribute Names", className, "Fitness", "Accuracy",
                             "Numerosity", "Avg Match Set Size", "TimeStamp GA", "Iteration Initialized",
                             "Specificity", "Deletion Probability", "Correct Count", "Match Count"])
            for c in self.population.popSet:
                names, values = [], []
                for a in range(numAttributes):
                    if a in c.specifiedAttList:
                        cond = c.condition[c.specifiedAttList.index(a)]
                        names.append(str(headerNames[a]))
                        values.append("[%s,%s]" % (cond[0], cond[1]) if isinstance(cond, list) else str(cond))
                phen = "%s,%s" % (c.phenotype[0], c.phenotype[1]) if isinstance(c.phenotype, list) else c.phenotype
                writer.writerow([", ".join(values), ", ".join(names), phen, c.fitness, c.accuracy, c.numerosity,
                                 c.aveMatchSetSize, c.timeStampGA, c.initTimeStamp,
                                 len(c.specifiedAttList) / numAttributes, c.deletionProb, c.correctCount, c.matchCount])
