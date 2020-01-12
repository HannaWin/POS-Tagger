#This script creates a confusion matrix and calculates micro and macro average for POS-tagged corpora
#it requires two documents which were tagged and are of the same length (to avoid different texts)
#the correctly tagged gold standard document needs to be the first argument, the predicted tags need to be the second argument

class Evaluation():

    def __init__(self, doc_gold, doc_pred):
        self.doc_pred = doc_pred
        self.doc_gold = doc_gold
        self.scores = []


    def corpusCheck(self):
        """checks if files (with correct tags and predcited tags) are of the same length """
        lengthCheck = False
        countGold = 0
        countPred = 0
        with open (self.doc_gold, "rt") as gold, open (self.doc_pred, "rt") as pred:
            for line in gold:
                countGold += 1
            for line in pred:
                countPred += 1
            if countGold == countPred:
                lengthCheck = True
        return lengthCheck



    def corpus (self):
        """returns a list of lists that contain each pair text - tag_gold - tag_predicted """
        corpus_list = []
        count_lists = 0
        count_disagreement = 0
        
        #writes text plus tag of gold standard into a list (creates list of lists)
        with open(self.doc_gold, "rt") as gold:
            for line in gold:
                if line:
                    corpus_list.append(line.split())


        #appends predicted tag to previous list            
        with open(self.doc_pred, "rt") as pred:
            for line in pred:
                if line:
                    if len(line.split()) == 2:
                        corpus_list[count_lists].append(line.split()[1])
                    else: 
                        corpus_list[count_lists].append("X")
                    count_lists += 1
        
        #prints the tags that are different in the two dev-files and counts the number of different tags          
        for tag in corpus_list:
            if tag[1] != tag[2]:
                count_disagreement += 1
                
        return corpus_list
            
            

    def count_tags(self, corpus_list):
        """Takes counts of fp, tp, fn for all tags and writes them into a dictionary """
        dic_counts = {}
        #tagList[1] corresponds to the gold standard's tag, tagList[2] corresponds to the predicted tag

        #creates a dictionary entry for each tag with the values being another dictionary that counts tp, fp, fn
        for tagList in corpus_list:
            if tagList[1] not in dic_counts:
                dic_counts[tagList[1]] = {"tp": 0, "fp": 0, "fn": 0}      
            if tagList[2] not in dic_counts:
                dic_counts[tagList[2]] = {"tp": 0, "fp": 0, "fn": 0}
                
            #adds counts to tp, fp, fn for each tag
            if tagList[1] == tagList[2]:
                dic_counts[tagList[1]]["tp"] += 1   
            else: 
                dic_counts[tagList[1]]["fn"] += 1
                dic_counts[tagList[2]]["fp"] += 1
    
        return dic_counts
          

        
    def microAverage(self, dic_counts):
        """Calculates micro average"""
        sum_tp = 0   #sum of all tp of all tags
        sum_fp = 0   #sum of all fp of all tags
        sum_fn = 0   #sum of all fn of all tags

        #counts tp, fp, fn of all tags
        for tag in dic_counts:
            sum_tp += dic_counts[tag]["tp"]
            sum_fp += dic_counts[tag]["fp"]
            sum_fn += dic_counts[tag]["fn"]

        microPrecision = sum_tp / (sum_tp + sum_fp)
        microRecall = sum_tp / (sum_tp + sum_fn)
        microFScore = 2 * microRecall * microPrecision / (microRecall + microPrecision)

        self.scores.append(microPrecision)
        self.scores.append(microRecall)
        self.scores.append(microFScore)
        #print("Micro Precision: " + str(round(microPrecision, 3)))
        #print("Micro Recall: " + str(round(microRecall, 3)))
        #print("Micro F-Score: " + str(round(microFScore, 3)))

        return self.scores



    def macroAverage(self, dic_counts):
        """Calculates macro average"""
        sum_Prec = 0
        sum_Rec = 0

        for tag in dic_counts:
            try:
            	#calculates precision for each tag
                precision_tag = dic_counts[tag]["tp"] / (dic_counts[tag]["tp"] + dic_counts[tag]["fp"])
                sum_Prec += precision_tag
                #calculates recall for each tag
                recall_tag = dic_counts[tag]["tp"] / (dic_counts[tag]["tp"] + dic_counts[tag]["fn"])
                sum_Rec += recall_tag
            #in case the counts for either tp+fp (precision) or tp+fn (recall) are zero
            except ZeroDivisionError:
                pass

        macroPrecision = sum_Prec / len(dic_counts)
        macroRecall = sum_Rec / len(dic_counts)
        macroFScore = 2 * macroRecall * macroPrecision / (macroPrecision + macroRecall)

        self.scores.append(macroPrecision)
        self.scores.append(macroRecall)
        self.scores.append(macroFScore)
        #print("Macro Precision: " + str(round(macroPrecision, 3)))
        #print("Macro Recall: " + str(round(macroRecall, 3)))
        #print("Macro F-Score: " + str(round(macroFScore, 3)))

        return self.scores


    def main(self):
    	#if Eval is imported as module, only this function needs to be called with the according documents as arguments
    	#checks if documents are of same length
        tags = self.corpus()
        counted_tags = self.count_tags(tags)

        if self.corpusCheck():
            self.microAverage(counted_tags)
            self.macroAverage(counted_tags)
            print(self.scores)
        else:
            print("The documents are of different length.")



if __name__ == '__main__':
    import sys
    doc_gold = sys.argv[1]
    doc_pred = sys.argv[2]

    execute = Evaluation(doc_gold, doc_pred)
    execute.main()

   