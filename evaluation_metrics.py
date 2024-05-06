def calculate_hss(true_positives, false_positives, true_negatives, false_negatives):
    E = ((true_positives + false_negatives) * (true_positives + false_positives) + 
         (false_negatives + true_negatives) * (false_positives + true_negatives)) / (true_positives + true_negatives + false_positives + false_negatives)
    HSS = (true_positives + true_negatives - E) / (true_positives + true_negatives + false_positives + false_negatives - E)
    return HSS

def calculate_tss(true_positives, false_positives, true_negatives, false_negatives):
    TSS = (true_positives / (true_positives + false_negatives)) - (false_positives / (false_positives + true_negatives))
    return TSS
