

# data = """
# # cascade_13    -numNeg 7000 -numStages 7  -featureType Haar    -numStages 5: Precision: 0.7777777777777778, Recall: 0.7659574468085106   -numStages 7: Precision: 0.8894009216589862, Recall: 0.8853211009174312
# # cascade_17    -numNeg 3500 -numStages 7  -featureType Haar    -numStages 5: Precision: 0.8073394495412844, Recall: 0.8073394495412844   -numStages 7: Precision: 0.9036697247706422, Recall: 0.9036697247706422
# # cascade_14    -numNeg 1750 -numStages 7  -featureType Haar    -numStages 5: Precision: 0.7247706422018348, Recall: 0.7247706422018348   -numStages 7: Precision: 0.8623853211009175, Recall: 0.8623853211009175
# # cascade_12    -numNeg 1750 -numStages 7  -featureType LBP     -numStages 5: Precision: 0.7978723404255319, Recall: 0.7978723404255319   -numStages 7: Precision: 0.8976744186046511, Recall: 0.8853211009174312
# # cascade_16    -numNeg 3500 -numStages 7  -featureType LBP     -numStages 5: Precision: 0.8783570300157978, Recall: 0.8449848024316109   -numStages 7: Precision: 0.9476439790575916, Recall: 0.8302752293577982
# # cascade_15    -numNeg 7000 -numStages 7  -featureType LBP     -numStages 5: Precision: 0.9045226130653267, Recall: 0.8256880733944955   -numStages 7: Precision: 0.9677419354838710, Recall: 0.8256880733944955
# """


# cascade_13:   -numNeg: 7000 -featureType: Haar    -numStages 5: Precision: 0.7777777777777778, Recall: 0.7659574468085106   -numStages 7: Precision: 0.8894009216589862, Recall: 0.8853211009174312
import re
import matplotlib.pyplot as plt

def extract_metrics(data):
    # Extrahiere Precision und Recall für numStages 5 und 7
    precision_5 = re.search(r'-numStages 5: Precision: (\d+\.\d+), Recall: (\d+\.\d+)', data)
    precision_7 = re.search(r'-numStages 7: Precision: (\d+\.\d+), Recall: (\d+\.\d+)', data)

    if precision_5 and precision_7:
        return float(precision_5.group(1)), float(precision_5.group(2)), float(precision_7.group(1)), float(precision_7.group(2))
    else:
        return None

def main():
    # Die bereitgestellten Informationen
    data = """
    cascade_13 -numNeg 7000 -numStages 7 -featureType Haar -numStages 5: Precision: 0.7777777777777778, Recall: 0.7659574468085106 -numStages 7: Precision: 0.8894009216589862, Recall: 0.8853211009174312
    cascade_17 -numNeg 3500 -numStages 7 -featureType Haar -numStages 5: Precision: 0.8073394495412844, Recall: 0.8073394495412844 -numStages 7: Precision: 0.9036697247706422, Recall: 0.9036697247706422
    cascade_14 -numNeg 1750 -numStages 7 -featureType Haar -numStages 5: Precision: 0.7247706422018348, Recall: 0.7247706422018348 -numStages 7: Precision: 0.8623853211009175, Recall: 0.8623853211009175
    cascade_12 -numNeg 1750 -numStages 7 -featureType LBP -numStages 5: Precision: 0.7978723404255319, Recall: 0.7978723404255319 -numStages 7: Precision: 0.8976744186046511, Recall: 0.8853211009174312
    cascade_16 -numNeg 3500 -numStages 7 -featureType LBP -numStages 5: Precision: 0.8783570300157978, Recall: 0.8449848024316109 -numStages 7: Precision: 0.9476439790575916, Recall: 0.8302752293577982
    cascade_15 -numNeg 7000 -numStages 7 -featureType LBP -numStages 5: Precision: 0.9045226130653267, Recall: 0.8256880733944955 -numStages 7: Precision: 0.9677419354838710, Recall: 0.8256880733944955
    """

    # Extrahiere Metrics
    precision_5, recall_5, precision_7, recall_7 = extract_metrics(data)

    if precision_5 is not None and recall_5 is not None and precision_7 is not None and recall_7 is not None:
        # Erstelle einen Graphen
        algorithms = ['cascade_13', 'cascade_17', 'cascade_14', 'cascade_12', 'cascade_16', 'cascade_15']
        precision_values_5 = [0.7777777777777778, 0.8073394495412844, 0.7247706422018348, 0.7978723404255319, 0.8783570300157978, 0.9045226130653267]
        recall_values_5 = [0.7659574468085106, 0.8073394495412844, 0.7247706422018348, 0.7978723404255319, 0.8449848024316109, 0.8256880733944955]
        precision_values_7 = [0.8894009216589862, 0.9036697247706422, 0.8623853211009175, 0.8976744186046511, 0.9476439790575916, 0.9677419354838710]
        recall_values_7 = [0.8853211009174312, 0.9036697247706422, 0.8623853211009175, 0.8853211009174312, 0.8302752293577982, 0.8256880733944955]

# Precision Werte
        plt.subplot(2, 1, 1)
        plt.bar(algorithms, precision_values_5, color='blue', label='Precision - 5 Iterationen')
        plt.ylabel('Precision')
        plt.legend()

        # Recall Werte
        plt.subplot(2, 1, 2)
        plt.bar(algorithms, recall_values_5, color='green', label='Recall - 5 Iterationen')
        plt.xlabel('Algorithmen')
        plt.ylabel('Recall')
        plt.legend()

        plt.suptitle('Vergleich der Precision und des Recall für 5 Iterationen')
        plt.show()

        # Precision und Recall für 7 Iterationen
        plt.figure()

        # Precision Werte
        plt.subplot(2, 1, 1)
        plt.bar(algorithms, precision_values_7, color='blue', label='Precision - 7 Iterationen')
        plt.ylabel('Precision')
        plt.legend()

        # Recall Werte
        plt.subplot(2, 1, 2)
        plt.bar(algorithms, recall_values_7, color='green', label='Recall - 7 Iterationen')
        plt.xlabel('Algorithmen')
        plt.ylabel('Recall')
        plt.legend()

        plt.suptitle('Vergleich der Precision und des Recall für 7 Iterationen')
        plt.show()


    else:
        print("Fehler beim Extrahieren der Metriken.")

if __name__ == "__main__":
    main()



