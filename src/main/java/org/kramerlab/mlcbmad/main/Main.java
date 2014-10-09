package org.kramerlab.mlcbmad.main;

import java.util.ArrayList;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluator;
import mulan.evaluation.MultipleEvaluation;
import mulan.evaluation.measure.ExampleBasedAccuracy;
import mulan.evaluation.measure.Measure;
import org.kramerlab.mlcbmad.classifier.MLCBMaD;
import weka.classifiers.trees.RandomForest;

public class Main{

    public static void main(String[] args) throws Exception{

	// maybe llog, medical, enron

	System.err.println("Running...");
	
	MultiLabelInstances mli = new MultiLabelInstances(args[0] + ".arff", args[0] + ".xml");
	
	
	
	    
	for (int k = 2; k < mli.getNumLabels(); k++) {
	    for (double t = 0.1; t <= 1.0; t+=0.1) {
		MLCBMaD classif = new MLCBMaD(new RandomForest());
		
		classif.setK(k);
		classif.setT(t);
		
		    
		Evaluator eval = new Evaluator();
		
		ArrayList<Measure> mes = new ArrayList<Measure>();
		
		mes.add(new ExampleBasedAccuracy());

		
		MultipleEvaluation meval = eval.crossValidate(classif,
							      mli, 
							      mes,
							      5);
		
		meval.calculateStatistics();
		
		double curperf  = meval.getMean("Example-Based Accuracy");
		System.out.println(k + " " + t+ " " + curperf);
		
	    }
	}
	
	
	
    }


}
