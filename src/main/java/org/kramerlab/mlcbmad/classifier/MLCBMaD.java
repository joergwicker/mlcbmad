/*
 * MLC-BMaD Copyright (C) 2009-2012 Joerg Wicker (wicker@uni-mainz.de)
 * 
 * This program is free software: you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 * 
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
 * details.
 * 
 * You should have received a copy of the GNU General Public License along with
 * this program. If not, see <http://www.gnu.org/licenses/>.
 */

package org.kramerlab.mlcbmad.classifier;


import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.Vector;
import mulan.classifier.MultiLabelLearner;
import mulan.classifier.MultiLabelOutput;
import mulan.classifier.transformation.BinaryRelevance;
import mulan.classifier.transformation.MultiLabelStacking;
import mulan.classifier.transformation.TransformationBasedMultiLabelLearner;
import mulan.data.LabelNode;
import mulan.data.LabelNodeImpl;
import mulan.data.LabelsMetaDataImpl;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluator;
import mulan.evaluation.MultipleEvaluation;
import org.kramerlab.bmad.algorithms.BooleanMatrixDecomposition;
import org.kramerlab.bmad.general.Tuple;
import weka.classifiers.Classifier;
import weka.classifiers.evaluation.NominalPrediction;
import weka.classifiers.trees.RandomForest;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.core.TechnicalInformation;
import weka.core.Utils;
import weka.core.converters.ArffLoader;
import weka.core.converters.ArffSaver;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.instance.SparseToNonSparse;

/**
 * Describe class <code>MLCBMaD</code> here.
 * 
 * @author <a href="mailto:wicker@uni-mainz.de">Joerg Wicker</a>
 * @version 1.0
 */

public class MLCBMaD extends TransformationBasedMultiLabelLearner {
    
    public static void main( String[] args) throws Exception {
	String datasetbase = Utils.getOption("dataset", args);
        
	MultiLabelInstances dataset = new MultiLabelInstances(datasetbase
							      + ".arff", datasetbase + ".xml");
        
	RandomForest rf = new RandomForest();
        
        
        

        for (double t = 0.9; t >= 0.1; t -= 0.1) {
            for (int k = dataset.getLabelIndices().length-1; k >= 2; k--) {
		MLCBMaD mlcbmad = new MLCBMaD(rf);
		mlcbmad.setDebug(true);
        
		mlcbmad.setK(k);
		mlcbmad.setT(t);
		
		Evaluator eval = new Evaluator();
		MultipleEvaluation res = eval.crossValidate(mlcbmad,
							    dataset, 3);
                System.out.println("\n======\nt=" + t + "\nk=" + k + "\n"
				   + res.toString());
            }
        }
    }
    

    public Instances getDecomp(){
	return this.decomp;
    }
    public Instances getUpper(){
	return this.uppermatrix;
    }


    
    protected BinaryRelevance basebr;
    protected Classifier baseClassifier;
    protected Instances decomp;
    protected Instances decompLabels;
    protected int[] features;
    protected Instances featuresAndDecomp;
    protected AttributeSelection[] filter;
    protected int[] indicesLabelsDecomp;
    protected int[] labelsdecomp;
    
    protected Instances uppermatrix;
    
    protected int k;
    protected double t;
    
    /**
     * An empty constructor
     */
    public MLCBMaD() {
        // empty
    }
    
    /**
     * A constructor with 2 arguments
     * 
     * @param baseClassifier
     *            the classifier used in the base-level
     * @throws Exception
     */
    public MLCBMaD( Classifier baseClassifier) throws Exception {
        super(baseClassifier);
        this.baseClassifier = baseClassifier;
    }
    
    /**
     * Returns an instance of a TechnicalInformation object, containing detailed
     * information about the technical background of this class, e.g., paper
     * reference or book this class is based on.
     * 
     * @return the technical information about this class
     */
    @Override
    public TechnicalInformation getTechnicalInformation() {
        
	TechnicalInformation result;
        result = new TechnicalInformation(Type.PROCEEDINGS);
        result.setValue(Field.AUTHOR, "Wicker, Joerg and Pfahringer, Bernhard and Kramer, Stefan");
        result.setValue(Field.TITLE, "Multi-Label Classification using Boolean Matrix Decomposition");
        result.setValue(Field.YEAR, "2012");
        result.setValue(Field.PAGES, "498-505");        
	result.setValue(Field.LOCATION, "Riva del Garda");
        result.setValue(Field.PUBLISHER , "ACM");        

        result.setValue(Field.SERIES, "ACM Symposium on Applied Computing");
        return result;


    }




    
    /**
     * Returns a string describing classifier.
     * 
     * @return a description suitable for displaying in a future
     *         explorer/experimenter gui
     */
    public String globalInfo() {
        return "";
    }

    
    
    public void saveObject( String filename) {
        try {
	    ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(filename));
            out.writeObject(this);
        } catch ( IOException ex) {
	    ex.printStackTrace();
        }
    }
    
    

    public void setK(int k){
	this.k = k;
    }

    public void setT(double t){
	this.t = t;
    }
    


    /**
     * Builds the classifier.
     * 
     * @param trainingSet
     * @throws Exception
     */
    protected void buildInternal( MultiLabelInstances trainingSet)
	throws Exception {
        
	// This step is necessary as there are problems with the 
	// attribute indexes in WEKA when merging instances
        Instances train = this.copyInstances(trainingSet.getDataSet());
        
	 
	
	debug("Learning model...");
	debug("Parameter Setting k = " + k + 
			   " and t = " + t + 
			   " ..."  );
				
        // remove the features, so we make a matrix decomposition only of
        // the labels
        
	Remove rem0 = new Remove();
	int[] features0 = trainingSet.getFeatureIndices();
        rem0.setAttributeIndicesArray(features0);
        rem0.setInputFormat(train);
        train = Filter.useFilter(train, rem0);
        
        Instances decompData;
        
        // lets do the decomposition

        
        // first save the arff in non sparse form
        
	SparseToNonSparse spfilter = new SparseToNonSparse();
        spfilter.setInputFormat(train);
	Instances out = Filter.useFilter(train, spfilter);
        

	
	BooleanMatrixDecomposition bmd = BooleanMatrixDecomposition.BEST_CONFIGURED(this.t);
	Tuple<Instances, Instances> res = bmd.decompose(out, this.k);
	
	decompData = res._1;
        uppermatrix = res._2;
        

        // get indices
        
        decomp = decompData;
        
	int[] features = trainingSet.getFeatureIndices();
        
	int[] decompindices = new int[decompData.numAttributes()];
        
        int countf = 0;
        for (int i = features.length; i < (decompData.numAttributes() + features.length); i++) {
            decompindices[countf] = i;
            countf++;
        }
        labelsdecomp = decompindices;
        
        // get features from training set
        
	Instances copied = this.copyInstances(trainingSet.getDataSet());
        
	Remove rem = new Remove();
        
        rem.setAttributeIndicesArray(features);
        rem.setInvertSelection(true);
        rem.setInputFormat(copied);
        
	Instances onlyFeatures = Filter.useFilter(copied, rem);
        
        // merge features with matrix decomposition
	
	if (onlyFeatures.numInstances() != decompData.numInstances()) {
	    //sthg went wrong when decomposing
	    throw new Exception("Problem when decomposing");
	}
        
        
	
        featuresAndDecomp = Instances.mergeInstances(onlyFeatures, this.copyInstances(decompData));
        
	Instances trainset = featuresAndDecomp;
        
	LabelsMetaDataImpl trainlmd = new LabelsMetaDataImpl();
        for ( int lab : labelsdecomp) {
	    LabelNode lni = new LabelNodeImpl(trainset.attribute(lab)
					      .name());
            trainlmd.addRootNode(lni);
        }
        
	MultiLabelInstances trainMulti = new MultiLabelInstances(
								 trainset, trainlmd);
        

	
        // build br for decomposed label prediction
        
        basebr = new BinaryRelevance(baseClassifier);
        
        basebr.build(trainMulti);
	
	debug("Model trained... all done.");

    }
    
    @Override
    protected MultiLabelOutput makePredictionInternal( Instance instance)
	throws Exception {

	MultiLabelOutput baseout = basebr.makePrediction(instance);
        
	boolean[] bipartition = new boolean[uppermatrix.numAttributes()];
	double[] confidences = new double[uppermatrix.numAttributes()];
	

        for (int i = 0; i < bipartition.length; i++) {
	    int index1 = uppermatrix.attribute(i).value(0).equals("0") ? 1 : 0;
            for (int j = 0; j < baseout.getBipartition().length; j++) {
		double matval = uppermatrix.instance(j).value(i);
                bipartition[i] = bipartition[i]
		    || (baseout.getBipartition()[j] && (matval == index1));
            
		if (matval == index1) {
		    confidences[i] = Math.max(confidences[i], baseout.getConfidences()[j]);

		}


	    }
            
        }
	MultiLabelOutput mlo = new MultiLabelOutput(bipartition, confidences);
        return mlo;
    }
    

    /**
     * This is a rather pointless method, but it avoids a bug in WEKA. The Instances are
     * serialized and read back in, otherwise, there are problems with the indices of the 
     * Instances.
     * 
     * TODO fixme
     *
     * @param inst The instances to copy.
     * @return A copy of the instances.
     */
    private Instances copyInstances(Instances inst) throws Exception{

	ByteArrayOutputStream bos = new ByteArrayOutputStream();
	
	ObjectOutputStream out = new ObjectOutputStream(bos);

	out.writeObject(inst);

	//De-serialization of object

	ByteArrayInputStream bis = new   ByteArrayInputStream(bos.toByteArray());

	ObjectInputStream in = new ObjectInputStream(bis);

	Instances copied = (Instances) in.readObject();

	return copied;
    }
    
}
