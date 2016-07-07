package umt.ml.nnw;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.StringTokenizer;

public class Classifier {
	private double[][] data;
	private int[] label;
	private String dataPath;
	private final int SAMPLESIZE=150;
	private List<Integer> structure;
	
	public Classifier(String path,List<Integer> st) throws IOException{
		this.dataPath=path;
		this.data=new double[150][4];
		this.label=new int[150];
		this.structure=st;
		loadData();
	}
	public void loadData() throws IOException{
		File csvData= new File(this.dataPath);
		BufferedReader br=new BufferedReader(new FileReader(csvData));
		String line="";
		int index=-1;
		while((line=br.readLine())!=null){
			if(index==-1){
				index++;continue;
			}
			//Use a tokenizer to split each line of data from the csv file
			StringTokenizer st=new StringTokenizer(line,",");
			for(int i=0;i<4;i++){
				data[index][i]=Double.valueOf(st.nextToken());
			}
			label[index]=Integer.parseInt(st.nextToken());
			index++;
		}
		br.close();
	}
	public double run(){
		shuffle();//randomly change the order of the data
		int epochs=100;//each one has 50 times of training by using the training dataset
		int times=10;//do 10 times
		//build a neural network
		NeuralNet nnet=new NeuralNet(structure);
		nnet.initialize();
		double accuracy=0;//avverage accuracy
		for(int i=0;i<times;i++){
			//do training
			for(int j=0;j<epochs;j++){
				for(int k=0;k<135;k++){
					List<Double> input=Arrays.asList(data[k][0],data[k][1],data[k][2],data[k][3]);
					List<Double> target=new ArrayList<Double>();
					for(int l=0;l<3;l++){
						if(l==label[k]-1) target.add(1d);
						else target.add(0d);
					}
					//training
					nnet.feedForward(input);
					nnet.backProp(target);
				}
			}
			//do test
			int right=0;
			for(int k=0;k<135;k++){
				List<Double> input=Arrays.asList(data[k][0],data[k][1],data[k][2],data[k][3]);
				nnet.feedForward(input);
				int index=0;
				double max=nnet.outputs[0];
				for(int l=1;l<3;l++){
					//find the max output
					if(nnet.outputs[l]>max){
						index=l;
						max=nnet.outputs[l];
					}
				}
				//3 output nodes, the one with max output will be the answer
				if(index+1==label[k]) right++;
			}
			System.out.println("Epoch "+i+" current misclassifications: "+(135-right));
		}
		//do test
		int right=0;
		int[][] cmatrix=new int[3][3];//the confusion matrix
		for(int k=135;k<150;k++){
			List<Double> input=Arrays.asList(data[k][0],data[k][1],data[k][2],data[k][3]);
			nnet.feedForward(input);
			int index=0;
			double max=nnet.outputs[0];
			for(int l=1;l<3;l++){
				//find the max output
				if(nnet.outputs[l]>max){
					index=l;
					max=nnet.outputs[l];
				}
			}
			//3 output nodes, the one with max output will be the answer
			if(index+1==label[k]) right++;
			cmatrix[index][label[k]-1]++;//build the confusion matrix
		}
		System.out.println("Total misclassifications in test dataset: "+(15-right));
		System.out.println("Confusion Matrix:");
		//print the confusion matrix
		for(int a=0;a<3;a++){
			for(int b=0;b<3;b++){
				System.out.print(cmatrix[a][b]);
				if(b==2){
					System.out.println();
				}else System.out.print(",");
			}
		}
		accuracy=right/15.0;
		//return the average of current fold, there are 10 folds in the main function
		return accuracy;
	}
	public void swap(int i,int j){
		//swap the data
		double[] t=data[i];
		data[i]=data[j];
		data[j]=t;
		//swap the class label
		int tc=label[i];
		label[i]=label[j];
		label[j]=tc;
	}
	public void shuffle(){
		//for each sample, generate a number between 1-150 and swap the current data with the data at the random position
		for(int i=0;i<SAMPLESIZE;i++){
			swap(i,(int)(Math.random()*SAMPLESIZE));
		}
	}
}
