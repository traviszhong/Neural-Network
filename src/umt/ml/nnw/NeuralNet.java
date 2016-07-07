package umt.ml.nnw;

import java.util.ArrayList;
import java.util.List;
/**
 * 
 * @author Zhong Ziyue
 *
 * @email zhongzy@strongit.com.cn
 * 
 * May 1, 2016
 */
public class NeuralNet {
	private List<List<Neuron>> network;
	double[] inputs;
	double[] outputs;
	private int numOfLayers;
	private List<Integer> numOfNeuron;
	private double eta;
	private double defaultWeight;
	
	/**
	 * Constructor
	 * @param neuronNum
	 */
	public NeuralNet(List<Integer> neuronNum){
		this.network=new ArrayList<List<Neuron>>();
		this.inputs=new double[4];
		this.outputs=new double[3];
		this.numOfLayers=neuronNum.size();
		this.numOfNeuron=neuronNum;
		//The default eta is 0.05
		this.eta=0.05;
		//the default weight is 0.05
		this.defaultWeight=0.05;
	}
	/**
	 * Do the set up everything to default value
	 */
	public void initialize(){
		this.network.clear();
		//set up layers
		for(int i=0;i<numOfLayers;i++) network.add(new ArrayList<Neuron>());
		//loop over the network
		for(int i=0;i<numOfLayers;i++){
			int curNum=numOfNeuron.get(i);
			for(int j=0;j<curNum;j++){
				//if it the node at the first layer
				if(i==0){
					//# of weights will be 4
					network.get(i).add(new Neuron(i,j,eta,4,numOfNeuron.get(i+1)));
					//set the weights to default value
					for(int k=0;k<4;k++) network.get(i).get(j).weights[k]=defaultWeight;
				//if it is in the output layer
				}else if(i==numOfLayers-1){
					//the output vector has 3 values
					network.get(i).add(new Neuron(i,j,eta,numOfNeuron.get(i-1),3));
					for(int k=0;k<numOfNeuron.get(i-1);k++) network.get(i).get(j).weights[k]=defaultWeight;
					network.get(i).get(j).inOutputLayer=true;
				//others
				}else{
					network.get(i).add(new Neuron(i,j,eta,numOfNeuron.get(i-1),numOfNeuron.get(i+1)));
					for(int k=0;k<numOfNeuron.get(i-1);k++) network.get(i).get(j).weights[k]=defaultWeight;
				}
			}
		}
	}
	/**
	 * This function do the feed forward in the whole network
	 * @param record: the input of a record from the dataset
	 */
	public void feedForward(List<Double> record){
		//convert a list into a array
		for(int i=0;i<4;i++) this.inputs[i]=record.get(i);
		//set input values to the first layer
		for(int i=0;i<numOfNeuron.get(0);i++) network.get(0).get(i).inputs=this.inputs;
		//for each node in the network do the feed forward within the neuron
		for(int i=0;i<numOfLayers;i++){
			for(int j=0;j<numOfNeuron.get(i);j++){
				network.get(i).get(j).feedForward();
				double curOut=network.get(i).get(j).output;
				if(i==numOfLayers-1){
					outputs[j]=curOut;
				}else{
					for(int k=0;k<numOfNeuron.get(i+1);k++){
						network.get(i+1).get(k).inputs[j]=curOut;
					}
				}
			}
		}
	}
	/**
	 * This function do the backprop in the whole network
	 * @param target
	 */
	public void backProp(List<Double> target){
		//convert the list of target into a array
		for(Neuron n:network.get(this.numOfLayers-1)){
			for(int i=0;i<3;i++){
				n.backDelta[i]=target.get(i);
			}
		}
		//loop through each neuron in the network, do the back prop in each neuron
		for(int i=this.numOfLayers-1;i>=0;i--){
			List<Neuron> curLyr=network.get(i);
			for(int j=0;j<curLyr.size();j++){
				Neuron curNode=curLyr.get(j);
				curNode.backProp();
				if(i!=0){
					List<Neuron> prevLyr=network.get(i-1);
					for(int k=0;k<prevLyr.size();k++){
						//put the delta of current node to the back delta list of each node in the previous
						// layer
						prevLyr.get(k).backDelta[j]=curNode.delta*curNode.weights[k];
					}
				}
			}
		}
		//updata the weights
		for(int i=0;i<numOfLayers;i++){
			//loop through each neuron node in the network
			List<Neuron> curLyr=network.get(i);
			for(int j=0;j<curLyr.size();j++){
				Neuron curNode=curLyr.get(j);
				for(int k=0;k<curNode.weights.length;k++){
					//update the weights of each node
					curNode.weights[k]+=curNode.eta*curNode.delta*curNode.inputs[k];
				}
				curNode.biasWeight+=curNode.eta*curNode.delta;
			}
		}
	}
}
