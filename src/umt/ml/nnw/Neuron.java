package umt.ml.nnw;
/**
 * 
 * @author Zhong Ziyue
 *
 * @email zhongzy@strongit.com.cn
 * 
 * May 1, 2016
 */
public class Neuron {
	private int layerId;
	private int nodeId;
	double[] weights;
	double[] inputs;
	double[] backDelta;
	double output;
	double delta;
	double eta;
	double biasWeight;
	boolean inOutputLayer;
	/**
	 * Constructor of a Neuron
	 * @param layer: layer id
	 * @param node: node id
	 * @param et: eta
	 * @param prev: # of nodes in the previous layer
	 * @param next: # of nodes in the next layer
	 */
	public Neuron(int layer,int node,double et,int prev,int next){
		this.layerId=layer;
		this.nodeId=node;
		this.weights=new double[prev];
		this.inputs=new double[prev];
		this.backDelta=new double[next];
		this.output=0;
		this.delta=0;
		this.eta=et;
		this.biasWeight=0.05;
		this.inOutputLayer=false;
	}
	/**
	 * Sigmoid function
	 * @param val
	 * @return
	 */
	public double sigmoid(double val){
		return 1d / (1d + Math.exp(-val));
	}
	/**
	 * This function takes an input, and do the feed forward within a single neuron
	 */
	public void feedForward(){
		double sum=0;
		int n=this.inputs.length;
		//go through each input, use the input times the weight and get the sum
		for(int i=0;i<n;i++){
			sum+=weights[i]*inputs[i];
		}
		//add the bias to the sum
		sum+=biasWeight;
		//use sigmoid function to calculate the output
		output=sigmoid(sum);
	}
	/**
	 * This function do the back prop with a single neuron
	 */
	public void backProp(){
		//to see if the current neuron is in the output layer
		if(this.inOutputLayer){
			this.delta=this.output*(1-this.output)*(backDelta[this.nodeId]-this.output);
		}else{
			this.delta=this.output*(1-this.output);
			double sum=0;
			//if the current neuron is not in the output layer use another formula
			for(double b:backDelta) sum+=b;
			this.delta*=sum;
		}
	}
	
}
