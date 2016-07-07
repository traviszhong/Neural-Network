package umt.ml.nnw;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;

public class RunItHere {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		List<Integer> st=Arrays.asList(4,4,3);//This is the configuration of the network
//		List<Integer> st=Arrays.asList(13,8,8,3);
		try {
			Classifier cf=new Classifier("src/iris.csv",st);
			double sum=0;
			for(int i=0;i<10;i++){//10-folds cross validation
				sum+=cf.run();
			}
			System.out.println("Average Accuracy: "+sum/10);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
	}

}
