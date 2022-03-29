import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import scala.Int;
import scala.Tuple2;

import java.io.IOException;
import java.util.*;

public class GP021 {

    public static void main(String[] args) throws IOException {

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // CHECKING NUMBER OF CMD LINE PARAMETERS
        // Parameters are: num_partitions, <path_to_file>
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        if (args.length != 4) {
            throw new IllegalArgumentException("USAGE: num_partitions file_path");
        }

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // SPARK SETUP
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        SparkConf conf = new SparkConf(true).setAppName("WordCount");
        JavaSparkContext sc = new JavaSparkContext(conf);
        sc.setLogLevel("WARN");

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // INPUT READING
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        // Read number of partitions
        int K = Integer.parseInt(args[0]);
        int H = Integer.parseInt(args[1]);
        String S = args[2];

        // Read input file and subdivide it into K random partitions
        JavaRDD<String> rawData = sc.textFile(args[3]).repartition(K).cache();

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // SETTING GLOBAL VARIABLES
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        long numdocs, numwords;
        numdocs = rawData.count();
        System.out.println("Number of rows = " + numdocs);

        JavaPairRDD<String, Integer> productCustomer = rawData.flatMapToPair((it) -> {
            String[] items = it.split(",");
            String theP = items[1];
            String theC = items[6];
            String theQ = items[3];
            String theS = items[7];
            if(Integer.parseInt(theQ) > 0 && (S.equals("all") || S.equals(theS)))
                return Collections.singletonList(new Tuple2<>(new Tuple2<>(theP, Integer.parseInt(theC)),items)).iterator();
            return Collections.emptyListIterator();
        }).groupByKey().flatMapToPair((it)-> Collections.singletonList(it._1).iterator());

        System.out.println("Product-Customer Pairs = " + productCustomer.count());

        JavaPairRDD<String,Integer> productPopularity1 = productCustomer.flatMapToPair((it) ->
            Collections.singletonList(new Tuple2<>(it._1, it._2)).iterator() // not working
        ).groupByKey().flatMapToPair((it) ->

             Collections.singletonList(new Tuple2<>(it._1, 1)).iterator() // not working
        ).reduceByKey(Integer::sum);

        System.out.println("productPopularity1");

        for(Tuple2<String, Integer> item : productPopularity1.collect()) {
            System.out.print("Product: "+ item._1 + " Popularity: "+ item._2+",");
        }
        System.out.println();


    }

}
