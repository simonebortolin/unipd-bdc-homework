import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import scala.Int;
import scala.Tuple2;

import java.io.IOException;
import java.io.Serializable;
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

        long numdocs;
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
                        Collections.singletonList(new Tuple2<>(it._1, 1)).iterator()
                ).groupByKey().mapPartitionsToPair((it) -> {
                    Map<String, Integer>  map = new HashMap<>();

                    for (; it.hasNext(); ) {
                        Tuple2<String, Iterable<Integer>> i = it.next();
                        int s = 0;
                        for (int j : i._2) {
                            s += j;
                        }
                        map.put(i._1, map.getOrDefault(i._1,0) + s);
                    }
                    List<Tuple2<String, Integer>> pairs = new ArrayList<>();
                    for (Map.Entry<String, Integer> e : map.entrySet()) {
                        pairs.add(new Tuple2<>(e.getKey(), e.getValue()));
                    }
                    return pairs.iterator();
                });

        System.out.println("productPopularity1");
        /*List<Tuple2<String ,Integer>> output=new ArrayList<>(H);
        for(int i=0;i<H;i++) {
            productPopularity1.max(Comparator.comparingInt(a -> a._2));
        }*/
        if(H != 0) {
            System.out.println("Top "+H+" Products and their Popularities");
            for (Tuple2<String, Integer> item : productPopularity1.takeOrdered(H, new SerializableComparator<Tuple2<String, Integer>>() {
                @Override
                public int compare(Tuple2<String, Integer> o1, Tuple2<String, Integer> o2) {
                    if(!Objects.equals(o2._2, o1._2))
                        return o2._2 - o1._2;
                    return o2._1.compareTo(o1._1);
                }
            })) {
                System.out.print("Product: " + item._1 + " Popularity: " + item._2 + "; ");
            }
            System.out.println();
        } else {
            //productPopularity1.sortByKey();
            for (Tuple2<String, Integer> item : productPopularity1.sortByKey().collect()) {
                System.out.print("Product: " + item._1 + " Popularity: " + item._2 + "; ");
            }
            System.out.println();
        }

       /* JavaPairRDD<String,Integer> productPopularity1 = productCustomer.flatMapToPair((it) ->
            Collections.singletonList(new Tuple2<>(it._1, it._2)).iterator() // not working
        ).groupByKey().flatMapToPair((it) ->
             Collections.singletonList(new Tuple2<>(it._1, 1)).iterator() // not working
        ).reduceByKey(Integer::sum);
        System.out.println("productPopularity1");
        for(Tuple2<String, Integer> item : productPopularity1.collect()) {
            System.out.print("Product: "+ item._1 + " Popularity: "+ item._2+",");
        }
        System.out.println();*/


    }

    public interface SerializableComparator<T> extends Comparator<T>, Serializable {

        static <T> SerializableComparator<T> serialize(SerializableComparator<T> comparator) {
            return comparator;
        }

    }

}