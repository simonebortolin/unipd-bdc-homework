import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaDoubleRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.BLAS;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import scala.Tuple2;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class G021HW3
{

    static List<Double> rs = new ArrayList<>();
// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// MAIN PROGRAM
// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    public static void main(String[] args) throws Exception {

        if (args.length != 4) {
            throw new IllegalArgumentException("USAGE: filepath k z L");
        }

        // ----- Initialize variables
        String filename = args[0];
        int k = Integer.parseInt(args[1]);
        int z = Integer.parseInt(args[2]);
        int L = Integer.parseInt(args[3]);
        long start, end; // variables for time measurements

        // ----- Set Spark Configuration
        Logger.getLogger("org").setLevel(Level.OFF);
        Logger.getLogger("akka").setLevel(Level.OFF);
        SparkConf conf = new SparkConf(true).setAppName("MR k-center with outliers");
        JavaSparkContext sc = new JavaSparkContext(conf);
        sc.setLogLevel("WARN");

        // ----- Read points from file
        start = System.currentTimeMillis();
        JavaRDD<Vector> inputPoints = sc.textFile(args[0], L)
                .map(x-> strToVector(x))
                .repartition(L)
                .cache();
        long N = inputPoints.count();
        end = System.currentTimeMillis();

        // ----- Pring input parameters
        System.out.println("File : " + filename);
        System.out.println("Number of points N = " + N);
        System.out.println("Number of centers k = " + k);
        System.out.println("Number of outliers z = " + z);
        System.out.println("Number of partitions L = " + L);
        System.out.println("Time to read from file: " + (end-start) + " ms");

        // ---- Solve the problem
        ArrayList<Vector> solution = MR_kCenterOutliers(inputPoints, k, z, L);

        // ---- Compute the value of the objective function
        start = System.currentTimeMillis();
        double objective = computeObjective(inputPoints, solution, z);
        end = System.currentTimeMillis();
        System.out.println("Objective function = " + objective);
        System.out.println("Time to compute objective function: " + (end-start) + " ms");

    }

// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// AUXILIARY METHODS
// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// Method strToVector: input reading
// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    public static Vector strToVector(String str) {
        String[] tokens = str.split(",");
        double[] data = new double[tokens.length];
        for (int i=0; i<tokens.length; i++) {
            data[i] = Double.parseDouble(tokens[i]);
        }
        return Vectors.dense(data);
    }

// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// Method euclidean: distance function
// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    public static double euclidean(Vector a, Vector b) {
        return Math.sqrt(Vectors.sqdist(a, b));
    }

// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// Method MR_kCenterOutliers: MR algorithm for k-center with outliers
// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    public static ArrayList<Vector> MR_kCenterOutliers (JavaRDD<Vector> points, int k, int z, int L)
    {

        long start, end;

        //------------- ROUND 1 ---------------------------

        start = System.currentTimeMillis();

        JavaRDD<Tuple2<Vector,Long>> coreset = points.mapPartitions(x ->
        {
            ArrayList<Vector> partition = new ArrayList<>();
            while (x.hasNext()) partition.add(x.next());
            ArrayList<Vector> centers = kCenterFFT(partition, k+z+1);
            ArrayList<Long> weights = computeWeights(partition, centers);
            ArrayList<Tuple2<Vector,Long>> c_w = new ArrayList<>();
            for(int i =0; i < centers.size(); ++i)
            {
                Tuple2<Vector, Long> entry = new Tuple2<>(centers.get(i), weights.get(i));
                c_w.add(i,entry);
            }
            return c_w.iterator();
        });
        // END OF ROUND 1



        //------------- ROUND 2 ---------------------------



        ArrayList<Tuple2<Vector, Long>> elems = new ArrayList<>((k+z)*L);
        elems.addAll(coreset.collect());
        end = System.currentTimeMillis();

        System.out.println("Time Round 1: " + (end-start) + "ms");

        start = System.currentTimeMillis();
        ArrayList<Vector> v=new ArrayList<>();
        ArrayList<Long> w=new ArrayList<>();
        for (Tuple2<Vector,Long> p:elems) {
            v.add(p._1());
            w.add(p._2());
        }

        end = System.currentTimeMillis();

        System.out.println("Time Round 2: " + (end-start) + "ms");

        return (ArrayList<Vector>) seqWeightedOutliers(v,w,k,z,2);
    }

// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// Method kCenterFFT: Farthest-First Traversal
// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    public static ArrayList<Vector> kCenterFFT (ArrayList<Vector> points, int k) {

        final int n = points.size();
        double[] minDistances = new double[n];
        Arrays.fill(minDistances, Double.POSITIVE_INFINITY);

        ArrayList<Vector> centers = new ArrayList<>(k);

        Vector lastCenter = points.get(0);
        centers.add(lastCenter);
        double radius =0;

        for (int iter=1; iter<k; iter++) {
            int maxIdx = 0;
            double maxDist = 0;

            for (int i = 0; i < n; i++) {
                double d = euclidean(points.get(i), lastCenter);
                if (d < minDistances[i]) {
                    minDistances[i] = d;
                }

                if (minDistances[i] > maxDist) {
                    maxDist = minDistances[i];
                    maxIdx = i;
                }
            }

            lastCenter = points.get(maxIdx);
            centers.add(lastCenter);
        }
        return centers;
    }

// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// Method computeWeights: compute weights of coreset points
// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    public static ArrayList<Long> computeWeights(ArrayList<Vector> points, ArrayList<Vector> centers)
    {
        Long weights[] = new Long[centers.size()];
        Arrays.fill(weights, 0L);
        for(int i = 0; i < points.size(); ++i)
        {
            double tmp = euclidean(points.get(i), centers.get(0));
            int mycenter = 0;
            for(int j = 1; j < centers.size(); ++j)
            {
                if(euclidean(points.get(i),centers.get(j)) < tmp)
                {
                    mycenter = j;
                    tmp = euclidean(points.get(i), centers.get(j));
                }
            }
            // System.out.println("Point = " + points.get(i) + " Center = " + centers.get(mycenter));
            weights[mycenter] += 1L;
        }
        ArrayList<Long> fin_weights = new ArrayList<>(Arrays.asList(weights));
        return fin_weights;
    }

// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// Method SeqWeightedOutliers: sequential k-center with outliers
// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    //
    // ****** ADD THE CODE FOR SeqWeightedOuliers from HW2
    //
    private static List<Integer> cb(List<Vector> Z, Vector center, double v) {
        List<Integer> list = new ArrayList<>();
        for (int i = 0; i < Z.size(); i++) {
            if (Math.sqrt(Vectors.sqdist(center, Z.get(i))) <= v) {
                list.add(i);
            }
        }
        return list;
    }
    public static double getMinD(int size, ArrayList<Vector> inputPoints) {
        double d = Double.MAX_VALUE;
        for (int i = 0; i < size - 1; i++) {
            for (int j = i + 1; j < size; j++) {
                d = Math.min(d, Math.sqrt(Vectors.sqdist(inputPoints.get(i), inputPoints.get(j))));
            }
        }
        return d;
    }
    public static List<Vector> seqWeightedOutliers(ArrayList<Vector> inputPoints, ArrayList<Long> w, int k, int z, int alpha) {
        double r = getMinD(k + z + 1, inputPoints) / 2;
        System.out.println("Initial guess "+r);
        rs.add(r);
        //Vector[] P = inputPoints.toArray(new Vector[0]);


        while (true) { // to prevent infinite loops
            List<Vector> Z = new ArrayList<>(inputPoints);
            List<Long> Zw = new ArrayList<>(w);
            List<Vector> S = new ArrayList<>(k);

            while (S.size() < k && Zw.stream().reduce(Long::sum).orElse(0L) > 0) {
                double max = 0;
                Vector newCenter = null;
                for (Vector x : inputPoints) {
                    List<Integer> ball = cb(Z, x, (1 + 2 * alpha) * r);
                    long ballWeight = ball.stream().map(w::get).reduce(Long::sum).orElse(0L);
                    if (ballWeight > max) {
                        max = ballWeight;
                        newCenter = x;
                    }
                }
                if (newCenter != null) {
                    S.add(newCenter);
                    List<Integer> ball = cb(Z, newCenter, (3 + 4 * alpha) * r);
                    ball.sort((a, b) -> a - b);
                    for (int i = ball.size() - 1; i >= 0; i--) {
                        int index = ball.get(i);
                        Z.remove(index);
                        Zw.remove(index);
                    }
                }
            }

            if (Zw.stream().reduce(Long::sum).orElse(0L) <= z) {
                return S;
            } else {
                r *= 2;
                rs.add(r);
            }
            System.out.println("Final guess "+r);
            System.out.println("Number of guess "+rs.size());
        }

    }


// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// Method computeObjective: computes objective function
// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    public static double computeObjective (JavaRDD<Vector> points, ArrayList<Vector> centers, int z)
    {
       List<Vector> l = points.collect();
       ArrayList<Double> d = new ArrayList<>(l.size());


      for (int i = 0, size = l.size(); i < size; i++) {
           double min = Double.MAX_VALUE;
           for (Vector j : centers) {
               min = Math.min(min, Math.sqrt(Vectors.sqdist(l.get(i), j)));
           }
           d.add(min);

       }
      d.sort((a, b) -> -Double.compare(a, b));
       if (z < d.size())
           return d.get(z);
        return d.get(d.size() - 1);

        //
        // ****** ADD THE CODE FOR computeObjective
        //


    }

}